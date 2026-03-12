import numpy as np
from transformers import BertTokenizer
import sys
import os
import argparse
# 保证项目根目录在 path 中，便于正确 import macbert、SimCSE、span_src 等
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import json
import torch
import faiss
import time

# ONNX 推理：若 onnx_output/ 下存在三个 .onnx 文件则优先使用 ONNX
ONNX_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'onnx_output')
ONNX_MAX_LENGTH = 128

def _use_onnx():
    for name in ('macbert.onnx', 'simcse.onnx', 'span.onnx'):
        if not os.path.exists(os.path.join(ONNX_OUTPUT_DIR, name)):
            return False
    return True

try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except ImportError:
    _HAS_ONNXRUNTIME = False

from SimCSE.model import TextBackbone
from span_src.config import Args
from span_src.model import build_model
from utils import generate_input, span_decode, edit_distance
from macbert.utils import Inference as MacBERTInference

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def _generate_input_onnx(text, tokenizer, max_length=ONNX_MAX_LENGTH):
    """生成 ONNX 推理用的 numpy 输入，固定 max_length。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("_generate_input_onnx: text 不能为空，请在上层跳过空实体后再调用。")
    enc = tokenizer.encode_plus(
        list(text),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        is_pretokenized=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return {k: v.numpy().astype(np.int64) for k, v in enc.items()}


class Correction():
    def __init__(self, macbert_model_path=None, macbert_bert_path=None, use_onnx=None):
        self.use_onnx = (use_onnx if use_onnx is not None else (_HAS_ONNXRUNTIME and _use_onnx()))
        if self.use_onnx:
            logger.info('使用 ONNX 推理（onnx_output/ 下已存在三个 .onnx 模型）')
        self.dim = 128
        # 0: MacBERT 通用纠错
        self.macbert_model = None
        self.macbert_onnx_session = None
        self.macbert_tokenizer = None
        self.init_macbert(macbert_model_path, macbert_bert_path)
        # 1: NER（span）
        self.init_ner()
        # 2: SimCSE
        self.init_simcse()
        self.init_index()

    # 初始化 MacBERT 通用纠错模型（若权重/ONNX 不存在则跳过）
    def init_macbert(self, model_path=None, bert_path=None):
        if bert_path is None:
            bert_path = os.path.join(PROJECT_ROOT, 'model', 'macbert_org')
        if not os.path.exists(bert_path):
            logger.warning('MacBERT 预训练权重目录未找到 ({}), 跳过通用纠错初始化。'.format(bert_path))
            return
        onnx_path = os.path.join(ONNX_OUTPUT_DIR, 'macbert.onnx')
        if self.use_onnx and _HAS_ONNXRUNTIME and os.path.exists(onnx_path):
            logger.info('initialize MacBERT (ONNX)......')
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.macbert_onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            self.macbert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            return
        model_path = model_path or os.path.join(PROJECT_ROOT, 'macbert', 'output', 'bert4csc', 'best_model.pt')
        if not os.path.exists(model_path):
            logger.warning('MacBERT 权重未找到 ({}), 将仅使用 NER+SimCSE 专有名称纠错。'.format(model_path))
            return
        logger.info('initialize MacBERT general correction model......')
        self.macbert_model = MacBERTInference(model_path=model_path, bert_path=bert_path)

    # 初始化NER模型的函数（tokenizer、ent2id 与 ONNX/PyTorch 共用）
    def init_ner(self):
        logger.info('initialize ner model......')
        opt = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, 'model', 'bert-base-chinese'))
        with open(os.path.join(PROJECT_ROOT, 'span_src', 'data', 'span_ent2id.json'), encoding='utf-8') as f:
            self.ent2id = json.load(f)
        self.id2ent = {v: k for k, v in self.ent2id.items()}

        span_onnx_path = os.path.join(ONNX_OUTPUT_DIR, 'span.onnx')
        if self.use_onnx and _HAS_ONNXRUNTIME and os.path.exists(span_onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.ner_onnx_session = ort.InferenceSession(span_onnx_path, providers=providers)
            self.ner_model = None
            return
        opt.bert_dir = os.path.join(PROJECT_ROOT, 'model', 'bert-base-chinese')
        self.ner_model = build_model('span', opt.bert_dir, opt,
                                     num_tags=len(self.ent2id) + 1,
                                     dropout_prob=opt.dropout_prob,
                                     loss_type=opt.loss_type)
        ner_model_path = os.path.join(PROJECT_ROOT, 'span_src', 'output', 'best_model.pt')
        self.ner_model.load_state_dict(torch.load(ner_model_path, map_location='cuda:0'), strict=True)
        self.ner_model.cuda()
        self.ner_model.eval()
        self.ner_onnx_session = None  # PyTorch 分支无 ONNX

    # 初始化 SimCSE 模型
    def init_simcse(self):
        logger.info('initialize simcse model......')
        simcse_onnx_path = os.path.join(ONNX_OUTPUT_DIR, 'simcse.onnx')
        if self.use_onnx and _HAS_ONNXRUNTIME and os.path.exists(simcse_onnx_path):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.simcse_onnx_session = ort.InferenceSession(simcse_onnx_path, providers=providers)
            self.simcse_model = None
            return
        self.simcse_model = TextBackbone(bert_path=os.path.join(PROJECT_ROOT, 'model', 'bert-base-chinese')).cuda()
        simcse_model_path = os.path.join(PROJECT_ROOT, 'SimCSE', 'output', 'sup_model.pt')
        self.simcse_model.load_state_dict(torch.load(simcse_model_path, map_location='cuda:0'), strict=True)
        self.simcse_model.eval()
        self.simcse_onnx_session = None  # PyTorch 分支无 ONNX

    def simcse_get_emb(self, text):
        if getattr(self, 'simcse_onnx_session', None) is not None:
            inp = _generate_input_onnx(text.strip(), self.tokenizer)
            out = self.simcse_onnx_session.run(None, {
                'input_ids': inp['input_ids'],
                'attention_mask': inp['attention_mask'],
                'token_type_ids': inp['token_type_ids'],
            })
            return np.array(out[0], dtype=np.float32)
        text = list(text.strip())
        input = self.tokenizer.encode_plus(text, return_tensors='pt').to('cuda:0')
        emb = self.simcse_model.predict(input)
        return emb

    def ner_predict(self, text):
        if getattr(self, 'ner_onnx_session', None) is not None:
            inp = _generate_input_onnx(text, self.tokenizer)
            start_logits, end_logits = self.ner_onnx_session.run(None, {
                'input_ids': inp['input_ids'],
                'attention_mask': inp['attention_mask'],
                'token_type_ids': inp['token_type_ids'],
            })
            start_logits = start_logits[0][1:-1]
            end_logits = end_logits[0][1:-1]
            return span_decode(start_logits, end_logits, text, self.id2ent)
        inputs = generate_input(text, self.tokenizer)
        decode_output = self.ner_model(**inputs)
        start_logits = decode_output[0].detach().cpu().numpy()[0][1:-1]
        end_logits = decode_output[1].detach().cpu().numpy()[0][1:-1]
        return span_decode(start_logits, end_logits, text, self.id2ent)

    # 注意：这是类内函数，属于Correction()类，需要有代码推进
    # 真正进行文本纠错的函数。
    # scope: "all"=领域+通用都做（默认）, "entity"=仅领域名词（span_src+SimCSE）, "general"=仅通用纠错（MacBERT）
    def correct(self, text, mode="distance_L", scope="all"):
        if not text or not text.strip():
            return text
        do_entity = scope in ("all", "entity")
        do_general = scope in ("all", "general")
        # 1: 专有股票名称纠错（NER 实体抽取 + SimCSE 对齐）
        if do_entity:
            res = self.ner_predict(text)
            if res:
                for item in res:
                    if not item or not item.strip():
                        continue
                    if item in self.stock_dic:
                        new_item = item
                    else:
                        new_item, score = self.faiss_search(item, mode)
                    text = text.replace(item, new_item, 1)
        # 2: 通用文本纠错（错别字、语法等）
        if do_general:
            if getattr(self, 'macbert_onnx_session', None) is not None:
                text = self._macbert_onnx_predict(text)
            elif self.macbert_model is not None:
                text = self.macbert_model.predict(text)
        return text

    def _macbert_onnx_predict(self, text):
        enc = self.macbert_tokenizer(
            text,
            max_length=ONNX_MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        inputs = {k: v.numpy().astype(np.int64) for k, v in enc.items()}
        _, logits = self.macbert_onnx_session.run(None, inputs)
        seq_len = int(enc['attention_mask'].sum(axis=1).item()) - 1
        pred_ids = np.argmax(logits[0], axis=-1)[1:seq_len]
        return self.macbert_tokenizer.decode(pred_ids).replace(' ', '')
    
    def init_index(self):
        logger.info('build faiss index......')
        embeddings = []
        texts = []
        # 将训练simcse模型时得到的文本相似度张量，以文件的模式加载进来
        with open(os.path.join(PROJECT_ROOT, 'SimCSE', 'output', 'doc_embedding'), mode='r', encoding='utf-8') as f:
            for line in f:
                text, emb = line.strip().split('\t')
                # 文本相似度张量是128维度，以','分隔的向量，要转换成float类型
                emb = [float(x) for x in emb.strip().split(',')]
                # 确认一下读进来的张量维度，和当初训练simcse的张量维度一致!!!
                assert len(emb) == self.dim
                embeddings.append(emb)
                texts.append(text)
        # 将文本和张量对应成映射字典
        embeddings = np.array(embeddings, dtype='float32')
        text2emb = {k: v for k, v in zip(texts, embeddings)}
        # 采用faiss的精确匹配索引模式，建立索引
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)
        # 正确的股票名称字典，以集合的形式初始化
        self.stock_dic = set(texts)
        # stock_name属性不是一个股票的名称，而是所有正确股票名称的总和列表
        self.stock_name = texts
    def faiss_search(self, text, mode='distance_L', k=15):
        # 待纠错的文本text，返回纠错后的正确文本以及得分
        # 1: 得到错误文本 text 的 SimCSE 向量
        e = self.simcse_get_emb(text)
        emb = np.asarray(e).squeeze()
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
        emb = np.array([emb], dtype='float32')
        # emb: [[ 0.11050283  0.01384767  0.0433474   0.22737686  0.10507135
        # -0.07436085
        #  0.02455004
        # -0.04649724  0.1542542  -0.11817002  0.01897967 -0.11548531
        #  0.04549803 -0.24729787 -0.0332848  -0.07336468 -0.0299028  ......]]
        #
        # 2: 在faiss已经构建好的索引张量中，召回top-k个候选结果
        _, results = self.index.search(emb, k)
        # results: [[ 867  866 4978 4977  591 3888 3889 4475 6132 9276 7677 1795 3489
        # 6762 6287]]

        # 将召回的top-k的候选结果所对应的"正确文本实体ents"组装成比较的列表
        ents = [self.stock_name[int(x)] for x in results[0]]
        # ents1: ['徐家汇', '徐家汇', '名家汇', '名家汇', '名家汇', '家润多', '同花顺', '同花顺', '家家悦', '顾家家居', '合富中国', '同兴天桥', 'ST和佳', '荣之联', '御家汇', '富满微']

        # text: 丽人丽妆
        # ents: ['丽人丽妆', '丽人申购', '丽尚国潮', '爱丽家居', '华丽家族', '美尔雅', '美尔雅', '美尚生态', '华录百纳', '华兰生物', 'ST美尚', '贵人鸟', '美尚生态', '华孚时尚', '美瑞新材']

        # 3: 如果采用第一种模式：经典最小编辑距离算法，则默认Ranking的参数mode = 'distance_L'
        res, score = self.Ranking(text, ents, mode)
        # res: 丽人丽妆, score: 2
        return res, score
    
    def Ranking(self, ent, candidates, mode='distance_L'):
        max_score, best_res = 10000, None
        # 将ner模型提取出来的"待纠错实体ent"，和从faiss库中搜索到的"正确的实体召回集合candidates"做评分对比
        for candi in candidates:
            score = edit_distance(ent, candi)  # distance_L 与 Levenshtein 均用编辑距离排序
            if score < max_score:
                max_score = score
                best_res = candi
        # 返回分数最高的"正确的候选实体candi"，和最高分数
        return best_res, max_score

if __name__ == '__main__':
    # 先解析 Correction 自有参数（parse_known_args），避免 span_src.Args().get_parser() 收到 --scope/--mode 报错
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Levenshtein', help='纠错模式：Levenshtein 或 distance_L')
    parser.add_argument('--scope', type=str, default='all', help='纠错范围：all=领域+通用, entity=仅领域(span+SimCSE), general=仅通用(MacBERT)')
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0]]  # 清掉命令行参数，防止 init_ner 里 span_src.Args().get_parser() 报 unrecognized
    corr = Correction()
    # scope: "all" 领域+通用, "entity" 仅领域名词(span+SimCSE), "general" 仅通用(MacBERT)
    with open(file='./demo.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            start_time = time.time()
            new_t = corr.correct(t, mode=args.mode, scope=args.scope)
            end_time = time.time()
            cost_time = end_time - start_time
            print('{}\t{}\t{}ms'.format(t, new_t, cost_time*1000))