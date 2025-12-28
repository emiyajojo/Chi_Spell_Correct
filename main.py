import numpy as np
from transformers import BertTokenizer
import sys
sys.path.append(["/hy-tmp/CSC_all"])
from SimCSE.model import TextBackbone
from span_src.config import Args
from span_src.model import build_model
from utils import generate_input, span_decode, edit_distance
import logging
import json
import os
import torch

import pdb
import faiss
import time

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

class Correction():
    def __init__(self):
        # 1: 初始化已经训练好的NER模型，用于对待纠错文本进行实体抽取
        self.init_ner()
        # 2: 初始化已经训练好的simcse模型，用于对下抽取出来的待纠错entity进行文本匹配
        self.init_simcse()
        # 在训练simcse模型的时候，已经设置了相似语义张量的维度为128，这里要和simcse模型参数保持一致
        self.dim = 128
        self.init_index()

    # 初始化NER模型的函数
    def init_ner(self):
        logger.info('initialize ner model......')
        opt = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert')

        with open('/hy-tmp/CSC_all/span_src/span_data/span_ent2id.json', encoding='utf-8') as f:
            self.ent2id = json.load(f)
        self.id2ent = {v:k for k,v in self.ent2id.items()}

        # 在本类中，NER采用span指针的格式，本质上提前训练好，此处只做推理用!!!
        opt.bert_dir = '/hy-tmp/bert'
        self.ner_model = build_model('span', opt.bert_dir, opt,
                                  num_tags=len(self.ent2id) + 1,
                                  dropout_prob=opt.dropout_prob,
                                  loss_type=opt.loss_type)

        # 将已经训练好的NER模型加载进来
        ner_model_path = '/hy-tmp/CSC_all/span_src/Spanmodel-out/best_model.pt'
        self.ner_model.load_state_dict(torch.load(ner_model_path, map_location="cuda:0"),
                                     strict=True)

        # 放置到GPU上，并设置为预测模式
        self.ner_model.cuda()
        self.ner_model.eval()

    # 初始化simcse模型的函数
    def init_simcse(self):
        logger.info('initialize simcse model......')
        # 在本类中，本质上也是将提前训练好的simcse模型加载进来，用于比较文本相似度的预测模型来使用!!!
        self.simcse_model = TextBackbone().cuda()

        simcse_model_path = '/hy-tmp/CSC_all/SimCSE/SimSCE-output/sup_model.pt'
        self.simcse_model.load_state_dict(torch.load(simcse_model_path,map_location="cuda:0"), strict=True)

        # 放置到GPU上，并设置为预测模式
        self.simcse_model.cuda()
        self.simcse_model.eval()

    # 将待纠错的文本get_emb，通过simcse模型直接预测出相似文本张量，并返回
    def simcse_get_emb(self, text):
        text = list(text.strip())
        input = self.tokenizer.encode_plus(text, return_tensors='pt').to('cuda:0')
        emb = self.simcse_model.predict(input)
        return emb
    def ner_predict(self, text):
        # 1: 调用tokenizer将text进行切割处理
        inputs = generate_input(text, self.tokenizer)

        # 2: 调用NER模型执行实体抽取，此处模型采用span指针的模式
        decode_output = self.ner_model( **inputs)

        # 3: 获取起始位置start的概率分布，结束位置end的概率分布
        start_logits = decode_output[0].detach().cpu().numpy()[0][1:-1]
        end_logits = decode_output[1].detach().cpu().numpy()[0][1:-1]

        # 4: 执行span解码函数处理，真正的获取到从待纠错文本text中提取到的entities
        predict = span_decode(start_logits, end_logits, text, self.id2ent)

        return predict

    # 注意：这是类内函数，属于Correction()类，需要有代码推进
    # 真正进行文本纠错的函数，此处为框架“伪代码”
    def correct(self, text, mode="distance_L"):
        
        # 1: 第一步对有错误的文本text执行NER预测，将实体提取出来
        res = self.ner_predict(text)
        # 2: 如果没有提取出实体，则文本text不需要纠错
        if not res:
            return text
        
        # 3: 遍历所有提取出来的实体
        for item in res:
            # 3.1: 如果实体本身就是"正确的股票名称"，则保留不变
            if item in self.stock_dic:
                new_item = item

            else:

                new_item, score = self.faiss_search(item, mode)

            text = text.replace(item, new_item, 1)
                # text: 徐家汇怎么样
        # 返回纠错完毕的"正确文本text"
        return text
    
    def init_index(self):
        logger.info('build faiss index......')
        embeddings = []
        texts = []
        # 将训练simcse模型时得到的文本相似度张量，以文件的模式加载进来
        with open('/hy-tmp/CSC_all/SimCSE/SimSCE-output/doc_embedding', mode='r', encoding='utf-8') as f:
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
        # 1: 首先得到错误文本text对应的simcse相似度张量
        # text: 徐家汇
        emb = self.simcse_get_emb(text).squeeze().detach().cpu().numpy().tolist()
        # 移动到CPU上，转换成numpy数据类型，再转换成list类型后，就可以封装成numpy的array张量
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
            if mode == 'Levenshtein':
                score = edit_distance(ent, candi)
            # 迭代更新最优分数和最优解实体
            if score < max_score:
                max_score = score
                best_res = candi
        # 返回分数最高的"正确的候选实体candi"，和最高分数
        return best_res, max_score

if __name__ == '__main__':
    # 实例化纠错类Correction()
    corr = Correction()
    with open(file='./demo.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            t = line.strip()
            start_time = time.time()
            # 如果采用编辑距离的模式
            new_t = corr.correct(t, mode='Levenshtein')
            end_time = time.time()
            cost_time = end_time - start_time
            print('{}\t{}\t{}'.format(t, new_t, cost_time))