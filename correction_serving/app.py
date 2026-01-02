from transformers import BertTokenizer
import onnxruntime
import json
import os
import numpy as np
import logging
import time
import sys
from configs import Args
import argparse
import faiss
from function_utils import CharFuncs

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)


class Correction_Serving():

    def __init__(self):
        self.base_config_init()
        self.onnx_ner_init()
        self.onnx_vector_model_init()
        self.faiss_index()
        self.macbert4csc_init()

    def onnx_ner_init(self, ):
        logger.info('init onnx ner model......')
        self.ner_onnx_session = onnxruntime.InferenceSession(
            self.opt.ner_onnx_path, providers=self.providers)

    def onnx_vector_model_init(self, ):
        logger.info('init onnx vector model......')
        self.vector_onnx_session = onnxruntime.InferenceSession(
            self.opt.simcse_onnx_path, providers=self.providers)

    def faiss_index(self, ):
        logger.info('build faiss index......')
        embeddings = []
        texts = []
        with open(file=self.opt.embe_file, mode='r', encoding='utf-8') as f:
            for line in f:
                text, emb = line.strip().split('\t')
                emb = [float(x) for x in emb.strip().split(',')]
                assert len(emb) == 128
                embeddings.append(emb)
                texts.append(text)
        embeddings = np.array(embeddings, dtype='float32')
        text2emb = {k: v for k, v in zip(texts, embeddings)}
        self.index = faiss.IndexFlatL2(128)
        self.index.add(embeddings)
        self.stock_dic = set(texts)
        self.stock_name = texts

    def macbert4csc_init(self, ):
        logger.info('init onnx macbert4csc model......')
        self.csc_onnx_session = onnxruntime.InferenceSession(
            self.opt.csc_onnx_path, providers=self.providers)

    def base_config_init(self, ):
        logger.info('init basic config......')
        self.opt = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained(self.opt.tokenizer_dir)
        with open(self.opt.span_ent2id, encoding='utf-8') as f:
            ent2id = json.load(f)
        self.id2ent = {v: k for k, v in ent2id.items()}
        # providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.providers = ['CUDAExecutionProvider']
        # self.providers = ['CPUExecutionProvider']
        self.char_feature = CharFuncs(self.opt.char_feature_file)

    def span_decode(self, pred_onnx, raw_text, id2ent):
        start_logits = pred_onnx[0][0][1:-1]
        end_logits = pred_onnx[1][0][1:-1]
        predict = []
        start_pred = np.argmax(start_logits, -1)
        end_pred = np.argmax(end_logits, -1)
        for i, s_type in enumerate(start_pred):
            if s_type == 0:
                continue
            for j, e_type in enumerate(end_pred[i:]):
                if s_type == e_type:
                    tmp_ent = raw_text[i:i + j + 1]
                    predict.append((''.join(tmp_ent), i, i + j, s_type))
                    break
        tmp = []
        for item in predict:
            if not tmp:
                tmp.append(item)
            else:
                if item[1] > tmp[-1][2]:
                    tmp.append(item)
        res = []
        for ent, _, _, _ in tmp:
            res.append(ent)
        return res

    def ner_infer(self, text):
        onnx_input = self.tokenizer.encode_plus(text=list(text),
                                                return_tensors='pt')
        pred_onnx = self.ner_onnx_session.run(
            None, {
                'input_ids': onnx_input['input_ids'].numpy(),
                'token_type_ids': onnx_input['token_type_ids'].numpy(),
                'attention_mask': onnx_input['attention_mask'].numpy(),
            })
        predict = self.span_decode(pred_onnx, text, self.id2ent)
        return predict

    # 将NER模型提取出来的"待纠错实体ent", 和从faiss库中搜索到的"正确的实体召回集合candidates"做评分对比
    def Ranking(self, ent, candidates, mode='distance_L'):
        max_score, best_res = 100000, None
        # 遍历从faiss中召回的候选集,计算得出最高分数的"正确实体"
        for candi in candidates:  # 采用第一种模式,计算最小编辑距离的方法
            if mode == 'distance_L':
                score = edit_distance(ent, candi)
                # 迭代更新最优分数和最优解实体
                if score < max_score:
                    max_score = score
                    best_res = candi
        # 返回分数最高的"正确的候选实体candi",和最高分数
        return best_res, max_score

    def simcse_get_emb(self, text):
        onnx_input = self.tokenizer.encode_plus(text=list(text),
                                                return_tensors='pt')
        pred_onnx = self.vector_onnx_session.run(
            None, {
                'input_ids': onnx_input['input_ids'].numpy(),
                'token_type_ids': onnx_input['token_type_ids'].numpy(),
                'attention_mask': onnx_input['attention_mask'].numpy(),
            })
        return pred_onnx

    def faiss_search(self, text, k=10):
        emb = self.simcse_get_emb(text)[0][0]
        emb = np.array([emb], dtype='float32')
        results = self.index.search(emb, k)
        ents = [self.stock_name[int(x)] for x in results[0]]
        res, score = self.Ranking(text, ents)
        return res, score

    def csc_predict(self, text):
        onnx_input = self.tokenizer.encode_plus(text=list(text),
                                                return_tensors='pt')
        pred_onnx = self.csc_onnx_session.run(
            None, {
                'input_ids': onnx_input['input_ids'].numpy(),
                'token_type_ids': onnx_input['token_type_ids'].numpy(),
                'attention_mask': onnx_input['attention_mask'].numpy(),
            })
        _y_hat = np.argmax(pred_onnx[1], -1)
        res = self.tokenizer.decode(_y_hat[0][1:-1]).replace(' ', '')
        return res

    def correct(self, text):
        res = self.ner_infer(text)
        for item in res:
            if item in self.stock_dic:
                new_item = item
            else:
                new_item, score = self.faiss_search(item)
            text = text.replace(item, new_item, 1)
        res = self.csc_predict(text)
        return res


if __name__ == '__main__':
    cor = Correction_Serving()
    demo = ['安佘尔没有主力资金为神么股价上张'] * 5
    for item in demo:
        s = time.time()
        res = cor.correct(item)
        e = time.time()
        print('{}\t{}\t{}ms'.format(cor, res, (e - s) * 1000))
