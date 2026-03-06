import torch
import onnxruntime
import sys

sys.path.append('../')
import logging
import time
import json
from config import Args
from model import build_model
from transformers import BertTokenizer
import numpy as np
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)


class OnnxTrans():

    def __init__(self, ):
        self.init_ner_model()
        self.init_onnx()

    # 实例化NER模型, 采用Span Pointer架构
    def init_ner_model(self, ):
        logger.info('init ner model.........')
        opt = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
        with open('./data/span_ent2id.json', encoding='utf-8') as f:
            self.ent2id = json.load(f)
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.ner_model = build_model('span',
                                     opt.bert_dir,
                                     opt,
                                     num_tags=len(self.ent2id) + 1,
                                     dropout_prob=opt.dropout_prob,
                                     loss_type=opt.loss_type)
        # 将已经训练好的模型加载进来, 后续是在best_model.pt上进行ONNX加速
        self.ner_model.load_state_dict(torch.load('./out/best_model.pt',
                                                  map_location='cpu'),
                                       strict=True)
        self.ner_model.eval()

    # 核心函数, 将NER模型导出成ONNX格式
    def init_onnx(self, ):
        logger.info('convert torch model to onnx model......')
        # 导出的ONNX模型的路径名称
        self.onnx_model_path = './out/model.onnx'
        # 明确使用torch的底层C语言包进行ONNX的导出操作
        operator_export_type = torch._C._onnx.OperatorExportTypes.ONNX
        # 构造一个"虚拟输入"
        onnx_input = self.make_onnx_infer_input()
        # 明确模型的输入, 输出, 动态轴的设置
        dynamic_axes = {
            'input_ids': [1],
            'attention_mask': [1],
            'token_type_ids': [1],
            'start_logits': [1],
            'end_logits': [1]
        }
        # 将NER模型导出成ONNX模型, 注意opset_version版本的兼容性
        out = torch.onnx.export(
            self.ner_model,
            onnx_input,
            self.onnx_model_path,
            verbose=False,
            operator_export_type=operator_export_type,
            opset_version=17,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['start_logits', 'end_logits'],
            dynamic_axes=dynamic_axes)
        print('out:', out)

    # 构造一个"虚拟输入"
    def make_onnx_infer_input(self, ):
        onnx_input_ids = torch.LongTensor([[13, 21, 34, 67]])
        onnx_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
        onnx_input_mask = torch.LongTensor([[1, 1, 1, 0]])
        return (onnx_input_ids, onnx_input_mask, onnx_token_type_ids)


if __name__ == '__main__':
    onnx = OnnxTrans()
