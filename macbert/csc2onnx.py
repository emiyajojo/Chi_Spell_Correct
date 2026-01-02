import torch
import onnxruntime
import sys
sys.path.append('../')
sys.path.append('.')
import logging
import time
import json
from bert4csc import Bert4Csc
from config import Args
from transformers import BertTokenizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
datefmt='%m/%d/%Y %H:%M:%S',
level=logging.INFO)
class OnnxTrans():
    def __init__(self):
        self.init_model()
        self.init_onnx()

    def init_model(self):
        logger.info('init csc model......')
        self.args = Args().get_parser()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_dir)
        # 实例化macBERT模型
        self.model = Bert4Csc(self.args, self.tokenizer)
        # 将已经训练好的模型加载进来, 后续是在best_model.pt上进行ONNX加速
        self.model.load_state_dict(torch.load('./output/bert4csc/best_model.pt', 
                                              map_location='cpu'), strict=True)
        self.model.eval()

    # 核心函数, 将macBERT模型导出成ONNX格式
    def init_onnx(self):
        logger.info('convert torch model to onnx model......')
        # 导出的ONNX模型的路径名称
        self.onnx_model_path = './output/bert4csc/model.onnx'
        # 明确使用torch的底层C语言包进行ONNX的导出操作
        operator_export_type = torch._C._onnx.OperatorExportTypes.ONNX
        # 构造一个"虚拟输入"
        onnx_input = self.make_onnx_infer_input()
        # 明确模型的输入, 输出, 动态轴的设置
        dynamic_axes = {'input_ids': [1], 'attention_mask': [1], 'token_type_ids': [1],
                       'output1': [1], 'output2': [1]}
        # 将macBERT模型导出成ONNX模型, 注意opset_version版本的兼容性
        out = torch.onnx.export(self.model,
                                onnx_input,
                                self.onnx_model_path,
                                verbose = False,
                                operator_export_type = operator_export_type,
                                opset_version=17,
                                input_names = ['input_ids', 'attention_mask', 'token_type_ids'],
                                output_names = ['output1', 'output2'],
                                dynamic_axes = dynamic_axes
                               )
        print('out:', out)

    # 构造一个"虚拟输入"
    def make_onnx_infer_input(self):
        onnx_input_ids = torch.LongTensor([[13, 21, 34, 67]])
        onnx_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
        onnx_input_mask = torch.LongTensor([[1, 1, 1, 0]])
        return (onnx_input_ids, onnx_input_mask, onnx_token_type_ids)

if __name__ == '__main__':
    onnx = OnnxTrans()