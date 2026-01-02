import time
import os
import sys
sys.path.append("/hy-tmp/")
import json
import logging
import torch
from torch.utils.data import DataLoader
from src.trainer import train
from src.config import Args
from src.model import build_model
from src.dataset_utils import NERDataset
from src.functions_utils import set_seed, Data_Processor

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 训练基础函数, 真正的训练流程在最后一行的train()函数中
def train_base(config, train_feature, dev_feature=None, test_feature=None):
    # 加载实体映射表
    with open(os.path.join(config.raw_data_dir, f'{config.task_type}_ent2id.json'), encoding='utf-8') as f:
        ent2id = json.load(f)
    # print('ent2id: ', ent2id)
    
    # 进行数据集的封装处理
    train_dataset = NERDataset(train_feature, config, ent2id)
    dev_dataset = NERDataset(dev_feature, config, ent2id)
    
    # 本项目的主要模型采用Span Pointer
    model = build_model('span', config.bert_dir, config, num_tags=len(ent2id) + 1,
                        dropout_prob=config.dropout_prob, loss_type=config.loss_type)
    logger.info('training start......')
    train(config, model, train_dataset, dev_dataset, ent2id)

def training(config):
    # 实例化数据处理对象
    processor = Data_Processor(config.raw_data_dir)
    # 调用processor的内部函数处理后
    train, dev = processor.get_label_data()
    # 调用train_base为主训练函数
    train_base(config, train, dev)

if __name__ == '__main__':
    args = Args().get_parser()
    set_seed(args.seed)
    training(args)