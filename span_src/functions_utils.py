import os
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from datetime import timedelta
from src.dataset_utils import NERDataset
import time
from torch.utils.data import DataLoader, RandomSampler
from src.model import build_model
import logging
import pdb
import json
import jieba
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

# 加载模型 & 放置到GPU中(单卡/多卡)
def load_model_and_parallel(model, gpu_ids, ckpt_path=None, strict=True):
    # 当服务器只有单一GPU时, 可以写下面一行代码
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 如果服务器有多块GPU时, 可以指定某一块GPU作为主设备
    device = torch.device('cpu' if gpu_ids == '-1' else 'cuda:' + gpu_ids[0])
    
    # 如果已经有训练好的模型存放在服务器上, 直接加载即可
    if ckpt_path is not None:
        logger.info(f'Load ckpt from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=strict)
    
    model.to(device)
    
    # 如果服务器上有多块GPU, 可以采用分布式训练的模式
    if gpu_ids != '-1' and len(gpu_ids) > 1:
        logger.info(f'Use multi gpus in: {gpu_ids}')
        gpu_ids = [int(x) for x in gpu_ids]
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        logger.info(f'Use single gpu in: {gpu_ids}')
    return model, device

# 获取模型存放路径的函数
def get_model_path_list(base_dir):
    # 从文件夹中获取model.pt的路径
    tmp = os.listdir(base_dir)
    tmp = ['checkpoint_{}_epoch.pt'.format(i) for i in range(21) if 'checkpoint_{}_epoch.pt'.format(i) in tmp]
    model_lists = [os.path.join(base_dir, x) for x in tmp]
    return model_lists

# SWA滑动平均模型的函数代码
def swa(model, model_dir):
    # swa: 滑动平均模型, 一般在训练平稳阶段再使用SWA
    model_path_list = get_model_path_list(model_dir)
    swa_model = copy.deepcopy(model)
    swa_n = 0.
    with torch.no_grad():
        for _ckpt in model_path_list:
            logger.info(f'Load model from {_ckpt}')
            # 加载模型
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            # 获取模型的命名参数字典
            tmp_para_dict = dict(model.named_parameters())
            # 按照公式计算alpha系数
            alpha = 1. / (swa_n + 1.)
            # 遍历模型的参数, 按照公式对参数进行SWA操作
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1
    return swa_model

# 数据预处理的类代码
class Data_Processor():
    def __init__(self, path):
        self.path = path

    def get_label_data(self):
        # 读取原始训练文件数据
        data = json.load(open(os.path.join(self.path, 'train.json'), mode='r', encoding='utf-8'))
        res = []
        # 遍历数据集, 每条样本包含2部分, 原始文本text, 命名实体的字符串stock_name
        for item in data:
            text, stock_names = item['text'], item['stock_name']
            tmp_text = text
            tmp_label = []
            # text: *st恒康重组会停牌多久
            # 遍历股票名称实体
            for stock_name in stock_names:
                # 查询股票名称的起始索引, 结束索引
                start_pos = tmp_text.find(stock_name)
                end_pos = start_pos + len(stock_name) - 1
                # stock_name: *st恒康
                # start_pos: 0
                # end_pos: 4
                # 以三元组的形式添加加结果列表中
                tmp_label.append((stock_name, start_pos, end_pos))
                # tmp_label: [('*st恒康', 0, 4)]
                
                # 将原始文本中的股票名称替换成'$', 避免名称重叠的问题
                tmp_text = tmp_text.replace(stock_name, '$' * len(stock_name), 1)
                # tmp_text: $$$$$重组会停牌多久
            
            # 以json数据格式添加进结果列表中
            res.append({'text': text, 'stock_name': tmp_label})
            # res: [{'text': '*st恒康重组会停牌多久', 'stock_name': [('*st恒康', 0, 4)]}]
        
        # for循环遍历结束后, 选择90%的数据作为训练集, 10%的数据作为验证集
        train = [x for i, x in enumerate(res) if i % 10 != 0]
        dev = [x for i, x in enumerate(res) if i % 10 == 0]
        return train, dev