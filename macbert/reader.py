import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DataCollator:
    def __init__(self, tokenizer):
        # 数据处理类中的tokenizer, 采⽤参数模式传⼊
        self.tokenizer = tokenizer

    def __call__(self, data):
        # 提取源⽂件数据, 共3部分
        # original_texts: 原始待纠错的⽂本
        # correct_texts: 正确的⽂本
        # wrong_idss: 错误字符所在的下标集合
        original_texts, correct_texts, wrong_idss = zip(*data)
        encoded_texts = [list(t) for t in original_texts]
        # 获取最⼤⽂本⻓度, 后⾯+2是为了给[CLS], [SEP]预留位置
        max_len = max([len(t) for t in encoded_texts]) + 2
        # 初始化错误字符的位置矩阵, 全零矩阵
        wrong_labels = torch.zeros(len(original_texts), max_len).long()
        # 代码构造⼆维矩阵wrong_labels, 作为待纠错下标的数据矩阵
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            # 遍历错误下标列表, 将错误字符的位置标记成1
            for idx in wrong_ids:
                # 相当于⼿动构造了one-hot格式的纠错标签张量
                wrong_labels[i, idx + 1] = 1
        # 以字典类型返回待纠错⽂本, 正确⽂本, 错误字符位置矩阵
        return {'texts': original_texts, 'cor_labels': correct_texts, 'det_labels': wrong_labels}

class CscDataset(Dataset):
    def __init__(self, file_path):
        self.data = json.load(open(file_path, 'r', encoding='utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 返回原始数据⽂件中的JSON信息, 3个key值为⽂件中的原值, 必须保持⼀致
        return self.data[index]['original_text'], self.data[index]['correct_text'], self.data[index]['wrong_ids']

def make_loaders(collate_fn, train_path='', valid_path='', test_path='', batch_size=32, num_workers=4):
    train_loader, valid_loader, test_loader = None, None, None
    # 创建训练集的数据迭代器
    if train_path and os.path.exists(train_path):
        train_loader = DataLoader(CscDataset(train_path),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    # 创建验证集的数据迭代器
    if valid_path and os.path.exists(valid_path):
        valid_loader = DataLoader(CscDataset(valid_path),
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    # 创建测试集的数据迭代器
    if test_path and os.path.exists(test_path):
        test_loader = DataLoader(CscDataset(test_path),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
