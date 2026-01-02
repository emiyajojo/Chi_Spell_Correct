import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os
from src.evaluate import span_decode
import numpy as np
import random

# 构建NER数据集的类代码
class NERDataset(Dataset):
    def __init__(self, train_feature, config, ent2id):
        # train_feature: 数据集
        # config: 配置信息, 以parser命令参数配置的模式传入
        # ent2id: 实体类型的映射字典, 本项目中只有stock_name一种实体类型, 未来可以添加更多的实体类型
        self.data = train_feature
        self.nums = len(train_feature)
        # 分词器采用BERT自带的分词器
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(config.bert_dir))
        self.ent2id = ent2id
        self.config = config

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        return self.data[index]

    # 根据原始文本和索引数据, 构造NER的标签列表数据
    def get_bieo_data(self, text, data):
        # 初始化构造为全'O'的列表
        labels = ['O'] * len(text)
        # 遍历索引信息, 将命名实体赋值为BI*E的格式
        for _, s, e in data:
            labels[s] = 'B-stock_name'
            labels[e] = 'E-stock_name'
            for i in range(s + 1, e):
                labels[i] = 'I-stock_name'
        return labels

    # 个性化数据处理函数
    def collate_fn(self, batch_data):
        # 获取当前批次数据的最长文本长度 + 2, 赋值给max_len, 为后续批次数据的padding操作确定尺度
        max_len = max([len(x['text']) for x in batch_data]) + 2
        # max_len = 21
        # 初始化若干变量, 赋值成空列表
        input_ids, token_type_ids, attention_mask, labels, raw_text = [], [], [], [], []
        start_ids, end_ids, bieo_labels = [], [], []
        
        # 遍历批次数据, 只有2个内置元素, 原始文本text和实体名称stock_name
        for sample in batch_data:
            text = sample['text']
            label = sample['stock_name']
            # 调用BERT的分词器中的encode_plus()函数, 除了返回数字化张量input_ids,
            # 还可以完成补齐, 截断, 并返回segment和attention mask张量
            # 不同的参数组合会产生不同的效果, 具体可以查询huggingface文档, 小心有些组合会报错!
            encode_dict = self.tokenizer.encode_plus(text=list(text),
                                                     max_length=max_len,
                                                     padding='max_length',
                                                     truncation=True,
                                                     is_split_into_words=True,
                                                     return_token_type_ids=True,
                                                     return_attention_mask=True)
            # encode_dict: {'input_ids': [101, ...], 'token_type_ids': [...], 'attention_mask': [...]}
            # 将字典中的数据分别添加进结果列表中
            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            raw_text.append(text)
            
            # 如果任务模式采用Span Pointer的模型
            if self.config.task_type == 'span':
                # 初始化起始索引列表, 结束索引列表
                start_id, end_id = [0] * len(text), [0] * len(text)
                # 调用类内函数, 获取NER任务的标签列表
                bieo_label = self.get_bieo_data(text, label)
                bieo_labels.append(bieo_label)
                # bieo_label: ['B-stock_name', 'I-stock_name', 'I-stock_name', 'E-stock_name', 'O', 'O']
                
                # 遍历标签索引数据, 将起始位置和结束位置, 分别赋值成stock_name的id值, 此处赋值为1
                for _, s, e in label:
                    end_id[e] = self.ent2id['stock_name']
                    start_id[s] = self.ent2id['stock_name']
                
                # 添加CLS, SEP, 本质上添加0即可
                start_id = [0] + start_id + [0]
                end_id = [0] + end_id + [0]
                # start_id: [0, 1, 0, 0, 0, 0, 0, 0]
                # end_id: [0, 0, 0, 0, 1, 0, 0, 0]
                
                # 进行padding操作, 将每条数据的长度补齐到max_len
                if len(start_id) < max_len:
                    start_id = start_id + [0] * (max_len - len(start_id))
                    end_id = end_id + [0] * (max_len - len(end_id))
                
                start_ids.append(start_id)
                end_ids.append(end_id)
        
        # for循环结束, 进行批次数据的Tensor封装和类型设置
        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).float()
        start_ids = torch.tensor(start_ids).long()
        end_ids = torch.tensor(end_ids).long()
        
        # 如果任务采用Span Pointer的模型, 则返回如下字典类型数据
        if self.config.task_type == 'span':
            result = ['input_ids', 'token_type_ids', 'attention_mask',
                      'raw_text', 'start_ids', 'end_ids', 'bieo_labels']
            return dict(zip(result, [input_ids, token_type_ids, attention_mask, raw_text,
                                     start_ids, end_ids, bieo_labels]))