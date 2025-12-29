import operator
from loguru import logger
import torch
import torch.nn as nn
import numpy as np
import os
import sys
sys.path.append('..')

# 实现FocalLoss的类
class FocalLoss(nn.Module):
    def __init__(self, num_labels, activation_type='sigmoid', gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        # 1: 分类问题的标签数量
        self.num_labels = num_labels
        # 2: FocalLoss算法中的指数参数
        self.gamma = gamma
        # 3: FocalLoss算法中的外围系数
        self.alpha = alpha
        # 4: 除法防⽌分⺟为零的极⼩正数
        self.epsilon = epsilon
        # 5: ⼆分类/多分类的不同计算策略
        self.activation_type = activation_type

    def forward(self, input, target):
        # 解决⼆分类问题, 损失函数设置为sigmoid
        multi_hot_key = target
        # 对于⼆分类问题, 直接在input上应⽤sigmoid即可
        logits = torch.sigmoid(input)
        zero_hot_key = 1 - multi_hot_key
        
        # 按照⼆分类的计算公式, loss需要加和两部分, 分别是标签为1对应的alpha
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        
        # loss还需要加和标签为0对应的(1 - alpha)
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        
        # 最后返回批次的平均损失值
        return loss.mean()
