import os
import sys
sys.path.append('../..')
import torch
from config import Args
from transformers import BertTokenizer, BertForMaskedLM
import argparse
from collections import OrderedDict
import json
from torch.cuda.amp import autocast as ac
import copy
from loguru import logger
from reader import make_loaders, DataCollator
from bert4csc import Bert4Csc
from utils import build_optimizer_and_scheduler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from utils import model_evaluate, save_model
import logging
from tqdm import tqdm

# 真实的训练流程逻辑函数
def train_process(args):
    # 实例化模型的BERT⾃带tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # 实例化数据类的定制化数据处理函数collate_fn
    collator = DataCollator(tokenizer=tokenizer)
    # 创建训练集, 验证集, 测试集上的数据迭代器对象
    train_loader, valid_loader, test_loader = make_loaders(
        collator,
        train_path=args.train_data,
        valid_path=args.valid_data,
        test_path=args.test_data,
        batch_size=args.train_batch_size
    )
    # 指定训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 实例化macBERT4CSC模型对象
    model = Bert4Csc(args, tokenizer).to(device)
    # 构建优化器对象和调节器对象
    t_total = len(train_loader) * args.train_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, t_total)
    
    # 是否使⽤混合精度训练
    scaler = None
    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        
    os.makedirs(args.output_dir, exist_ok=True)
    best_score = 0
    avg_loss = 0.
    fgm, pgd = None, None
    print('Train begin!')
    
    for epoch in range(args.train_epochs):
        # 为了防⽌显存溢出, 每⼀轮epoch训练开始前清空缓存
        torch.cuda.empty_cache()
        # ⾮常重要的⼀步, 将模型设定为训练模式
        model.train()
        for batch, batch_data in enumerate(tqdm(train_loader)):
            batch_data['device'] = device
            # 训练流程, "⽼三样"的第1步
            optimizer.zero_grad()
            # 如果采⽤混合精度训练
            if args.use_fp16:
                with ac():
                    loss = model(**batch_data)[0]
            else:
                loss = model(**batch_data)[0]
            
            if args.use_fp16:
                # 如果采⽤混合精度训练, 在模型输出损失之后, 进⾏反向传播之前, 要对loss进⾏scale操作
                scaler.scale(loss).backward()
            else:
                # 训练流程, "⽼三样"的第2步
                loss.backward()
                
            if args.use_fp16:
                scaler.unscale_(optimizer)
                # 对梯度进⾏裁剪, 防⽌发⽣梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 训练流程, "⽼三样"的第3步
                optimizer.step()
                
            # 调节器在最后使⽤
            scheduler.step()
            avg_loss += loss.item()
            
        # ⼀个epoch循环结束后, 在验证集上执⾏⼀次模型评估, 并记录⽇志, 同时保存最优模型
        # 注意: 这⾥⽤的评估函数是utils.py⽂件中的model_evaluate()函数, ⽽不是evaluate()函数!!!
        detect_f1, correct_f1, final_score = model_evaluate(model, device, args)
        
        # 关键信息写⽇志
        logger.info('epoch {}/{}, avg_loss: {}'.format(epoch + 1, args.train_epochs, round(avg_loss, 4)))
        # 屏幕打印显示若⼲关键信息
        print('epoch {}/{}, avg_loss: {}'.format(epoch + 1, args.train_epochs, round(avg_loss, 4)))
        
        # 如果有更好的模型指标, 则保存最优模型
        if final_score > best_score:
            best_score = final_score
            best_model = copy.deepcopy(model)
            save_model(args, best_model, epoch)
        
        avg_loss = 0.
        # 关键信息写⽇志
        logger.info('detect_f1: {}, correct_f1: {}, final_score: {}, best_score: {}'.format(round(detect_f1, 4), round(correct_f1, 4), round(final_score), round(best_score, 4)))
        # 屏幕打印显示若⼲关键信息
        print('detect_f1: {}, correct_f1: {}, final_score: {}, best_score: {}'.format(round(detect_f1, 4), round(correct_f1, 4), round(final_score), round(best_score, 4)))
        
    # 为了防⽌显存溢出, 训练结束时清空缓存
    torch.cuda.empty_cache()
    logger.info('Train done!')

if __name__ == '__main__':
    args = Args().get_parser()
    train_process(args)
