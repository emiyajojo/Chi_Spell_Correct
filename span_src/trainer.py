# 导入相关工具包
import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
# from src.attack_train_utils import FGM, PGD # 文档中未提供此文件
from src.functions_utils import load_model_and_parallel, swa
from src.evaluate import model_evaluate
from src.cal_metric import calculate
import pdb
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 保存模型的函数代码
def save_model(config, model, epoch):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    model_to_save = (model.module if hasattr(model, 'module') else model)
    # 保存文件命名为"best_model.pt"
    torch.save(model_to_save.state_dict(), os.path.join(config.output_dir, 'best_model.bin'))

# 构建优化器和控制器
def build_optimizer_and_scheduler(config, model, t_total):
    module = (model.module if hasattr(model, 'module') else model)
    # 差分学习率
    no_decay = ['bias', 'LayerNorm.weight']
    model_param = list(module.named_parameters())
    bert_param_optimizer = []
    other_param_optimizer = []
    
    # 区分BERT模块和其他模块
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))
            
    optimizer_grouped_parameters = [
        # bert相关模块
        {
            'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
            'lr': config.lr
        },
        {
            'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.lr
        },
        # 其他模块, 差分学习率
        {
            'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
            'lr': config.other_lr
        }, 
        {
            'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.other_lr
        }
    ]
    
    # 分别设置优化器和控制器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.warmup_proportion * t_total),
                                                num_training_steps=t_total)
    return optimizer, scheduler

# 外层的主函数main.py -> train_base() -> train(), 这里才是整个训练的主函数逻辑代码
def train(config, model, train_dataset, dev_dataset, ent2id):
    # 本项目中采用Span Pointer模型进行NER任务实现
    if config.task_type in ['span', 'crf']:
        fn = train_dataset.collate_fn
    else:
        # fn = train_dataset.collate_fn_mrc # 文档中未提供MRC相关代码
        pass
        
    # 构建训练集的数据迭代器
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.train_batch_size,
                              num_workers=8,
                              collate_fn=fn,
                              shuffle=True)
    # 构建验证集的数据迭代器
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=config.train_batch_size,
                            num_workers=8,
                            collate_fn=fn,
                            shuffle=False)
                            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    swa_raw_model = copy.deepcopy(model)
    
    # 设置优化器和调节器
    t_total = len(train_loader) * config.train_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(config, model, t_total)
    
    # 进行日志写入, 追踪
    logger.info('batch_size: {}, epochs:{}, train_nums: {}, dev_nums: {}'.format(
        config.train_batch_size, config.train_epochs, len(train_dataset), len(dev_dataset)))
        
    avg_loss = 0.
    best_f1 = 0
    best_model = None
    
    # 经典的双重for循环训练模式
    for epoch in range(config.train_epochs):
        # 每一个epoch开始前清空CUDA缓存, 节省GPU资源
        torch.cuda.empty_cache()
        model.zero_grad()
        for step, batch_data in enumerate(tqdm(train_loader)):
            # 将模型设置为训练模式
            model.train()
            # 尽最大可能节省GPU显存资源
            del batch_data['raw_text']
            del batch_data['bieo_labels']
            # 将数据的有效信息传至GPU上
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            
            # 模型前向计算, 并反向传播
            loss = model(**batch_data)[0]
            loss.backward()
            
            # 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            avg_loss += loss.item()
            
        # 当一轮epoch训练结束后, 在验证集上进行一次评估
        f1 = model_evaluate(model, dev_loader, config, device, ent2id)
        # 直接调用计算关键字指标的函数
        f1 = calculate(config)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = copy.deepcopy(model)
            save_model(config, best_model, epoch)
            
        torch.cuda.empty_cache()
        logger.info('Train done')