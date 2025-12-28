import os
import sys
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
from tqdm import tqdm
import transformers
from transformers import  get_linear_schedule_with_warmup
from torch.optim import AdamW

# 导入自定义模块
from data import Supervised, Infer
from model import TextBackbone
from utils import swa, FGM, PGD  # 文档中FGM, PGD未提供实现

# 设置⽇志的相关配置
transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# 通过命令⾏传⼊训练, 推理模式
# 注意：实际运行时需确保 sys.argv 有参数，例如 python main.py train
mode = sys.argv[1] if len(sys.argv) > 1 else 'train'

# 对⽐学习的损失函数(有监督学习版本)
def sup_loss(y_pred, lamda=0.05):
    # y_pred.shape: [batch_size * 3, 128] -> [192, 128] if batch=64
    row = torch.arange(0, y_pred.shape[0], 3, device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0, len(col), 2, device='cuda')
    
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = torch.index_select(similarities, 0, row)
    similarities = torch.index_select(similarities, 1, col)
    
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)

# 对⽇志⽂件做准备⼯作, 设置路径和写⼊格式
def prepare():
    os.makedirs('/hy-tmp/SimCSE/SimSCE-output', exist_ok=True)
    now = datetime.now()
    log_file = now.strftime('%Y_%m_%d_%H_%M_%S') + '_log.txt'
    return '/hy-tmp/SimCSE/SimSCE-output' + log_file

def train(dataloader, model, optimizer, schedular, criterion, log_file, mode='unsup', attack_train=' '):
    if attack_train == 'fgm':
        fgm = FGM(model=model)
    
    # ⽆监督训练版本, 数据采⽤(x, x+)格式, 为⼆元组
    num = 2
    # 有监督训练版本, 数据采⽤(x, pos, neg)格式, 为三元组
    if mode == 'sup':
        num = 3
    
    all_loss = []
    # 遍历训练集数据
    for idx, data in enumerate(tqdm(dataloader)):
        # 将3个输⼊张量的shape转变成特定的格式
        # data['input_ids'].shape = [64, 3, 15] -> [192, 15]
        input_ids = data['input_ids'].view(len(data['input_ids']) * num, -1).cuda()
        attention_mask = (data['attention_mask'].view(len(data['attention_mask']) * num, -1).cuda())
        token_type_ids = (data['token_type_ids'].view(len(data['token_type_ids']) * num, -1).cuda())
        
        pred = model(input_ids, attention_mask, token_type_ids)
        
        optimizer.zero_grad()
        loss = criterion(pred)
        all_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        schedular.step()
        
        # 每隔30个batch进⾏⼀次缓存清空, 并将信息写⼊⽇志⽂件
        if idx % 30 == 0:
            torch.cuda.empty_cache()
            with open(log_file, 'a+') as f:
                t = sum(all_loss) / len(all_loss)
                info = str(idx) + ' == {} == '.format(mode) + str(t) + '\n'
                f.write(info)
            all_loss = []

# ⼊⼝主函数
if __name__ == '__main__':
    # 实例化对⽐学习模型的对象model
    model = TextBackbone().cuda()
    
    # 训练阶段
    if mode == 'train':
        logger.info('make sup simcse train.....')
        log_file = prepare()
        # 准备"三元组"格式的监督学习模式数据
        dataset = Supervised()
        # 构建数据迭代器
        batch_size = 64
        logger.info('batch_size:{},train_num:{}'.format(batch_size, len(dataset)))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # 设定模型参数的分模块优化策略
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
            },
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        # 设定优化器对象
        optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
        epochs = 10
        num_train_steps = int(len(dataloader) * epochs)
        
        # 设定调节器对象
        schedular = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0.05 * num_train_steps,
                                                    num_training_steps=num_train_steps)
        # 设定损失函数
        criterion = sup_loss
        
        # 外层循环epoch开启训练主流程
        for epoch in range(1, epochs + 1):
            logger.info('Epoch:{}/{}\n'.format(epoch, epochs))
            # 调⽤真实的训练函数
            train(dataloader, model, optimizer, schedular, criterion, log_file, mode='sup')
            # 每⼀个epoch轮次训练结束后, 对模型进⾏⼀次保存
            torch.save(model.state_dict(), '/hy-tmp/SimCSE/SimSCE-output/sup_model.pt')
            
    else:
        # 进⼊测试阶段(推理阶段)
        logger.info('make predict......')
        # 加载已经训练好的模型参数
        model.load_state_dict(torch.load('/hy-tmp/SimCSE/SimSCE-output/sup_model.pt', map_location='cpu'), strict=True)
        
        # 将上⼀次存在的embedding张量⽂件删除
        if os.path.exists('doc_embedding'):
            os.remove('doc_embedding')
            
        # ⾮常重要的⼀步: 推理阶段将模型设置为推理模式
        model.eval()
        
        # 实例化推理类的对象
        infer = Infer(model)
        # 获取所有的股票公司的名称
        companys = infer.get_companys()
        
        # 写⼊embedding张量⽂件
        with open(file='/hy-tmp/SimCSE/SimSCE-output/doc_embedding', mode='w', encoding='utf-8') as f:
            for text in tqdm(companys):
                # 调⽤推理对象infer, 获取股票名称text的数字化张量emb
                emb = infer.get_emb(text).squeeze().detach().cpu().numpy().tolist()
                # 保留8位有效数字, 并转换为字符类型
                y = [str(round(i, 8)) for i in emb]
                info = text.strip() + '\t'
                info = info + ','.join(y)
                f.write(info + '\n')