import torch
import logging
import numpy as np
from collections import defaultdict
import pdb

logger = logging.getLogger(__name__)

# Span Pointer的核心解码函数, 此处采用"非重叠最短匹配策略"
def span_decode(start_logits, end_logits, raw_text, id2ent):
    predict = []
    ## text: *st恒康重组会停牌多久
    start_pred = np.argmax(start_logits, -1)
    end_pred = np.argmax(end_logits, -1)
    
    # 循环解码综合考虑不同的标签 种类s_type, 和不同的下标i
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i: i + j + 1]
                predict.append((''.join(tmp_ent), i, i + j, s_type))
                break
                
    # 如果抽取出多个命名实体, 依次按照不重叠原则提取至tmp列表中
    tmp = []
    for item in predict:
        if not tmp:
            tmp.append(item)
        else:
            if item[1] > tmp[-1][2]:
                tmp.append(item)
                
    # 以原始文本为基准, 初始化全'O'的标签列表
    result = ['O'] * len(raw_text)
    for item in tmp:
        # 提取起始索引s, 结束索引e, 实体标签flag
        s, e, flag = item[1], item[2], id2ent[item[3]]
        # 如果结束索引在起始索引的右侧, 则可以组装成BI*E的格式
        if e > s:
            result[s] = 'B-{}'.format(flag)
            result[e] = 'E-{}'.format(flag)
            # 中间字符全部设置为I标签
            if e - s > 1:
                for i in range(s + 1, e):
                    result[i] = 'I-{}'.format(flag)
        # 如果结束索引==起始索引, 说明是单一字符, 设置为S标签
        if e == s:
            result[s] = 'S-{}'.format(flag)
    return result

# 模型评估的主代码函数
def model_evaluate(model, dev_load, config, device, ent2id):
    with open(file='./tmp_dev_evaluate_{}'.format(config.task_type), mode='w', encoding='utf-8') as f:
        id2ent = {v: k for k, v in ent2id.items()}
        # 将模型设置为评估模式
        model.eval()
        with torch.no_grad():
            for batch, batch_data in enumerate(dev_load):
                # 提取数据并最大程度节省GPU显存资源
                # batch_data: ['input_ids', 'token_type_ids','attention_mask', 'raw_text','start_ids','end_ids','bieo_labels']
                raw_text = batch_data['raw_text']
                del batch_data['raw_text']
                labels = batch_data['bieo_labels']
                del batch_data['bieo_labels']
                
                # 将其余有效数据信息传至GPU上
                for key in batch_data.keys():
                    batch_data[key] = batch_data[key].to(device)
                    
                # 如果解码采用Span Pointer模式
                if config.task_type == 'span':
                    decode_output = model(**batch_data)
                    # decode_output: (start_logits, end_logits)
                    start_logits = decode_output[0].cpu().numpy()
                    end_logits = decode_output[1].cpu().numpy()
                    
                    # 一条条样本独立处理解码
                    for tmp_start_logits, tmp_end_logits, text, label in zip(start_logits, end_logits, raw_text, labels):
                        tmp_start_logits = tmp_start_logits[1: 1 + len(text)]
                        tmp_end_logits = tmp_end_logits[1: 1 + len(text)]
                        
                        # 预测阶段的最重要一步: 预测解码函数span_decode
                        predict = span_decode(tmp_start_logits, tmp_end_logits, text, id2ent)
                        tmp_label = label[:len(text)]
                        
                        # 写入评估文件的数据格式分3列, (原始中文字符, 真实标签, 预测标签)
                        for char, true, pre in zip(text, tmp_label, predict):
                            f.write('{}\n'.format(' '.join([char, true, pre])))
                        f.write('\n')
        