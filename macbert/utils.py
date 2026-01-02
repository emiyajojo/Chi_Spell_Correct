# 导入相关工具包
import os
import sys
sys.path.append('../..')
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from bert4csc import Bert4Csc
from config import Args
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import re
import time
from loguru import logger
import jieba
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from pathlib import Path
import os
import Levenshtein
from string import punctuation as en_pun
from zhon.hanzi import punctuation as zh_pun



# 编写个性化词典类, 目的是为了丰富jieba的个性化词库, 将已有的股票名称添加到jieba词库中
class MyToken():
    def __init__(self):
        # 1: 获取"股票名称.csv"文件中的所有股票名称
        self.dic = self.get_dic()
        
        # 2: 遍历字典dic, 将所有真实可信的股票名称添加进jieba字典中
        for w in self.dic:
            jieba.add_word(w)

    def get_dic(self):
        dic = []
        # 读取文件"股票名称.csv", 遍历每一行将有效股票名称信息写入列表dic中
        with open(file='../raw_data/股票名称.csv', mode='r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # 跳过第一行表头
                if i == 0:
                    continue

                # 忽略"不详", "None", 将有效股票名称信息保留
                line = [x for x in line.strip().split('\t') if x != '不详' and x != 'None' and x]
                dic += line

        dic = set([x.lower() for x in dic if 'ST' in x] + dic)

        return dic

    def my_cut(self, text):
        # 将分词后的文本中, 股票名称信息的实体过滤并保留下来
        res = [x for x in jieba.lcut(text) if x in self.dic]
        return res


class Inference():
    def __init__(self, model_path='./output/bert4csc/best_model.pt', bert_path='../macbert_chinese_base/'):
        # 实例化BERT模型的tokenizer对象
        print('Initializer the Inference model......')
        start_time = time.time()
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        # 设定当前服务器环境的设备变量GPU or CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 实例化macBERT4CSC类对象
        args = Args().get_parser()
        self.model = Bert4Csc(args, self.tokenizer)

        # 如果有已经训练好的maxbert4csc模型, 直接加载进model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
        else:
            raise ValueError('model not found!')

        # 将加载好的模型放到GPU上, 并设置为推理模式
        self.model.to(self.device)
        self.model.eval()
        end_time = time.time()
        print('The Inference model initialized success, cost time: {}s.'.format(end_time - start_time))

    # 类内预测函数
    def predict(self, sentence_list):
        # 输入值sentence_list: 输入文本列表, 类型list
        # 返回值: corrected_texts, 纠错后的正确文本, 类型list
        is_str = False
        
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]

        # 调用macBERT4CSC模型中的predict函数直接完成预测
        corrected_texts = self.model.predict(sentence_list, self.device)
        
        if is_str:
            return corrected_texts[0]
        
        return corrected_texts
    
    # 类内预测函数(要求函数返回值带错误位置信息等细节)
    def predict_with_error_detail(self, sentence_list):
        # 文本纠错模型预测, 结果带错误位置信息
        # sentence_list: list, 输入文本列表
        # Returns: corrected_texts(list), details(list)
        details = []
        def get_errors(corrected_text, origin_text):
        # 使用 Levenshtein.opcodes 计算从 origin_text 到 corrected_text 的编辑操作
        # opcodes 返回一个列表，每个元素为元组 (tag, i1, i2, j1, j2)
        # tag: 操作类型 ('replace', 'delete', 'insert', 'equal')
        # i1, i2: origin_text 中的起始和结束下标
        # j1, j2: corrected_text 中的起始和结束下标
            edits = Levenshtein.opcodes(origin_text, corrected_text)
            details = []

            for edit in edits:
                tag = edit[0]
                src_start = edit[1]
                src_end = edit[2]
                tgt_start = edit[3]
                tgt_end = edit[4]

                # 保持与 convert_from_sentpair2edits 逻辑一致，如果目标文本片段包含句号则跳过
                if '。' in corrected_text[tgt_start: tgt_end]:
                    continue

                if tag == 'replace':
                    # 别字: 原始文本 src_start:src_end 错误，应替换为 corrected_text tgt_start:tgt_end
                    details.append((src_start, '别字', origin_text[src_start: src_end], corrected_text[tgt_start: tgt_end]))
                elif tag == 'delete':
                    # 冗余: 原始文本 src_start:src_end 是多余的
                    details.append((src_start, '冗余', origin_text[src_start: src_end], ''))
                elif tag == 'insert':
                    # 缺失: 原始文本 src_start 位置缺失了内容，应插入 corrected_text tgt_start:tgt_end
                    details.append((src_start, '缺失', '', corrected_text[tgt_start: tgt_end]))
                    
            return corrected_text, details
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        
        # 将待预测文本以列表格式输出模型类中的predict函数, 得到纠错后的文本
        corrected_texts = self.model.predict(sentence_list)

        # 遍历原始文本, 以及纠错后的文本
        for corrected_text, text in zip(corrected_texts, sentence_list):
            # 调用get_errors()函数, 通过比较得到具体的纠错信息
            corrected_text, sub_details = get_errors(corrected_text, text)
            details.append(sub_details)
        
        if is_str:
            return corrected_texts[0], details[0]
        
        return corrected_texts, details


# 构建优化器和调节器的函数
def build_optimizer_and_scheduler(config, model, t_total):
    module = (model.module if hasattr(model, 'module') else model)

    # 差分学习率
    no_decay = ['bias', 'LayerNorm.weight']
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    # 未来参数优化策略, 会区分"BERT模块"和"other 模块", 因此将分别放进2个参数列表中
    for name, para in model_param:
        space = name.split('.')
        # 1: name中带bert的, 放进"BERT模块"中
        if 'bert' in space:
            bert_param_optimizer.append((name, para))
        # 2: name中不带bert的, 放进"other 模块"中
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
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
        {
            'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': config.weight_decay,
            'lr': config.other_lr
        },
        {
            'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.other_lr
        }]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    return optimizer, scheduler


# 这里面的model_evaluate()函数, 是train.py中真正调用的函数
def model_evaluate(model, device, args):
    # path = './output/predict_dev_file.txt'
    path = args.predict_dev_path
    res_path = args.res_path

    with open(file=path, mode='w', encoding='utf-8') as f:
        data = json.load(open(args.valid_data, 'r', encoding='utf-8'))
        
        # 遍历验证集dev, 得到纠错后的文本
        for item in tqdm(data):
            original_text = item['original_text']
            
            # print('----------------START------------------')
            # original_text: 国星光电

            # 直接调用predict函数, 得到纠错后的文本res
            res = model.predict([original_text], device)
            # res: ['国星光电']
            
            # 向结果文件中写入的3条信息: 原始错误文本, 原始正确文本, 模型预测文本
            f.write('{}\t{}\t{}\n'.format(original_text, item['correct_text'], res[0]))

    # 对预测结果文件path, 执行评估计算, 得到关键指标F1分数. 调用函数中会产生检错 + 纠错的完整细节!!!
    detect_f1, correct_f1, final_score = do_calculate(path, res_path)
    
    return detect_f1, correct_f1, final_score


# 去除掉声调, 语音字符
def remove_pun(text):
    _text = ''
    for uchar in text:
        if uchar in en_pun + zh_pun + " 　":
            continue    
        _text += uchar
    
    return _text


# 读取"原始错误文本"的函数, 本质上是读取input_file = './output/predict_dev_file.txt'的第1列数据
def read_input_file(input_file):
    pid_to_text = {}

    # predict_dev_file.txt文件中的每一行包含3个字段: "原始错误文本", "原始正确文本", "模型预测文本"
    with open(input_file, 'r') as f:
        for id, line in enumerate(f):
            line = line.strip().split('\t')
            
            line = [remove_pun(item.strip()) for item in line]
            
            # 字典中只保留第一个字段: "原始错误文本"
            pid = str(id)
            text = line[0]
            pid_to_text[pid] = text

    return pid_to_text


# 本函数绝不仅仅是为了读取文件数据, 而是产生若干重要信息的函数
# 通过具体的"纠错过程字符串"的信息, 来还原出每条数据的"纠错集合"(4个元素), "错误集合"(3个元素), "正确集合"(2个元素)
def read_label_file(pid_to_text, label_file_list):
    # 读取纠正结果
    error_set, det_set, cor_set = set(), set(), set()
    
    for line in label_file_list: 
        # 调用convert_from_sentpair2edits()函数时, 纠错流程是以字符串写入的, 不同的操作用','分隔
        terms = line.strip().split(',')
        terms = [t.strip() for t in terms]
        
        pid = terms[0]
        # pid不存在直接跳过
        if pid not in pid_to_text:
            continue
        
        # 最后是'-1', 说明最小编辑距离没有计算出结果, 直接跳过, 不处理该条数据
        if len(terms) == 2 and terms[-1] == '-1':
            continue
        
        # text: 代表"原始错误文本", 即预测结果文件中的第一列信息
        text = pid_to_text[pid]
        
        # len(terms) - 2是应对开始的sid, 和结尾的', '
        # % 4 是为了验证纠错流程字符串中包含的信息数量是正确的
        if (len(terms) - 2) % 4 == 0:
            # 整除4后的结果, 代表着需要进行纠错的数量
            error_num = int((len(terms) - 2) / 4)
            
            for i in range(error_num):
                # 每次截取出来4个字段, +1是为了跳过','的分隔
                # 分别是需要纠错的起始位置loc, 错误的类型type, 错误的子串wrong, 正确的子串correct
                # 错误的类型type对应三种情况: insert - "缺失", replace - "别字", delete - "冗余"
                loc, _, wrong, correct = terms[i * 4 + 1: (i + 1) * 4 + 1]
                loc = int(loc)
                
                # 利用相关信息拼接出正确的结果字符串cor_text, 利用text: "原始错误文本"
                cor_text = text[:loc] + correct + text[loc + len(wrong):]
                
                # 纠错集合中添加4个元素的元组信息
                error_set.add((pid, loc, wrong, cor_text))
                
                # 错误集合中添加3个元素的元组信息
                det_set.add((pid, loc, wrong))
                
                # 正确集合中添加2个元素的元组信息
                cor_set.add((pid, cor_text))
        # 如果无法整除4, 说明读入的数据格式有错误
        else:
            raise Exception('check your data format: {}'.format(line))
    
    return error_set, det_set, cor_set


# 计算关键指标F1值的函数
def cal_f1(ref_num, pred_num, right_num):
    # 计算precision的值
    precision = float(right_num) / (pred_num + 0.0001)
    
    # 计算recall的值
    recall = float(right_num) / (ref_num + 0.0001)
    
    # 如果两个值太小, 则返回0
    if precision + recall < 1e-6:
        return 0.0
    
    # 按照计算公式得到F1的值
    f1 = 2 * precision * recall / (precision + recall + 0.0001)
    
    # 以%的形式返回
    return f1 * 100



# 真实的项目流程中, train.py中进行评估的函数是model_evaluate()函数, 而不是此处的evaluate()函数!!!
def evaluate(input_file, ref_file_list, pred_file_list):
    # 读取预测结果文件, input_file = './output/predict_dev_file.txt'
    # input_file文件中每行包含3条信息: 原始错误文本, 原始正确文本, 模型预测文本
    # pid_to_text: 字典类型, id: 原始错误文本 (其中只包含文件中的第一列信息)
    pid_to_text = read_input_file(input_file)

    # ref_file_list: 包含"原始错误文本" -> "原始正确文本", 所需的详细操作步骤
    # 返回具体纠错集合(4个元素), 错误集合(3个元素), 正确集合(2个元素)

    # print('---------------------444----------------------')
    # pid_to_text: {'0': '国星光电', '1': '韵达股份2021年报每股收益', '2': '集合竞价和直真科技一样的', '3': '华鑫证卷客服号码多少', '4': '哈焊华通技术走向', '5': 'a股宝钢股份', '6': '牧原股份筹码', '7': '广汇能源网上热', '8': 'a股金鹰股份2021最高价', '9': '澄江县妹子保健11693371微信当做', '10': '宁波能源', ......, '3140': '002163主力控盘缣决度', '3141': '东方电热分时均线叁线合一', '3142': '昨日涨停板去除st去除科创板去除创业板按涨跌幅排续', '3143': '昨天股价收阳昨天上影线大于实体2倍昨忝没有下影线昨天低开且创最近10日新低非新股非次新股非ST非科创', '3144': '002905主力资金妞向', '3145': '000753SZ主边增仓占比', '3146': '股价四元至六十元涨幅01至9净利润一千万以上流通市值五十亿以上量比大于一近一个月有上涨百分之九以上今天开盘价高于昨天开盘价和收盘价027薏上昨天开盘价低于前一天收盘价和开盘价', '3147': '002909技术型态', '3148': '最后的价值工椰部股票被饿死了北上', '3149': '广晟有色怎么操炸', '3150': '牧原股份机构关注情矿'}

    # ref_file_list: ['0, -1', '1, -1', '2, -1', '3, 3, 别字, 卷, 券,', '4, -1', '5, -1', '6, -1', '7, -1', '8, -1', '9, 18, 别字, 做, 作,', '10, -1', '11, 0, 冗余, n, , 4, 缺失, , 南,', '12, -1', '13, -1', '14, -1', '15, -1', '16, -1', '17, -1', '18, -1', '19, -1', '20, -1', '21, -1', '22, -1', '23, 1, 别字, 园, 圆,', ......, '3140, 10, 别字, 缣, 坚,', '3141, 8, 别字, 叁, 三,', '3142, 24, 别字, 续, 序,', '3143, 18, 别字, 忝, 天,', '3144, 10, 别字, 妞, 流,', '3145, 9, 别字, 边, 力,', '3146, 68, 别字, 薏, 以,', '3147, 8, 别字, 型, 形,', '3148, 6, 别字, 椰, 业,', '3149, 7, 别字, 炸, 作,', '3150, 9, 别字, 矿, 况,']

    ref_error_set, ref_det_set, ref_cor_set = read_label_file(pid_to_text, ref_file_list)
    
    # ref_error_set: {('825', 7, '钩', '300506机构活跃度'), ('324', 0, '金田', '今天北向资金'), ('2844', 6, '勒', '杉杉股份估值类型机会'), ('582', 6, '糁', '浙江建投今日散户数量多少'), ('2623', 7, '凭', '汇顶科技热门点评'), ('816', 6, '搭', '威胜信息重大合同点评'), ('1327', 3, '耻', '七一二止盈位'), ('677', 9, '符', '603630竞价涨幅'), ......, ('1874', 6, '晾', '闻泰科技价跌量升非跌停'), ('1603', 6, '曹', '陕鼓动力怎么操作'), ('1325', 4, '既', '兰石重装机构关注情况'), ('1958', 7, '钩', '688396机构持股比例'), ('3121', 10, '各', '北新建材定02转股价格'), ('2231', 7, '诃', '航天发展走势如何'), ('1933', 7, '开', '晋控煤业估值类型机会')}
    # ref_det_set: {('2231', 7, '诃'), ('3132', 3, '罪'), ('1042', 15, '愆'), ('1138', 5, '注'), ('2597', 4, '嫜'), ('594', 33, '房'), ('577', 8, '铘'), ('627', 4, '甑'), ('2053', 9, '铜'), ('2346', 13, '漂'), ('960', 118, '苻'), ('1353', 8, '辎'), ('2619', 12, '渖'), ('1086', 11, '战'), ('1345', 11, '乒'), ('2019', 13, '闰'), ......, ('204', 8, '集'), ('1401', 8, '纷'), ('1525', 3, '甸'), ('2735', 7, '往'), ('1987', 10, '液'), ('2018', 4, '岍'), ('1217', 34, '禁'), ('1081', 5, '芴'), ('615', 2, '正')}
    # ref_cor_set: {('1483', '600084机构持股比例'), ('1630', '688508主力资金流向'), ('1098', '宝新能源题材要点'), ('2442', '000887主力资金流向'), ('2824', '紫金矿业散户数量'), ('2102', '天赐材料分红细则'), ......, ('1227', '601991撑压分析'), ('2438', '得润电子加速上涨形')}

    # len(ref_error_set): 2940
    # len(ref_det_set): 2940
    # len(ref_cor_set): 2940

    # pred_file_list: 包含"原始错误文本" -> "模型预测文本", 所需的详细操作步骤
    # 返回具体纠错集合(4个元素), 错误集合(3个元素), 正确集合(2个元素)
    pred_error_set, pred_det_set, pred_cor_set = read_label_file(pid_to_text, pred_file_list)
    
    # pred_error_set: {('3107', 5, '纺', '怡合达分配方案点评'), ('733', 5, '悒', '上峰水泥可以买吗'), ('948', 5, '炉', '启明星辰散户dd'), ('988', 10, '烬', '祥和实业股份过去五年盈现比平均值'), ('1454', 14, '搡', '世纪天鸿的资金流出为主力还是散户'), ('2920', 7, '品', '绿城水务解禁点评'), ('554', 3, '觞', '比亚迪上方压力位'), ('2489', 12, '涠', '000650支撑位和压力位'), ('1124', 17, '斜', 'wr指标超卖市净率小于2国企控股剔除st'), ('2115', 4, '伞', '天齐锂业散户'), ......}
    # pred_det_set: {('1903', 3, '潺'), ('3121', 10, '各'), ('186', 7, '昌'), ('568', 7, '鹄'), ('2669', 6, '村'), ('1207', 7, '炸'), ('2098', 3, '缯'), ('2964', 6, '隈'), ('1245', 6, '榆'), ('2547', 5, '棺'), ('2277', 29, '窗'), ('2615', 17, '缙'), ('650', 39, '仙'), ('2091', 6, '祷'), ('1601', 15, '卣'), ('3043', 6, '嘈'), ......}
    # pred_cor_set: {('1204', '华利集团支撑位压力位'), ('1549', '迎驾贡酒资金强度'), ('315', '牧原股份散户'), ('1034', '锦泓集团是国企吗'), ('990', '聚灿光电怎么操作'), ......}
    
    # len(pred_error_set): 2925
    # len(pred_det_set): 2925
    # len(pred_cor_set): 2925
    # print('---------------------555----------------------')

    # 评估数据中"原始正确文本"中"应该存在的正确文本数量"
    ref_num = len(ref_cor_set)
    # red_num = 2940

    # 评估数据中"模型预测文本"中"模型预测出来的正确文本数量"
    pred_num = len(pred_cor_set)
    # pred_num = 2925

    # 初始化"纠错正确的数量"
    det_right_num = 0
    # ref_error_set代表纠错集合, 集合中每个元素的形式(pid, loc, wrong, cor_text)
    for error in ref_error_set:
        pid, loc, wrong, cor_text = error
        # 如果"纠错集合的信息"在"模型预测的错误集合"中
        # 或者"纠错后的文本信息"在"模型预测的正确集合"中
        if (pid, loc, wrong) in pred_det_set or (pid, cor_text) in pred_cor_set:
            # 纠错正确的文本数量+1
            det_right_num += 1

    # "应该正确的原始文本数量", "模型预测出来的正确文本数量", "纠错正确的文本数量"
    # 共同计算"检测正确的F1值"
    # det_right_num = 2801
    detect_f1 = cal_f1(ref_num, pred_num, det_right_num)
    # detect_f1 = 95.51076856328194
    # print('-----------------666-------------------')

    # 交集代表"模型预测正确的文本数量"
    cor_right_num = len(ref_cor_set & pred_cor_set)
    # cor_right_num = 2449
    
    # 共同计算"纠错正确的文本数量"
    correct_f1 = cal_f1(ref_num, pred_num, cor_right_num)
    # correct_f1 = 83.50735895054461

    # 按照检测80%权重, 纠错20%权重, 计算总分数
    final_score = 0.8 * detect_f1 + 0.2 * correct_f1
    # final_score = 93.11008664073448

    # 将3个重要评估值返回
    return detect_f1, correct_f1, final_score


# 将原始数据中的3个字段进行"最小编辑距离"计算
def convert_from_sentpair2edits(lines_sid, lines_src, lines_tgt):
    # 首先确认传入的3个原始数据列表的长度一样
    assert len(lines_src) == len(lines_tgt) == len(lines_sid)
    
    # print('--------------------222-----------------')
    # len(lines_tgt): 3151

    convert_result = []
    for i in range(len(lines_src)):
        src_line = lines_src[i].strip()
        tgt_line = lines_tgt[i].strip()
        sid = lines_sid[i].strip()
        
        # 直接调用Levenshtein中的opcodes函数, 进行最小编辑距离的计算
        edits = Levenshtein.opcodes(src_line, tgt_line)
        # opcodes()函数会给出所有第一个字符串转换成第二个字符串需要执行的操作和操作详情会给出一个列表, 列表的值为元组, 每个元组中有5个值
        # [('delete', 0, 1, 0, 0), ('equal', 1, 3, 0, 2), ('insert', 3, 3, 2, 3), ('replace', 3, 4, 3, 4)]
        # 第一个值是需要进行的操作, 例如下标1是要删除的操作, 下标2和3是第一个字符串需要改变的切片起始位和结束位, 例如第一个元组是删除第一字符串的0 - 1这个下标的元素
        # 下标4和5是第二个字符串需要改变的切片起始位和结束位, 例如第一个元组是删除第一字符串的0 - 1这个下标的元素, 所以对应于第二个字符串不需要删除, 对应位置是0 - 0即可
        
        result = []
        for edit in edits:
            # 如果第二个字符串的目标位置是'。', 则跳过不处理
            if '。' in tgt_line[edit[3]: edit[4]]:
                continue

            # 1: 如果进行插入操作 - 'insert'
            if edit[0] == 'insert':
                # 意味着第一个字符串的1位置, '缺失'第二个字符串的对应位置3-4的子串
                result.append((str(edit[1]), '缺失', '', tgt_line[edit[3]: edit[4]]))
            # 2: 如果进行替换操作 - 'replace'
            elif edit[0] == 'replace':
                # 意味着第一个字符串的1位置, 是'别字', 具体的别字是1-2的子串, 而正确的应该是第二个字符串中3-4的子串
                result.append((str(edit[1]), '别字', src_line[edit[1]: edit[2]], tgt_line[edit[3]: edit[4]]))
            # 3: 如果进行删除操作 - 'delete'
            elif edit[0] == 'delete':
                # 意味着第一个字符串的1位置, 有'冗余', 具体的冗余是第一个字符串中1-2的子串
                result.append((str(edit[1]), '冗余', src_line[edit[1]: edit[2]], ''))

        # 将连续的操作流程, 以同一个字符串的格式进行拼接, 中间全部用', '分隔.
        out_line = ''
        for res in result:
            out_line += ', '.join(res) + ', '

        # 如果有对应的纠错流程, 则用sid + 纠错流程字符串存储进列表中
        if out_line:
            convert_result.append(sid + ', ' + out_line.strip())
        # 如果没有纠错流程, 也就是两个字符串一样, 则用sid + ', -1'来标识
        else:
            convert_result.append(sid + ', -1')
    
    return convert_result


def do_calculate(path, res_path):
    # path = './output/predict_dev_file.txt', 本质上就是模型对于验证集的predict结果的持久化存储文件
    # path文件中每行包含3条信息: 原始错误文本, 原始正确文本, 模型预测文本
    pred_id, pred_src, pred_prd = [], [], []
    
    # 接下来的循环是为了获取到第0列, 第2列的数据, 从而构造出pred_convert_result
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.lower().split('\t')
            
            # 确保读取的预测结果文件有3列信息
            if len(line) != 3:
                raise ValueError('line {} length must be 3 after split by \\t, line: {}'.format(idx, line))
            
            pred_id.append(str(idx))
            
            # 将原始错误文本line[0]添加进列表, 即添加"原始错误文本"
            pred_src.append(remove_pun(line[0].strip()))
            
            # 将模型预测文本line[2]添加进列表, 即添加"模型预测文本"
            pred_prd.append(remove_pun(line[2].strip()))

    # 调用Levenshtein工具包的opcodes()函数, 得到具体的最小编辑距离的操作流程
    # pred_convert_result是一个列表, 每一个元素都对应一条样本如何将"错误文本"转换成"正确文本"的操作步骤
    # 计算得到"原始错误文本" -> "模型预测文本", 转换所需的步骤(中间原理为最小编辑距离)
    # print('----------------111---------------------')
    pred_convert_result = convert_from_sentpair2edits(pred_id, pred_src, pred_prd)
    
    # pred_convert_result: ['0, -1', '1, -1', '2, -1', '3, -1', '4, 1, 别字, 焊, 京,', '5, -1', '6, -1', '7, -1', '8, -1', '9, 18, 别字, 做, 作,', '10, -1', '11, -1', '12, -1', '13, -1', '14, -1', '15, -1', '16, -1', '17, -1', '18, -1', '19, -1', '20, -1', '21, -1', '22, -1', '23, -1', '24, -1', '25, -1', '26, -1', '27, -1', '28, -1', '29, -1', '30, -1', '31, 2, 别字, 蒽, 斯,', '32, -1', '33, -1', '34, -1', '35, -1', '36, -1', '37, -1', '38, -1', '39, -1', '40, -1', '41, -1', '42, -1', '43, -1', '44, -1', '45, -1', '46, -1', '47, -1', '48, -1', '49, -1', '50, -1', '51, -1', '52, -1', '53, -1', '54, -1', '55, -1', '56, -1', '57, -1', '58, -1', '59, -1', '60, -1', '61, -1', '62, 2, 别字, 荒, 洋,', '63, -1', '64, -1', '65, -1', '66, -1', '67, 3, 别字, 自, 旗,', '68, -1', '69, -1', '70, -1', '71, -1', '72, -1', '73, -1', '74, 1, 别字, 太, 泰,', '75, -1', '76, -1', '77, -1', '78, -1', '79, -1', '80, -1', '81, -1', '82, -1', '83, -1', '84, -1', '85, -1', '86, 6, 别字, 1, 日,', '87, -1', '88, -1', '89, -1', '90, -1', '91, -1', '92, -1', '93, -1', '94, -1', '95, -1', '96, -1', '97, -1', '98, -1', '99, -1', '100, -1', ......, '3140, 10, 别字, 缣, 坚,', '3141, 8, 别字, 叁, 三,', '3142, -1', '3143, 18, 别字, 忝, 天,', '3144, 10, 别字, 妞, 流,', '3145, 9, 别字, 边, 力,', '3146, 68, 别字, 薏, 以,', '3147, 8, 别字, 型, 形,', '3148, 6, 别字, 椰, 一,', '3149, 7, 别字, 炸, 作,', '3150, 9, 别字, 矿, 况,']

    # 将检错 + 纠错的细节写入文件中
    with open(res_path, mode='w', encoding='utf-8') as f:
        for text_src, line, text_pred in zip(pred_src, pred_convert_result, pred_prd):
            f.write(text_src + '\t' + line + '\t' + text_pred + '\n')
    
    ref_id, ref_src, ref_prd = [], [], []
    # 接下来的循环是为了获取到第0列, 第1列的数据, 从而构造出ref_convert_result
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.lower().split('\t')
            
            ref_id.append(str(idx))
            
            # 将原始错误文本line[0]添加进列表
            ref_src.append(remove_pun(line[0].strip()))
            
            # 将原始正确文本line[1]添加进列表
            ref_prd.append(remove_pun(line[1].strip()))
    
    # 计算得到"原始错误文本" -> "原始正确文本", 转换所需的步骤(中间原理为最小编辑距离)
    ref_convert_result = convert_from_sentpair2edits(ref_id, ref_src, ref_prd)
    # ref_convert_result: ['0, -1', '1, -1', '2, -1', '3, 3, 别字, 卷, 券,', '4, -1', '5, -1', '6, -1', '7, -1', '8, -1', '9, 18, 别字, 做, 作,', '10, -1', '11, 0, 冗余, n, , 4, 缺失, , 南,', '12, -1', '13, -1', '14, -1', '15, -1', '16, -1', '17, -1', '18, -1', '19, -1', '20, -1', '21, -1', '22, -1', '23, 1, 别字, 园, 圆,', '24, -1', '25, -1', '26, -1', '27, -1', '28, -1', '29, -1', '30, -1', '31, -1', '32, 3, 别字, 业, 渔,', '33, -1', '34, -1', '35, -1', '36, -1', '37, -1', '38, -1', '39, -1', '40, -1', '41, -1', '42, -1', '43, -1', '44, -1', '45, -1', '46, -1', '47, -1', '48, -1', '49, -1', '50, -1', '51, -1', '52, -1', '53, -1', '54, -1', '55, -1', '56, -1', '57, 1, 缺失, , 林, 3, 冗余, 业, ,', '58, -1', '59, -1', '60, -1', '61, -1', '62, -1', '63, 4, 别字, 势, 市,', '64, 5, 别字, 微, 为,', '65, -1', '66, -1', '67, -1', '68, -1', '69, -1', '70, -1', '71, -1', '72, -1', '73, -1', '74, 1, 别字, 太, 泰,', '75, -1', '76, -1', '77, -1', '78, -1', '79, -1', '80, -1', '81, -1', '82, -1', '83, -1', '84, -1', '85, -1', '86, -1', '87, -1', '88, -1', '89, -1', '90, -1', '91, -1', '92, -1', '93, -1', '94, -1', '95, -1', '96, -1', '97, -1', '98, -1', '99, -1', '100, -1', ...... , '3140, 10, 别字, 缣, 坚,', '3141, 8, 别字, 叁, 三,', '3142, 24, 别字, 续, 序,', '3143, 18, 别字, 忝, 天,', '3144, 10, 别字, 妞, 流,', '3145, 9, 别字, 边, 力,', '3146, 68, 别字, 薏, 以,', '3147, 8, 别字, 型, 形,', '3148, 6, 别字, 椰, 业,', '3149, 7, 别字, 炸, 作,', '3150, 9, 别字, 矿, 况,']

    # 函数调用链: train.py -> model_evaluate() -> do_calculate() -> evaluate()
    detect_f1, correct_f1, final_score = evaluate(path, ref_convert_result, pred_convert_result)
    
    return detect_f1, correct_f1, final_score


def save_model(opt, model,epoch):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    model_to_save = (model.module if hasattr(model, 'module') else model)
    torch.save(model_to_save.state_dict(), os.path.join(opt.output_dir, 'best_model.pt'))

