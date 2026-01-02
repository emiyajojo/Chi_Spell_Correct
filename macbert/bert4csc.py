import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from Chi_Spell_Correct.macbert.base_model import FocalLoss

# 构建macBert4CSC模型类
class Bert4Csc(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(Bert4Csc, self).__init__()
        self.args = args
        # 此处初始化的原始模型是哈⼯⼤开源的macBERT
        self.bert = BertForMaskedLM.from_pretrained(args.bert_dir)
        # 检测⽹络对应的是从768向1的映射全连接层
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        # ⼆分类的计算函数
        self.sigmoid = nn.Sigmoid()
        # tokenizer采⽤参数模式传⼊
        self.tokenizer = tokenizer
        # 默认配置⽂件中, 将hyper_params设置为0.2
        self.w = args.hyper_params

    # 批次数据的编码函数, 主要为了将原始⽂本编码成BERT需要的若⼲重要张量
    def batch_encode(self, batch_data):
        # 最⼤⻓度+2, 为[CLS], [SEP]预留空间
        max_len = max([len(x) for x in batch_data]) + 2
        encoded_inputs = self.tokenizer.batch_encode_plus(
            batch_data,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True
        )
        # 最后以字典类型返回3个张量
        return {'input_ids': encoded_inputs['input_ids'], 
                'attention_mask': encoded_inputs['attention_mask'], 
                'token_type_ids': encoded_inputs['token_type_ids']}

    # macBert4CSC模型的前向计算逻辑
    def forward(self, texts, cor_labels=None, det_labels=None, device=None):
        # 如果传⼊了正确的标签
        if cor_labels is not None:
            # 对正确的标签进⾏batch级别的编码
            text_labels = self.batch_encode(cor_labels)['input_ids']
            # 对于标签编码等于0的位置(本质上是PAD的位置), 赋值成-100, 未来计算损失时会起到忽略不计的效果
            text_labels[text_labels == 0] = -100
            text_labels = text_labels.to(device)
        # 如果没有传⼊正确的标签, 则将标签张量text_labels赋值为None
        else:
            text_labels = None

        # 对传⼊⽂本进⾏batch级别的编码
        encoded_text = self.batch_encode(texts)
        
        # encoded_text作为字典类型, 拥有3种张量
        for key in encoded_text.keys():
            encoded_text[key] = encoded_text[key].to(device)

        # 直接将编码后的张量送⼊macBERT模型中, 得到输出张量
        bert_outputs = self.bert(**encoded_text,
                                 labels=text_labels,
                                 return_dict=True,
                                 output_hidden_states=True)
        
        # 最后⼀层隐藏层的输出张量, 送⼊Detection⽹络, 得到检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        # 如果没有传⼊正确的标签, predict函数调⽤, 推理阶段
        if text_labels is None:
            # 最后返回2个张量: Detection⽹络的检错概率, 和macBERT⽹络的输出概率分布
            outputs = (prob, bert_outputs.logits)
        # 如果传⼊了正确的标签, 训练阶段
        else:
            # det_labels: 数据迭代器中, 已经⼿动构建好的错误字符标签的one-hot格式张量
            det_labels = det_labels.to(device)
            # 设置损失函数的计算规则为FocalLoss的⼆分类模式sigmoid
            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid').cuda()
            
            # pad部分不计算损失, 只把mask == 1的位置计算有效损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            # Detection⽹络的检错概率, 进⾏mask掩码后, 作为有效的检错概率分布张量active_probs
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            # 检错标签, 进⾏mask掩码后, 作为有效的检错分布标签
            active_labels = det_labels[active_loss]
            
            # Detection⽹络的有效输出概率active_probs, 和有效检错标签active_labels, 进⾏FocalLoss计算
            det_loss = det_loss_fct(active_probs, active_labels.float())
            
            # 按照macBert4CSC计算公式, macBERT的输出损失, 和检错⽹络的FocalLoss损失, 进⾏加权求和.
            loss = self.w * bert_outputs.loss + (1 - self.w) * det_loss
            
            # 最后返回3个张量: ⽹络的总损失loss, Detection⽹络的检错概率, macBERT⽹络的输出概率分布
            outputs = (loss, self.sigmoid(prob).squeeze(-1), bert_outputs.logits)
        return outputs

    # 预测函数, 本质上执⾏的是推理过程
    def predict(self, texts, device):
        # ⾸先对原始⽂本进⾏编码, ⽣成输⼊张量inputs
        inputs = self.batch_encode(texts)
        with torch.no_grad():
            # 得到Detection⽹络的检错输出, 以及macBERT⽹络的纠错输出
            outputs = self.forward(texts, device=device)
            # 在本函数中, outputs[1]代表macBERT⽹络的输出概率分布张量
            y_hat = torch.argmax(outputs[1], dim=-1)
            
            # 对attention_mask求和得出有效⽂本⻓度, -1是为了删除最后⾯的[SEP]
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1
            
            # 初始化batch解码的结果列表
            res = []
            # 每⼀条⽂本的有效⻓度不同, 因此⼀⼀对应的处理
            for t_len, _y_hat in zip(expand_text_lens, y_hat):
                t_len = t_len.long()
                # 直接调⽤tokenizer.decode()对检错⽹络按照贪⼼算法argmax()来进⾏解码, 并将空格全部删除
                res.append(self.tokenizer.decode(_y_hat[1: t_len]).replace(' ', ''))
        return res
