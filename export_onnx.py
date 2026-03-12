"""
将 MacBERT、SimCSE、Span NER 三个模型分别导出为 ONNX 格式，便于跨平台/加速推理。
运行方式：在项目根目录执行 python export_onnx.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

# 默认导出用的序列长度，与推理时一致
ONNX_MAX_LENGTH = 128
ONNX_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'onnx_output')


def export_macbert_onnx():
    """导出 MacBERT (Bert4Csc) 为 ONNX：输入为编码后的 input_ids/attention_mask/token_type_ids，输出为 logits（用于 argmax 解码）。"""
    from transformers import BertTokenizer
    from macbert.config import Args
    from macbert.bert4csc import Bert4Csc

    bert_path = os.path.join(PROJECT_ROOT, 'model', 'macbert_org')
    model_pt = os.path.join(PROJECT_ROOT, 'macbert', 'output', 'bert4csc', 'best_model.pt')
    if not os.path.exists(model_pt):
        print('[export_onnx] 跳过 MacBERT：未找到 {}'.format(model_pt))
        return
    if not os.path.exists(bert_path):
        print('[export_onnx] 跳过 MacBERT：未找到 bert 目录 {}'.format(bert_path))
        return

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    args = Args().get_parser()
    args.bert_dir = bert_path
    model = Bert4Csc(args, tokenizer)
    model.load_state_dict(torch.load(model_pt, map_location='cpu'), strict=True)
    model.eval()

    # 仅导出「张量进 -> 张量出」子图，便于 ONNX；推理时由 main 里用 tokenizer 编码/解码
    class MacBERTEncoderWrapper(nn.Module):
        def __init__(self, bert4csc):
            super().__init__()
            self.bert = bert4csc.bert
            self.detection = bert4csc.detection
            self.sigmoid = bert4csc.sigmoid

        def forward(self, input_ids, attention_mask, token_type_ids):
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
                output_hidden_states=True
            )
            prob = self.detection(bert_out.hidden_states[-1])
            # 推理时用 logits 做 argmax 解码，与原 predict 一致
            return self.sigmoid(prob), bert_out.logits

    wrapper = MacBERTEncoderWrapper(model)
    dummy_input_ids = torch.randint(0, 21128, (1, ONNX_MAX_LENGTH), dtype=torch.long)
    dummy_attention = torch.ones(1, ONNX_MAX_LENGTH, dtype=torch.long)
    dummy_token_types = torch.zeros(1, ONNX_MAX_LENGTH, dtype=torch.long)

    out_path = os.path.join(ONNX_OUTPUT_DIR, 'macbert.onnx')
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention, dummy_token_types),
        out_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['prob', 'logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'token_type_ids': {0: 'batch', 1: 'seq_len'},
            'prob': {0: 'batch', 1: 'seq_len'},
            'logits': {0: 'batch', 1: 'seq_len'},
        },
        opset_version=14,
    )
    print('[export_onnx] MacBERT 已导出: {}'.format(out_path))


def export_simcse_onnx():
    """导出 SimCSE (TextBackbone) 为 ONNX：输入同上，输出为归一化后的句向量。"""
    from transformers import BertTokenizer
    from SimCSE.model import TextBackbone

    bert_path = os.path.join(PROJECT_ROOT, 'model', 'bert-base-chinese')
    model_pt = os.path.join(PROJECT_ROOT, 'SimCSE', 'output', 'sup_model.pt')
    if not os.path.exists(model_pt):
        print('[export_onnx] 跳过 SimCSE：未找到 {}'.format(model_pt))
        return
    if not os.path.exists(bert_path):
        print('[export_onnx] 跳过 SimCSE：未找到 bert 目录 {}'.format(bert_path))
        return

    model = TextBackbone(bert_path=bert_path)
    model.load_state_dict(torch.load(model_pt, map_location='cpu'), strict=True)
    model.eval()
    model = model.cpu()

    dummy_input_ids = torch.randint(0, 21128, (1, ONNX_MAX_LENGTH), dtype=torch.long)
    dummy_attention = torch.ones(1, ONNX_MAX_LENGTH, dtype=torch.long)
    dummy_token_types = torch.zeros(1, ONNX_MAX_LENGTH, dtype=torch.long)

    out_path = os.path.join(ONNX_OUTPUT_DIR, 'simcse.onnx')
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention, dummy_token_types),
        out_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'token_type_ids': {0: 'batch', 1: 'seq_len'},
            'embedding': {0: 'batch'},
        },
        opset_version=14,
    )
    print('[export_onnx] SimCSE 已导出: {}'.format(out_path))


def export_span_onnx():
    """导出 Span NER 为 ONNX：输入同上，输出为 start_logits, end_logits。"""
    import json
    from transformers import BertTokenizer
    from span_src.config import Args as SpanArgs
    from span_src.model import build_model

    bert_path = os.path.join(PROJECT_ROOT, 'model', 'bert-base-chinese')
    ent2id_path = os.path.join(PROJECT_ROOT, 'span_src', 'data', 'span_ent2id.json')
    model_pt = os.path.join(PROJECT_ROOT, 'span_src', 'output', 'best_model.pt')
    if not os.path.exists(model_pt):
        print('[export_onnx] 跳过 Span：未找到 {}'.format(model_pt))
        return
    if not os.path.exists(ent2id_path):
        print('[export_onnx] 跳过 Span：未找到 {}'.format(ent2id_path))
        return

    with open(ent2id_path, encoding='utf-8') as f:
        ent2id = json.load(f)
    opt = SpanArgs().get_parser()
    opt.bert_dir = bert_path
    model = build_model(
        'span', opt.bert_dir, opt,
        num_tags=len(ent2id) + 1,
        dropout_prob=opt.dropout_prob,
        loss_type=opt.loss_type,
    )
    model.load_state_dict(torch.load(model_pt, map_location='cpu'), strict=True)
    model.eval()

    dummy_input_ids = torch.randint(0, 21128, (1, ONNX_MAX_LENGTH), dtype=torch.long)
    dummy_attention = torch.ones(1, ONNX_MAX_LENGTH, dtype=torch.long)
    dummy_token_types = torch.zeros(1, ONNX_MAX_LENGTH, dtype=torch.long)

    out_path = os.path.join(ONNX_OUTPUT_DIR, 'span.onnx')
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention, dummy_token_types),
        out_path,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['start_logits', 'end_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'token_type_ids': {0: 'batch', 1: 'seq_len'},
            'start_logits': {0: 'batch', 1: 'seq_len'},
            'end_logits': {0: 'batch', 1: 'seq_len'},
        },
        opset_version=14,
    )
    print('[export_onnx] Span NER 已导出: {}'.format(out_path))


if __name__ == '__main__':
    print('开始导出 ONNX 模型，输出目录: {}'.format(ONNX_OUTPUT_DIR))
    # export_macbert_onnx()
    export_simcse_onnx()
    export_span_onnx()
    print('导出结束。')
