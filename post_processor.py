# -*- coding: utf-8 -*-
"""
纠错后处理：UNK 回填、大小写还原，支持单条与 batch。
在模型输出后调用，避免 tokenizer 缺陷导致 [UNK] 与大小写错误影响评估。
"""

from __future__ import annotations

UNK_TOKEN = "[UNK]"


def _unk_flashback_one(original: str, predicted: str) -> str:
    """将 predicted 中的 [UNK] 按位置用 original 对应字符回填。"""
    if not original:
        return predicted
    orig_idx = 0
    result = []
    i = 0
    n = len(predicted)
    while i < n:
        if i + len(UNK_TOKEN) <= n and predicted[i : i + len(UNK_TOKEN)] == UNK_TOKEN:
            result.append(original[orig_idx] if orig_idx < len(original) else "?")
            orig_idx += 1
            i += len(UNK_TOKEN)
        else:
            result.append(predicted[i])
            orig_idx += 1
            i += 1
    return "".join(result)


def _case_restore_one(original: str, processed: str) -> str:
    """按 original 各位置大小写还原 processed 中的英文字母。"""
    if not original or not processed:
        return processed
    out = list(processed)
    for i in range(min(len(original), len(out))):
        if not out[i].isalpha():
            continue
        if original[i].isupper():
            out[i] = out[i].upper()
        elif original[i].islower():
            out[i] = out[i].lower()
    return "".join(out)


def post_process(original: str, predicted: str) -> str:
    """
    单条后处理：UNK 回填 + 大小写还原。
    - original: 模型输入（纠错前）文本
    - predicted: 模型输出文本
    """
    if not predicted:
        return predicted
    s = _unk_flashback_one(original, predicted)
    s = _case_restore_one(original, s)
    return s


def post_process_batch(originals: list[str], predicted_list: list[str]) -> list[str]:
    """Batch 后处理，保持与输入顺序一致。"""
    assert len(originals) == len(predicted_list)
    return [post_process(orig, pred) for orig, pred in zip(originals, predicted_list)]
