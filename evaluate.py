#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融纠错评测脚本：读取黄金评测集与模型预测结果，计算 P/R/F1/F0.5、分类别报告、过纠率，并表格化输出。
预留对 Correction.correct 的调用逻辑，可直接跑模型或从文件读预测。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval_data.json"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "eval_predictions.json"


def load_eval_data(path: Path) -> list[dict]:
    """加载黄金评测集。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_correction_on_eval(eval_data: list[dict], mode: str = "distance_L") -> list[str]:
    """
    调用纠错接口对评测集逐条预测。
    接口对齐：Correction.py 中 correct(self, text, mode="distance_L")。
    """
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from Correction import Correction

    corr = Correction()
    predictions = []
    for item in eval_data:
        text = (item.get("text") or "").strip()
        pred = corr.correct(text, mode=mode)
        predictions.append(pred if pred is not None else text)
    return predictions


def load_predictions(path: Path) -> list[str]:
    """从文件加载预测结果（每行一条，与 eval_data 顺序一致，空行也保留）。"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def compute_metrics(eval_data: list[dict], predictions: list[str]) -> dict:
    """
    计算核心指标与分类别统计。
    - Precision: 在所有「模型做了修改」的样本中，修改后等于 reference 的比例。
    - Recall: 在所有「需要纠错」(reference != text) 的样本中，被正确纠错的比例。
    - F1, F0.5: F0.5 = (1+0.5^2)*P*R / ((0.5^2)*P + R)，强调准确率。
    - 过纠率: noise_type=='correct' 的样本中，模型错误修改的比例。
    """
    assert len(eval_data) == len(predictions)
    # 需要纠错的样本（reference != text）
    need_correct = [(i, d) for i, d in enumerate(eval_data) if d["reference"] != d["text"]]
    # 模型做了修改的样本（pred != text）
    model_changed = [(i, d) for i, d in enumerate(eval_data) if predictions[i] != d["text"]]
    # 修改正确：在「做了修改」的样本中 pred == reference 的个数
    correct_changes = sum(1 for i, _ in model_changed if predictions[i] == eval_data[i]["reference"])
    # 需要纠错且纠对
    correct_fixes = sum(1 for i, d in need_correct if predictions[i] == d["reference"])

    n_need = len(need_correct)
    n_changed = len(model_changed)
    # Precision: 做了修改的样本中，修改正确的比例
    precision = correct_changes / n_changed if n_changed else 0.0
    # Recall: 需要纠错的样本中，被纠对的比例
    recall = correct_fixes / n_need if n_need else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    # F0.5 = (1 + 0.5^2) * P * R / ((0.5^2) * P + R)
    denom = (0.5 ** 2) * precision + recall
    f05 = (1 + 0.5 ** 2) * precision * recall / denom if denom else 0.0

    # 过纠率：noise_type == 'correct' 的样本中，pred != text 的比例
    correct_type_samples = [i for i, d in enumerate(eval_data) if d.get("noise_type") == "correct"]
    n_correct_type = len(correct_type_samples)
    over_correct = sum(1 for i in correct_type_samples if predictions[i] != eval_data[i]["text"])
    over_correction_rate = over_correct / n_correct_type if n_correct_type else 0.0

    # 分类别召回：按 noise_type 统计「该类型样本中 pred==reference 的比例」
    breakdown = {}
    for nt in ["correct", "phonetic", "order", "missing", "extra"]:
        indices = [i for i, d in enumerate(eval_data) if d.get("noise_type") == nt]
        if not indices:
            breakdown[nt] = {"count": 0, "correct": 0, "recall": 0.0}
            continue
        correct_count = sum(1 for i in indices if predictions[i] == eval_data[i]["reference"])
        breakdown[nt] = {
            "count": len(indices),
            "correct": correct_count,
            "recall": correct_count / len(indices),
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f05": f05,
        "over_correction_rate": over_correction_rate,
        "n_total": len(eval_data),
        "n_need_correct": n_need,
        "n_model_changed": n_changed,
        "breakdown": breakdown,
    }


def print_report(metrics: dict):
    """使用 tabulate 打印整齐表格。"""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print("\n========== 核心指标 ==========")
    core = [
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1-Score", f"{metrics['f1']:.4f}"],
        ["F0.5-Score", f"{metrics['f05']:.4f}"],
        ["过纠率 (noise_type=correct)", f"{metrics['over_correction_rate']:.4f}"],
    ]
    if tabulate:
        print(tabulate(core, headers=["Metric", "Value"], tablefmt="simple"))
    else:
        for row in core:
            print(f"  {row[0]}: {row[1]}")

    print("\n========== 分类别召回 (Breakdown) ==========")
    rows = []
    for nt, v in metrics["breakdown"].items():
        rows.append([nt, v["count"], v["correct"], f"{v['recall']:.4f}"])
    if tabulate:
        print(tabulate(rows, headers=["noise_type", "count", "correct", "recall"], tablefmt="simple"))
    else:
        print("  noise_type  count  correct  recall")
        for r in rows:
            print(f"  {r[0]}  {r[1]}  {r[2]}  {r[3]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="金融纠错评测：P/R/F1/F0.5 + 分类别 + 过纠率")
    parser.add_argument("--eval-data", type=Path, default=EVAL_DATA_PATH, help="黄金评测集 JSON 路径")
    parser.add_argument("--predictions", type=Path, default=None, help="预测结果文件（每行一条）；不传则调用 Correction.correct 现场预测")
    parser.add_argument("--mode", type=str, default="distance_L", help="纠错模式，传入 Correction.correct(text, mode=...)")
    parser.add_argument("--output-predictions", type=Path, default=None, help="若使用模型预测，将预测结果写入该文件（每行一条）")
    args = parser.parse_args()

    if not args.eval_data.exists():
        print(f"评测集不存在: {args.eval_data}，请先运行 scripts/generate_eval_data.py 生成。")
        sys.exit(1)

    eval_data = load_eval_data(args.eval_data)
    if args.predictions is not None and args.predictions.exists():
        predictions = load_predictions(args.predictions)
        print(f"已从文件加载预测: {args.predictions}, 共 {len(predictions)} 条")
    else:
        print("未提供预测文件，正在调用 Correction.correct 进行预测……")
        predictions = run_correction_on_eval(eval_data, mode=args.mode)
        if args.output_predictions:
            args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_predictions, "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write((p or "").replace("\n", " ") + "\n")
            print(f"预测结果已写入: {args.output_predictions}")

    if len(predictions) != len(eval_data):
        print(f"预测条数 {len(predictions)} 与评测集 {len(eval_data)} 不一致，请检查。")
        sys.exit(1)

    # --- 插入以下 Debug 代码 ---
    print("\n" + "!"*30)
    print("🔍 [Pro Debug] 正在抽样分析过纠原因...")
    debug_count = 0
    for i, item in enumerate(eval_data):
        # 专门找那些 noise_type 是 correct（本不该改），但模型却改了的样本
        if item.get("noise_type") == "correct" and predictions[i] != item["text"]:
            print(f"\n[Case {debug_count+1}] 索引: {i}")
            print(f"原句 (Input):  |{item['text']}|")
            print(f"预测 (Pred):   |{predictions[i]}|")
            
            # 检查长度和字符差异
            if len(item['text']) != len(predictions[i]):
                print(f"⚠️ 长度不一致! 原句:{len(item['text'])} char, 预测:{len(predictions[i])} char")
            
            # 检查是否包含特殊 Token
            for token in ["[CLS]", "[SEP]", "[PAD]", " "]:
                if token in predictions[i]:
                    print(f"⚠️ 预测结果中发现了干扰字符: '{token}'")
            
            debug_count += 1
            if debug_count >= 50: # 打印 50 条足够了
                break
    print("!"*30 + "\n")
    # ----------------------------
    metrics = compute_metrics(eval_data, predictions)
    print_report(metrics)


if __name__ == "__main__":
    main()
