#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成金融纠错黄金评测集 eval_data.json（1500 条）。
5 种噪声类型（Correct, Phonetic, Order, Missing, Extra）各 300 条，严格标记 noise_type。
支持从 checkpoint 复用模板或通过 LangChain+DeepSeek 生成语境。
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_CSV = PROJECT_ROOT / "SimCSE" / "data" / "stocks_name.csv"
CHECKPOINT_TEMPLATES = PROJECT_ROOT / "data" / "checkpoint_templates.json"
EVAL_DATA_PATH = PROJECT_ROOT / "data" / "eval_data.json"

# 评测集规模：5 类 × SAMPLES_PER_TYPE；默认 300 条/类 = 1500 条（若需「各 100 条」可传 --samples-per-type 100，共 500 条）
NOISE_TYPES = ["correct", "phonetic", "order", "missing", "extra"]
SAMPLES_PER_TYPE = 300
TOTAL_SAMPLES = len(NOISE_TYPES) * SAMPLES_PER_TYPE

SCENES = ["金融新闻", "推介文案", "报表摘要", "研报摘要", "股吧讨论", "客服咨询", "公告解读"]

FINANCIAL_CHARS = "股票资金涨跌买卖市值仓单盘量价盈亏"
SHAPE_CONFUSION = {
    "未": "末", "末": "未", "已": "己", "己": "已", "日": "曰", "曰": "日",
    "人": "入", "入": "人", "土": "士", "士": "土", "干": "千", "千": "干",
    "王": "玉", "玉": "王", "大": "太", "太": "大", "金": "全", "全": "金",
    "稳": "隐", "隐": "稳", "健": "建", "建": "健", "华": "化", "化": "华",
    "海": "每", "每": "海", "淮": "海", "油": "由", "由": "油", "业": "亚",
    "亚": "业", "止": "正", "正": "止",
}
HOMOPHONE_GROUPS = [
    "稳健隐", "华化划话", "海淮", "业叶", "金今", "股古", "涨张章",
    "跌叠", "资子自", "持迟", "仓苍", "买卖", "报保", "券权", "未位",
    "已以", "止只", "型形", "净静", "龙隆", "油由", "正整", "克可",
]


def load_entities(csv_path: Path) -> list[str]:
    """从 stocks_name.csv 读取实体（股票简称列）。"""
    entities = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row:
                continue
            name = (row[0] or "").strip()
            if name and name not in ("不详", "None"):
                entities.append(name)
    return list(dict.fromkeys(entities))


class EvalNoiseGenerator:
    """评测集专用噪声：按类型生成 Correct / Phonetic / Order / Missing / Extra。"""

    def __init__(self, seed: int | None = None):
        self.financial_chars = list(FINANCIAL_CHARS)
        self.rng = random.Random(seed)
        try:
            import pypinyin
            self._pypinyin = True
        except ImportError:
            self._pypinyin = False

    def _phonetic(self, s: str) -> str:
        if len(s) < 2:
            return s
        chars = list(s)
        for i, c in enumerate(chars):
            if c in SHAPE_CONFUSION:
                chars[i] = SHAPE_CONFUSION[c]
                return "".join(chars)
        for group in HOMOPHONE_GROUPS:
            if chars[0] in group:
                cands = [x for x in group if x != chars[0]]
                if cands:
                    chars[0] = self.rng.choice(cands)
                    return "".join(chars)
        idx = self.rng.randint(0, len(chars) - 1)
        for group in HOMOPHONE_GROUPS:
            if chars[idx] in group:
                cands = [x for x in group if x != chars[idx]]
                if cands:
                    chars[idx] = self.rng.choice(cands)
                    return "".join(chars)
        return s

    def _order(self, s: str) -> str:
        if len(s) < 3:
            return s
        chars = list(s)
        i = self.rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _missing(self, s: str) -> str:
        if len(s) <= 2:
            return s
        n = self.rng.randint(1, min(2, len(s) - 1))
        idx = sorted(self.rng.sample(range(len(s)), n), reverse=True)
        chars = list(s)
        for i in idx:
            chars.pop(i)
        return "".join(chars)

    def _extra(self, s: str) -> str:
        chars = list(s)
        pos = self.rng.randint(0, len(chars))
        chars.insert(pos, self.rng.choice(self.financial_chars))
        return "".join(chars)

    def add_noise(self, entity: str, noise_type: str) -> str:
        if noise_type == "correct":
            return entity
        if noise_type == "phonetic":
            return self._phonetic(entity)
        if noise_type == "order":
            return self._order(entity)
        if noise_type == "missing":
            return self._missing(entity)
        if noise_type == "extra":
            return self._extra(entity)
        return entity


def load_templates_from_checkpoint() -> list[dict]:
    """从 data_augmentation 的 checkpoint 加载模板。"""
    if not CHECKPOINT_TEMPLATES.exists():
        return []
    with open(CHECKPOINT_TEMPLATES, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [t for t in data if isinstance(t, dict) and "[ENTITY]" in (t.get("template") or "")]


async def generate_templates_llm(count: int) -> list[dict]:
    """使用 LangChain + DeepSeek 生成语境模板。"""
    import os
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI
    from langchain_core.messages import HumanMessage

    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    llm = ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.8,
        max_tokens=1024,
    )
    templates = []
    batch_size = 10
    need = count
    while need > 0:
        n = min(batch_size, need)
        prompt = f"""你是一位金融文本数据构造专家。请生成 {n} 个中文金融语境模板，用于评测纠错系统。
要求：每个模板必须且仅有一处占位符 [ENTITY]；风格多样：金融新闻、推介文案、报表摘要、研报摘要、股吧讨论等。
直接输出 JSON 数组，每项：{{"template": "含 [ENTITY] 的句子", "style": "风格名"}}。不要其他说明。"""
        try:
            raw = await llm.ainvoke([HumanMessage(content=prompt)])
            text = (raw.content or "").strip()
            start, end = text.find("["), text.rfind("]") + 1
            if start >= 0 and end > start:
                arr = json.loads(text[start:end])
            else:
                arr = []
            for item in arr:
                if isinstance(item, dict) and "[ENTITY]" in (item.get("template") or ""):
                    templates.append(item)
                    need -= 1
                    if need <= 0:
                        break
        except Exception as e:
            print(f"  [WARN] LLM 模板生成失败: {e}")
            break
    return templates


def build_sentence(template: str, entity: str) -> str:
    return template.replace("[ENTITY]", entity)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成黄金评测集 eval_data.json")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--use-llm", action="store_true", help="若 checkpoint 模板不足则用 LLM 生成")
    parser.add_argument("--samples-per-type", type=int, default=SAMPLES_PER_TYPE)
    args = parser.parse_args()
    random.seed(args.seed)

    entities = load_entities(STOCKS_CSV)
    if not entities:
        raise SystemExit("未读取到实体，请检查 SimCSE/data/stocks_name.csv")
    print(f"[1] 已加载 {len(entities)} 个实体。")

    templates = load_templates_from_checkpoint()
    need_templates = max(50, (args.samples_per_type * len(NOISE_TYPES)) // 20)
    if len(templates) < need_templates and args.use_llm:
        print(f"[2] 当前模板 {len(templates)} 条，需约 {need_templates} 条，正在用 LLM 补充……")
        new_t = asyncio.run(generate_templates_llm(need_templates - len(templates)))
        templates.extend(new_t)
    if len(templates) < 10:
        # 兜底：写死少量模板
        templates = [
            {"template": "受 [ENTITY] 近期财报影响，股价波动剧烈。", "style": "金融新闻"},
            {"template": "投资者应关注 [ENTITY] 的股权质押风险。", "style": "研报摘要"},
            {"template": "建议逢低布局 [ENTITY] 等龙头标的。", "style": "推介文案"},
            {"template": "[ENTITY] 今日主力资金净流入。", "style": "报表摘要"},
            {"template": "[ENTITY] 这只票还能拿吗？", "style": "股吧讨论"},
        ] + templates
    print(f"[2] 使用 {len(templates)} 个语境模板。")

    noise_gen = EvalNoiseGenerator(seed=args.seed)
    out = []
    for noise_type in NOISE_TYPES:
        for _ in range(args.samples_per_type):
            t = random.choice(templates)
            e = random.choice(entities)
            scene = random.choice(SCENES)
            noised_entity = noise_gen.add_noise(e, noise_type)
            reference = build_sentence(t["template"], e)
            text = build_sentence(t["template"], noised_entity)
            out.append({
                "text": text,
                "reference": reference,
                "noise_type": noise_type,
                "entity": e,
                "scene": scene,
            })
    random.shuffle(out)

    EVAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[3] 已写入 {len(out)} 条 -> {EVAL_DATA_PATH}")


if __name__ == "__main__":
    main()
