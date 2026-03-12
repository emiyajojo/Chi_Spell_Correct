#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金融语境数据增强脚本：LLM 模板生成 + 离线噪声注入。
生成 span_src/train.json (NER, ~15000) 与 SimCSE/train.json (~5000)。
"""

from __future__ import annotations

import asyncio
import csv
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 项目根目录（脚本在 scripts/ 下）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_CSV = PROJECT_ROOT / "SimCSE" / "data" / "stocks_name.csv"
SPAN_SRC_TRAIN = PROJECT_ROOT / "span_src" / "data" / "train.json"
SIMCSE_TRAIN = PROJECT_ROOT / "SimCSE" / "data" / "train.json"
CHECKPOINT_DIR = PROJECT_ROOT / "data"
TEMPLATE_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_templates.json"

# 目标数量
NUM_TEMPLATES = 1500
SIMCSE_TARGET = 5000
SPAN_SRC_TARGET = 15000
CORRECT_RATIO_SIMCSE = 0.20  # SimCSE 中 20% 为完全正确样本

# 金融干扰字（用于 Extra 噪声）
FINANCIAL_CHARS = "股票资金涨跌买卖市值仓单盘量价盈亏"

# 形近字混淆表（部分常见）
SHAPE_CONFUSION = {
    "未": "末", "末": "未", "已": "己", "己": "已", "日": "曰", "曰": "日",
    "人": "入", "入": "人", "土": "士", "士": "土", "干": "千", "千": "干",
    "王": "玉", "玉": "王", "大": "太", "太": "大", "析": "折", "折": "析",
    "金": "全", "全": "金", "稳": "隐", "隐": "稳", "健": "建",
    "建": "健", "华": "化", "化": "华", "海": "每", "每": "海", "淮": "海",
    "油": "由", "由": "油", "业": "亚", "亚": "业", "止": "正", "正": "止",
}


# ---------- Pydantic 模型 ----------
class ContextTemplate(BaseModel):
    """LLM 输出的语境模板，需包含 [ENTITY] 占位符。"""
    template: str = Field(..., description="含 [ENTITY] 的句子模板")
    style: str = Field(..., description="风格标签：新闻报道/研报摘要/社交媒体/客服咨询等")


# ---------- 第一阶段：LLM 生成语境模板 ----------
def _get_llm():
    """获取 DeepSeek（OpenAI 兼容）或本地配置的 LLM。"""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI
    import os
    base_url = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    api_key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    return ChatOpenAI(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        openai_api_key=api_key,
        openai_api_base=base_url,
        temperature=0.8,
        max_tokens=1024,
    )


def _template_prompt(batch_id: int) -> str:
    return f"""你是一位金融文本数据构造专家。请生成 5 个**高度多样化**的中文金融语境模板，用于后续填入股票/实体名称。

要求：
1. 每个模板中必须且仅有一处占位符 [ENTITY]，表示将要填入的实体（如股票名、公司名）。
2. 风格需多样：新闻报道、研报摘要、社交媒体讨论、客服咨询、股吧讨论、公告解读等，尽量不重复。
3. 句子通顺、贴近真实金融场景，长度建议 10～30 字。
4. 直接输出 JSON 数组，每项格式：{{"template": "句子内容", "style": "风格名"}}

示例：
[{{"template": "受 [ENTITY] 近期财报影响，股价波动剧烈。", "style": "新闻报道"}}, {{"template": "投资者应关注 [ENTITY] 的股权质押风险。", "style": "研报摘要"}}]

请生成第 {batch_id} 批的 5 个模板（仅输出 JSON 数组，不要其他说明）："""


async def _generate_templates_batch(llm, batch_id: int, sem: asyncio.Semaphore) -> list[ContextTemplate]:
    """单批生成 5 个模板，带 Pydantic 校验。"""
    async with sem:
        from langchain_core.messages import HumanMessage
        try:
            prompt = _template_prompt(batch_id)
            raw = await llm.ainvoke([HumanMessage(content=prompt)])
            text = (raw.content or "").strip()
            # 抽取 JSON 数组
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                arr = json.loads(text[start:end])
            else:
                arr = json.loads(text) if text.startswith("[") else []
            if not isinstance(arr, list):
                arr = [arr]
            out = []
            for item in arr:
                try:
                    if not isinstance(item, dict):
                        continue
                    t = ContextTemplate(**item)
                    if "[ENTITY]" in t.template:
                        out.append(t)
                except Exception:
                    continue
            return out
        except json.JSONDecodeError as e:
            print(f"  [WARN] batch {batch_id} JSON 解析失败: {e}")
            return []
        except Exception as e:
            print(f"  [WARN] batch {batch_id} 失败: {e}")
            return []


async def phase1_generate_templates(resume: bool = True) -> list[dict]:
    """第一阶段：生成 1500 个模板，支持断点续传。"""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    templates = []
    if resume and TEMPLATE_CHECKPOINT.exists():
        with open(TEMPLATE_CHECKPOINT, "r", encoding="utf-8") as f:
            templates = json.load(f)
        print(f"[Phase1] 从断点恢复，已存在 {len(templates)} 个模板。")

    need = NUM_TEMPLATES - len(templates)
    if need <= 0:
        print(f"[Phase1] 模板数量已满足 {NUM_TEMPLATES}，跳过生成。")
        return templates

    llm = _get_llm()
    batch_size = 5
    num_batches = (need + batch_size - 1) // batch_size
    sem = asyncio.Semaphore(8)

    async def one_batch(i: int):
        return await _generate_templates_batch(llm, len(templates) // batch_size + i + 1, sem)

    for start in range(0, num_batches, 10):
        batch_indices = list(range(start, min(start + 10, num_batches)))
        tasks = [one_batch(i) for i in batch_indices]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                print(f"  [WARN] batch 异常: {r}")
                continue
            for t in r:
                templates.append(t.model_dump())
        with open(TEMPLATE_CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(templates, f, ensure_ascii=False, indent=0)
        print(f"[Phase1] 已生成 {len(templates)}/{NUM_TEMPLATES} 模板。")
        if len(templates) >= NUM_TEMPLATES:
            break

    return templates[:NUM_TEMPLATES]


# ---------- 第二阶段：读取实体与噪声注入 ----------
def load_entities(csv_path: Path) -> list[str]:
    """从 stocks_name.csv 读取实体名（股票简称列）。"""
    entities = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = (row[0] or "").strip()
            if name and name != "不详" and name != "None":
                entities.append(name)
    return list(dict.fromkeys(entities))


class NoiseGenerator:
    """离线噪声注入：Correct / Typos / Order / Missing / Extra 各 20%。"""

    def __init__(self, financial_chars: str = FINANCIAL_CHARS, seed: int | None = None):
        self.financial_chars = list(financial_chars)
        self.rng = random.Random(seed)
        self._pypinyin_available = False
        try:
            import pypinyin
            self._pypinyin_available = True
        except ImportError:
            pass

    def _typos_with_pypinyin(self, s: str) -> str:
        """同音/形近字替换：优先形近字，否则用 pypinyin 找同音字替换单字。"""
        if len(s) < 2:
            return s
        chars = list(s)
        idx = self.rng.randint(0, len(chars) - 1)
        c = chars[idx]
        if c in SHAPE_CONFUSION:
            chars[idx] = SHAPE_CONFUSION[c]
            return "".join(chars)
        if self._pypinyin_available:
            try:
                import pypinyin
                py_list = pypinyin.pinyin(c, style=pypinyin.Style.NORMAL)
                if py_list:
                    py = py_list[0][0]
                    # 同音字表（常见金融/实体相关）
                    homophones = self._get_homophones(c, py)
                    if homophones:
                        chars[idx] = self.rng.choice(homophones)
                        return "".join(chars)
            except Exception:
                pass
        return self._typos_shape(s)

    def _get_homophones(self, c: str, py: str) -> list[str]:
        """根据拼音返回常见同音字列表（不含自身）。"""
        # 常见同音字组，便于 Typos 替换
        homophone_groups = [
            "稳健隐", "华化划话", "海淮", "业叶", "金今", "股古", "涨张章",
            "跌叠", "资子自", "持迟", "仓苍", "买卖", "报保", "券权", "未位",
            "已以", "止只", "型形", "净静", "龙隆", "油由", "正整", "克可",
        ]
        for group in homophone_groups:
            if c in group:
                return [x for x in group if x != c]
        return []

    def _typos_shape(self, s: str) -> str:
        """形近字替换。"""
        chars = list(s)
        candidates = [i for i, c in enumerate(chars) if c in SHAPE_CONFUSION]
        if not candidates:
            idx = self.rng.randint(0, len(chars) - 1)
            for k, v in SHAPE_CONFUSION.items():
                if k == chars[idx]:
                    chars[idx] = v
                    return "".join(chars)
            return s
        idx = self.rng.choice(candidates)
        chars[idx] = SHAPE_CONFUSION[chars[idx]]
        return "".join(chars)

    def _order(self, s: str) -> str:
        """随机交换相邻字符顺序。"""
        if len(s) < 3:
            return s
        chars = list(s)
        i = self.rng.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _missing(self, s: str) -> str:
        """随机删除 1～2 个字符。"""
        if len(s) <= 2:
            return s
        n = self.rng.randint(1, min(2, len(s) - 1))
        idx = sorted(self.rng.sample(range(len(s)), n), reverse=True)
        chars = list(s)
        for i in idx:
            chars.pop(i)
        return "".join(chars)

    def _extra(self, s: str) -> str:
        """随机插入金融干扰字。"""
        chars = list(s)
        pos = self.rng.randint(0, len(chars))
        char = self.rng.choice(self.financial_chars)
        chars.insert(pos, char)
        return "".join(chars)

    def add_noise(
        self,
        entity: str,
        mode: Literal["correct", "typos", "order", "missing", "extra"],
    ) -> str:
        if mode == "correct":
            return entity
        if mode == "typos":
            return self._typos_with_pypinyin(entity)
        if mode == "order":
            return self._order(entity)
        if mode == "missing":
            return self._missing(entity)
        if mode == "extra":
            return self._extra(entity)
        return entity

    def sample_noise_type(self) -> Literal["correct", "typos", "order", "missing", "extra"]:
        return self.rng.choice(["correct", "typos", "order", "missing", "extra"])


# ---------- 第三阶段：数据对齐与 Adapter 导出 ----------
def build_sentence(template: str, entity: str) -> str:
    """将模板中的 [ENTITY] 替换为实体。"""
    return template.replace("[ENTITY]", entity)


def adapter_span_src(text: str, stock_name: list[str]) -> dict:
    """生成 span_src 单条格式。"""
    return {"text": text, "stock_name": stock_name}


def adapter_simcse(
    source: str,
    target: str,
    source_company: str,
    target_company: str,
) -> dict:
    """生成 SimCSE 单条格式。"""
    return {
        "source": source,
        "target": target,
        "source_company": source_company,
        "target_company": target_company,
    }


def phase3_export(
    templates: list[dict],
    entities: list[str],
    noise_gen: NoiseGenerator,
    simcse_target: int = SIMCSE_TARGET,
    span_src_target: int = SPAN_SRC_TARGET,
    correct_ratio: float = CORRECT_RATIO_SIMCSE,
) -> tuple[list[dict], list[dict]]:
    """
    根据模板与实体生成两条数据流，并截断到目标数量。
    - SimCSE: 20% 正确样本（source == target），其余为噪声→正确。
    - span_src: 全部为「带噪声句子 + 噪声实体列表」。
    """
    simcse_data: list[dict] = []
    span_src_data: list[dict] = []

    n_correct_simcse = int(simcse_target * correct_ratio)
    n_noisy_simcse = simcse_target - n_correct_simcse

    # SimCSE：先 20% 完全正确，再 80% 噪声→正确
    for _ in range(n_correct_simcse):
        t = random.choice(templates)
        e = random.choice(entities)
        sent = build_sentence(t["template"], e)
        simcse_data.append(
            adapter_simcse(source=sent, target=sent, source_company=e, target_company=e)
        )
    for _ in range(n_noisy_simcse):
        t = random.choice(templates)
        e = random.choice(entities)
        mode = noise_gen.sample_noise_type()
        noisy_e = noise_gen.add_noise(e, mode)
        noisy_sent = build_sentence(t["template"], noisy_e)
        correct_sent = build_sentence(t["template"], e)
        simcse_data.append(
            adapter_simcse(
                source=noisy_sent,
                target=correct_sent,
                source_company=noisy_e,
                target_company=e,
            )
        )

    # span_src: 15000 条，每条 = 模板填噪声实体，stock_name 为 [噪声实体]
    for _ in range(span_src_target):
        t = random.choice(templates)
        e = random.choice(entities)
        mode = noise_gen.sample_noise_type()
        noisy_e = noise_gen.add_noise(e, mode)
        text = build_sentence(t["template"], noisy_e)
        span_src_data.append(adapter_span_src(text, [noisy_e]))

    return simcse_data[:simcse_target], span_src_data[:span_src_target]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="金融语境数据增强")
    parser.add_argument("--skip-phase1", action="store_true", help="跳过 LLM 模板生成，使用已有 checkpoint")
    parser.add_argument("--shuffle-only", action="store_true", help="仅打乱当前 train.json，不生成新数据、不调 API")
    parser.add_argument("--no-resume", action="store_true", help="Phase1 不续传，重新生成")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    # 仅打乱模式：读现有文件 -> 打乱 -> 写回
    if args.shuffle_only:
        def _load(path: Path) -> list:
            if not path.exists():
                print(f"[shuffle-only] 文件不存在，跳过: {path}")
                return []
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        simcse = _load(SIMCSE_TRAIN)
        span_src = _load(SPAN_SRC_TRAIN)
        if simcse:
            random.shuffle(simcse)
            SIMCSE_TRAIN.parent.mkdir(parents=True, exist_ok=True)
            with open(SIMCSE_TRAIN, "w", encoding="utf-8") as f:
                json.dump(simcse, f, ensure_ascii=False, indent=4)
            print(f"[shuffle-only] 已打乱并写回 SimCSE: {len(simcse)} 条 -> {SIMCSE_TRAIN}")
        if span_src:
            random.shuffle(span_src)
            SPAN_SRC_TRAIN.parent.mkdir(parents=True, exist_ok=True)
            with open(SPAN_SRC_TRAIN, "w", encoding="utf-8") as f:
                json.dump(span_src, f, ensure_ascii=False, indent=4)
            print(f"[shuffle-only] 已打乱并写回 span_src: {len(span_src)} 条 -> {SPAN_SRC_TRAIN}")
        if not simcse and not span_src:
            print("[shuffle-only] 未找到任何 train.json，无操作。")
        return

    # Phase1
    if args.skip_phase1:
        if not TEMPLATE_CHECKPOINT.exists():
            raise FileNotFoundError(f"未找到模板 checkpoint: {TEMPLATE_CHECKPOINT}，请先运行 Phase1。")
        with open(TEMPLATE_CHECKPOINT, "r", encoding="utf-8") as f:
            templates = json.load(f)
        print(f"[Phase1] 已加载 {len(templates)} 个模板。")
    else:
        templates = asyncio.run(phase1_generate_templates(resume=not args.no_resume))

    if len(templates) < 100:
        print("[WARN] 模板过少，建议至少 100 个再跑 Phase3。")

    # Phase2
    entities = load_entities(STOCKS_CSV)
    print(f"[Phase2] 已加载 {len(entities)} 个实体。")
    noise_gen = NoiseGenerator(seed=args.seed)

    # Phase3
    simcse_list, span_src_list = phase3_export(
        templates, entities, noise_gen,
        simcse_target=SIMCSE_TARGET,
        span_src_target=SPAN_SRC_TARGET,
        correct_ratio=CORRECT_RATIO_SIMCSE,
    )

    # 与原有数据合并并打乱
    def load_existing(path: Path, default: list) -> list:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] 读取原有数据失败 {path}: {e}，仅使用新数据。")
        return default

    existing_simcse = load_existing(SIMCSE_TRAIN, [])
    existing_span_src = load_existing(SPAN_SRC_TRAIN, [])
    simcse_merged = existing_simcse + simcse_list
    span_src_merged = existing_span_src + span_src_list
    random.shuffle(simcse_merged)
    random.shuffle(span_src_merged)

    SPAN_SRC_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    SIMCSE_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(SPAN_SRC_TRAIN, "w", encoding="utf-8") as f:
        json.dump(span_src_merged, f, ensure_ascii=False, indent=4)
    with open(SIMCSE_TRAIN, "w", encoding="utf-8") as f:
        json.dump(simcse_merged, f, ensure_ascii=False, indent=4)
    print(f"[Phase3] 已合并原有+新数据并打乱 -> span_src: {len(existing_span_src)} + {len(span_src_list)} = {len(span_src_merged)} 条 -> {SPAN_SRC_TRAIN}")
    print(f"[Phase3] 已合并原有+新数据并打乱 -> SimCSE: {len(existing_simcse)} + {len(simcse_list)} = {len(simcse_merged)} 条 -> {SIMCSE_TRAIN}")


if __name__ == "__main__":
    main()
