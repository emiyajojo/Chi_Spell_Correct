# 评测集与评估脚本说明

## Task 1: 黄金评测集 (eval_data.json)

### 生成脚本

- **脚本**: `scripts/generate_eval_data.py`
- **输出**: `data/eval_data.json`（默认 1500 条：5 种噪声类型各 300 条）

### 格式

每条样本包含：

- `text`: 输入句子（带噪或正确）
- `reference`: 标准答案
- `noise_type`: `correct` | `phonetic` | `order` | `missing` | `extra`
- `entity`: 正确实体（股票名）
- `scene`: 场景标签（金融新闻、推介文案、报表摘要等）

### 运行

```bash
cd /hy-tmp/Chi_Spell_Correct
# 使用已有模板（data/checkpoint_templates.json）生成，不调 API
python scripts/generate_eval_data.py

# 若模板不足，用 LangChain+DeepSeek 补充模板（需设置 DEEPSEEK_API_KEY）
python scripts/generate_eval_data.py --use-llm

# 每类 100 条，共 500 条
python scripts/generate_eval_data.py --samples-per-type 100
```

---

## Task 2: 评估逻辑 (evaluate.py)

### 脚本位置

- **脚本**: 项目根目录 `evaluate.py`

### 指标

- **Precision / Recall / F1 / F0.5**：句子级纠错正确率，F0.5 公式：  
  `F0.5 = (1 + 0.5^2) * P * R / ((0.5^2)*P + R)`
- **分类别召回**：按 `noise_type` 统计各类召回率。
- **过纠率**：在 `noise_type == 'correct'` 的样本中，模型错误修改的比例。

### 纠错接口对接

评估时默认会调用 **`Correction.correct(text, mode="distance_L")`**（见 `Correction.py:174`）。  
若不传 `--predictions`，脚本会：

1. 将项目根加入 `sys.path`
2. `from Correction import Correction`
3. 实例化 `Correction()` 并对每条 `eval_data` 的 `text` 调用 `correct(text, mode=...)`

若已有预测结果文件（每行一条，与 `eval_data.json` 顺序一致），可传入 `--predictions` 跳过模型调用。

### 运行

```bash
cd /hy-tmp/Chi_Spell_Correct

# 使用 Correction.correct 现场预测并计算指标
python evaluate.py

# 指定纠错模式
python evaluate.py --mode distance_L

# 将本次预测结果写入文件（每行一条），便于复跑评估
python evaluate.py --output-predictions data/eval_predictions.txt

# 仅从已有预测文件计算指标（不加载模型）
python evaluate.py --predictions data/eval_predictions.txt
```

### 依赖

- 评估表格输出依赖 `tabulate`：`pip install tabulate`
- 若未安装，会退化为简单打印。
