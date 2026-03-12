# 数据增强脚本使用说明

## 功能

- **Phase1**：用 LangChain + DeepSeek 生成 1500 个金融语境模板（含 `[ENTITY]` 占位符），支持断点续传。
- **Phase2**：从 `SimCSE/data/stocks_name.csv` 读取实体，经 `NoiseGenerator` 做五类离线噪声（Correct/Typos/Order/Missing/Extra 各 20%）。
- **Phase3**：通过 Adapter 生成并写入：
  - `span_src/data/train.json`：约 15000 条 NER 格式（`text`, `stock_name`）。
  - `SimCSE/data/train.json`：约 5000 条，其中 20% 为完全正确样本（source == target）。

## 环境

```bash
cd /hy-tmp/Chi_Spell_Correct/scripts
pip install -r requirements_data_augmentation.txt
```

需设置 DeepSeek API（Phase1 用）：

```bash
export DEEPSEEK_API_KEY="your-key"
# 可选
export DEEPSEEK_API_BASE="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-chat"
```

## 运行

```bash
# 完整流程（Phase1+2+3）
python data_augmentation.py

# 仅用已有模板做 Phase2+3（不调 API）
python data_augmentation.py --skip-phase1

# 重新生成模板（不续传）
python data_augmentation.py --no-resume

# 指定随机种子
python data_augmentation.py --seed 123
```

断点文件：`Chi_Spell_Correct/data/checkpoint_templates.json`。首次运行会先执行 Phase1 并保存模板，中断后再次运行会从该文件恢复。
