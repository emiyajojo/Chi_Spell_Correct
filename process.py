import json
import os

def count_specific_field(file_path, field_name):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    total_count = 0
    
    def recursive_search(data, target_field):
        """递归搜索字典或列表中的目标字段"""
        nonlocal total_count
        if isinstance(data, dict):
            for key, value in data.items():
                if key == target_field:
                    total_count += 1
                recursive_search(value, target_field)
        elif isinstance(data, list):
            for item in data:
                recursive_search(item, target_field)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 兼容 JSONL 格式（大文件常用）
            first_char = f.read(1)
            f.seek(0)
            
            # 如果是 JSONL (每行一个对象)
            if first_char != '[' and first_char != '{':
                for line in f:
                    if line.strip():
                        try:
                            line_data = json.loads(line)
                            recursive_search(line_data, field_name)
                        except:
                            continue
            else:
                # 标准 JSON 格式
                data = json.load(f)
                recursive_search(data, field_name)

        print(f"📊 文件: {file_path}")
        print(f"🔍 字段 [{field_name}] 出现次数: {total_count}")
        return total_count

    except Exception as e:
        print(f"🔥 统计出错: {e}")

if __name__ == "__main__":
    # 示例用法
    # 统计 SimCSE 数据中的错误实体数量
    count_specific_field("/hy-tmp/Chi_Spell_Correct/SimCSE/data/train.json", "source_company")
    
    # 统计 span_src 数据中的实体列表字段
    # 注意：如果 stock_name 是个列表，这里统计的是这个列表出现的次数
    count_specific_field("/hy-tmp/Chi_Spell_Correct/span_src/data/train.json", "stock_name")