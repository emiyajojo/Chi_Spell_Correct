"""
本机调用纠错服务进行测试。
请先启动服务：uvicorn serve:app --host 127.0.0.1 --port 8000
然后在本目录执行：python test.py
"""
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    r.raise_for_status()
    print("健康检查:", r.json())
    return r.json()


def test_correct(text: str, mode: str = "Levenshtein"):
    r = requests.post(
        f"{BASE_URL}/correct",
        json={"text": text, "mode": mode},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    print(f"原文: {data['original']}")
    print(f"纠错: {data['corrected']}")
    print(f"耗时: {data['time_ms']} ms\n")
    return data


if __name__ == "__main__":
    print("=== 文本纠错服务测试 ===\n")
    try:
        test_health()
    except requests.exceptions.RequestException as e:
        print(f"请先启动服务: uvicorn serve:app --host 127.0.0.1 --port 8000\n错误: {e}")
        exit(1)

    # 单条测试
    test_correct("俆家汇怎么样")
    test_correct("华鑫证卷的号吗是多少？")

    # 可选：从 demo.txt 读多行测试
    try:
        with open("demo.txt", "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if not t:
                    continue
                test_correct(t)
    except FileNotFoundError:
        pass
