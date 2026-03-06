"""
FastAPI 封装文本纠错推理服务。
启动方式（在项目根目录执行）：
  uvicorn serve:app --host 0.0.0.0 --port 8000
本机测试：python test.py
"""
import sys
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 延迟导入，确保 path 已设置
from main import Correction

app = FastAPI(
    title="文本纠错 API",
    description="专有名称（股票）+ 通用文本纠错：先 NER+SimCSE 再 MacBERT",
    version="1.0",
)

# 启动时加载一次模型，全局复用
corrector = None


@app.on_event("startup")
def startup():
    global corrector
    corrector = Correction()


class CorrectRequest(BaseModel):
    text: str = Field(..., description="待纠错文本")
    mode: str = Field(default="Levenshtein", description="专有实体匹配模式：Levenshtein 或 distance_L")


class CorrectResponse(BaseModel):
    original: str = Field(..., description="原始输入")
    corrected: str = Field(..., description="纠错结果")
    time_ms: float = Field(..., description="耗时（毫秒）")


@app.get("/")
def root():
    return {"service": "text-correction", "docs": "/docs", "correct": "POST /correct"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": corrector is not None}


@app.post("/correct", response_model=CorrectResponse)
def correct(req: CorrectRequest):
    if corrector is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    t0 = time.perf_counter()
    try:
        corrected = corrector.correct(text, mode=req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    time_ms = (time.perf_counter() - t0) * 1000
    return CorrectResponse(original=text, corrected=corrected, time_ms=round(time_ms, 2))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
