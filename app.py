from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from model import QwenModel
from fastapi.responses import StreamingResponse
from transformers.generation.streamers import TextIteratorStreamer
import asyncio
from threading import Thread


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen API Service")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    history: list = None


# 响应模型
class ChatResponse(BaseModel):
    response: str
    status: str = "success"


# 全局模型实例
qwen_model = None


def generate_stream(request: ChatRequest):
    """流式生成响应"""
    global model, tokenizer

    # 创建streamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 准备输入
    inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)

    # 生成参数
    generation_kwargs = dict(
        inputs=inputs,
        streamer=streamer,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        do_sample=True,
    )

    # 在新线程中生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 异步迭代streamer
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    response_text = ""
    for new_text in streamer:
        response_text += new_text
        # SSE格式
        yield f" {new_text}\n\n"

    # 结束标记
    yield " [DONE]\n\n"


@app.on_event("startup")
async def startup_event():
    global qwen_model
    try:
        logger.info("Loading Qwen model...")
        qwen_model = QwenModel()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if qwen_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        response = qwen_model.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream")
async def chat(request: ChatRequest):
    try:
        if qwen_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        return StreamingResponse(
            generate_stream(request),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": qwen_model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
