from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
from model import QwenModel

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
    max_length: int = 512
    temperature: float = 0.7
    history: list = None


# 响应模型
class ChatResponse(BaseModel):
    response: str
    status: str = "success"


# 全局模型实例
qwen_model = None


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
            max_length=request.max_length,
            temperature=request.temperature
        )

        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": qwen_model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
