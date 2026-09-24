"""FastAPI wrapper for OpenAI-compatible API with auth and streaming."""
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, AsyncGenerator
import json
import asyncio
from contextlib import asynccontextmanager

from serve.vllm_engine import VLLMEngine
from serve.sglang_engine import SGLangEngine


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = {"demo-key-123"}  # In production, load from env/config


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.95, ge=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: dict


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[dict]


engine = None  # Will be initialized on startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    import os
    engine_type = os.environ.get("ENGINE_TYPE", "vllm").lower()
    model_path = os.environ.get("MODEL_PATH", "./merged_16bit")

    if engine_type == "vllm":
        engine = VLLMEngine(model_path)
    elif engine_type == "sglang":
        engine = SGLangEngine(model_path)
    else:
        raise ValueError(f"Unknown ENGINE_TYPE: {engine_type}")

    await engine.start()
    yield
    await engine.shutdown()


app = FastAPI(
    title="Regulon API",
    version="1.0.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": type(engine).__name__}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            stream_response(messages, request),
            media_type="text/event-stream",
        )

    response = await engine.generate(
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{response.request_id}",
        created=int(response.created),
        model=request.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=Message(role="assistant", content=response.text),
            finish_reason=response.finish_reason,
        )],
        usage=response.usage,
    )


async def stream_response(messages: List[dict], request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    request_id = f"chatcmpl-{asyncio.current_task().get_name()}"
    created = int(asyncio.get_event_loop().time())

    async for chunk in engine.stream_generate(
        messages=messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    ):
        yield f"data: {json.dumps(ChatCompletionChunk(
            id=request_id,
            created=created,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {"content": chunk.text} if chunk.text else {},
                "finish_reason": chunk.finish_reason,
            }],
        ).model_dump())}\n\n"

    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)