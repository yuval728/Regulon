"""SGLang engine wrapper."""
import os
import asyncio
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
import sglang as sgl
from sglang import function, system, user, assistant, gen, set_default_backend


@dataclass
class GenerationResponse:
    request_id: str
    text: str
    finish_reason: str
    usage: dict
    created: int


@dataclass
class StreamChunk:
    text: str
    finish_reason: Optional[str]


class SGLangEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.runtime = None

    async def start(self):
        # Start SGLang runtime server
        self.runtime = sgl.Runtime(
            model_path=self.model_path,
            port=30000,  # Internal port
            tokenizer_path=self.model_path,
            trust_remote_code=True,
            mem_fraction_static=0.90,
        )
        await self.runtime.start()
        print(f"SGLang runtime started with model: {self.model_path}")

    async def shutdown(self):
        if self.runtime:
            self.runtime.shutdown()

    async def generate(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> GenerationResponse:
        # Use SGLang's native chat template
        prompt = self._format_messages(messages)

        @sgl.function
        def generate_fn(s, prompt):
            s += prompt + sgl.gen("response", max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)

        state = generate_fn.run(prompt)
        response_text = state["response"]

        return GenerationResponse(
            request_id=f"req-{os.urandom(8).hex()}",
            text=response_text,
            finish_reason="stop",
            usage={  # SGLang doesn't expose token counts easily
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            created=int(asyncio.get_event_loop().time()),
        )

    async def stream_generate(
        self,
        messages: List[dict],
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        prompt = self._format_messages(messages)

        @sgl.function
        def stream_fn(s, prompt):
            s += prompt + sgl.gen("response", max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop, stream=True)

        # SGLang streaming is handled differently - use the stream option
        state = stream_fn.run(prompt, stream=True)
        
        last_text = ""
        for output in state.text_iter("response"):
            new_text = output[len(last_text):]
            last_text = output
            if new_text:
                yield StreamChunk(text=new_text, finish_reason=None)

        yield StreamChunk(text="", finish_reason="stop")

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages using SGLang's expected format."""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)