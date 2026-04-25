from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from backend.config import Settings
from backend.rag import search


class LLMResponder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _lookup_context(self, transcript: str) -> str:
        return await asyncio.to_thread(search, transcript, self.settings.rag_top_k)

    def _messages(self, transcript: str, kb_context: str) -> list[dict[str, str]]:
        system_prompt = (
            f"{self.settings.llm_system_prompt}\n\n"
            "You must answer only from the provided IST University knowledge base context. "
            "Do not use outside knowledge. If the answer is not clearly supported by the context, "
            'say exactly: "I do not have that information in the IST knowledge base." '
            "Keep answers concise and factual."
        )
        user_prompt = (
            "Knowledge base context:\n"
            f"{kb_context or 'No relevant context found.'}\n\n"
            "User question:\n"
            f"{transcript}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def stream_reply(self, transcript: str) -> AsyncIterator[str]:
        kb_context = await self._lookup_context(transcript)
        stream = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._messages(transcript, kb_context),
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta