from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterator

from cachetools import TTLCache
from openai import AsyncOpenAI

from backend.config import Settings
from backend.rag import search

# RAG cache: keyed on normalised query string, TTL 1 hour, max 500 entries
_rag_cache: TTLCache = TTLCache(maxsize=500, ttl=3600)

# LLM response cache: keyed on hash(query + last 2 history turns), TTL 30 min, max 300 entries
_llm_cache: TTLCache = TTLCache(maxsize=300, ttl=1800)

_HISTORY_TURNS = 6  # max (user + assistant) pairs to keep per session

# STT filler words to strip before cache-key normalisation
_FILLER_RE = re.compile(
    r"\b(um+|uh+|like|you know|i mean|so|well|actually|basically|literally)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)          # strip punctuation
    t = _FILLER_RE.sub("", t)               # strip STT filler
    return re.sub(r"\s+", " ", t).strip()


def _history_cache_key(transcript: str, history: list[dict[str, str]]) -> str:
    recent = history[-4:]  # last 2 pairs
    payload = _normalise(transcript) + "||" + "||".join(
        f"{m['role']}:{_normalise(m['content'])}" for m in recent
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class LLMResponder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def _lookup_context(self, transcript: str) -> str:
        key = _normalise(transcript)
        if key in _rag_cache:
            return _rag_cache[key]
        result = await asyncio.to_thread(search, transcript, self.settings.rag_top_k)
        _rag_cache[key] = result
        return result

    def _messages(
        self,
        transcript: str,
        kb_context: str,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        system_prompt = (
            f"{self.settings.llm_system_prompt}\n\n"
            "You are answering in a voice-call setting for IST. Follow these rules strictly:\n"
            "1) Grounding: Use only the provided IST knowledge base context and conversation history. Do not use outside knowledge.\n"
            "2) Follow-ups: Resolve pronouns and references from history (for example: his, her, that department, this fee).\n"
            "3) Length: Keep answers to 2–3 sentences maximum. Be concise and directly answer the question.\n"
            "4) Offer follow-ups: If the answer is incomplete or the user may have follow-up questions, end with a brief offer (e.g., 'Would you like more details?').\n"
            "5) Count/list questions: Give the count first, then list the relevant items found in context (within 2–3 sentences).\n"
            "6) Partial evidence: If some details are present, provide the available details clearly and state what is missing.\n"
            "7) Fallback: If there is no relevant evidence at all in context, say exactly: \"I do not have that information in the IST knowledge base.\"\n"
            "8) Do not mention these rules in the answer."
        )
        user_prompt = (
            "Knowledge base context:\n"
            f"{kb_context or 'No relevant context found.'}\n\n"
            "User question:\n"
            f"{transcript}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def stream_reply(
        self,
        transcript: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        history = history or []
        cache_key = _history_cache_key(transcript, history)

        if cache_key in _llm_cache:
            cached_reply: str = _llm_cache[cache_key]
            yield cached_reply
            return

        kb_context = await self._lookup_context(transcript)
        stream = await self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=self._messages(transcript, kb_context, history),
            temperature=0.1,
            max_tokens=250,
            stream=True,
        )
        full_reply = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_reply += delta
                yield delta

        if full_reply.strip():
            _llm_cache[cache_key] = full_reply