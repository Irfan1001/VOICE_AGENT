from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from openai import AsyncOpenAI

from backend.config import Settings


logger = logging.getLogger("ist_voice_agent.tts")

SendJson = Callable[[dict[str, Any]], Awaitable[None]]


class TTSResponder:
    def __init__(self, settings: Settings, send_json: SendJson) -> None:
        self.settings = settings
        self.send_json = send_json
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.active_reply_id: int | None = None
        self.active_task: asyncio.Task[None] | None = None
        self.closed = False

    async def stream_reply(self, reply_id: int, text: str) -> None:
        if self.closed or not text.strip():
            return

        await self.stop_current_reply(reason="superseded")
        self.active_reply_id = reply_id

        async def _run() -> None:
            await self.send_json(
                {
                    "type": "tts_started",
                    "reply_id": reply_id,
                    "format": "pcm",
                    "sample_rate": self.settings.tts_sample_rate,
                }
            )
            try:
                async with self.client.audio.speech.with_streaming_response.create(
                    model=self.settings.tts_model,
                    voice=self.settings.tts_voice,
                    input=text,
                    instructions=self.settings.tts_instructions,
                    response_format="pcm",
                    speed=self.settings.tts_speed,
                    stream_format="audio",
                ) as response:
                    async for chunk in response.iter_bytes(chunk_size=self.settings.tts_chunk_bytes):
                        if self.closed or reply_id != self.active_reply_id:
                            return
                        if not chunk:
                            continue

                        await self.send_json(
                            {
                                "type": "tts_chunk",
                                "reply_id": reply_id,
                                "audio": base64.b64encode(chunk).decode("ascii"),
                            }
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TTS streaming failed")
                if reply_id == self.active_reply_id:
                    await self.send_json({"type": "tts_error", "reply_id": reply_id, "msg": "TTS streaming failed"})
            finally:
                if reply_id == self.active_reply_id:
                    self.active_reply_id = None
                    await self.send_json({"type": "tts_done", "reply_id": reply_id})

        self.active_task = asyncio.create_task(_run())

    async def stop_current_reply(self, reason: str = "barge_in") -> bool:
        reply_id = self.active_reply_id
        if reply_id is None:
            return False

        self.active_reply_id = None

        if self.active_task is not None and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass

        self.active_task = None
        await self.send_json({"type": "tts_stopped", "reply_id": reply_id, "reason": reason})
        return True

    async def close(self) -> None:
        self.closed = True
        await self.stop_current_reply(reason="session_closed")