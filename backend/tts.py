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
        self.active_generation = 0
        self.transition_lock = asyncio.Lock()
        self.closed = False

    async def stream_reply(self, reply_id: int, text: str) -> None:
        if self.closed or not text.strip():
            return

        async with self.transition_lock:
            await self._stop_current_locked(reason="superseded")
            self.active_reply_id = reply_id
            self.active_generation += 1
            generation = self.active_generation

        async def _run() -> None:
            await self.send_json({"type": "tts_buffer_clear", "reply_id": reply_id})
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
                        if self.closed or reply_id != self.active_reply_id or generation != self.active_generation:
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
                        # Yield so STT timers and other tasks are not starved by tight chunk loops.
                        await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("TTS streaming failed")
                if reply_id == self.active_reply_id:
                    await self.send_json({"type": "tts_error", "reply_id": reply_id, "msg": "TTS streaming failed"})
            finally:
                if reply_id == self.active_reply_id and generation == self.active_generation:
                    self.active_reply_id = None
                    await self.send_json({"type": "tts_done", "reply_id": reply_id})

        self.active_task = asyncio.create_task(_run())

    async def _stop_current_locked(self, reason: str = "barge_in") -> bool:
        reply_id = self.active_reply_id
        if reply_id is None:
            return False

        self.active_reply_id = None
        self.active_generation += 1

        if self.active_task is not None and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass

        self.active_task = None
        await self.send_json({"type": "tts_stopped", "reply_id": reply_id, "reason": reason})
        return True

    async def stop_current_reply(self, reason: str = "barge_in") -> bool:
        async with self.transition_lock:
            return await self._stop_current_locked(reason=reason)

    # ------------------------------------------------------------------
    # Sentence-pipelined TTS: begin → push_sentence* → end_streaming_reply
    # Allows TTS to start on the first sentence while LLM is still
    # generating subsequent sentences, reducing perceived latency.
    # ------------------------------------------------------------------

    async def begin_streaming_reply(self, reply_id: int) -> None:
        """Cancel any current TTS, then start draining sentences as they arrive."""
        if self.closed:
            return
        async with self.transition_lock:
            await self._stop_current_locked(reason="superseded")
            self.active_reply_id = reply_id
            self.active_generation += 1
            generation = self.active_generation

        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.active_task = asyncio.create_task(
            self._drain_sentences(reply_id, generation)
        )

    async def push_sentence(self, reply_id: int, sentence: str) -> None:
        """Enqueue a sentence for TTS. No-op if this reply is no longer active."""
        if self.closed or reply_id != self.active_reply_id:
            return
        queue: asyncio.Queue[str | None] | None = getattr(self, "_sentence_queue", None)
        if queue is not None:
            await queue.put(sentence)

    async def end_streaming_reply(self, reply_id: int) -> None:
        """Signal that no more sentences will be pushed; drain task will finish."""
        if reply_id != self.active_reply_id:
            return
        queue: asyncio.Queue[str | None] | None = getattr(self, "_sentence_queue", None)
        if queue is not None:
            await queue.put(None)  # sentinel

    async def _drain_sentences(self, reply_id: int, generation: int) -> None:
        """Consume sentences from _sentence_queue and TTS each one sequentially."""
        await self.send_json({"type": "tts_buffer_clear", "reply_id": reply_id})
        await self.send_json(
            {
                "type": "tts_started",
                "reply_id": reply_id,
                "format": "pcm",
                "sample_rate": self.settings.tts_sample_rate,
            }
        )
        try:
            queue: asyncio.Queue[str | None] = self._sentence_queue
            while True:
                sentence = await queue.get()
                if sentence is None:
                    # Sentinel — no more sentences
                    break
                if self.closed or reply_id != self.active_reply_id or generation != self.active_generation:
                    break
                if not sentence.strip():
                    continue
                try:
                    async with self.client.audio.speech.with_streaming_response.create(
                        model=self.settings.tts_model,
                        voice=self.settings.tts_voice,
                        input=sentence,
                        instructions=self.settings.tts_instructions,
                        response_format="pcm",
                        speed=self.settings.tts_speed,
                        stream_format="audio",
                    ) as response:
                        async for chunk in response.iter_bytes(chunk_size=self.settings.tts_chunk_bytes):
                            if self.closed or reply_id != self.active_reply_id or generation != self.active_generation:
                                return
                            if chunk:
                                await self.send_json(
                                    {
                                        "type": "tts_chunk",
                                        "reply_id": reply_id,
                                        "audio": base64.b64encode(chunk).decode("ascii"),
                                    }
                                )
                                # Yield so STT timers and other tasks are not starved by tight chunk loops.
                                await asyncio.sleep(0)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("TTS sentence streaming failed for reply_id=%s", reply_id)
        except asyncio.CancelledError:
            raise
        finally:
            if reply_id == self.active_reply_id and generation == self.active_generation:
                self.active_reply_id = None
                await self.send_json({"type": "tts_done", "reply_id": reply_id})

    async def close(self) -> None:
        self.closed = True
        await self.stop_current_reply(reason="session_closed")