from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlencode

from fastapi import WebSocket, WebSocketDisconnect
import websockets

from backend.config import Settings
from backend.llm import LLMResponder
from backend.tts import TTSResponder


logger = logging.getLogger("ist_voice_agent.stt")


def is_likely_human_speech(text: str, confidence: float, settings: Settings) -> bool:
    cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
    return len(cleaned) >= 2 and confidence >= settings.vad_min_confidence


def has_transcript_candidate(utterance_parts: list[str], latest_transcript: str) -> bool:
    return bool(utterance_parts or latest_transcript.strip())


class STTSession:
    def __init__(self, client_ws: WebSocket, settings: Settings, llm_responder: LLMResponder | None = None) -> None:
        self.client_ws = client_ws
        self.settings = settings
        self.llm_responder = llm_responder
        self.tts_responder = TTSResponder(settings, self.send_json)
        self.send_lock = asyncio.Lock()
        self.deepgram_ws: Any | None = None
        self.deepgram_task: asyncio.Task[None] | None = None
        self.deepgram_keepalive_task: asyncio.Task[None] | None = None
        self.pending_finalize_task: asyncio.Task[None] | None = None
        self.turn_finalize_task: asyncio.Task[None] | None = None
        self.llm_worker_task: asyncio.Task[None] | None = None
        self.pending_llm_requests: asyncio.Queue[str] = asyncio.Queue()
        self.last_audio_sent_at = time.monotonic()
        self.utterance_parts: list[str] = []
        self.latest_transcript = ""
        self.turn_parts: list[str] = []
        self.pending_speech_start = False
        self.speech_votes = 0
        self.user_speaking = False
        self.closed = False
        self.reply_counter = 0

    async def speak_agent_text(self, text: str) -> None:
        message = text.strip()
        if not message:
            return

        self.reply_counter += 1
        reply_id = self.reply_counter
        await self.send_json({"type": "agent_started"})
        await self.send_json({"type": "agent_text", "text": message})
        await self.tts_responder.stream_reply(reply_id, message)
        await self.send_json({"type": "agent_done"})

    def _deepgram_url(self) -> str:
        params = {
            "model": self.settings.deepgram_model,
            "language": self.settings.deepgram_language,
            "encoding": "linear16",
            "sample_rate": self.settings.sample_rate,
            "channels": 1,
            "interim_results": "true",
            "smart_format": "true",
            "vad_events": "true",
            "utterance_end_ms": self.settings.utterance_end_ms,
        }
        return f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self.send_lock:
            try:
                await self.client_ws.send_text(json.dumps(payload))
            except (WebSocketDisconnect, RuntimeError):
                self.closed = True

    def cancel_pending_finalize(self) -> None:
        if self.pending_finalize_task and not self.pending_finalize_task.done():
            self.pending_finalize_task.cancel()
        self.pending_finalize_task = None

    def cancel_turn_finalize(self) -> None:
        if self.turn_finalize_task and not self.turn_finalize_task.done():
            self.turn_finalize_task.cancel()
        self.turn_finalize_task = None

    async def finalize_utterance(self) -> None:
        self.user_speaking = False
        self.pending_speech_start = False
        self.speech_votes = 0
        segment = " ".join(self.utterance_parts).strip() or self.latest_transcript.strip()
        self.utterance_parts.clear()
        self.latest_transcript = ""
        await self.send_json({"type": "speech_ended"})
        if segment:
            self.turn_parts.append(segment)
        # Start/reset the turn silence timer — emit combined transcript after turn_silence_ms
        self.cancel_turn_finalize()
        self.turn_finalize_task = asyncio.create_task(self._emit_turn())

    async def _emit_turn(self) -> None:
        """Wait for turn_silence_ms of silence then emit the full combined transcript."""
        await asyncio.sleep(self.settings.turn_silence_ms / 1000)
        if not self.turn_parts:
            return
        full_text = " ".join(self.turn_parts).strip()
        self.turn_parts.clear()
        if full_text:
            await self.send_json({"type": "transcript", "text": full_text})
            if self.llm_responder is not None:
                await self.pending_llm_requests.put(full_text)

    async def llm_worker(self) -> None:
        try:
            while not self.closed:
                transcript = await self.pending_llm_requests.get()
                if not transcript:
                    self.pending_llm_requests.task_done()
                    continue

                reply_id: int | None = None
                await self.send_json({"type": "agent_started"})
                try:
                    full_reply = ""
                    self.reply_counter += 1
                    reply_id = self.reply_counter
                    if self.llm_responder is not None:
                        async for chunk in self.llm_responder.stream_reply(transcript):
                            full_reply += chunk
                            await self.send_json({"type": "agent_delta", "text": chunk})
                except Exception:
                    logger.exception("LLM generation failed")
                    await self.tts_responder.stop_current_reply(reason="agent_error")
                    await self.send_json({"type": "agent_error", "msg": "LLM generation failed"})
                else:
                    if full_reply.strip():
                        await self.send_json({"type": "agent_text", "text": full_reply.strip()})
                        # Start TTS only after full response is generated (not partial)
                        if reply_id is not None:
                            await self.tts_responder.stream_reply(reply_id, full_reply.strip())
                finally:
                    await self.send_json({"type": "agent_done"})
                    self.pending_llm_requests.task_done()
        except asyncio.CancelledError:
            raise

    async def finalize_with_grace(self, immediate: bool = False) -> None:
        if not immediate:
            await asyncio.sleep(self.settings.utterance_end_grace_ms / 1000)
        if self.user_speaking:
            return
        await self.finalize_utterance()

    async def deepgram_keepalive(self) -> None:
        assert self.deepgram_ws is not None
        try:
            while not self.closed:
                await asyncio.sleep(1.0)
                if self.closed or self.deepgram_ws is None:
                    return
                idle_for = time.monotonic() - self.last_audio_sent_at
                if idle_for >= self.settings.deepgram_idle_keepalive_seconds:
                    await self.deepgram_ws.send(json.dumps({"type": "KeepAlive"}))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Deepgram keepalive loop failed")

    async def handle_result(self, event: dict[str, Any]) -> None:
        alternatives = event.get("channel", {}).get("alternatives", [])
        if not alternatives:
            return

        transcript = (alternatives[0].get("transcript") or "").strip()
        confidence = float(alternatives[0].get("confidence") or 0.0)

        if transcript:
            self.latest_transcript = transcript

        if transcript and self.pending_speech_start and not self.user_speaking:
            if is_likely_human_speech(transcript, confidence, self.settings):
                self.speech_votes += 1
            else:
                self.speech_votes = max(0, self.speech_votes - 1)

        if self.pending_speech_start and not self.user_speaking and self.speech_votes >= self.settings.vad_start_votes:
            self.user_speaking = True
            self.pending_speech_start = False
            self.speech_votes = 0
            # Do NOT clear utterance_parts — early is_final chunks before VAD confirmation must be kept
            await self.tts_responder.stop_current_reply(reason="barge_in")
            await self.send_json({"type": "speech_started"})

        is_final = bool(event.get("is_final"))
        speech_final = bool(event.get("speech_final"))

        if transcript and not is_final:
            # Interim result — show real-time words while speaking
            await self.send_json({"type": "transcript_interim", "text": transcript})

        if transcript and is_final:
            self.utterance_parts.append(transcript)

        # speech_final means Deepgram is confident this chunk is done, but it fires on every
        # short pause mid-sentence. Do NOT finalize here — wait for UtteranceEnd (after
        # utterance_end_ms of silence) so minor pauses don't split one sentence into many transcripts.
        # Just reset the speaking flag so SpeechStarted can re-arm it for the next chunk.
        if speech_final:
            self.user_speaking = False

    async def deepgram_reader(self) -> None:
        assert self.deepgram_ws is not None
        try:
            async for raw in self.deepgram_ws:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")
                if event_type == "SpeechStarted":
                    # Don't interrupt TTS on raw VAD event (too sensitive to noise).
                    # Instead, just arm vote collection. Actual barge-in happens
                    # after confidence-gated speech is confirmed (in handle_result).
                    self.cancel_pending_finalize()
                    self.cancel_turn_finalize()
                    if not self.user_speaking:
                        if not self.utterance_parts:
                            # Fresh speech — start vote accumulation
                            self.pending_speech_start = True
                            self.speech_votes = 0
                    continue

                if event_type == "UtteranceEnd":
                    if (
                        self.user_speaking
                        or self.pending_speech_start
                        or has_transcript_candidate(self.utterance_parts, self.latest_transcript)
                    ):
                        self.user_speaking = False
                        self.pending_speech_start = False
                        self.cancel_pending_finalize()
                        # UtteranceEnd fires after silence_ms — finalize immediately, no extra grace
                        self.pending_finalize_task = asyncio.create_task(self.finalize_with_grace(immediate=True))
                    continue

                if event_type == "Results":
                    await self.handle_result(event)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            logger.info("Deepgram websocket closed: %s", exc)

    async def connect_deepgram(self) -> None:
        self.deepgram_ws = await websockets.connect(
            self._deepgram_url(),
            additional_headers={"Authorization": f"Token {self.settings.deepgram_api_key}"},
            open_timeout=30,
        )
        self.last_audio_sent_at = time.monotonic()
        self.deepgram_task = asyncio.create_task(self.deepgram_reader())
        self.deepgram_keepalive_task = asyncio.create_task(self.deepgram_keepalive())

    async def run(self) -> None:
        try:
            await self.connect_deepgram()
        except Exception:
            logger.exception("Deepgram connection failed")
            await self.send_json({"type": "error", "msg": "Deepgram connection failed"})
            return

        if self.llm_responder is not None:
            self.llm_worker_task = asyncio.create_task(self.llm_worker())

        await self.send_json({"type": "ready"})
        # Small delay to ensure frontend audio context is fully initialized
        await asyncio.sleep(0.3)
        await self.speak_agent_text(self.settings.welcome_message)

        try:
            while True:
                msg = await self.client_ws.receive()
                raw_bytes = msg.get("bytes")
                raw_text = msg.get("text")

                if raw_bytes and self.deepgram_ws is not None:
                    try:
                        await self.deepgram_ws.send(raw_bytes)
                        self.last_audio_sent_at = time.monotonic()
                    except websockets.ConnectionClosed as exc:
                        logger.info("Deepgram websocket closed while forwarding audio: %s", exc)
                        return
                    continue

                if raw_text:
                    try:
                        event = json.loads(raw_text)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") in {"hangup", "stop"}:
                        return
        except (WebSocketDisconnect, RuntimeError):
            return

    async def close(self) -> None:
        self.closed = True
        self.pending_speech_start = False
        self.speech_votes = 0
        self.cancel_pending_finalize()
        self.cancel_turn_finalize()
        await self.tts_responder.close()

        if self.deepgram_ws is not None:
            try:
                await self.deepgram_ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
            await self.deepgram_ws.close()

        if self.deepgram_keepalive_task and not self.deepgram_keepalive_task.done():
            self.deepgram_keepalive_task.cancel()
            try:
                await self.deepgram_keepalive_task
            except asyncio.CancelledError:
                pass

        if self.deepgram_task and not self.deepgram_task.done():
            self.deepgram_task.cancel()
            try:
                await self.deepgram_task
            except asyncio.CancelledError:
                pass

        if self.llm_worker_task and not self.llm_worker_task.done():
            self.llm_worker_task.cancel()
            try:
                await self.llm_worker_task
            except asyncio.CancelledError:
                pass