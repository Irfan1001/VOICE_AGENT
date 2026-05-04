from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any
from urllib.parse import urlencode

# Matches a sentence boundary: ./?/! followed by whitespace (for mid-stream splitting)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Timestamp logging for event ordering
import time as time_module
def _ts_log(logger_obj: logging.Logger, event: str, details: str = "") -> None:
    ts_ms = int(time_module.time() * 1000) % 100000
    msg = f"[TS={ts_ms:>5}ms] {event}"
    if details:
        msg += f" | {details}"
    logger_obj.info(msg)

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
        self.llm_worker_task: asyncio.Task[None] | None = None
        self.pending_llm_requests: asyncio.Queue[str] = asyncio.Queue()
        self.last_audio_sent_at = time.monotonic()
        self.utterance_parts: list[str] = []
        self.latest_transcript = ""
        self.pending_speech_start = False
        self.speech_votes = 0
        self.user_speaking = False
        self.closed = False
        self.reply_counter = 0
        self.reply_id_lock = asyncio.Lock()
        # Conversation history: list of {role, content} dicts, capped at _MAX_HISTORY_TURNS pairs
        self._conversation_history: list[dict[str, str]] = []
        self._MAX_HISTORY_TURNS = 6  # max (user+assistant) pairs kept

    async def _next_reply_id(self) -> int:
        async with self.reply_id_lock:
            self.reply_counter += 1
            return self.reply_counter

    async def speak_agent_text(self, text: str) -> None:
        message = text.strip()
        if not message:
            return

        reply_id = await self._next_reply_id()
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

    async def finalize_utterance(self) -> None:
        self.user_speaking = False
        self.pending_speech_start = False
        self.speech_votes = 0
        segment = " ".join(self.utterance_parts).strip() or self.latest_transcript.strip()
        self.utterance_parts.clear()
        self.latest_transcript = ""
        await self.send_json({"type": "speech_ended"})
        if segment:
            await self.send_json({"type": "transcript", "text": segment})
            if self.llm_responder is not None:
                await self.pending_llm_requests.put(segment)

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
                    reply_id = await self._next_reply_id()
                    # Start TTS pipeline immediately — first sentence will begin playing
                    # before LLM finishes generating the rest.
                    await self.tts_responder.begin_streaming_reply(reply_id)
                    _ts_log(logger, "TTS_BEGIN_STREAMING", f"reply_id={reply_id}")

                    if self.llm_responder is not None:
                        history_snapshot = list(self._conversation_history)
                        pending_sentence = ""
                        first_tts_pushed = False
                        delta_count = 0
                        async for chunk in self.llm_responder.stream_reply(transcript, history_snapshot):
                            full_reply += chunk
                            pending_sentence += chunk
                            delta_count += 1
                            if delta_count == 1:
                                _ts_log(logger, "AGENT_DELTA_FIRST", f"chunk_len={len(chunk)}")
                            await self.send_json({"type": "agent_delta", "text": chunk})
                            # Detect sentence boundary: flush completed sentence to TTS
                            parts = _SENT_BOUNDARY.split(pending_sentence)
                            if len(parts) > 1:
                                # All parts except the last are complete sentences
                                for sentence in parts[:-1]:
                                    sentence = sentence.strip()
                                    if len(sentence) >= 10:
                                        _ts_log(logger, "SENTENCE_BOUNDARY", f"len={len(sentence)}")
                                        await self.tts_responder.push_sentence(reply_id, sentence)
                                        first_tts_pushed = True
                                pending_sentence = parts[-1]

                            # Fallback for long first sentence without punctuation:
                            # flush once we have enough words so audio can begin earlier.
                            if not first_tts_pushed:
                                early = pending_sentence.strip()
                                if len(early) >= 60 and len(early.split()) >= 10:
                                    _ts_log(logger, "EARLY_FLUSH_FALLBACK", f"len={len(early)}")
                                    await self.tts_responder.push_sentence(reply_id, early)
                                    pending_sentence = ""
                                    first_tts_pushed = True

                        # Flush any remaining text after the stream ends
                        if pending_sentence.strip():
                            await self.tts_responder.push_sentence(reply_id, pending_sentence.strip())

                    # Signal no more sentences; drain task will finish and send tts_done
                    await self.tts_responder.end_streaming_reply(reply_id)

                except Exception:
                    logger.exception("LLM generation failed")
                    await self.tts_responder.stop_current_reply(reason="agent_error")
                    await self.send_json({"type": "agent_error", "msg": "LLM generation failed"})
                else:
                    if full_reply.strip():
                        _ts_log(logger, "AGENT_TEXT_FINAL", f"len={len(full_reply)}")
                        await self.send_json({"type": "agent_text", "text": full_reply.strip()})
                        # Append to history and cap length
                        self._conversation_history.append({"role": "user", "content": transcript})
                        self._conversation_history.append({"role": "assistant", "content": full_reply.strip()})
                        max_msgs = self._MAX_HISTORY_TURNS * 2
                        if len(self._conversation_history) > max_msgs:
                            self._conversation_history = self._conversation_history[-max_msgs:]
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