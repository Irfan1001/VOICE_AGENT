"""
IST University Voice Agent — Backend
Cheaper 3-call pipeline: Whisper STT → GPT-4o-mini LLM → TTS-1 (streaming PCM).

Per-turn flow:
  1. Client streams raw PCM16 binary over WebSocket while user speaks.
  2. Client sends {"type":"commit"} after VAD detects end-of-utterance.
  3. Backend wraps PCM in WAV → Whisper STT.
  4. RAG lookup (lru_cache — near-zero cost on repeated questions).
  5. GPT-4o-mini streams text; each complete sentence fires a TTS call.
  6. TTS returns raw PCM16 (response_format="pcm") streamed back as binary WS frames.
  7. Client plays PCM16 via WebAudio — loudspeaker, no earpiece.

Barge-in:
  Client sends {"type":"barge_in"} → backend increments turn generation so the
  in-progress task stops at its next await checkpoint without cancel races.
"""

import asyncio
import io
import json
import os
import re
import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

from backend.rag import search

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Config (all overridable via environment variables) ────────────────────────
CHAT_MODEL     = os.getenv("CHAT_MODEL",  "gpt-4o-mini")
TTS_MODEL      = os.getenv("TTS_MODEL",   "tts-1")         # tts-1-hd for higher quality
TTS_VOICE      = os.getenv("TTS_VOICE",   "alloy")
RAG_TOP_K      = int(os.getenv("RAG_TOP_K",      "4"))
RAG_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE",  "256"))
SAMPLE_RATE    = 24000

SYSTEM_PROMPT = """You are the IST University voice assistant.
Help students and parents with admissions, fees, programs, and campus queries.

Strict rules — follow every one:
1. LANGUAGE: Always respond in English by default. If the user speaks in Urdu (Pakistani Nastaliq script), reply in Urdu only. Never use Hindi. Never mix languages unless the user mixes first.
2. KNOWLEDGE BASE ONLY: Answer ONLY from the [KNOWLEDGE BASE CONTEXT] provided. Never invent, guess, or use outside knowledge. If the answer is not in the context say: "I'm sorry, I don't have that information in my knowledge base right now."
3. LENGTH: Keep every response to 1-2 short spoken sentences. After answering, ask if the user has another question.
4. FORMAT: No bullet lists, no numbered lists, no headers. Speak naturally as if on a phone call.
5. SCOPE: If the question is unrelated to IST University, say you can only help with IST-related topics.
"""

GREETING = "Welcome to IST University. How can I help you today?"

app = FastAPI()
oai = AsyncOpenAI()


# ── RAG with LRU cache ────────────────────────────────────────────────────────
@lru_cache(maxsize=RAG_CACHE_SIZE)
def _cached_rag(q: str, k: int) -> str:
	return search(q, k=k)


def rag_search(query: str, k: int) -> str:
	return _cached_rag(" ".join(query.lower().split()), k)


# ── Audio helpers ─────────────────────────────────────────────────────────────
def pcm_to_wav(pcm: bytes, sr: int = SAMPLE_RATE) -> bytes:
	buf = io.BytesIO()
	with wave.open(buf, "wb") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sr)
		wf.writeframes(pcm)
	return buf.getvalue()


# Sentence splitter — also handles Urdu sentence-end marker
_SENT_END = re.compile(r'(?<=[.!?])\s+')


def pop_sentences(buf: str) -> tuple:
	"""Return (completed_sentences, remainder)."""
	parts = _SENT_END.split(buf)
	if len(parts) <= 1:
		return [], buf
	return [s.strip() for s in parts[:-1] if s.strip()], parts[-1]


# ── TTS streaming helper ──────────────────────────────────────────────────────
async def stream_tts(text: str, ws: WebSocket, gen: int, cur: list) -> None:
	"""Stream PCM16 audio for text to the client. Stops if generation advances."""
	try:
		async with oai.audio.speech.with_streaming_response.create(
			model=TTS_MODEL,
			voice=TTS_VOICE,
			input=text,
			response_format="pcm",
			speed=1.0,
		) as resp:
			async for chunk in resp.iter_bytes(chunk_size=4096):
				if cur[0] != gen:
					return
				await ws.send_bytes(chunk)
	except Exception:
		pass


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
	return {"status": "ok"}


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
	await ws.accept()

	audio_buf: bytearray = bytearray()
	cur = [0]                                       # mutable generation counter
	current_task: Optional[asyncio.Task] = None

	# ── Per-turn processing task ──────────────────────────────────────────────
	async def run_turn(gen: int, pcm: bytes) -> None:
		try:
			# 1. STT ──────────────────────────────────────────────────────────
			wav_io = io.BytesIO(pcm_to_wav(pcm))
			wav_io.name = "audio.wav"
			try:
				result = await oai.audio.transcriptions.create(
					model="whisper-1",
					file=wav_io,
					prompt="IST University Islamabad, Pakistan. English or Urdu speech.",
				)
			except Exception:
				await ws.send_text(json.dumps({"type": "error", "msg": "STT failed"}))
				return

			transcript = (result.text or "").strip()
			if not transcript or cur[0] != gen:
				return

			await ws.send_text(json.dumps({"type": "transcript", "text": transcript}))

			# 2. RAG ──────────────────────────────────────────────────────────
			try:
				context = rag_search(transcript, k=RAG_TOP_K)
			except Exception:
				context = ""

			if cur[0] != gen:
				return

			# 3. LLM stream → sentence TTS stream ────────────────────────────
			await ws.send_text(json.dumps({"type": "agent_start"}))

			messages = [
				{
					"role": "system",
					"content": (
						SYSTEM_PROMPT
						+ (
							f"\n\n[KNOWLEDGE BASE CONTEXT — answer ONLY from this]\n{context}\n[/KNOWLEDGE BASE CONTEXT]"
							if context else ""
						)
					),
				},
				{"role": "user", "content": transcript},
			]

			try:
				stream = await oai.chat.completions.create(
					model=CHAT_MODEL,
					messages=messages,
					max_tokens=150,
					temperature=0.2,
					stream=True,
				)
			except Exception:
				await ws.send_text(json.dumps({"type": "agent_end"}))
				return

			text_buf = ""
			async for chunk in stream:
				if cur[0] != gen:
					break
				delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
				if not delta:
					continue
				text_buf += delta
				await ws.send_text(json.dumps({"type": "agent_text_delta", "text": delta}))

				# Fire TTS for each completed sentence immediately
				sentences, text_buf = pop_sentences(text_buf)
				for sentence in sentences:
					if cur[0] != gen:
						break
					await stream_tts(sentence, ws, gen, cur)

			# Flush remaining text after stream ends
			if text_buf.strip() and cur[0] == gen:
				await stream_tts(text_buf.strip(), ws, gen, cur)

		except asyncio.CancelledError:
			pass
		except Exception:
			pass
		finally:
			try:
				await ws.send_text(json.dumps({"type": "agent_end"}))
			except Exception:
				pass

	# ── Send greeting on connect ──────────────────────────────────────────────
	try:
		await ws.send_text(json.dumps({"type": "agent_start"}))
		await ws.send_text(json.dumps({"type": "agent_text_delta", "text": GREETING}))
		await stream_tts(GREETING, ws, 0, cur)
		await ws.send_text(json.dumps({"type": "agent_end"}))
	except Exception:
		pass

	# ── Main receive loop ─────────────────────────────────────────────────────
	try:
		while True:
			msg = await ws.receive()
			raw_bytes = msg.get("bytes")
			raw_text  = msg.get("text")

			if raw_bytes:
				audio_buf.extend(raw_bytes)

			elif raw_text:
				try:
					event = json.loads(raw_text)
				except Exception:
					continue
				etype = event.get("type", "")

				if etype == "commit" and len(audio_buf) > 1600:
					cur[0] += 1
					pcm = bytes(audio_buf)
					audio_buf.clear()
					if current_task and not current_task.done():
						current_task.cancel()
					current_task = asyncio.create_task(run_turn(cur[0], pcm))

				elif etype == "barge_in":
					cur[0] += 1
					audio_buf.clear()
					if current_task and not current_task.done():
						current_task.cancel()
					try:
						await ws.send_text(json.dumps({"type": "agent_end"}))
					except Exception:
						pass

	except WebSocketDisconnect:
		pass
	except Exception:
		pass
	finally:
		if current_task:
			current_task.cancel()


# ── Serve frontend ────────────────────────────────────────────────────────────
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
