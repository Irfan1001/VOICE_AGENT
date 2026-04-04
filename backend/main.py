"""
IST University Voice Agent — Backend
Proxies to OpenAI Realtime API (manual turn-detection).
Flow: client streams PCM16 → RMS VAD on client → commit signal →
      backend gets transcript → RAG → update session context → response.create
"""

import asyncio
import base64
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from backend.rag import search

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Config (all overridable via env vars) ─────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")
REALTIME_URL   = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"
TTS_VOICE      = os.getenv("TTS_VOICE", "alloy")
RAG_TOP_K      = int(os.getenv("RAG_TOP_K", "4"))
RAG_CACHE_SIZE = int(os.getenv("RAG_CACHE_SIZE", "256"))

SYSTEM_PROMPT = """You are the IST University voice assistant.
Help students and parents with admissions, fees, programs, and campus queries.

Strict rules — follow every one:
1. LANGUAGE: Always respond in English by default. If the user speaks in Urdu (the Pakistani language written in Nastaliq script), reply in Urdu only. Never use Hindi. Never mix languages unless the user mixes first.
2. KNOWLEDGE BASE ONLY: Answer ONLY from the [KNOWLEDGE BASE CONTEXT] provided. Never invent, guess, or use outside knowledge. If the answer is not in the context, say exactly: "I'm sorry, I don't have that information in my knowledge base right now."
3. LENGTH: Keep every response to 1–2 short spoken sentences maximum. After answering, ask if the user has another question.
4. FORMAT: No bullet lists, no numbered lists, no headers. Speak naturally as if on a phone call.
5. SCOPE: If the question is completely unrelated to IST University, politely say you can only help with IST-related topics.
"""


@lru_cache(maxsize=RAG_CACHE_SIZE)
def _cached_rag(query_key: str, k: int) -> str:
	"""Cache RAG results by (query_key, k). query_key is a normalised lowercase query."""
	return search(query_key, k=k)


def rag_search(query: str, k: int) -> str:
	"""Normalise query and return cached RAG context."""
	normalised = " ".join(query.lower().split())
	return _cached_rag(normalised, k)

app = FastAPI()


@app.get("/health")
def health():
	return {"status": "ok"}


@app.websocket("/ws")
async def ws_proxy(client_ws: WebSocket):
	await client_ws.accept()

	oai_headers = {
		"Authorization": f"Bearer {OPENAI_API_KEY}",
		"OpenAI-Beta": "realtime=v1",
	}

	# Session with manual turn-detection — we commit audio and create responses
	# ourselves so we can inject RAG context before each response.
	session_config = {
		"type": "session.update",
		"session": {
			"instructions": SYSTEM_PROMPT,
			"voice": TTS_VOICE,
			"input_audio_format": "pcm16",
			"output_audio_format": "pcm16",
			"input_audio_transcription": {"model": "whisper-1"},
			"turn_detection": None,
			"modalities": ["text", "audio"],
		},
	}

	try:
		async with websockets.connect(
			REALTIME_URL,
			additional_headers=oai_headers,
			max_size=None,
			ping_interval=20,
		) as oai_ws:
			await oai_ws.send(json.dumps(session_config))

			# Welcome greeting — always in English, 1-2 sentences
			await oai_ws.send(json.dumps({
				"type": "response.create",
				"response": {
					"instructions": (
						"Greet the user in English only. "
						"Say: 'Welcome to IST University. How can I help you today?' "
						"Do not say anything else."
					),
				},
			}))

			pending_commit = False  # True while we are waiting for a transcription event

			async def from_client():
				nonlocal pending_commit
				try:
					while True:
						msg = await client_ws.receive()
						raw_bytes = msg.get("bytes")
						raw_text  = msg.get("text")

						if raw_bytes:
							# PCM16 audio chunk — forward to Realtime API
							await oai_ws.send(json.dumps({
								"type": "input_audio_buffer.append",
								"audio": base64.b64encode(raw_bytes).decode(),
							}))

						elif raw_text:
							data = json.loads(raw_text)
							event_type = data.get("type", "")

							if event_type == "commit":
								# Client detected end-of-utterance — commit audio buffer
								pending_commit = True
								await oai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

							elif event_type == "barge_in":
								# User spoke over the agent — cancel current response
								pending_commit = False
								await oai_ws.send(json.dumps({"type": "response.cancel"}))

				except Exception:
					pass

			async def from_oai():
				nonlocal pending_commit
				try:
					async for raw in oai_ws:
						payload = raw if isinstance(raw, str) else raw.decode()
						event   = json.loads(payload)
						etype   = event.get("type", "")

						# Transcription completed → inject RAG context → trigger response
						if etype == "conversation.item.input_audio_transcription.completed" and pending_commit:
							pending_commit = False
							transcript = event.get("transcript", "").strip()

							if transcript:
								try:
									context = rag_search(transcript, k=RAG_TOP_K)
									await oai_ws.send(json.dumps({
										"type": "session.update",
										"session": {
											"instructions": (
												SYSTEM_PROMPT
												+ f"\n\n[KNOWLEDGE BASE CONTEXT — answer ONLY from this]\n{context}\n[/KNOWLEDGE BASE CONTEXT]"
											),
										},
									}))
								except Exception:
									pass  # RAG failure — model will say it doesn't have the info

							await oai_ws.send(json.dumps({"type": "response.create"}))

						elif etype == "conversation.item.input_audio_transcription.failed" and pending_commit:
							# Transcription failed — respond anyway
							pending_commit = False
							await oai_ws.send(json.dumps({"type": "response.create"}))

						# Forward every Realtime event to the browser as-is
						try:
							await client_ws.send_text(payload)
						except Exception:
							break

				except Exception:
					pass

			await asyncio.gather(from_client(), from_oai())

	except WebSocketDisconnect:
		pass
	except Exception:
		try:
			await client_ws.close()
		except Exception:
			pass


# Serve frontend — must be mounted after all API/WS routes.
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
