import uuid
import json
import io
import os
import tempfile
from pathlib import Path
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from backend.rag import search

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI()
client = OpenAI()

sessions = {}
session_order = deque()


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return int(value)
	except ValueError:
		return default


def _env_float(name: str, default: float) -> float:
	value = os.getenv(name)
	if value is None:
		return default
	try:
		return float(value)
	except ValueError:
		return default


APP_CONFIG = {
	"chat_model": os.getenv("CHAT_MODEL", "gpt-4o-mini"),
	"transcribe_model": os.getenv("TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
	"tts_model": os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
	"retrieval_k": _env_int("RAG_TOP_K", 6),
	"max_tokens": _env_int("CHAT_MAX_TOKENS", 220),
	"temperature": _env_float("CHAT_TEMPERATURE", 0.2),
	"max_sessions": _env_int("MAX_SESSIONS", 500),
	"history_cap": _env_int("SESSION_HISTORY_CAP", 13),
}

SYSTEM_PROMPT = """You are the IST University voice assistant.
Your role is to help students and parents with admissions, fee structure, programs, campus life, and all IST-related queries.

Guidelines:
1. Answer using ONLY information from the provided KB context. Never invent or guess facts.
2. If the context contains a clear answer, respond concisely and helpfully.
3. If the context partially covers the question, answer what you can and note what is not covered.
4. If the question is ambiguous or unclear, ask the user to rephrase or clarify before saying you don't know.
5. Only if the topic is genuinely absent from the KB after doing your best, say: "I don't have specific information on that right now. I can help connect you to a human agent if you'd like."
6. Keep responses short and conversational — suitable for voice output. Avoid bullet lists.
7. If the user asks in English, reply only in English.
8. If the user asks in Urdu, reply only in Urdu.
9. Only support English and Urdu. If the user asks in any other language, reply: "Sorry, I can only help in English or Urdu."
10. Do not mix English and Urdu unless the user mixes them first.
"""
WELCOME_MESSAGE = "Welcome to IST University. How may I help you today?"



def get_history(session_id):
	return sessions.get(session_id, [])


def save_message(session_id, role, content):
	sessions.setdefault(session_id, []).append({"role": role, "content": content})
	# Keep only recent history to avoid prompt bloat and latency spikes.
	history_cap = max(APP_CONFIG["history_cap"], 3)
	if len(sessions[session_id]) > history_cap:
		sessions[session_id] = [sessions[session_id][0], *sessions[session_id][-(history_cap - 1):]]


def register_session(session_id: str):
	session_order.append(session_id)
	max_sessions = max(APP_CONFIG["max_sessions"], 50)
	while len(session_order) > max_sessions:
		old_id = session_order.popleft()
		sessions.pop(old_id, None)


def build_user_prompt(context: str, question: str) -> str:
	return f"""
Answer only from context.
If the user asks about a person and their name appears in context, provide their role/affiliation from context.
If context is insufficient, ask a short clarification question.

Context:
{context}

Question:
{question}
"""


@app.get("/health")
def health():
	return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
	data = await audio.read()
	if not data:
		raise HTTPException(status_code=400, detail="Empty audio payload")

	buffer = io.BytesIO(data)
	buffer.name = audio.filename or "utterance.webm"

	try:
		result = client.audio.transcriptions.create(
			model=APP_CONFIG["transcribe_model"],
			file=buffer,
			prompt=(
				"This is an IST university admissions support call from Pakistan. "
				"Recognize English, Urdu, and mixed code-switch speech accurately. "
				"Preserve names, abbreviations, and phone numbers exactly as spoken."
			),
		)
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}") from exc

	text = (getattr(result, "text", "") or "").strip()
	return {"text": text}


class TTSRequest(BaseModel):
	text: str
	lang: str = "en-US"


@app.post("/tts")
async def synthesize_tts(payload: TTSRequest):
	text = (payload.text or "").strip()
	if not text:
		raise HTTPException(status_code=400, detail="Text is required")

	instructions = "Speak clearly and naturally."
	if payload.lang.lower().startswith("ur"):
		instructions = "Speak naturally in Urdu with clear pronunciation and moderate pace."

	try:
		result = client.audio.speech.create(
			model=APP_CONFIG["tts_model"],
			voice="alloy",
			input=text,
			instructions=instructions,
		)
		with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
			tmp_path = tmp.name
		result.stream_to_file(tmp_path)
		audio_bytes = Path(tmp_path).read_bytes()
	except Exception as exc:
		raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc
	finally:
		if "tmp_path" in locals() and os.path.exists(tmp_path):
			os.remove(tmp_path)

	return Response(content=audio_bytes, media_type="audio/mpeg")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
	await ws.accept()

	session_id = str(uuid.uuid4())
	sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
	register_session(session_id)
	await ws.send_text(WELCOME_MESSAGE)
	save_message(session_id, "assistant", WELCOME_MESSAGE)

	try:
		while True:
			question = (await ws.receive_text()).strip()
			if not question:
				continue

			context = search(question, k=max(APP_CONFIG["retrieval_k"], 1))
			user_prompt = build_user_prompt(context=context, question=question)

			# Inject retrieval context only for the current turn (do not persist it in chat history).
			messages = [
				*get_history(session_id),
				{"role": "user", "content": user_prompt},
			]

			try:
				response = client.chat.completions.create(
					model=APP_CONFIG["chat_model"],
					messages=messages,
					temperature=APP_CONFIG["temperature"],
					max_tokens=APP_CONFIG["max_tokens"],
					stream=True,
				)
			except Exception:
				fallback = "I am having trouble answering right now. Please try again in a moment."
				await ws.send_text(json.dumps({"type": "stream_start"}))
				await ws.send_text(json.dumps({"type": "stream_delta", "text": fallback}))
				await ws.send_text(json.dumps({"type": "stream_end"}))
				save_message(session_id, "user", question)
				save_message(session_id, "assistant", fallback)
				continue

			await ws.send_text(json.dumps({"type": "stream_start"}))
			parts = []
			for chunk in response:
				delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
				if not delta:
					continue
				parts.append(delta)
				await ws.send_text(json.dumps({"type": "stream_delta", "text": delta}))
			await ws.send_text(json.dumps({"type": "stream_end"}))

			answer = "".join(parts).strip() or "I could not generate an answer."

			save_message(session_id, "user", question)
			save_message(session_id, "assistant", answer)
	except WebSocketDisconnect:
		sessions.pop(session_id, None)
		try:
			session_order.remove(session_id)
		except ValueError:
			pass


# Serve the frontend — must be mounted AFTER all API/WebSocket routes.
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
