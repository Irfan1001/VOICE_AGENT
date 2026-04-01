import uuid
import json
import io
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

from backend.rag import search

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI()
client = OpenAI()

sessions = {}
SYSTEM_PROMPT = """You are the IST University voice assistant.
Your role is to help students and parents with admissions, fee structure, programs, campus life, and all IST-related queries.

Guidelines:
1. Answer using ONLY information from the provided KB context. Never invent or guess facts.
2. If the context contains a clear answer, respond concisely and helpfully.
3. If the context partially covers the question, answer what you can and note what is not covered.
4. If the question is ambiguous or unclear, ask the user to rephrase or clarify before saying you don't know.
5. Only if the topic is genuinely absent from the KB after doing your best, say: "I don't have specific information on that right now. I can help connect you to a human agent if you'd like."
6. Keep responses short and conversational — suitable for voice output. Avoid bullet lists.
"""
WELCOME_MESSAGE = "Welcome to IST University. How may I help you today?"



def get_history(session_id):
	return sessions.get(session_id, [])


def save_message(session_id, role, content):
	sessions.setdefault(session_id, []).append({"role": role, "content": content})
	# Keep only recent history to avoid prompt bloat and latency spikes.
	if len(sessions[session_id]) > 13:
		sessions[session_id] = [sessions[session_id][0], *sessions[session_id][-12:]]


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
			model="gpt-4o-transcribe",
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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
	await ws.accept()

	session_id = str(uuid.uuid4())
	sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
	await ws.send_text(WELCOME_MESSAGE)
	save_message(session_id, "assistant", WELCOME_MESSAGE)

	try:
		while True:
			question = await ws.receive_text()
			context = search(question, k=6)

			user_prompt = f"""
Answer only from context.
If the user asks about a person and their name appears in context, provide their role/affiliation from context.
If context is insufficient, ask a short clarification question.

Context:
{context}

Question:
{question}
"""

			# Inject retrieval context only for the current turn (do not persist it in chat history).
			messages = [
				*get_history(session_id),
				{"role": "user", "content": user_prompt},
			]

			response = client.chat.completions.create(
				model="gpt-4o-mini",
				messages=messages,
				temperature=0.2,
				max_tokens=220,
				stream=True,
			)

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


# Serve the frontend — must be mounted AFTER all API/WebSocket routes.
_frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
