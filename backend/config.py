from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    deepgram_api_key: str
    openai_api_key: str
    deepgram_model: str
    llm_model: str
    welcome_message: str
    tts_model: str
    tts_voice: str
    tts_instructions: str
    tts_speed: float
    tts_sample_rate: int
    tts_chunk_bytes: int
    llm_system_prompt: str
    rag_top_k: int
    deepgram_language: str
    sample_rate: int
    utterance_end_grace_ms: int
    utterance_end_ms: int
    turn_silence_ms: int
    deepgram_idle_keepalive_seconds: float
    vad_min_confidence: float
    vad_start_votes: int
    frontend_dir: Path


settings = Settings(
    deepgram_api_key=_require_env("DEEPGRAM_API_KEY"),
    openai_api_key=_require_env("OPENAI_API_KEY"),
    deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
    llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),
    welcome_message=os.getenv("WELCOME_MESSAGE", "Welcome to IST University. How may I help you?"),
    tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
    tts_voice=os.getenv("TTS_VOICE", "marin"),
    tts_instructions=os.getenv(
        "TTS_INSTRUCTIONS",
        "Speak clearly, warmly, and conversationally. Keep a professional university helpdesk tone.",
    ),
    tts_speed=float(os.getenv("TTS_SPEED", "1.0")),
    tts_sample_rate=int(os.getenv("TTS_SAMPLE_RATE", "24000")),
    tts_chunk_bytes=int(os.getenv("TTS_CHUNK_BYTES", "4096")),
    llm_system_prompt=os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are the IST University voice assistant. Answer clearly, briefly, and helpfully. For generic questions like 'Can you hear me?' or 'Are you there?', respond conversationally (e.g., 'Yes, I can hear you. How can I help?'). For IST-specific questions, use the knowledge base context provided. Avoid saying 'I don't have that information' unless absolutely necessary.",
    ),
    rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
    deepgram_language=os.getenv("DEEPGRAM_LANGUAGE", "en-US"),
    sample_rate=int(os.getenv("SAMPLE_RATE", "48000")),
    utterance_end_grace_ms=int(os.getenv("UTTERANCE_END_GRACE_MS", "300")),
    utterance_end_ms=int(os.getenv("UTTERANCE_END_MS", "1500")),  # 1.2s silence to confirm full utterance before finalizing
    turn_silence_ms=int(os.getenv("TURN_SILENCE_MS", "500")),   # Wait after UtteranceEnd before sending to LLM (short, since utterance_end_ms already waited)
    deepgram_idle_keepalive_seconds=float(os.getenv("DEEPGRAM_IDLE_KEEPALIVE_SECONDS", "3")),
    vad_min_confidence=float(os.getenv("VAD_MIN_CONFIDENCE", "0.5")),  # Increased from 0.4 to filter noise better
    vad_start_votes=int(os.getenv("VAD_START_VOTES", "2")),  # Increased from 1 to 2 for confirmation
    frontend_dir=BASE_DIR / "frontend",
)