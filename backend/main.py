import logging
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.llm import LLMResponder
from backend.rag import preload_models
from backend.stt import STTSession

logger = logging.getLogger("ist_voice_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI()
llm_responder = LLMResponder(settings)


@app.on_event("startup")
async def startup_event() -> None:
    try:
        preload_models(preload_reranker=True)
        logger.info("RAG models preloaded successfully")
    except Exception as exc:
        logger.warning("RAG model preload failed, continuing without warm cache: %s", exc)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    session = STTSession(ws, settings, llm_responder=llm_responder)
    try:
        await session.run()
    finally:
        await session.close()


app.mount("/", StaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
