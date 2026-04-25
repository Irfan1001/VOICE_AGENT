import logging
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.llm import LLMResponder
from backend.stt import STTSession

logger = logging.getLogger("ist_voice_agent")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

app = FastAPI()
llm_responder = LLMResponder(settings)


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
