"""FastAPI server for the chess commentary app.

    uvicorn webapp.server:app --reload          (from the chess-annotator folder)

Endpoints:
  POST /api/annotate   body {"pgn": "...", "factcheck": true}  -> NDJSON event stream
  GET  /api/health     engine + budget status
  GET  /               the web page (webapp/static/index.html)
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .llm import GroqClient, LlamaServerClient, MockClient, StudentClient
from .pipeline import GamePipeline, PGNError, parse_single_game
from .store import Store

STATIC = Path(__file__).parent / "static"


class AnnotateRequest(BaseModel):
    pgn: str = Field(..., min_length=1)
    factcheck: bool | None = None


class RateLimiter:
    """Sliding one-hour window of game starts per client IP (in memory)."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.hits = defaultdict(deque)

    def allow(self, ip: str) -> bool:
        now = time.time()
        dq = self.hits[ip]
        while dq and now - dq[0] > 3600:
            dq.popleft()
        if len(dq) >= self.per_hour:
            return False
        dq.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = Settings()
    app.state.settings = s
    app.state.store = Store(s.db_path)
    if s.commentator_backend == "mock":
        app.state.llm = MockClient()
    elif s.commentator_backend == "student":
        print(f"Loading student: {s.student_base_model} + {s.student_adapter} (first run downloads the base model)...")
        app.state.llm = StudentClient(s.student_base_model, s.student_adapter, s.student_max_tokens)
        print(f"Student ready on {app.state.llm.device}.")
    elif s.commentator_backend == "student_gguf":
        app.state.llm = LlamaServerClient(s.student_gguf_url, s.student_gguf, s.llama_server, s.llama_port,
                                          s.student_threads, s.student_max_tokens)
        s.model = f"student-v3/{app.state.llm.name}"
        s.factcheck_default = False
        print(f"Student ready: {app.state.llm.device} at {app.state.llm.url}.")
        s.model = f"student-v3/{app.state.llm.name}"  # shown in /api/health and part of the cache key
        s.factcheck_default = False
    else:
        app.state.llm = GroqClient(s.groq_api_key, s.reasoning_effort)
    app.state.llm_sem = asyncio.Semaphore(s.llm_concurrency)
    app.state.game_sem = asyncio.Semaphore(s.max_concurrent_games)
    app.state.limiter = RateLimiter(s.games_per_ip_per_hour)
    yield
    await app.state.llm.aclose()


app = FastAPI(title="Chess commentary", lifespan=lifespan)


def _client_ip(request: Request) -> str:
    if request.app.state.settings.trust_proxy_headers:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.get("/api/health")
async def health(request: Request):
    s, store = request.app.state.settings, request.app.state.store
    return {"ok": True, "model": s.model, "backend": s.commentator_backend,
            "spent_today_usd": round(store.spent_today(), 4), "daily_budget_usd": s.daily_budget_usd}


@app.post("/api/annotate")
async def annotate(req: AnnotateRequest, request: Request):
    st = request.app.state
    s: Settings = st.settings
    try:
        game = parse_single_game(req.pgn, s)
    except PGNError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if st.store.spent_today() >= s.daily_budget_usd:
        raise HTTPException(status_code=503, detail="The daily commentary budget has been used up. Please try again tomorrow.")
    if not st.limiter.allow(_client_ip(request)):
        raise HTTPException(status_code=429, detail="Too many games from your address this hour. Please wait a bit.")
    try:
        await asyncio.wait_for(st.game_sem.acquire(), timeout=s.queue_wait_s)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="The server is busy analysing other games. Please try again shortly.")

    factcheck = s.factcheck_default if req.factcheck is None else req.factcheck
    if s.commentator_backend.startswith("student"):
        factcheck = False
    pipeline = GamePipeline(s, st.llm, st.store, st.llm_sem)
    cancel = threading.Event()

    async def stream():
        events = pipeline.run(game, factcheck, cancel)
        try:
            async for ev in events:
                if await request.is_disconnected():
                    cancel.set()
                    break
                yield json.dumps(ev) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "error", "message": f"Unexpected server error: {e}"}) + "\n"
        finally:
            cancel.set()
            await events.aclose()
            st.game_sem.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    page = STATIC / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"message": "Frontend not built yet. POST a PGN to /api/annotate."})
