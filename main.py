"""
YouTube Summarizer — FastAPI backend
Serves the single-page frontend and exposes the /process endpoint.
"""

import asyncio
import json
import logging
import logging.handlers
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── Logging setup ──────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)

_file = logging.handlers.RotatingFileHandler(
    LOG_DIR / "app.log",
    maxBytes=10 * 1024 * 1024,   # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_file.setFormatter(_fmt)
_file.setLevel(logging.DEBUG)    # full detail in file

logging.basicConfig(level=logging.INFO, handlers=[_console, _file])
logger = logging.getLogger(__name__)
logger.info(f"Log file: {LOG_DIR / 'app.log'}")

STATIC_DIR = Path(__file__).parent / "static"


def _get_local_ips() -> list[str]:
    """Return non-loopback IPv4 addresses for all NICs (LAN, Tailscale, etc.)."""
    import socket
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                ips.append(addr)
    except Exception:
        pass
    return sorted(set(ips))


@asynccontextmanager
async def lifespan(app: FastAPI):
    port = 8000
    logger.info("=" * 50)
    logger.info("YouTube Summarizer started")
    logger.info(f"  Local:      http://localhost:{port}")
    for ip in _get_local_ips():
        label = "Tailscale" if ip.startswith("100.") else "Network"
        logger.info(f"  {label:10s}  http://{ip}:{port}")
    logger.info("=" * 50)
    yield


app = FastAPI(title="YouTube Summarizer", lifespan=lifespan)

# Serve static files (index.html etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ProcessRequest(BaseModel):
    url: str
    mode: str           # "transcript" | "prompt"
    language: str       # "ja" | "en" | "auto"
    engine: str = "groq"
    groq_api_key: str = ""

# Accepted audio/video MIME types and extensions
ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".aac", ".opus", ".weba", ".mp4", ".webm", ".mkv", ".mov",
}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.post("/process")
async def process(req: ProcessRequest):
    """
    SSE-streaming endpoint. Sends JSON lines:
    {"type": "status", "message": "..."}
    {"type": "result",  "text": "..."}
    {"type": "error",   "message": "..."}
    """
    if req.mode not in ("transcript", "prompt"):
        raise HTTPException(status_code=400, detail="mode must be 'transcript' or 'prompt'")
    if req.language not in ("ja", "en", "auto"):
        raise HTTPException(status_code=400, detail="language must be 'ja', 'en', or 'auto'")

    # Use a queue so that callbacks from sync threads can feed into async SSE
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def status_cb(msg: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "status", "message": msg})

    def progress_cb(value: int):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "value": value})

    async def run_pipeline():
        """Run the blocking pipeline in a thread pool and stream events."""
        try:
            # --- Step 1: Get transcript ---
            from transcriber import get_transcript

            logger.info("/process: starting get_transcript")
            groq_key = req.groq_api_key if req.engine == "groq" else ""
            transcript_data = await asyncio.to_thread(
                get_transcript, req.url, req.language, status_cb, progress_cb, groq_key
            )
            logger.info("/process: get_transcript returned")
            transcript_text = transcript_data["text"]
            method = transcript_data["method"]

            if not transcript_text.strip():
                queue.put_nowait({"type": "error", "message": "Transcript is empty."})
                return

            method_label = {'captions': 'YouTube captions', 'groq': 'Groq Whisper API'}.get(method, method)
            status_cb(
                f"Transcript obtained via {method_label} "
                f"({len(transcript_text):,} chars). Processing..."
            )
            progress_cb(90)

            # --- Step 2: Process with Claude ---
            from summarizer import process_transcript

            logger.info(f"/process: starting process_transcript (mode={req.mode}, chars={len(transcript_text):,})")
            result = await asyncio.to_thread(
                process_transcript, transcript_text, req.mode, req.language, status_cb
            )

            logger.info(f"/process: process_transcript returned ({len(result):,} chars)")
            progress_cb(100)
            queue.put_nowait({"type": "result", "text": result})

        except ValueError as e:
            logger.error(f"/process pipeline ValueError: {e}")
            queue.put_nowait({"type": "error", "message": str(e)})
        except BaseException as e:
            logger.exception(f"/process pipeline error ({type(e).__name__})")
            queue.put_nowait({"type": "error", "message": f"Processing failed: {e}"})
        finally:
            logger.info("/process: pipeline finally — putting sentinel")
            queue.put_nowait(None)  # sentinel

    async def event_generator():
        task = asyncio.create_task(run_pipeline())
        pending_get = asyncio.ensure_future(queue.get())
        try:
            while True:
                done, _ = await asyncio.wait({pending_get}, timeout=KEEPALIVE_INTERVAL)
                if not done:
                    yield ": keepalive\n\n"
                    continue
                item = pending_get.result()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                pending_get = asyncio.ensure_future(queue.get())
        except GeneratorExit:
            logger.info("SSE client disconnected (/process) — pipeline continues in background")
            if not pending_get.done():
                pending_get.cancel()
            return
        finally:
            if not pending_get.done():
                pending_get.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


KEEPALIVE_INTERVAL = 20  # seconds — prevents browser SSE timeout during long Whisper jobs


def _make_sse_pipeline(queue: asyncio.Queue, coro):
    """
    Shared SSE streaming wrapper with safe keepalive.

    Uses asyncio.wait (not asyncio.shield + wait_for) so queue items are
    NEVER silently consumed and discarded on timeout.
    """
    async def event_generator():
        task = asyncio.create_task(coro)
        # One persistent Future per queue item — never cancelled on keepalive
        pending_get = asyncio.ensure_future(queue.get())
        try:
            while True:
                done, _ = await asyncio.wait({pending_get}, timeout=KEEPALIVE_INTERVAL)
                if not done:
                    # Timeout — item not ready yet, ping the browser and keep waiting
                    yield ": keepalive\n\n"
                    continue
                item = pending_get.result()
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
                # Create fresh Future for next item
                pending_get = asyncio.ensure_future(queue.get())
        except GeneratorExit:
            # Client disconnected mid-stream — let pipeline finish (don't waste GPU work)
            logger.info("SSE client disconnected — pipeline continues in background")
            if not pending_get.done():
                pending_get.cancel()
            return
        finally:
            if not pending_get.done():
                pending_get.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/process-file")
async def process_file(
    file: UploadFile = File(...),
    mode: str = Form(...),
    language: str = Form(...),
    engine: str = Form(default="groq"),
    groq_api_key: str = Form(default=""),
):
    """
    SSE-streaming endpoint for uploaded audio files.
    Accepts multipart/form-data: file + mode + language.
    """
    if mode not in ("transcript", "prompt"):
        raise HTTPException(status_code=400, detail="mode must be 'transcript' or 'prompt'")
    if language not in ("ja", "en", "auto"):
        raise HTTPException(status_code=400, detail="language must be 'ja', 'en', or 'auto'")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )

    # Read file content immediately (before the generator runs)
    file_bytes = await file.read()
    original_name = file.filename or f"audio{ext}"

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def status_cb(msg: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "status", "message": msg})

    def progress_cb(value: int):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "value": value})

    async def run_pipeline():
        tmp_path = None
        try:
            # Save upload to temp file with correct extension
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            status_cb(f"File received: {original_name} ({len(file_bytes) / 1_048_576:.1f} MB). Transcribing...")
            progress_cb(2)

            from transcriber import get_transcript_from_file
            from summarizer import process_transcript

            # Run transcription + summarization in ONE thread.
            # Two separate asyncio.to_thread calls had a timing gap between them
            # where the coroutine could be swallowed by uvicorn's connection lifecycle.
            def run_sync() -> str:
                logger.info(f"[THREAD] full pipeline start: {original_name}")

                groq_key = groq_api_key if engine == "groq" else ""
                tdata = get_transcript_from_file(tmp_path, language, status_cb, progress_cb, groq_key)
                text  = tdata["text"]
                logger.info(f"[THREAD] transcription done: {len(text):,} chars")

                # Push status/progress from thread (call_soon_threadsafe is thread-safe)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "status",   "message": f"Transcription complete ({len(text):,} chars). Processing..."},
                )
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "value": 90})

                result = process_transcript(text, mode, language, status_cb)
                logger.info(f"[THREAD] processing done: {len(result):,} chars")
                return result

            logger.info(f"/process-file: launching combined pipeline thread (mode={mode})")
            result = await asyncio.to_thread(run_sync)
            logger.info(f"/process-file: thread returned — {len(result):,} chars")

            progress_cb(100)
            queue.put_nowait({"type": "result", "text": result})

        except BaseException as e:
            logger.exception(f"/process-file pipeline error ({type(e).__name__})")
            queue.put_nowait({"type": "error", "message": f"Processing failed: {e}"})
        finally:
            logger.info("/process-file: pipeline finally — cleaning up")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            queue.put_nowait(None)

    return _make_sse_pipeline(queue, run_pipeline())


@app.get("/health")
async def health():
    return {"status": "ok"}
