# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-page web app that turns a YouTube URL (or an uploaded audio/video file) into
either a clean transcript or a ready-to-paste summary prompt. FastAPI backend + a single
static `index.html` frontend. The UI text is primarily Japanese. Deployed to Render
(see `render.yaml` / `Procfile`).

## Commands

```bash
pip install -r requirements.txt          # install deps
uvicorn main:app --reload --port 8000    # run locally (http://localhost:8000)
uvicorn main:app --host 0.0.0.0 --port $PORT   # production (Render uses this)
```

There is no test suite, linter, or build step configured. The frontend is served directly
from `static/index.html` — no bundler, no npm.

## Architecture

The request flow is a three-stage pipeline, each stage in its own module:

1. **`main.py`** — FastAPI app. Two endpoints do the real work:
   - `POST /process` (JSON body: `url`, `mode`, `language`, `engine`, `groq_api_key`) for URLs
   - `POST /process-file` (multipart) for uploaded audio/video
   Both stream **Server-Sent Events** back to the browser as JSON lines of shape
   `{"type": "status"|"progress"|"result"|"error", ...}`. `GET /` serves the SPA;
   `GET /health` is a liveness probe.

2. **`transcriber.py`** — `get_transcript()` (URL) and `get_transcript_from_file()` (upload)
   return `{"text": ..., "method": ...}`. The URL path tries fallbacks **in order**:
   `youtube-transcript-api` (free captions) → Supadata API (needs `SUPADATA_API_KEY`,
   handles captionless videos) → `yt-dlp` audio download + Groq Whisper. File uploads go
   straight to Groq Whisper.

3. **`summarizer.py`** — `process_transcript()`. Despite the "summarizer" name and stale
   docstrings mentioning a "Claude API", **no LLM call happens here**. `mode="transcript"`
   returns the raw transcript as-is; `mode="prompt"` returns a formatted prompt string the
   user pastes into a chat assistant themselves. Chunking helpers exist but are currently unused.

### Cross-cutting conventions

- **SSE + threads bridge:** The blocking pipeline runs in `asyncio.to_thread`. Sync code
  reports progress via `status_callback`/`progress_callback`, which use
  `loop.call_soon_threadsafe(queue.put_nowait, ...)` to feed an `asyncio.Queue` that the SSE
  generator drains. A `None` sentinel on the queue signals completion. When editing the
  streaming logic, preserve the "one persistent Future per queue item" pattern in
  `_make_sse_pipeline` / `event_generator` — it deliberately avoids `wait_for`/`shield` so
  queue items are never silently dropped on a keepalive timeout (`KEEPALIVE_INTERVAL = 20s`).
- **Client disconnect:** On `GeneratorExit` the pipeline is intentionally left running in the
  background (don't waste a long Whisper job) rather than cancelled.
- **`mode`** is always `"transcript"` or `"prompt"`; **`language`** is always `"ja"`, `"en"`,
  or `"auto"`. Both are validated at the endpoint. `engine` defaults to `"groq"` (the only
  real engine).
- **Groq API key** comes from the browser (stored in `localStorage`, sent per-request), not
  from server env. Groq Whisper uses model `whisper-large-v3-turbo` with a hard **25 MB**
  file limit. Supadata is the only key read from server env (`SUPADATA_API_KEY`).
- **Status messages** shown to users are Japanese strings; keep new ones consistent.
- **Logging:** rotating file handler writes full DEBUG to `logs/app.log` (gitignored);
  console is INFO. Use the module `logger`, not `print`.

## Deployment notes

`yt-dlp` audio download often fails on cloud IPs (YouTube bot-blocking). The code detects
"Sign in"/"bot"/"cookies" errors and tells the user to set `SUPADATA_API_KEY` or use a
captioned video — so on Render, Supadata is effectively the primary fallback for captionless
videos, not yt-dlp.
