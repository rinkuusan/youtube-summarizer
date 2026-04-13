"""
Transcript extraction pipeline (cloud edition):
1. youtube-transcript-api (fast, free)
2. Supadata API fallback (handles captionless videos via AI transcription)
3. yt-dlp + Groq Whisper (last resort, may be blocked on cloud IPs)
"""

import json
import os
import re
import subprocess
import tempfile
import time
import logging
from typing import Optional, Callable

import httpx

logger = logging.getLogger(__name__)

SUPADATA_API_URL = "https://api.supadata.ai/v1/youtube/transcript"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_preferred_languages(language: str) -> list[str]:
    if language == "ja":
        return ["ja", "en"]
    elif language == "en":
        return ["en", "ja"]
    else:
        return ["ja", "en", "ko", "zh", "fr", "de", "es", "it", "pt"]


# ─────────────────────────────────────────────
# YouTube captions (free, no API key)
# ─────────────────────────────────────────────

def fetch_youtube_captions(video_id: str, language: str) -> Optional[str]:
    from youtube_transcript_api import YouTubeTranscriptApi

    preferred = get_preferred_languages(language)

    def _join(entries) -> str:
        texts = []
        for e in entries:
            if isinstance(e, dict):
                texts.append(e.get("text", ""))
            else:
                texts.append(getattr(e, "text", ""))
        return " ".join(texts).strip()

    # ── Strategy 1: instance-based API (>=0.6.x) ──
    try:
        api = YouTubeTranscriptApi()
        for lang in preferred:
            try:
                entries = api.fetch(video_id, languages=[lang])
                text = _join(entries)
                if text:
                    logger.info(f"Got captions via api.fetch (lang={lang})")
                    return text
            except Exception:
                pass
        try:
            entries = api.fetch(video_id)
            text = _join(entries)
            if text:
                logger.info("Got captions via api.fetch (any lang)")
                return text
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Instance-based fetch failed: {e}")

    # ── Strategy 2: class-method API (<0.6.x) ──
    try:
        for lang in preferred:
            try:
                entries = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                text = _join(entries)
                if text:
                    logger.info(f"Got captions via get_transcript (lang={lang})")
                    return text
            except Exception:
                pass
        try:
            entries = YouTubeTranscriptApi.get_transcript(video_id)
            text = _join(entries)
            if text:
                logger.info("Got captions via get_transcript (any lang)")
                return text
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"Class-method fetch failed: {e}")

    logger.info("No captions available for this video")
    return None


# ─────────────────────────────────────────────
# Supadata API (handles captionless videos)
# ─────────────────────────────────────────────

def fetch_supadata_transcript(
    video_id: str,
    language: str,
    groq_api_key: str,
    status_callback=None,
    progress_callback=None,
) -> Optional[str]:
    """Fetch transcript via Supadata API with auto mode (AI fallback for captionless videos)."""

    # Use Groq API key as Supadata key if it starts with expected prefix,
    # otherwise check for SUPADATA_API_KEY env var
    supadata_key = os.environ.get("SUPADATA_API_KEY", "")
    if not supadata_key:
        logger.info("No SUPADATA_API_KEY set, skipping Supadata fallback")
        return None

    def status(msg):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    def prog(v):
        if progress_callback:
            progress_callback(v)

    lang_param = None if language == "auto" else language
    url = f"https://www.youtube.com/watch?v={video_id}"

    status("Supadata APIで文字起こし中（字幕なし動画対応）...")
    prog(20)

    try:
        params = {"url": url, "text": "true", "mode": "auto"}
        if lang_param:
            params["lang"] = lang_param

        headers = {"x-api-key": supadata_key}

        with httpx.Client(timeout=30) as client:
            resp = client.get(SUPADATA_API_URL, params=params, headers=headers)

        if resp.status_code == 202:
            # Async job — poll for result
            job = resp.json()
            job_id = job.get("jobId", "")
            if not job_id:
                logger.warning("Supadata returned 202 but no jobId")
                return None

            status("Supadata: AI文字起こし処理中（長い動画は時間がかかります）...")
            poll_url = f"{SUPADATA_API_URL}/{job_id}"
            for attempt in range(120):  # max ~4 min
                time.sleep(2)
                prog(20 + min(60, attempt))
                with httpx.Client(timeout=15) as client:
                    poll_resp = client.get(poll_url, headers=headers)
                if poll_resp.status_code == 200:
                    data = poll_resp.json()
                    content = data.get("content", "")
                    if content:
                        logger.info(f"Supadata async job complete: {len(content)} chars")
                        return content
                    return None
                elif poll_resp.status_code == 202:
                    continue  # still processing
                else:
                    logger.warning(f"Supadata poll error: {poll_resp.status_code}")
                    return None

            logger.warning("Supadata job timed out")
            return None

        elif resp.status_code == 200:
            data = resp.json()
            content = data.get("content", "")
            if content:
                logger.info(f"Supadata immediate result: {len(content)} chars")
                prog(85)
                return content
            return None

        else:
            logger.warning(f"Supadata error: {resp.status_code} {resp.text[:200]}")
            return None

    except Exception as e:
        logger.warning(f"Supadata API failed: {e}")
        return None


# ─────────────────────────────────────────────
# yt-dlp audio download (may fail on cloud IPs)
# ─────────────────────────────────────────────

def download_audio_yt_dlp(video_id: str, output_dir: str) -> str:
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(output_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for ext in ["m4a", "webm", "opus", "ogg", "mp3", "wav", "flac"]:
        candidate = os.path.join(output_dir, f"audio.{ext}")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError("yt-dlp did not produce an audio file")


# ─────────────────────────────────────────────
# Groq Whisper API
# ─────────────────────────────────────────────

GROQ_FILE_LIMIT_BYTES = 25 * 1024 * 1024  # 25 MB


def transcribe_with_groq(
    audio_path: str,
    language: str,
    groq_api_key: str,
    status_callback=None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> str:
    """Transcribe audio via Groq Whisper API (no GPU required, 25 MB limit)."""
    from groq import Groq

    def status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    file_size = os.path.getsize(audio_path)
    if file_size > GROQ_FILE_LIMIT_BYTES:
        raise ValueError(
            f"File size {file_size / 1_048_576:.1f} MB exceeds Groq limit (25 MB). "
            "Try a shorter video or one with captions available."
        )

    status(f"Groq Whisper APIで文字起こし中 ({file_size / 1_048_576:.1f} MB)...")
    if progress_callback:
        progress_callback(20)

    client = Groq(api_key=groq_api_key)
    whisper_lang = None if language == "auto" else language

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3-turbo",
            language=whisper_lang,
            response_format="text",
        )

    if progress_callback:
        progress_callback(90)

    return result.strip() if isinstance(result, str) else result.text.strip()


# ─────────────────────────────────────────────
# Public entry points
# ─────────────────────────────────────────────

def get_transcript_from_file(
    file_path: str,
    language: str,
    status_callback=None,
    progress_callback: Optional[Callable[[int], None]] = None,
    groq_api_key: str = "",
) -> dict:
    def status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    if progress_callback:
        progress_callback(2)

    if not groq_api_key:
        raise ValueError("Groq APIキーが必要です。画面上部で入力してください。")

    text = transcribe_with_groq(file_path, language, groq_api_key, status_callback, progress_callback)
    if not text.strip():
        raise ValueError("Transcription returned empty result.")
    return {"text": text, "method": "groq"}


def get_transcript(
    url: str,
    language: str,
    status_callback=None,
    progress_callback: Optional[Callable[[int], None]] = None,
    groq_api_key: str = "",
) -> dict:
    def status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    def prog(v: int):
        if progress_callback:
            progress_callback(v)

    video_id = extract_video_id(url)

    # ── Step 1: Try youtube-transcript-api (free, fast) ──
    prog(5)
    status("字幕を取得中...")
    caption_text = fetch_youtube_captions(video_id, language)

    if caption_text:
        prog(85)
        status("字幕を取得しました。")
        return {"text": caption_text, "method": "captions", "video_id": video_id}

    # ── Step 2: Try Supadata API (handles captionless videos) ──
    prog(10)
    status("字幕なし。Supadata APIで文字起こしを試行中...")
    supadata_text = fetch_supadata_transcript(
        video_id, language, groq_api_key, status_callback, progress_callback
    )

    if supadata_text:
        prog(85)
        status("Supadata APIで文字起こし完了。")
        return {"text": supadata_text, "method": "supadata", "video_id": video_id}

    # ── Step 3: Try yt-dlp + Groq (may fail on cloud) ──
    if not groq_api_key:
        raise ValueError(
            "この動画には字幕がありません。音声文字起こしにはGroq APIキーが必要です。"
        )

    prog(15)
    status("音声をダウンロード中...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_audio_yt_dlp(video_id, tmpdir)
            prog(20)

            def scaled(p: int):
                prog(20 + int(p * 0.65))

            status("Groq Whisper APIで文字起こし中...")
            transcript_text = transcribe_with_groq(
                audio_path, language, groq_api_key, status_callback, scaled
            )
            if not transcript_text.strip():
                raise ValueError("Transcription returned empty result.")
    except Exception as e:
        err_msg = str(e)
        if "Sign in" in err_msg or "bot" in err_msg or "cookies" in err_msg:
            raise ValueError(
                "この動画には字幕がなく、YouTubeが音声取得をブロックしています。\n"
                "Supadata APIキーを設定するか、字幕付き動画を試してください。\n"
                "（Renderダッシュボードで環境変数 SUPADATA_API_KEY を設定）"
            ) from e
        raise

    prog(87)
    return {"text": transcript_text, "method": "groq", "video_id": video_id}
