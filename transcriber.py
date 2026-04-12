"""
Transcript extraction pipeline (cloud edition — no local GPU):
1. youtube-transcript-api (fast, free)
2. yt-dlp audio download → Groq Whisper API (cloud STT)
"""

import json
import os
import re
import subprocess
import tempfile
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)


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
# YouTube captions
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
# yt-dlp audio download
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

    prog(5)
    status("Fetching captions...")
    caption_text = fetch_youtube_captions(video_id, language)

    if caption_text:
        prog(85)
        status("Captions found.")
        return {"text": caption_text, "method": "captions", "video_id": video_id}

    # No captions — need Groq API key for audio transcription
    if not groq_api_key:
        raise ValueError(
            "この動画には字幕がありません。音声文字起こしにはGroq APIキーが必要です。"
        )

    prog(15)
    status("No captions found. Extracting audio...")
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

    prog(87)
    return {"text": transcript_text, "method": "groq", "video_id": video_id}
