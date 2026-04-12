"""
Transcript formatter and summary-prompt builder (no API required).
Handles long transcripts via chunking.
"""

import logging

logger = logging.getLogger(__name__)

CHUNK_SIZE = 80_000        # chars per chunk
OVERLAP = 500              # char overlap between chunks to preserve context


def lang_label(language: str) -> str:
    return {"ja": "Japanese", "en": "English", "auto": "the same language as the input"}.get(
        language, "the same language as the input"
    )


def build_summary_prompt(transcript: str, language: str) -> str:
    """
    Build a ready-to-paste prompt for Claude chat.
    No API call — just formats the text.
    """
    if language == "ja":
        instruction = (
            "以下の動画の文字起こしを日本語で要約してください。\n\n"
            "【出力形式】\n"
            "## 概要\n3文以内で動画の内容を説明する\n\n"
            "## 主要ポイント\n- 箇条書き5〜8個\n\n"
            "## 結論・学び\n- 箇条書き3個\n\n"
            "【文字起こし】"
        )
    elif language == "en":
        instruction = (
            "Please summarize the following video transcript in English.\n\n"
            "【Output format】\n"
            "## Overview\nUp to 3 sentences\n\n"
            "## Key Points\n- 5–8 bullets\n\n"
            "## Takeaways\n- 3 bullets\n\n"
            "【Transcript】"
        )
    else:  # auto
        instruction = (
            "以下の動画の文字起こしを要約してください。文字起こしと同じ言語で出力してください。\n\n"
            "【出力形式】\n"
            "## 概要 / Overview\n3文以内\n\n"
            "## 主要ポイント / Key Points\n- 5〜8箇条\n\n"
            "## 結論・学び / Takeaways\n- 3箇条\n\n"
            "【文字起こし】"
        )

    return f"{instruction}\n\n{transcript}"


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def process_transcript(
    transcript: str,
    mode: str,
    language: str,
    status_callback=None,
) -> str:
    """
    Main entry point.
    mode: "transcript" — clean via Claude API, return formatted text
    mode: "prompt"     — return ready-to-paste summary prompt (no API call)
    language: "ja" | "en" | "auto"
    """

    def status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    if mode == "prompt":
        status("Building summary prompt...")
        return build_summary_prompt(transcript, language)

    elif mode == "transcript":
        # Return the raw Whisper transcript as-is — no API call needed.
        status("Formatting transcript...")
        return transcript

    else:
        raise ValueError(f"Unknown mode: {mode}")
