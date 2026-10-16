"""ElevenLabs text-to-speech for the script lines."""
import os

import httpx

URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128"
VOICE_SETTINGS = {"stability": 0.4, "similarity_boost": 0.75}


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"環境変数 {name} を設定してください。")
    return value


def use_tags_default() -> bool:
    return os.environ.get("ELEVENLABS_TAGS", "1").strip() not in ("0", "", "false", "no")


def _line(item: dict, use_tags: bool) -> str:
    tag = item.get("tag") or ""
    return f"[{tag}] {item['text']}" if use_tags and tag else item["text"]


def render_text(lines: list, use_tags: bool) -> str:
    return "\n".join(_line(i, use_tags) for i in lines)


def chunk_lines(lines: list, use_tags: bool, max_chars: int) -> list:
    chunks, cur = [], ""
    for item in lines:
        text = _line(item, use_tags)
        while len(text) > max_chars:   # a single over-long line: hard split
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(text[:max_chars])
            text = text[max_chars:]
        if cur and len(cur) + 1 + len(text) > max_chars:
            chunks.append(cur)
            cur = ""
        cur = cur + "\n" + text if cur else text
    if cur:
        chunks.append(cur)
    return chunks


def synthesize(lines: list, use_tags=None) -> bytes:
    key, voice = _env("ELEVENLABS_API_KEY"), _env("ELEVENLABS_VOICE_ID")
    # "eleven_v3" is the model id as far as known; the exact id is unverified.
    model = os.environ.get("ELEVENLABS_MODEL", "").strip() or "eleven_v3"
    max_chars = int(os.environ.get("ELEVENLABS_MAX_CHARS", "800") or 800)   # limit unverified
    if use_tags is None:
        use_tags = use_tags_default()
    errors = {401: "ElevenLabsのAPIキーを確認してください。", 403: "ElevenLabsの利用権限・プランを確認してください。",
              422: "ElevenLabsが入力を受け付けませんでした（モデルID・ボイスID・タグを確認）。",
              429: "ElevenLabsの利用上限に達しました。時間を置いて再実行してください。"}
    audio = b""
    try:
        with httpx.Client(timeout=120) as client:
            for chunk in chunk_lines(lines, use_tags, max_chars):
                response = client.post(
                    URL.format(voice_id=voice),
                    headers={"xi-api-key": key, "Content-Type": "application/json"},
                    json={"text": chunk, "model_id": model, "voice_settings": VOICE_SETTINGS},
                )
                if response.status_code != 200:
                    raise ValueError("音声合成: " + errors.get(response.status_code, f"HTTP {response.status_code}。"))
                audio += response.content
    except httpx.HTTPError as exc:
        raise ValueError("ElevenLabsとの通信に失敗しました。") from exc
    return audio
