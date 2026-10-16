"""Small OpenAI-compatible chat client (Ollama by default)."""
import json
import os
import re

import httpx

DEFAULT_BASE = "http://localhost:11434/v1"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL", "").strip() or DEFAULT_BASE).rstrip("/")


def model_name() -> str:
    model = os.environ.get("SCRIPT_MODEL", "").strip()
    if not model:
        raise ValueError("環境変数 SCRIPT_MODEL（台本用のモデル名。例: Ollamaで取得したモデル名）を設定してください。")
    return model


def parse_script_json(content: str) -> list:
    """Return the list of line dicts from {"lines":[...]} (or a bare list), tolerating code fences."""
    s = content.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    data = None
    for pattern in (None, r"\{.*\}", r"\[.*\]"):
        cand = s if pattern is None else (re.search(pattern, s, re.S) or [None])[0]
        if cand is None:
            continue
        try:
            data = json.loads(cand)
            break
        except ValueError:
            continue
    if isinstance(data, dict):
        data = data.get("lines")
    if not isinstance(data, list):
        raise ValueError("台本の返答をJSONとして解釈できませんでした。")
    out = []
    for item in data:
        if isinstance(item, str):
            item = {"text": item, "tag": ""}
        if isinstance(item, dict) and str(item.get("text") or "").strip():
            out.append({"text": str(item["text"]).strip(), "tag": str(item.get("tag") or "").strip()})
    return out


def complete(messages: list) -> str:
    model = model_name()
    base = base_url()
    try:
        body = {"model": model, "temperature": 0.9, "messages": messages}
        # Retry once without response_format if the server rejects it with 400.
        bodies = [dict(body, response_format={"type": "json_object"}), body]
        with httpx.Client(timeout=300) as client:
            for b in bodies:
                response = client.post(
                    base + "/chat/completions",
                    # Ollama ignores the key, but OpenAI-style clients send one.
                    headers={"Authorization": "Bearer " + (os.environ.get("LLM_API_KEY", "").strip() or "ollama")},
                    json=b,
                )
                if response.status_code != 400:
                    break
        if response.status_code != 200:
            known = {401: "APIキーを確認してください。", 403: "このモデルの利用権限を確認してください。",
                     404: "モデル名（SCRIPT_MODEL）またはURLを確認してください。",
                     429: "利用上限に達しました。時間を置いて再実行してください。"}
            raise ValueError("台本LLM: " + known.get(response.status_code, f"HTTP {response.status_code}。"))
        content = (response.json()["choices"][0]["message"].get("content") or "").strip()
        if not content:
            raise ValueError("台本LLMが空の結果を返しました。")
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"台本LLMとの通信に失敗しました（{base}）。Ollamaは起動していますか？") from exc
    return content
