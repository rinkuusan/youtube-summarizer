"""Hallucination removal, speaker grouping and per-day cleanup."""
import re
from dataclasses import dataclass, field

from .parse import Block

HALLUCINATION = {
    "ありがとうございました", "ご視聴ありがとうございました", "ご視聴ありがとうございます",
    "次回予告", "はい", "うん", "エンド", "ん", "ありがとうございます",
    "ご清聴ありがとうございました", "おしまい", "ああ", "あ", "え", "ふう", "あふ",
}
TV = "テレビ・動画"
OTHER = "その他"
_PUNCT = "。、，．.!！?？ \t　…ー-"
_STRIP = re.compile(r"[。、，．.!！?？\s…]+$|^[。、，．.!！?？\s…]+")


def _is_filler(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    norm = s.strip(_PUNCT)
    if not norm:
        return True
    if norm in HALLUCINATION:
        return True
    # repeats such as 'うん。うん。うん。' / 'はい。はい'
    parts = [p.strip(_PUNCT) for p in re.split(r"[。、!！?？\s]+", s) if p.strip(_PUNCT)]
    return bool(parts) and all(p in HALLUCINATION for p in parts)


def clean_text(text: str) -> str:
    return "\n".join(l.strip() for l in text.splitlines() if not _is_filler(l)).strip()


def group_speaker(speaker: str) -> str:
    if speaker in ("純", "あやか"):
        return speaker
    if speaker == TV:
        return TV
    return OTHER


@dataclass
class DayClean:
    date: str
    weekday: str
    blocks: list = field(default_factory=list)   # kept Block (speaker grouped)
    watched: list = field(default_factory=list)  # ["HH:MM snippet"]
    counts: dict = field(default_factory=dict)


def clean_day(blocks: list[Block]) -> DayClean:
    date = blocks[0].date if blocks else ""
    weekday = blocks[0].weekday if blocks else ""
    day = DayClean(date, weekday)
    counts = {"純": 0, "あやか": 0, OTHER: 0, TV: 0, "low_conf": 0}
    for b in blocks:
        g = group_speaker(b.speaker)
        counts[g] += 1
        if b.low_conf:
            counts["low_conf"] += 1
        text = clean_text(b.text)
        if not text:
            continue
        if g == TV:
            flat = re.sub(r"\s+", " ", text).strip()
            day.watched.append(f"{b.time} {flat[:60]}")
        else:
            day.blocks.append(Block(b.date, b.weekday, b.time, g, b.low_conf, text))
    day.counts = counts
    return day
