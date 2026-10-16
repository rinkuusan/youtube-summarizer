"""Activity rhythm of 純 from block times (no LLM)."""
from .parse import Block


def _fmt(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def hist30(blocks: list[Block]) -> dict:
    out: dict = {}
    for b in blocks:
        if b.speaker != "純":
            continue
        key = _fmt(_minutes(b.time) // 30 * 30)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def jun_active(hist: dict) -> list:
    starts = sorted(_minutes(k) for k in hist)
    spans: list = []
    for s in starts:
        if spans and s - spans[-1][1] <= 60:   # gap between bucket starts
            spans[-1][1] = s
        else:
            spans.append([s, s])
    return [{"start": _fmt(a), "end": _fmt(b + 30)} for a, b in spans]


def notify_windows(blocks: list[Block], top: int = 3) -> list:
    by_day: dict = {}
    for b in blocks:
        by_day.setdefault(b.date, [])
        if b.speaker == "純":
            by_day[b.date].append(_minutes(b.time))
    total = len(by_day)
    if not total:
        return []
    scored = []
    for start in range(0, 24 * 60 - 59, 30):
        days = sum(1 for ts in by_day.values() if any(start <= t < start + 60 for t in ts))
        if days:
            scored.append((days / total, start))
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen: list = []
    for score, start in scored:   # greedy, non-overlapping
        if all(abs(start - c[1]) >= 60 for c in chosen):
            chosen.append((score, start))
        if len(chosen) == top:
            break
    return [{"start": _fmt(s), "end": _fmt(s + 60), "score": round(sc, 2)} for sc, s in chosen]
