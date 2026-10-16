"""Parse the recording transcript log into blocks."""
import re
from dataclasses import dataclass

DAY_RE = re.compile(r"^===\s*(\d{4}-\d{2}-\d{2})\s*(?:\((.*)\))?\s*===\s*$")
HEAD_RE = re.compile(r"^(⚠\s*)?\[(\d{1,2}):(\d{2})\]\s*(.*)$")
WEEKDAY_RE = re.compile(r"\(([月火水木金土日])\)")


@dataclass
class Block:
    date: str
    weekday: str
    time: str
    speaker: str
    low_conf: bool
    text: str


def _split_speaker(rest: str) -> tuple[str, str]:
    """Speaker is the text before the first ': ' if short and free of spaces/。."""
    for sep in (": ", "：", ":"):
        idx = rest.find(sep)
        if idx > 0:
            cand = rest[:idx]
            if len(cand) <= 12 and " " not in cand and "。" not in cand and "　" not in cand:
                return cand, rest[idx + len(sep):].strip()
            break
    return "不明", rest.strip()


def parse_log(text: str) -> list[Block]:
    blocks: list[Block] = []
    date = weekday = ""
    cur: Block | None = None
    for line in text.splitlines():
        line = line.rstrip()
        m = DAY_RE.match(line.strip())
        if m:
            date = m.group(1)
            wm = WEEKDAY_RE.search(m.group(2) or "")
            weekday = wm.group(1) if wm else ""
            cur = None
            continue
        h = HEAD_RE.match(line)
        if h:
            speaker, body = _split_speaker(h.group(4))
            cur = Block(date, weekday, f"{int(h.group(2)):02d}:{h.group(3)}",
                        speaker, bool(h.group(1)), body)
            blocks.append(cur)
            continue
        if cur is not None and line.strip():
            cur.text += "\n" + line
    return blocks
