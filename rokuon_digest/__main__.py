"""CLI: python -m rokuon_digest LOG.txt -o digest.json [--no-llm] [--days N]"""
import argparse
import datetime
import json
import os
import sys

from .clean import clean_day
from .parse import parse_log
from .rhythm import hist30, jun_active, notify_windows
from .summarize_day import needs_key, summarize_day


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rokuon_digest")
    ap.add_argument("log")
    ap.add_argument("-o", "--output", default="digest.json")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--days", type=int, default=0, help="only the latest N days")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args(argv)

    key = args.api_key.strip() or os.environ.get("GROQ_API_KEY", "").strip()
    if not key and not args.no_llm and needs_key():
        print("エラー: GROQ_API_KEY（または --api-key）が必要です（ローカルLLMなら LLM_BASE_URL を設定）。LLMなしなら --no-llm を指定してください。",
              file=sys.stderr)
        return 2
    with open(args.log, encoding="utf-8") as f:
        blocks = parse_log(f.read())
    dates = sorted({b.date for b in blocks if b.date}, reverse=True)
    if args.days > 0:
        dates = dates[:args.days]
    blocks = [b for b in blocks if b.date in dates]

    days = []
    for d in dates:
        day_blocks = [b for b in blocks if b.date == d]
        day = clean_day(day_blocks)
        h = hist30(day_blocks)
        summary = None
        if not args.no_llm and (day.blocks or day.watched):
            try:
                summary = summarize_day(day, key)
            except ValueError as exc:
                print(f"{d}: 要約失敗: {exc}", file=sys.stderr)
        c = day.counts
        print(f"{d} ({day.weekday}) 純={c['純']} あやか={c['あやか']} その他={c['その他']} "
              f"テレビ・動画={c['テレビ・動画']} low_conf={c['low_conf']}", file=sys.stderr)
        days.append({"date": d, "weekday": day.weekday, "jun_active": jun_active(h), "hist30": h,
                     "counts": c, "watched": day.watched, "summary": summary})

    out = {"generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "source": os.path.basename(args.log),
           "notify_windows": notify_windows(blocks), "days": days}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
