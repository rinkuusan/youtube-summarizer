"""CLI: python -m night_botami digest.json --stage N -o OUTDIR [--date YYYY-MM-DD] [--dry-run] [--minutes 1.5]"""
import argparse
import json
import os
import sys

from . import tts
from .script import day_hooks, generate_script


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="night_botami")
    ap.add_argument("digest")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (default: latest day)")
    ap.add_argument("--stage", type=int, required=True, choices=range(1, 6))
    ap.add_argument("-o", "--output", default="night_out")
    ap.add_argument("--dry-run", action="store_true", help="script only, no speech synthesis")
    ap.add_argument("--minutes", type=float, default=1.5)
    args = ap.parse_args(argv)

    if not os.environ.get("SCRIPT_MODEL", "").strip():
        print("エラー: 環境変数 SCRIPT_MODEL（台本用のモデル名）を設定してください。", file=sys.stderr)
        return 2
    if not args.dry_run:
        for name in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"):
            if not os.environ.get(name, "").strip():
                print(f"エラー: 環境変数 {name} を設定してください（台本だけなら --dry-run）。", file=sys.stderr)
                return 2
    with open(args.digest, encoding="utf-8") as f:
        days = json.load(f).get("days") or []
    day = next((d for d in days if d["date"] == args.date), None) if args.date else (days[0] if days else None)
    if day is None:
        print("エラー: 指定した日付のデータが digest にありません。", file=sys.stderr)
        return 2

    try:
        lines = generate_script(day, args.stage, minutes=args.minutes)
        os.makedirs(args.output, exist_ok=True)
        base = os.path.join(args.output, f"{day['date']}_stage{args.stage}")
        with open(base + "_script.json", "w", encoding="utf-8") as f:
            json.dump({"date": day["date"], "stage": args.stage, "hooks": day_hooks(day), "lines": lines},
                      f, ensure_ascii=False, indent=2)
        if args.dry_run:
            print("\n".join(i["text"] for i in lines))
            return 0
        with open(base + ".mp3", "wb") as f:
            f.write(tts.synthesize(lines))
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    print(base + ".mp3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
