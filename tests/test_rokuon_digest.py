import json

import httpx
import pytest

from rokuon_digest import __main__ as cli
from rokuon_digest.clean import clean_day
from rokuon_digest.parse import parse_log
from rokuon_digest.rhythm import hist30, jun_active, notify_windows
from rokuon_digest.summarize_day import summarize_day

LOG = """=== 2026-10-15 (10月15日(木)) ===
[4:05] テレビ・動画: ニュースです。今日は晴れ
はい
エンド
[7:05] ありがとうございました。はい

=== 2026-10-14 (10月14日(水)) ===
[8:00] 純: おはよう
今日は出かける
⚠ [8:10] あやか: 朝ごはんできたよ
[8:40] 純: うん。うん。うん。
はい
[9:10] 人物57: こんにちは
[13:30] 純: 仕事する
[13:45] 純: 休憩
"""


def blocks():
    return parse_log(LOG)


def test_parse():
    bs = blocks()
    assert len(bs) == 8
    assert bs[0].speaker == "テレビ・動画" and bs[0].time == "04:05" and bs[0].date == "2026-10-15"
    assert bs[0].weekday == "木"
    assert bs[1].speaker == "不明" and bs[1].text == "ありがとうございました。はい"
    assert bs[2].text == "おはよう\n今日は出かける"
    assert bs[3].low_conf and bs[3].speaker == "あやか" and not bs[2].low_conf


def test_hallucination_tv_counts():
    d1 = clean_day([b for b in blocks() if b.date == "2026-10-15"])
    assert d1.blocks == []
    assert d1.watched == ["04:05 ニュースです。今日は晴れ"]
    assert d1.counts["テレビ・動画"] == 1 and d1.counts["その他"] == 1
    d2 = clean_day([b for b in blocks() if b.date == "2026-10-14"])
    assert [b.time for b in d2.blocks] == ["08:00", "08:10", "09:10", "13:30", "13:45"]
    assert d2.counts == {"純": 4, "あやか": 1, "その他": 1, "テレビ・動画": 0, "low_conf": 1}
    assert d2.blocks[1].low_conf and d2.blocks[2].speaker == "その他"


def test_rhythm():
    bs = [b for b in blocks() if b.date == "2026-10-14"]
    h = hist30(bs)
    assert h == {"08:00": 1, "08:30": 1, "13:30": 2}
    assert jun_active(h) == [{"start": "08:00", "end": "09:00"}, {"start": "13:30", "end": "14:00"}]
    w = notify_windows(blocks())
    assert len(w) <= 3 and w[0] == {"start": "07:30", "end": "08:30", "score": 0.5}


def reply(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"finish_reason": finish, "message": {"content": content}}]})


GOOD = {"jun_did": ["仕事"], "jun_mood": "普通", "with_ayaka": [], "topics": [], "uncertain": []}


def test_summarize_fenced_and_errors(monkeypatch):
    day = clean_day([b for b in blocks() if b.date == "2026-10-14"])
    seen = {}

    def post(self, url, **kw):
        seen.update(kw["json"])
        return reply("```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```")
    monkeypatch.setattr(httpx.Client, "post", post)
    assert summarize_day(day, "k") == GOOD
    assert seen["response_format"] == {"type": "json_object"}
    assert "⚠ [08:10] あやか" in seen["messages"][1]["content"]
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: reply("not json"))
    with pytest.raises(ValueError):
        summarize_day(day, "k")
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: httpx.Response(429, json={}))
    with pytest.raises(ValueError, match="利用上限"):
        summarize_day(day, "k")
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **k: reply("{}", "length"))
    with pytest.raises(ValueError, match="途中"):
        summarize_day(day, "k")


def test_summarize_chunk_and_merge(monkeypatch):
    big = "\n".join(f"[{8 + i % 10}:{i % 60:02d}] 純: " + "あ" * 200 + str(i) for i in range(150))
    day = clean_day(parse_log("=== 2026-10-14 (10月14日(水)) ===\n" + big))
    calls = []

    def post(self, url, **kw):
        calls.append(kw["json"]["messages"][1]["content"])
        return reply(json.dumps(GOOD, ensure_ascii=False))
    monkeypatch.setattr(httpx.Client, "post", post)
    assert summarize_day(day, "k") == GOOD
    assert len(calls) >= 3 and calls[-1].startswith("[{")


def test_summarize_retries_without_response_format_on_400(monkeypatch):
    day = clean_day([b for b in blocks() if b.date == "2026-10-14"])
    payloads = []

    def post(self, url, **kw):
        payloads.append(kw["json"])
        if len(payloads) == 1:
            return httpx.Response(400, json={})
        return reply(json.dumps(GOOD, ensure_ascii=False))
    monkeypatch.setattr(httpx.Client, "post", post)
    assert summarize_day(day, "k") == GOOD
    assert len(payloads) == 2
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]


def test_cli_no_llm(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    src, out = tmp_path / "x_rokuon.txt", tmp_path / "digest.json"
    src.write_text(LOG, encoding="utf-8")
    assert cli.main([str(src), "-o", str(out)]) == 2
    assert cli.main([str(src), "-o", str(out), "--no-llm"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [d["date"] for d in data["days"]] == ["2026-10-15", "2026-10-14"]
    assert data["source"] == "x_rokuon.txt" and data["days"][0]["summary"] is None
    assert data["days"][1]["counts"]["純"] == 4
    assert "2026-10-14" in capsys.readouterr().err
    assert cli.main([str(src), "-o", str(out), "--no-llm", "--days", "1"]) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))["days"]) == 1
