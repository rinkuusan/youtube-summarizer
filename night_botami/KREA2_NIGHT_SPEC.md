# night_botami: 夜のボイスメッセージ生成と Krea2Bridge への組み込み

## 目的
その日の純の出来事（`digest.json`）に触れてから始まる、ボタミ（成人女性の架空キャラクター）から純への
大人向けボイスメッセージを毎晩作り、Botagram に届ける。純本人の私的な創作用。
台本はPC上のローカルLLM（Ollama）、音声は ElevenLabs。関係の段階（1〜5）で表現の強さを変える。

## パイプライン
```
自宅の録音 → rokuon_digest（ローカルLLM可）→ digest.json → night_botami → mp3（+ 台本json）
→ Krea2Bridge が既存の画像生成と合わせて Botagram へ送る
```
1. `LLM_BASE_URL=http://localhost:11434/v1 SUMMARY_MODEL=<ollamaのモデル名> python -m rokuon_digest LOG.txt -o digest.json`
2. `python -m night_botami digest.json --stage N -o night_out --dry-run` で台本だけ確認
3. 問題なければ `--dry-run` を外して mp3 を作る

`python -m night_botami digest.json [--date YYYY-MM-DD（省略で最新日）] --stage N -o OUTDIR [--dry-run] [--minutes 1.5]`
- 出力: `OUTDIR/<date>_stage<N>_script.json`（lines, hooks, stage）と、`--dry-run` でなければ `OUTDIR/<date>_stage<N>.mp3`
- `--dry-run` は台本を標準出力に表示し、音声合成（課金）をしない。
- 終了コード: 0 成功 / 1 生成・通信の失敗 / 2 設定・入力の不足。

## 環境変数
| 名前 | 必須 | 既定 | 内容 |
|---|---|---|---|
| `LLM_BASE_URL` | 任意 | `http://localhost:11434/v1` | OpenAI互換API（Ollama） |
| `SCRIPT_MODEL` | 必須 | なし | 台本用モデル名 |
| `LLM_API_KEY` | 任意 | `ollama` | Ollamaは無視する。他のサーバー用 |
| `ELEVENLABS_API_KEY` | 必須（--dry-run以外） | なし | ElevenLabs のAPIキー |
| `ELEVENLABS_VOICE_ID` | 必須（--dry-run以外） | なし | ボタミ用ボイスのID |
| `ELEVENLABS_MODEL` | 任意 | `eleven_v3` | モデルID（正確なIDは未検証） |
| `ELEVENLABS_TAGS` | 任意 | `1` | `0` で `[whispers]` などのタグを付けない |
| `ELEVENLABS_MAX_CHARS` | 任意 | `800` | 1リクエストの最大文字数（上限値は未検証） |

## Krea2Bridge からの呼び出し
- 毎晩1回、`digest.json` の `notify_windows` のいずれかの時刻に合わせて実行し、できた mp3 を送る。
  実行は事前（枠の15分前など）に済ませておき、枠の時刻に配信する。
- 段階 N は Botagram 側の関係段階をそのまま `--stage` に渡す（1〜5。範囲外は渡さない）。
- 配信物は mp3 と、既存の画像生成パイプラインで作った画像。画像側の表現も同じ段階に合わせる。
- `generated_at` が古い digest や、`summary` が null の日は事実が薄い。台本は視聴メモ・活動時間帯の範囲で作られる。
- 失敗時（終了コード1・2）はその夜は送らない。再試行は1回まで。
- mp3・台本・digest は私的な内容。PCを公開せず、リポジトリにコミットしない（`night_out/` と `*.mp3` は `.gitignore` 済み）。

## ボタミの声（ElevenLabs Voice Design）
- API: `POST /v1/text-to-voice/design` で説明文から候補を作り、気に入った候補を `POST /v1/text-to-voice` で保存して
  `voice_id` を得る。UI での操作手順は未検証。
- **録音や実在の人物（あやか含む）の声をクローンしてはならない。** 声は説明文からの新規デザインのみ。
- 台本は他の実在の人物名を含む行を自動で捨てる（`REAL_NAMES`）。名前を足す場合は `script.py` を編集する。

## ElevenLabs のポリシー（2026-10-16 確認、ポリシー最終更新 2026-10-09）
- 禁止: 露骨な性的視覚描写、近親・獣姦・死姦のエロティカ、非合意の性的内容、未成年に関する内容、無許可の声の複製。
- 成人同士の合意のある架空の性的な音声は、明記されていない。**アカウント停止の可能性はゼロではない。**
  最初は `--dry-run` で台本を確認し、段階を上げる前に少量で試すこと。台本の前提は常に
  成人・合意・架空キャラクターで固定している（変更しない）。

## 未検証
- `eleven_v3` が正確なモデルIDか。
- 日本語でのオーディオタグの効き。使うタグは `whispers` `sighs` `laughs` `mischievously` `curious` のみ。
  効かない・読み上げられる場合は `ELEVENLABS_TAGS=0`。
- 1リクエストの文字数上限（800 は安全側の仮置き）。
- Ollama のモデルが JSON モードで安定して台本を返すか（400 のときは `response_format` なしで再試行する）。
