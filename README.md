# 動画ノート / YouTube Summarizer

複数のYouTube URL（最大20本）を順番に処理し、全文・実際の要約・要約用指示文を作成します。

## Web / Android 1.1.0

- **要約**: Groq APIで実際の要約を生成。画面のGroq APIキー、またはサーバーの `GROQ_API_KEY` が必要です。`SUMMARY_MODEL` でモデルを指定でき、既定は `llama-3.3-70b-versatile` です。
- **全文**: 取得した字幕・音声文字起こしをそのまま返します。
- **要約用指示文**: 原文に指示文を添える従来機能です。APIによる要約は行いません。
- 長文は12,000文字単位、250文字の重なりで分割して全パートを要約し、必要なら複数段階で統合します。原文の末尾は切り捨てません。API制限、出力打ち切り、空の応答は失敗表示します。
- 取得直後の全文を要約と別に送信し、ブラウザ/WebViewのIndexedDBに保存します。「保存済みの全文」からTXT保存できます。端末容量不足やブラウザデータ削除には対応できないため、必要な全文はTXT保存してください。
- 複数URLは重複を除いて処理し、1本の失敗後も次へ進みます。「以降を停止」は処理中の1本が終わってから停止します。
- キーは現在の実装では端末の保存領域に保存され、要約・文字起こし時にAPIサーバーへ送信されます。配布物とログにキーを含めないでください。
- 従来の字幕取得フォールバック（サーバーのSupadata `auto`、Groq Whisper）は維持しています。Chrome拡張の既存字幕のみの処理とは別です。

サーバー:

```sh
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

既存Render設定は `render.yaml`。公開版の `/health` に `version: 1.1.0` と `summary` が表示されて初めて更新反映を確認できます。

Androidの更新・ビルド方法は [android/README.md](android/README.md)。ソースは `android/`、APKは署名鍵を含まない配布物として別納品します。

## Chrome拡張

1. `chrome-extension` フォルダを保存します。
2. Chromeで `chrome://extensions` を開き、デベロッパーモードをONにします。
3. 「パッケージ化されていない拡張機能を読み込む」でこのフォルダを選択します。
4. YouTubeの視聴ページを再読み込みします。右下の「動画ノート」を開きます。

手動の「全文を取得」、または前面で実際に再生した180秒で一度取得します。広告・停止・シーク・非表示時間は加算しません。2倍速でも実時間を数えます。ページを再読み込みすると視聴時間のカウントは0に戻ります。保存済み全文は動画IDと希望言語ごとに再利用します。

字幕の直接取得にはYouTubeの非公開仕様を使うため、将来の仕様変更や動画の制限で失敗することがあります。未取得の続きがある字幕、空の字幕、現在配信中のライブを全文完了とは扱いません。

直接取得が失敗したときだけ、別ボタンからSupadata既存字幕取得を利用できます。APIキーは設定画面に入力してください。毎回確認後に `mode=native` で取得し、AI文字起こしは行いません。API通信は拡張のバックグラウンド内で行い、YouTubeにはキーを渡しません。通信間隔は1.1秒以上です。

APIの受付結果が未確定な場合は再送を止め、重複消費を避けます。設定の「前回のAPI処理が未確定の場合」から、利用状況を確認後に解除できます。受理済みジョブIDは保存して30秒間隔で最大15分確認します。

## 検証

```sh
python -m pytest tests/test_summary.py -q
node tests/test-youtube-multi.cjs
node android/tests/browser-logic.cjs
node tests/test-extension.cjs
```

自動テストは実機操作や実際のAPIによる要約品質の確認を代替しません。配布時点の実確認状況は納品の確認結果を参照してください。

実装参照: [Chrome service worker lifecycle](https://developer.chrome.com/docs/extensions/develop/concepts/service-workers/lifecycle)、[Supadata transcript](https://docs.supadata.ai/get-transcript)、[Groq API](https://console.groq.com/docs/openai)。
