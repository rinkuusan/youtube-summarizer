# 5ch ビューアー (Android / Expo)

5ch の専用ブラウザ。React Native + Expo (SDK 57)。

**端末から 5ch へ直接通信する**。サーバもバックエンドも要らない。
リポジトリのルートにある FastAPI 製の YouTube 要約アプリとは完全に独立していて、
そちらは一切使わない。

## 手元の PC で動かす

必要なもの: Node.js 20 以上と Android 端末 (Expo Go をインストール)。

```bash
git clone <このリポジトリ>
cd youtube-summarizer/mobile
npm install
npx expo start
```

ターミナルに QR コードが出るので、Android の **Expo Go** アプリで読み取る。
PC と端末が同じ Wi-Fi にいれば繋がる。別ネットワークなら:

```bash
npx expo start --tunnel
```

ビルドも審査も Apple Developer 登録も不要。コードを保存すると即座に端末に反映される。

常用アプリとして端末に置きたくなったら、APK を作って直接インストールできる:

```bash
npx eas build -p android --profile preview
```

## テスト

実機もネットワークも使わずに検証できるようにしてある。実際の 5ch から取得した dat と
検索結果 HTML をフィクスチャに固定してあり、DB は Node 22 の `node:sqlite` で
本物の SQLite に対して走らせている。

```bash
npm test          # jest
npm run typecheck # tsc --noEmit
```

## 構成

```
src/
  net/      通信と Shift_JIS
    http.ts       expo/fetch のラッパ。バイト列で受け取る
    sjis.ts       Shift_JIS の復号・符号化・行境界の切り出し
    cookieJar.ts  投稿用の自前 Cookie 管理
    errors.ts     エラーの日本語化
  parse/    純関数のパーサ (テスト対象)
    datLine.ts    dat 1 行 -> Post
    body.ts       本文 -> 描画用トークン列
    findHtml.ts   find.5ch の検索結果
    entities.ts   HTML 実体参照
  api/      5ch のエンドポイント
    bbsmenu.ts    板一覧
    subject.ts    スレ一覧
    dat.ts        レス取得 (Range による差分取得)
    search.ts     スレタイ検索
    post.ts       bbs.cgi への書き込み
    postErrors.ts 応答の分類 (データ駆動)
    setting.ts    SETTING.TXT
    momentum.ts   勢い
  db/       expo-sqlite
    migrations.ts PRAGMA user_version の梯子
    threadRepo.ts 既読・お気に入り・履歴
    postRepo.ts   レスのキャッシュ
  filter/
    applyNg.ts    描画時の NG 判定 (連鎖対応)
  components/
  app/      expo-router のルート
```

## 実装上の勘所

コードを読む前に知っておくと早い点。

**Shift_JIS は `encoding-japanese` で扱う。** Hermes には `TextDecoder('shift_jis')` が
無く、既存の TextDecoder polyfill も UTF-8 系しか対応していない。

**`expo/fetch` を明示的に import する。** React Native 標準の `fetch` は
`.arrayBuffer()` が未実装で例外を投げる (facebook/react-native#34402)。
dat はバイト列で受け取らないと復号できないので、ここは代替が効かない。

**差分取得は行境界で切る。** Shift_JIS の 2 バイト目に `0x0A` (LF) は現れないので、
LF の直後でだけ切っている限り、Range のバイト境界で多バイト文字が割れることは
原理的に起こらない。不完全な行は保存もせず捨てて、次回取り直す
(`splitCompleteLines`)。

**レス番号 = 行インデックス + 1。** NG フィルタを掛ける前の生の行配列で確定させる。
ここを崩すとアプリ中のアンカーが全部ずれる。

**アンカーは dat の時点で HTML 化されている。** ただしタグ化されるのは先頭の 1 個だけで、
`>>4,8-10,13` の `,8-10,13` は生テキストのまま残る。だから `<a>` タグのパースと
生アンカーの正規表現の両方が要る。

**エンティティの復号はタグ抽出より後。** 先に復号すると、本文中に書かれた
`&lt;a&gt;` が本物のタグに化ける。

## 履歴は 2 種類

「開いたスレ」と「書き込んだスレ」を分けて自動で記録する。手で登録する操作は無い。

テーブルは分けず、`thread` に 2 本のタイムスタンプ列を置いている
(`last_opened_at` / `last_posted_at`)。スレは 1 つの実体で、お気に入りも既読も
履歴もその属性にすぎないため。

- 閲覧履歴 ... スレビューの mount で `threadRepo.touchOpened()`
- 書き込み履歴 ... 投稿成功時に `threadRepo.markPosted()`

刈り込みは非対称にしてある。**閲覧履歴は 500 件で打ち切るが、書き込み履歴は刈らない。**
自分が書いたスレは価値が高く、件数もたかが知れているので。

## 現状

閲覧・既読・お気に入り・履歴・検索・NG・書き込みまで一通り実装済み。

- スレを開くとキャッシュから即描画し、そのあと Range で差分だけ取る。
  **圏外でも取得済みのレスは読める**
- 未読の先頭に「ここまで読んだ」区切りを挟んで、その位置に復元する
- NG はワード / ID / 名前 / ワッチョイ、正規表現と連鎖に対応。描画時に適用するので
  切り替えに再取得が要らない
- 書き込みは下書きの自動保存、連投規制の先読み、確認画面の表示、
  「ブラウザで書き込む」の逃げ道つき

## 注意

- 生 dat の直読みは現在は通るが、5ch の裁量でいつでも塞がれうる。
- 読み取りの User-Agent は専ブラの慣例に従って `Monazilla/1.00 (...)` を名乗っている。
  自動更新の間隔を詰めすぎない、板を一括スクレイプしない、といった作法は守ること。
- 個人利用向け。ストア配信は想定していない。
