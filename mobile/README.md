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

パーサは実機なしで検証できる。実際の 5ch から取得した dat を
`src/parse/__fixtures__/sample.dat.b64` に固定してあるので、オフラインで走る。

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
    errors.ts     エラーの日本語化
  parse/    純関数のパーサ (テスト対象)
    datLine.ts    dat 1 行 -> Post
    body.ts       本文 -> 描画用トークン列
    entities.ts   HTML 実体参照
  api/      5ch のエンドポイント
    bbsmenu.ts    板一覧
    subject.ts    スレ一覧
    dat.ts        レス取得 (Range による差分取得)
    momentum.ts   勢い
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

## 現状

読む側は動く。板一覧 -> スレ一覧 (勢い順) -> レス表示、アンカーのポップアップ
(多段で辿れる)、逆参照、画像・URL のリンク化、Range による差分取得まで。

未実装: お気に入りと既読管理 (SQLite)、NG、検索、書き込み。

## 注意

- 生 dat の直読みは現在は通るが、5ch の裁量でいつでも塞がれうる。
- 読み取りの User-Agent は専ブラの慣例に従って `Monazilla/1.00 (...)` を名乗っている。
  自動更新の間隔を詰めすぎない、板を一括スクレイプしない、といった作法は守ること。
- 個人利用向け。ストア配信は想定していない。
