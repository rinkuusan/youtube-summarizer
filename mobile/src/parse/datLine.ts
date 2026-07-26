import { decodeEntities } from './entities';

/**
 * dat の 1 行をパースする。
 *
 * 実物の形式 (5 フィールド、`<>` 区切り):
 *   名前<>メール<>日付 ID<>本文<>スレタイ
 * スレタイは 1 レス目にのみ入る。
 *
 * 例:
 *   以下、5ちゃんねるからVIPがお送りします </b>(ﾜｯﾁｮｲW a587-tyyI)<b><><>
 *   2026/07/25(土) 19:09:14.630 ID:mMwNdz2O0<> 本文 <br> 本文 <>VIPでウマ娘
 */

export interface Post {
  /** レス番号。行インデックス + 1。 */
  res: number;
  /** 表示用の名前 (タグ除去済み)。 */
  name: string;
  /** (ﾜｯﾁｮｲW a587-tyyI) のような識別子。無ければ null。 */
  wacchoi: string | null;
  /** ◆ トリップ。無ければ null。 */
  trip: string | null;
  /** ★ 付き (運営キャップ)。 */
  isCap: boolean;
  mail: string;
  /** 日付欄の生文字列。 */
  dateText: string;
  /** JST として解釈した epoch ミリ秒。読めなければ null。 */
  timestamp: number | null;
  /** ID:xxxx。ID 非表示板では null になる。 */
  uid: string | null;
  be: string | null;
  /** 本文の生文字列 (<br> やエンティティを含む)。描画時に body.ts でトークン化する。 */
  body: string;
  /** あぼーん (削除済み) レス。 */
  isAbone: boolean;
}

/** dat 全体のパース結果。 */
export interface ParsedThread {
  title: string | null;
  posts: Post[];
}

const TAG_B = /<\/?b>/gi;
const TAG_SMALL = /<\/?small>/gi;

/**
 * 名前欄を分解する。
 *
 * read.cgi は名前欄全体を <b>...</b> で囲んで描画する。つまり dat 内の `</b>` は
 * 「ここから非太字」、`<b>` は「ここから再び太字」を意味し、その間に挟まれた部分が
 * ワッチョイやキャップなどの付加情報になっている。
 */
function parseName(raw: string): Pick<Post, 'name' | 'wacchoi' | 'trip' | 'isCap'> {
  let wacchoi: string | null = null;
  let isCap = false;

  // </b> と <b> に挟まれた部分 = 付加情報
  const between = /<\/b>(.*?)(?:<b>|$)/is.exec(raw);
  if (between) {
    const sub = between[1];
    const paren = /[（(]([^）)]*)[）)]/.exec(sub);
    if (paren) wacchoi = paren[1].trim() || null;
    if (sub.includes('★')) isCap = true;
  }
  if (raw.includes('★')) isCap = true;

  // タグを剥がす。<small> の中身 (どんぐりの階級など) は表示したいので中身は残す。
  let name = raw.replace(TAG_B, '').replace(TAG_SMALL, '');

  // ワッチョイ部分は別枠で出すので名前からは取り除く
  if (wacchoi) {
    name = name.replace(/[（(][^）)]*[）)]\s*$/, '');
  }

  let trip: string | null = null;
  const tripM = /◆[!-~]{8,12}/.exec(name);
  if (tripM) {
    trip = tripM[0];
    name = name.replace(tripM[0], '');
  }

  return { name: decodeEntities(name).trim(), wacchoi, trip, isCap };
}

/** 日付欄から日時・ID・BE を切り出す。 */
function parseDateField(raw: string): Pick<Post, 'dateText' | 'timestamp' | 'uid' | 'be'> {
  const uidM = /ID:([^\s<]+)/.exec(raw);
  const beM = /BE:([^\s<]+)/.exec(raw);

  let timestamp: number | null = null;
  const dm = /(\d{4})\/(\d{2})\/(\d{2})\([^)]*\)\s*(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?/.exec(raw);
  if (dm) {
    const [, y, mo, d, h, mi, s, ms] = dm;
    // dat の時刻は JST 固定。端末のタイムゾーンに依存させないため UTC に直して持つ。
    timestamp = Date.UTC(+y, +mo - 1, +d, +h - 9, +mi, +s, ms ? +ms.padEnd(3, '0').slice(0, 3) : 0);
  }

  return {
    dateText: raw.trim(),
    timestamp,
    uid: uidM ? uidM[1] : null,
    be: beM ? beM[1] : null,
  };
}

/** 1 行をパースする。res は呼び出し側が行インデックス+1 で渡す。 */
export function parseDatLine(line: string, res: number): Post | null {
  if (!line) return null;
  const f = line.split('<>');
  if (f.length < 4) return null;

  const [rawName, rawMail, rawDate, rawBody] = f;
  const isAbone = rawName === 'あぼーん' && rawBody === 'あぼーん';

  return {
    res,
    ...parseName(rawName),
    mail: decodeEntities(rawMail).trim(),
    ...parseDateField(rawDate),
    body: rawBody,
    isAbone,
  };
}

/**
 * dat 全体をパースする。
 *
 * 不変条件: レス番号 = 行インデックス + 1。NG フィルタを掛ける「前」の生の行配列で確定させる。
 * ここを崩すとアプリ中のアンカーが全部ずれるので、この関数の外でレス番号を振り直さないこと。
 */
export function parseDat(lines: string[], startRes = 1): ParsedThread {
  const posts: Post[] = [];
  let title: string | null = null;

  lines.forEach((line, i) => {
    const res = startRes + i;
    const post = parseDatLine(line, res);
    if (!post) return;
    if (res === 1) {
      const f = line.split('<>');
      if (f.length >= 5 && f[4]) title = decodeEntities(f[4]).trim();
    }
    posts.push(post);
  });

  return { title, posts };
}
