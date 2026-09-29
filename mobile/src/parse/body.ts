import { decodeEntities } from './entities';

/**
 * レス本文を描画用のトークン列に変換する。
 *
 * 実物の dat では、アンカーは既に read.cgi 向けの <a> タグに変換されて入っている:
 *   <a href="../test/read.cgi/news4vip/1784974154/4" ...>&gt;&gt;4</a>,8-10,13,16,21,26
 * ただしタグ化されるのは先頭の 1 個だけで、`,8-10,13,...` の続きは生テキストのまま残る。
 * したがって「タグのパース」と「生アンカーの正規表現」の両方が必要になる。
 *
 * 処理順が重要:
 *   1. <br> を改行に
 *   2. <a> タグを先に切り出す
 *   3. 残りのテキストでエンティティを復号し、生アンカーと URL を拾う
 * エンティティの復号を先にやると、本文中の `&lt;a&gt;` が偽のタグに化ける。
 */

export type Segment =
  | { type: 'text'; text: string }
  | { type: 'anchor'; from: number; to: number; text: string }
  | { type: 'link'; url: string; text: string }
  | { type: 'image'; url: string; text: string };

const A_TAG = /<a\s+[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>/gis;
const READ_CGI = /read\.cgi\/([^/]+)\/(\d+)\/(\d+)(?:-(\d+))?/i;

/** 生アンカー (>>12, >>12-15, >>4,8-10) と素の URL をまとめて拾う。 */
const INLINE = /(?:>|＞){2,}\s*(\d+(?:\s*[-‐ー－]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-‐ー－]\s*\d+)?)*)|(https?:\/\/[^\s<>"'）)】「」]+)/g;

/** 先頭が `,8-10,13` のような、直前のアンカーから続くレス番リストか。 */
const CONTINUATION = /^\s*((?:,\s*\d+(?:\s*[-‐ー－]\s*\d+)?)+)/;

const IMAGE_EXT = /\.(jpe?g|png|gif|webp|bmp|avif)(?:[?#]|$)/i;
const IMAGE_HOST = /^https?:\/\/(?:i\.imgur\.com|pbs\.twimg\.com|i\.gyazo\.com)\//i;

/** URL の末尾にくっつきがちな句読点を落とす。 */
function trimUrlTail(url: string): string {
  return url.replace(/[.,、。！？!?:;]+$/, '');
}

function isImageUrl(url: string): boolean {
  return IMAGE_EXT.test(url) || IMAGE_HOST.test(url);
}

/**
 * 5ch のリダイレクタを剥がして、本物の URL を取り出す。
 *
 * 5ch は外部リンクを `<a href="http://jump5.ch/?https://i.imgur.com/x.jpeg">` の
 * ように包む。表示テキストだけが本物なので、href をそのまま使うと画像が
 * リダイレクトページを読みに行って失敗する (しかも http なので Android の
 * 平文通信ブロックにも掛かる)。
 *
 * 形式は 2 通り。
 *   http://jump5.ch/?https://example.com/x.jpg   … クエリに丸ごと入る新しい形
 *   http://ime.nu/example.com/x.jpg              … パスに繋げる古い形 (scheme 無し)
 */
export function unwrapRedirect(url: string): string {
  const q = /^https?:\/\/(?:jump\.5ch\.net|jump5\.ch|jump\.2ch\.net|pinktower\.com)\/\?(.+)$/i.exec(url);
  if (q) return unwrapRedirect(q[1]);

  const p = /^https?:\/\/(?:ime\.nu|ime\.st|jump\.5ch\.net|jump5\.ch)\/(.+)$/i.exec(url);
  if (p) {
    const rest = p[1];
    // 古い形は scheme が落ちているので補う。
    return unwrapRedirect(/^https?:\/\//i.test(rest) ? rest : `http://${rest}`);
  }
  return url;
}

function urlSegment(url: string): Segment {
  const clean = unwrapRedirect(trimUrlTail(url));
  return isImageUrl(clean) ? { type: 'image', url: clean, text: clean } : { type: 'link', url: clean, text: clean };
}

interface Range {
  from: number;
  to: number;
  text: string;
}

/** `4,8-10,13` を個々の範囲に分解する。 */
function parseAnchorSpec(spec: string): Range[] {
  const out: Range[] = [];
  for (const part of spec.split(',')) {
    const t = part.trim();
    const m = /^(\d+)(?:\s*[-‐ー－]\s*(\d+))?$/.exec(t);
    if (!m) continue;
    const from = Number(m[1]);
    const to = m[2] ? Number(m[2]) : from;
    if (!Number.isFinite(from) || from < 1) continue;
    out.push({ from, to: Math.max(from, to), text: t });
  }
  return out;
}

/** 範囲リストをアンカーセグメント列にする。先頭にだけ `>>` を付ける。 */
function rangesToSegments(ranges: Range[], leadWithMarker: boolean): Segment[] {
  const out: Segment[] = [];
  ranges.forEach((r, i) => {
    if (i > 0) out.push({ type: 'text', text: ',' });
    const marker = i === 0 && leadWithMarker ? '>>' : '';
    out.push({ type: 'anchor', from: r.from, to: r.to, text: `${marker}${r.text}` });
  });
  return out;
}

function pushText(out: Segment[], text: string) {
  if (!text) return;
  const last = out[out.length - 1];
  if (last && last.type === 'text') last.text += text;
  else out.push({ type: 'text', text });
}

/**
 * <a> タグを除いた素のテキストを走査する。
 * @param afterAnchor 直前のセグメントがアンカーなら、先頭のレス番リストも継続として扱う
 */
function tokenizeText(raw: string, afterAnchor: boolean): Segment[] {
  const out: Segment[] = [];
  let text = decodeEntities(raw);

  if (afterAnchor) {
    const cont = CONTINUATION.exec(text);
    if (cont) {
      const ranges = parseAnchorSpec(cont[1].replace(/^\s*,/, ''));
      if (ranges.length) {
        out.push(...rangesToSegments(ranges, false));
        text = text.slice(cont[0].length);
      }
    }
  }

  let last = 0;
  INLINE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = INLINE.exec(text)) !== null) {
    pushText(out, text.slice(last, m.index));
    if (m[1] !== undefined) {
      const ranges = parseAnchorSpec(m[1]);
      if (ranges.length) out.push(...rangesToSegments(ranges, true));
      else pushText(out, m[0]);
    } else if (m[2] !== undefined) {
      out.push(urlSegment(m[2]));
    }
    last = m.index + m[0].length;
  }
  pushText(out, text.slice(last));

  return out;
}

/** 本文フィールドをトークン化する。 */
export function parseBody(rawBody: string): Segment[] {
  // 本文は前後に 1 個ずつパディングの空白が付く。AA の字下げを壊さないよう 1 個だけ剥がす。
  let s = rawBody;
  if (s.startsWith(' ')) s = s.slice(1);
  if (s.endsWith(' ')) s = s.slice(0, -1);

  // <br> の前後の空白もパディングなので 1 個ずつだけ吸収する。
  s = s.replace(/ ?<br> ?/gi, '\n');
  s = s.replace(/<hr\s*\/?>/gi, '\n');

  const out: Segment[] = [];
  let last = 0;
  A_TAG.lastIndex = 0;
  let m: RegExpExecArray | null;

  while ((m = A_TAG.exec(s)) !== null) {
    const before = s.slice(last, m.index);
    const prevWasAnchor = out.length > 0 && out[out.length - 1].type === 'anchor';
    out.push(...tokenizeText(before, prevWasAnchor && before.length > 0 ? true : false));

    const href = m[1];
    const inner = decodeEntities(m[2].replace(/<[^>]*>/g, ''));
    const rc = READ_CGI.exec(href);
    if (rc) {
      const from = Number(rc[3]);
      const to = rc[4] ? Number(rc[4]) : from;
      out.push({ type: 'anchor', from, to: Math.max(from, to), text: inner });
    } else if (/^https?:\/\//i.test(href)) {
      const seg = urlSegment(href);
      out.push({ ...seg, text: inner || seg.text } as Segment);
    } else {
      pushText(out, inner);
    }
    last = m.index + m[0].length;
  }

  const tail = s.slice(last);
  const prevWasAnchor = out.length > 0 && out[out.length - 1].type === 'anchor';
  out.push(...tokenizeText(tail, prevWasAnchor));

  return out;
}

/** アンカーが指すレス番号を列挙する。逆参照マップを作るのに使う。 */
export function anchorTargets(segments: Segment[], maxSpan = 50): number[] {
  const out: number[] = [];
  for (const seg of segments) {
    if (seg.type !== 'anchor') continue;
    // >>1-1000 のような広範囲は逆参照としては意味が薄いので展開しない
    if (seg.to - seg.from >= maxSpan) {
      out.push(seg.from);
      continue;
    }
    for (let n = seg.from; n <= seg.to; n++) out.push(n);
  }
  return out;
}

/** 表示用のプレーンテキスト (NG ワード判定やコピーに使う)。 */
export function segmentsToPlainText(segments: Segment[]): string {
  return segments.map((s) => s.text).join('');
}
