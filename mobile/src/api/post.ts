import type { SQLiteDatabase } from 'expo-sqlite';

import * as kvRepo from '../db/kvRepo';
import type { ThreadRef } from '../db/types';
import * as cookieJar from '../net/cookieJar';
import { fetchBytes, getHeaders, resolveUrl } from '../net/http';
import { buildSjisForm, decodeSjis } from '../net/sjis';
import { log } from '../net/log';
import { classifyPostResponse, type PostResult } from './postErrors';

/**
 * bbs.cgi への書き込み。
 *
 * 正直に言っておくと、ここは実装が正しくても書けないことが普通にある。
 * 5ch 側の規制 (どんぐり、SAMBA24、IP 規制) に阻まれるためで、
 * 成功を保証する作りにはしていない。応答の内容をそのまま見せて、
 * ユーザーが原因を判断できるようにするのが現実解。
 */

/**
 * 書き込み用の User-Agent。
 *
 * 読み取りは Monazilla を名乗るが (5ch 側で受理されることを実測済み)、bbs.cgi は
 * Cloudflare を通るので実ブラウザの UA を使う。加えて、どんぐりは身元を
 * IP + UA に紐付けるので、この値は**変えてはいけない**。変えると認証が切れる。
 */
export const POST_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36';

/** 直近の投稿時刻 (板ごと)。SAMBA24 を手元で先読みするために持つ。 */
const LAST_POST_KEY = 'post.lastPostAt';
/** 板ごとの連投間隔の目安。5ch 側の SAMBA24 は板により異なるので控えめに置く。 */
export const LOCAL_COOLDOWN_MS = 30_000;

export interface PostDraft {
  name: string;
  mail: string;
  message: string;
  /** スレ立てのときだけ入れる。既存スレへのレスでは使わない。 */
  subject?: string;
}

export interface PostOptions {
  /** 書き込み確認ページを承諾して再送する。ユーザーの明示的な操作を経てのみ true にする。 */
  accepted?: boolean;
  /**
   * 確認ページのフォームに入っていた値。承諾時にそのまま積み直す。
   *
   * これを送り返さないと、何度承諾しても確認ページが返り続ける。
   * 実測 (2026-07-28、kizuna.5ch.io/gamefight): 確認ページのフォームは
   * feature と submit の 2 つだけを持ち、こちらが組み立てた submit 文字列を
   * 送っても 5ch は承諾と認めない。使い捨てトークンである feature を
   * 返すことが条件になっている。
   */
  confirmFields?: Record<string, string>;
  /**
   * 確認ページのフォームの送信先。相対のことがある。
   * 実測 (2026-07-28) では `../test/bbs.cgi?guid=ON` で、この `?guid=ON` が無いと
   * 5ch は承諾と認めず確認ページを返し続ける。
   */
  confirmAction?: string | null;
}

function bbsCgiUrl(host: string): string {
  return `https://${host}/test/bbs.cgi`;
}

function refererUrl(ref: ThreadRef): string {
  // スレ立て時はまだスレが無いので板のトップを名乗る。
  return ref.key
    ? `https://${ref.host}/test/read.cgi/${ref.board}/${ref.key}/`
    : `https://${ref.host}/${ref.board}/`;
}

/** 手元で分かる連投規制。無駄に 5ch 側の規制カウントを踏ませない。 */
export async function checkLocalCooldown(
  db: SQLiteDatabase,
  host: string,
  board: string
): Promise<number> {
  const map = await kvRepo.getJson<Record<string, number>>(db, LAST_POST_KEY, {});
  const last = map[`${host}/${board}`];
  if (!last) return 0;
  const remain = LOCAL_COOLDOWN_MS - (Date.now() - last);
  return remain > 0 ? Math.ceil(remain / 1000) : 0;
}

async function recordPostAttempt(db: SQLiteDatabase, host: string, board: string): Promise<void> {
  const map = await kvRepo.getJson<Record<string, number>>(db, LAST_POST_KEY, {});
  map[`${host}/${board}`] = Date.now();
  await kvRepo.setJson(db, LAST_POST_KEY, map);
}

/**
 * レスを投稿する。
 *
 * 1 回目は普通に送る。5ch が「書き込み確認」を返したら、その本文をそのまま
 * 呼び出し側に渡す (outcome==='confirm')。UI がそれを改変せず表示し、
 * ユーザーが承諾したら accepted:true で呼び直す。
 *
 * 確認画面を自動で承諾しないのは、5ch が専ブラに対して明示的に求めている作法で、
 * 破るとクライアントごと弾かれる近道になるため。
 */
export async function submitPost(
  db: SQLiteDatabase,
  ref: ThreadRef,
  draft: PostDraft,
  opts: PostOptions = {}
): Promise<PostResult> {
  const base = bbsCgiUrl(ref.host);
  // 承諾時は確認ページが指す宛先へ送る。素の bbs.cgi に投げ直すと
  // ?guid=ON が落ちて、何度承諾しても確認ページが返ってくる。
  const url =
    opts.accepted && opts.confirmAction ? resolveUrl(base, opts.confirmAction) : base;

  const fields: Record<string, string> = {
    bbs: ref.board,
    // スレ立ては key の代わりに subject を送る。両方は送らない。
    ...(draft.subject ? { subject: draft.subject } : { key: ref.key }),
    // 新しすぎる値を弾くサーバがあるので 60 秒過去にする
    time: String(Math.floor(Date.now() / 1000) - 60),
    FROM: draft.name,
    mail: draft.mail,
    MESSAGE: draft.message,
    submit: draft.subject ? '新規スレッド作成' : '書き込む',
  };

  // 承諾して送り直すときは、確認ページのフォームの値を上書きで積む。
  // submit の文字列もページ側のものを使う (こちらで文字列を推測しない)。
  if (opts.accepted && opts.confirmFields) {
    Object.assign(fields, opts.confirmFields);
  }

  const headers: Record<string, string> = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'User-Agent': POST_UA,
    Referer: refererUrl(ref),
    Origin: `https://${ref.host}`,
  };
  const cookie = await cookieJar.header(db, ref.host);
  if (cookie) headers.Cookie = cookie;

  await recordPostAttempt(db, ref.host, ref.board);

  const res = await fetchBytes(url, {
    method: 'POST',
    headers,
    // 値は Shift_JIS でパーセントエンコードする。encodeURIComponent は UTF-8 に
    // なるので使えない。buildSjisForm が fallback:'html-entity' 込みで面倒を見る。
    body: buildSjisForm(fields),
  });

  // 応答の Set-Cookie を取り込む。確認ページの再送にはこれが要る。
  await cookieJar.store(db, ref.host, getHeaders(res.headers, 'set-cookie'));

  const html = decodeSjis(res.bytes);
  const result = classifyPostResponse(html, res.status);
  logPostExchange(url, fields, cookie, getHeaders(res.headers, 'set-cookie'), html, result, opts);
  return result;
}

/**
 * 書き込みの往復を記録する。
 *
 * 「書き込み確認が何度押しても返ってくる」のが最も多い詰まり方で、原因は
 * Cookie が保存/送信できていないか、確認フォームが要求する隠しフィールドを
 * 送り返せていないかのどちらか。画面には 5ch の文面しか出ないため、
 * どちらなのかを切り分けられる情報をここで残す。
 *
 * Cookie の値は認証情報そのものなので出さない。名前と長さだけにする。
 */
function logPostExchange(
  url: string,
  sentFields: Record<string, string>,
  sentCookie: string | null,
  setCookie: string[],
  html: string,
  result: PostResult,
  opts: PostOptions
): void {
  const names = (c: string) => c.split(/;\s*/).map((p) => p.split('=')[0]).filter(Boolean);
  const sent = sentCookie ? names(sentCookie) : [];
  const got = setCookie.map((line) => line.split('=')[0].trim());
  const title = /<title>([^<]*)<\/title>/i.exec(html)?.[1]?.trim() ?? '(title なし)';

  // 応答フォームの送信先。同意の宛先が bbs.cgi とは限らないので必ず見る。
  const actions = [...html.matchAll(/<form\b[^>]*\baction\s*=\s*"([^"]*)"/gi)].map((m) => m[1]);
  const inputs = [...html.matchAll(/<input\b[^>]*>/gi)].map((m) => m[0]);

  // logcat にも載るよう、切り分けに要る情報は msg 側に置く。
  log(
    'info',
    'post',
    `結果=${result.outcome} 承諾=${opts.accepted ? 'あり' : 'なし'} title=${title}` +
      ` | 送信フィールド=${Object.keys(sentFields).join(',')}` +
      ` | 送信先=${url}` +
      ` | 応答form action=${actions.length ? actions.join(' , ') : '(なし)'}` +
      ` | 応答input=${inputs.join(' ')}` +
      ` | 送信Cookie=${sent.length ? sent.join(',') : 'なし'}` +
      ` | 受信Set-Cookie=${got.length ? got.join(',') : 'なし'}`,
    html
  );
}

// --- 下書き ---

function draftKey(ref: ThreadRef): string {
  return `draft.${ref.host}/${ref.board}/${ref.key}`;
}

/** 規制エラーで長文を失わせないよう、入力のたびに保存する。 */
export async function saveDraft(
  db: SQLiteDatabase,
  ref: ThreadRef,
  draft: PostDraft
): Promise<void> {
  await kvRepo.setJson(db, draftKey(ref), draft);
}

export async function loadDraft(db: SQLiteDatabase, ref: ThreadRef): Promise<PostDraft | null> {
  return kvRepo.getJson<PostDraft | null>(db, draftKey(ref), null);
}

export async function clearDraft(db: SQLiteDatabase, ref: ThreadRef): Promise<void> {
  await kvRepo.remove(db, draftKey(ref));
}

// --- 自分のレスの後追い照合 ---

const MY_POST_KEY = 'post.pendingMine';

interface PendingMine {
  host: string;
  board: string;
  key: string;
  message: string;
  postedAt: number;
}

/**
 * bbs.cgi はレス番号を返さないので、投稿した本文を控えておき、
 * 次に dat を取ったときに一致するレスを探して印を付ける。
 */
export async function rememberMyPost(
  db: SQLiteDatabase,
  ref: ThreadRef,
  message: string
): Promise<void> {
  const list = await kvRepo.getJson<PendingMine[]>(db, MY_POST_KEY, []);
  list.push({ ...ref, message, postedAt: Date.now() });
  // 溜め込まない。古いものから落とす。
  await kvRepo.setJson(db, MY_POST_KEY, list.slice(-20));
}

export async function takePendingMine(
  db: SQLiteDatabase,
  ref: ThreadRef
): Promise<PendingMine[]> {
  const list = await kvRepo.getJson<PendingMine[]>(db, MY_POST_KEY, []);
  const mine = list.filter((p) => p.host === ref.host && p.board === ref.board && p.key === ref.key);
  if (mine.length > 0) {
    const rest = list.filter((p) => !mine.includes(p));
    await kvRepo.setJson(db, MY_POST_KEY, rest);
  }
  return mine;
}

/** 本文が一致するか。dat 側は <br> やエンティティを含むので緩く比べる。 */
export function bodyMatchesMine(datBody: string, myMessage: string): boolean {
  const norm = (s: string) =>
    s
      .replace(/<br>/gi, '')
      .replace(/&[a-z#0-9]+;/gi, '')
      .replace(/\s+/g, '')
      .trim();
  const a = norm(datBody);
  const b = norm(myMessage);
  return a.length > 0 && b.length > 0 && a === b;
}
