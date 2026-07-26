import type { SQLiteDatabase } from 'expo-sqlite';

import * as kvRepo from '../db/kvRepo';
import type { ThreadRef } from '../db/types';
import * as cookieJar from '../net/cookieJar';
import { fetchBytes, getHeaders } from '../net/http';
import { buildSjisForm, decodeSjis } from '../net/sjis';
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
}

export interface PostOptions {
  /** 書き込み確認ページを承諾して再送する。ユーザーの明示的な操作を経てのみ true にする。 */
  accepted?: boolean;
}

function bbsCgiUrl(host: string): string {
  return `https://${host}/test/bbs.cgi`;
}

function refererUrl(ref: ThreadRef): string {
  return `https://${ref.host}/test/read.cgi/${ref.board}/${ref.key}/`;
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
  const url = bbsCgiUrl(ref.host);

  const fields: Record<string, string> = {
    bbs: ref.board,
    key: ref.key,
    // 新しすぎる値を弾くサーバがあるので 60 秒過去にする
    time: String(Math.floor(Date.now() / 1000) - 60),
    FROM: draft.name,
    mail: draft.mail,
    MESSAGE: draft.message,
    submit: opts.accepted ? '上記全てを承諾して書き込む' : '書き込む',
  };

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
  return classifyPostResponse(html, res.status);
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
