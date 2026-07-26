import type { SQLiteDatabase } from 'expo-sqlite';

import * as kvRepo from '../db/kvRepo';

/**
 * 自前の Cookie ジャー。
 *
 * fetchBytes は既定で credentials:'omit' なので、OS の Cookie ストアには触らせない。
 * 自前で持つ理由:
 *  - iOS と Android で挙動が違い、中身が見えない
 *  - 「どんぐり Cookie の期限」を表示したり「Cookie を削除」を出したりできない。
 *    書き込み確認が延々ループする定番の詰まりは、Cookie を捨てれば直ることが多く、
 *    それをワンタップで出せるようにしておきたい
 *
 * Set-Cookie は http.ts の getHeaders() で取る。expo/fetch の _rawHeaders は
 * [name, value] の配列なので、同名ヘッダが複数あっても失われない
 * (標準の Headers に入れると連結されて壊れる)。
 */

export interface StoredCookie {
  name: string;
  value: string;
  /** epoch ms。無期限なら null。 */
  expiresAt: number | null;
}

type Jar = Record<string, StoredCookie[]>;

const KEY = 'net.cookies';

function jarKey(host: string): string {
  // 5ch は板ごとにサーバが分かれるが Cookie は 5ch.io 全体で共有される。
  const m = /([^.]+\.[^.]+)$/.exec(host);
  return m ? m[1] : host;
}

function parseSetCookie(line: string): StoredCookie | null {
  const [pair, ...attrs] = line.split(';');
  const eq = pair.indexOf('=');
  if (eq <= 0) return null;

  const name = pair.slice(0, eq).trim();
  const value = pair.slice(eq + 1).trim();
  if (!name) return null;

  let expiresAt: number | null = null;
  for (const attr of attrs) {
    const [k, v] = attr.split('=');
    const key = k.trim().toLowerCase();
    if (key === 'max-age') {
      const secs = Number(v);
      if (Number.isFinite(secs)) expiresAt = Date.now() + secs * 1000;
    } else if (key === 'expires' && expiresAt === null) {
      const t = Date.parse((v ?? '').trim());
      if (Number.isFinite(t)) expiresAt = t;
    }
  }
  return { name, value, expiresAt };
}

async function readJar(db: SQLiteDatabase): Promise<Jar> {
  return kvRepo.getJson<Jar>(db, KEY, {});
}

/** 応答の Set-Cookie を取り込む。 */
export async function store(
  db: SQLiteDatabase,
  host: string,
  setCookieLines: string[]
): Promise<void> {
  if (setCookieLines.length === 0) return;
  const jar = await readJar(db);
  const key = jarKey(host);
  const existing = jar[key] ?? [];
  const byName = new Map(existing.map((c) => [c.name, c]));

  for (const line of setCookieLines) {
    const cookie = parseSetCookie(line);
    if (!cookie) continue;
    // 値が空か既に期限切れなら削除指示とみなす
    if (!cookie.value || (cookie.expiresAt !== null && cookie.expiresAt <= Date.now())) {
      byName.delete(cookie.name);
    } else {
      byName.set(cookie.name, cookie);
    }
  }

  jar[key] = [...byName.values()];
  await kvRepo.setJson(db, KEY, jar);
}

/** リクエストに載せる Cookie ヘッダ。無ければ null。 */
export async function header(db: SQLiteDatabase, host: string): Promise<string | null> {
  const jar = await readJar(db);
  const now = Date.now();
  const live = (jar[jarKey(host)] ?? []).filter((c) => c.expiresAt === null || c.expiresAt > now);
  if (live.length === 0) return null;
  return live.map((c) => `${c.name}=${c.value}`).join('; ');
}

/** 設定画面で中身を見せるため。 */
export async function list(db: SQLiteDatabase, host: string): Promise<StoredCookie[]> {
  const jar = await readJar(db);
  return jar[jarKey(host)] ?? [];
}

/** 「Cookie を削除」。書き込み確認ループの定番の対処。 */
export async function clear(db: SQLiteDatabase, host?: string): Promise<void> {
  if (!host) {
    await kvRepo.remove(db, KEY);
    return;
  }
  const jar = await readJar(db);
  delete jar[jarKey(host)];
  await kvRepo.setJson(db, KEY, jar);
}
