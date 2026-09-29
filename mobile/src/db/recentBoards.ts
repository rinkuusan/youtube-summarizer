import type { SQLiteDatabase } from 'expo-sqlite';

import * as kvRepo from './kvRepo';

/**
 * 最近見た板。
 *
 * board テーブルに列を足さず kv に置いているのは、これが「順序付きの短い
 * リスト」でしかなく、関係クエリの対象にならないため。マイグレーションも要らない。
 */

const KEY = 'board.recent';
const MAX = 12;

export interface RecentBoard {
  host: string;
  board: string;
  name: string;
  at: number;
}

export async function list(db: SQLiteDatabase): Promise<RecentBoard[]> {
  return kvRepo.getJson<RecentBoard[]>(db, KEY, []);
}

/** 板を開いたときに呼ぶ。同じ板は先頭に繰り上げる。 */
export async function touch(
  db: SQLiteDatabase,
  entry: { host: string; board: string; name: string }
): Promise<void> {
  const current = await list(db);
  const rest = current.filter((r) => !(r.host === entry.host && r.board === entry.board));
  await kvRepo.setJson(db, KEY, [{ ...entry, at: Date.now() }, ...rest].slice(0, MAX));
}

export async function clear(db: SQLiteDatabase): Promise<void> {
  await kvRepo.remove(db, KEY);
}
