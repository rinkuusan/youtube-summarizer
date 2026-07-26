import type { SQLiteDatabase } from 'expo-sqlite';

/**
 * 小さな設定値の置き場。JSON を 1 列に入れるだけ。
 * AsyncStorage を足さずに済ませるためにこれで代用する。
 */

export async function getJson<T>(db: SQLiteDatabase, key: string, fallback: T): Promise<T> {
  const row = await db.getFirstAsync<{ v: string }>('SELECT v FROM kv WHERE k = ?', [key]);
  if (!row) return fallback;
  try {
    return JSON.parse(row.v) as T;
  } catch {
    // 壊れた値は無視して既定値に倒す。設定が壊れてアプリが起動しない事態を避ける。
    return fallback;
  }
}

export async function setJson(db: SQLiteDatabase, key: string, value: unknown): Promise<void> {
  await db.runAsync(
    'INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT (k) DO UPDATE SET v = excluded.v',
    [key, JSON.stringify(value)]
  );
}

export async function remove(db: SQLiteDatabase, key: string): Promise<void> {
  await db.runAsync('DELETE FROM kv WHERE k = ?', [key]);
}
