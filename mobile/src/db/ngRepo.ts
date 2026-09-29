import type { SQLiteDatabase } from 'expo-sqlite';

import type { NgHideMode, NgKind, NgRule } from './types';

/** NGID の既定の有効期限。ID は日替わりなので永久 NG は溜まるだけで意味が薄い。 */
export const NG_ID_TTL_MS = 24 * 60 * 60 * 1000;

export interface NewNgRule {
  kind: NgKind;
  pattern: string;
  scopeBoard?: string | null;
  isRegex?: boolean;
  hideMode?: NgHideMode;
  chain?: boolean;
  expiresAt?: number | null;
}

export async function add(db: SQLiteDatabase, rule: NewNgRule): Promise<void> {
  await db.runAsync(
    `INSERT INTO ng_rule (kind, pattern, scope_board, is_regex, hide_mode, chain, expires_at, created_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
    [
      rule.kind,
      rule.pattern,
      rule.scopeBoard ?? null,
      rule.isRegex ? 1 : 0,
      rule.hideMode ?? 'abone',
      rule.chain ? 1 : 0,
      rule.expiresAt ?? null,
      Date.now(),
    ]
  );
}

/** レス番タップのメニューから NGID を足すときの入口。既定で 24 時間で切れる。 */
export async function addNgId(
  db: SQLiteDatabase,
  uid: string,
  scopeBoard: string | null
): Promise<void> {
  await add(db, {
    kind: 'id',
    pattern: uid,
    scopeBoard,
    expiresAt: Date.now() + NG_ID_TTL_MS,
  });
}

export async function remove(db: SQLiteDatabase, id: number): Promise<void> {
  await db.runAsync('DELETE FROM ng_rule WHERE id = ?', [id]);
}

/** 期限切れを掃除する。アプリ起動時に 1 回呼ぶ。 */
export async function purgeExpired(db: SQLiteDatabase): Promise<number> {
  const res = await db.runAsync('DELETE FROM ng_rule WHERE expires_at IS NOT NULL AND expires_at < ?', [
    Date.now(),
  ]);
  return res.changes;
}

export async function listAll(db: SQLiteDatabase): Promise<NgRule[]> {
  return db.getAllAsync<NgRule>('SELECT * FROM ng_rule ORDER BY created_at DESC');
}

/**
 * ある板に効くルールだけを引く。全板共通 (scope_board IS NULL) と、
 * その板専用のものを合わせて返す。期限切れは除く。
 */
export async function listForBoard(
  db: SQLiteDatabase,
  host: string,
  board: string
): Promise<NgRule[]> {
  const scope = `${host}/${board}`;
  return db.getAllAsync<NgRule>(
    `SELECT * FROM ng_rule
     WHERE (scope_board IS NULL OR scope_board = ?)
       AND (expires_at IS NULL OR expires_at > ?)
     ORDER BY created_at DESC`,
    [scope, Date.now()]
  );
}
