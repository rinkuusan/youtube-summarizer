import type { SQLiteDatabase } from 'expo-sqlite';

import type { Board } from '../api/bbsmenu';
import type { BoardRow } from './types';

/**
 * 板一覧のキャッシュ。bbsmenu.json は 180KB あるので毎回取りに行かない。
 * 履歴やお気に入りの一覧で板名を出すためにも使う (thread テーブルと JOIN する)。
 */

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

export async function saveAll(db: SQLiteDatabase, boards: Board[]): Promise<void> {
  const now = Date.now();
  await db.withTransactionAsync(async () => {
    for (const b of boards) {
      await db.runAsync(
        `INSERT INTO board (host, id, name, category, category_order, updated_at)
         VALUES (?, ?, ?, ?, ?, ?)
         ON CONFLICT (host, id) DO UPDATE SET
           name = excluded.name, category = excluded.category,
           category_order = excluded.category_order, updated_at = excluded.updated_at`,
        [b.host, b.id, b.name, b.category, b.categoryOrder, now]
      );
    }
  });
}

export async function loadAll(db: SQLiteDatabase): Promise<Board[]> {
  const rows = await db.getAllAsync<BoardRow>('SELECT * FROM board');
  return rows.map((r) => ({
    host: r.host,
    id: r.id,
    name: r.name,
    category: r.category ?? 'その他',
    categoryOrder: r.category_order ?? 0,
  }));
}

/** キャッシュが新しいか。古ければ呼び出し側が取り直す。 */
export async function isFresh(db: SQLiteDatabase): Promise<boolean> {
  const row = await db.getFirstAsync<{ newest: number | null }>(
    'SELECT MAX(updated_at) AS newest FROM board'
  );
  if (!row?.newest) return false;
  return Date.now() - row.newest < CACHE_TTL_MS;
}

/** SETTING.TXT 由来の情報を足す (デフォルト名無し・最大文字数)。 */
export async function saveSettings(
  db: SQLiteDatabase,
  host: string,
  id: string,
  settings: { noname?: string | null; maxMessage?: number | null }
): Promise<void> {
  await db.runAsync(
    `UPDATE board SET noname = COALESCE(?, noname), max_message = COALESCE(?, max_message)
     WHERE host = ? AND id = ?`,
    [settings.noname ?? null, settings.maxMessage ?? null, host, id]
  );
}

export async function get(
  db: SQLiteDatabase,
  host: string,
  id: string
): Promise<BoardRow | null> {
  return db.getFirstAsync<BoardRow>('SELECT * FROM board WHERE host = ? AND id = ?', [host, id]);
}

// --- お気に入り板 ---

export async function listFavoriteBoards(db: SQLiteDatabase): Promise<Board[]> {
  const rows = await db.getAllAsync<BoardRow>(
    `SELECT b.* FROM fav_board f
     JOIN board b ON b.host = f.host AND b.id = f.board
     ORDER BY f.sort_order ASC`
  );
  return rows.map((r) => ({
    host: r.host,
    id: r.id,
    name: r.name,
    category: r.category ?? 'その他',
    categoryOrder: r.category_order ?? 0,
  }));
}

export async function isFavoriteBoard(
  db: SQLiteDatabase,
  host: string,
  board: string
): Promise<boolean> {
  const row = await db.getFirstAsync<{ n: number }>(
    'SELECT COUNT(*) AS n FROM fav_board WHERE host = ? AND board = ?',
    [host, board]
  );
  return (row?.n ?? 0) > 0;
}

export async function setFavoriteBoard(
  db: SQLiteDatabase,
  host: string,
  board: string,
  favorite: boolean
): Promise<void> {
  if (favorite) {
    await db.runAsync(
      `INSERT INTO fav_board (host, board, sort_order)
       VALUES (?, ?, (SELECT COALESCE(MAX(sort_order), 0) + 1 FROM fav_board))
       ON CONFLICT (host, board) DO NOTHING`,
      [host, board]
    );
  } else {
    await db.runAsync('DELETE FROM fav_board WHERE host = ? AND board = ?', [host, board]);
  }
}
