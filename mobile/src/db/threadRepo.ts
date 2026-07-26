import type { SQLiteDatabase } from 'expo-sqlite';

import type { DatCursor } from '../api/dat';
import type { ThreadListItem, ThreadRef, ThreadRow } from './types';

/**
 * スレッドの状態 (既読・お気に入り・履歴) の読み書き。UI から SQL を直接触らせない。
 *
 * 履歴は別テーブルにしていない。スレは 1 つの実体で、閲覧履歴も書き込み履歴も
 * その属性にすぎないため、2 本のタイムスタンプ列で表す:
 *   last_opened_at ... 開くたびに更新 (閲覧履歴)
 *   last_posted_at ... 投稿成功時に更新 (書き込み履歴)
 */

/** 閲覧履歴として保持する最大件数。これを超えた分は古い順に捨てる。 */
export const OPENED_HISTORY_LIMIT = 500;

const SELECT_LIST_ITEM = `
  SELECT t.host, t.board, t.key,
         COALESCE(t.title, '') AS title,
         b.name AS boardName,
         COALESCE(t.res_count, t.cached_count) AS resCount,
         t.read_count AS readCount,
         t.my_post_count AS myPostCount,
         t.last_opened_at AS lastOpenedAt,
         t.last_posted_at AS lastPostedAt,
         t.favorite AS favorite
  FROM thread t
  LEFT JOIN board b ON b.host = t.host AND b.id = t.board
`;

interface RawListItem {
  host: string;
  board: string;
  key: string;
  title: string;
  boardName: string | null;
  resCount: number | null;
  readCount: number;
  myPostCount: number;
  lastOpenedAt: number | null;
  lastPostedAt: number | null;
  favorite: number;
}

function toListItem(r: RawListItem): ThreadListItem {
  const resCount = r.resCount ?? 0;
  return {
    host: r.host,
    board: r.board,
    key: r.key,
    title: r.title,
    boardName: r.boardName,
    resCount,
    readCount: r.readCount,
    unread: Math.max(0, resCount - r.readCount),
    myPostCount: r.myPostCount,
    lastOpenedAt: r.lastOpenedAt,
    lastPostedAt: r.lastPostedAt,
    favorite: r.favorite === 1,
  };
}

/** 行が無ければ作る。既にあれば何もしない。 */
async function ensureRow(db: SQLiteDatabase, ref: ThreadRef, title?: string | null): Promise<void> {
  await db.runAsync(
    `INSERT INTO thread (host, board, key, title) VALUES (?, ?, ?, ?)
     ON CONFLICT (host, board, key) DO UPDATE SET
       title = COALESCE(excluded.title, thread.title)`,
    [ref.host, ref.board, ref.key, title ?? null]
  );
}

export async function get(db: SQLiteDatabase, ref: ThreadRef): Promise<ThreadRow | null> {
  return db.getFirstAsync<ThreadRow>(
    'SELECT * FROM thread WHERE host = ? AND board = ? AND key = ?',
    [ref.host, ref.board, ref.key]
  );
}

/**
 * スレを開いたことを記録する。閲覧履歴はこれだけで自動的に積まれる。
 * スレビューの mount から呼ぶ。
 */
export async function touchOpened(
  db: SQLiteDatabase,
  ref: ThreadRef,
  title?: string | null
): Promise<void> {
  await ensureRow(db, ref, title);
  await db.runAsync(
    'UPDATE thread SET last_opened_at = ? WHERE host = ? AND board = ? AND key = ?',
    [Date.now(), ref.host, ref.board, ref.key]
  );
}

/**
 * 書き込みに成功したことを記録する。書き込み履歴はこれだけで自動的に積まれる。
 * post.ts の成功経路から呼ぶ。
 */
export async function markPosted(db: SQLiteDatabase, ref: ThreadRef): Promise<void> {
  await ensureRow(db, ref);
  await db.runAsync(
    `UPDATE thread SET last_posted_at = ?, my_post_count = my_post_count + 1
     WHERE host = ? AND board = ? AND key = ?`,
    [Date.now(), ref.host, ref.board, ref.key]
  );
}

/** 差分取得のカーソルを保存する。 */
export async function saveCursor(
  db: SQLiteDatabase,
  ref: ThreadRef,
  cursor: DatCursor,
  opts: { resCount?: number | null; title?: string | null } = {}
): Promise<void> {
  await ensureRow(db, ref, opts.title);
  await db.runAsync(
    `UPDATE thread SET
       dat_bytes = ?, etag = ?, last_modified = ?, cached_count = ?,
       res_count = COALESCE(?, ?), last_fetched_at = ?
     WHERE host = ? AND board = ? AND key = ?`,
    [
      cursor.bytes,
      cursor.etag,
      cursor.lastModified,
      cursor.lineCount,
      opts.resCount ?? null,
      cursor.lineCount,
      Date.now(),
      ref.host,
      ref.board,
      ref.key,
    ]
  );
}

/** subject.txt 由来の総レス数だけを更新する (お気に入りの新着チェック用)。 */
export async function updateResCount(
  db: SQLiteDatabase,
  ref: ThreadRef,
  resCount: number
): Promise<void> {
  await db.runAsync(
    'UPDATE thread SET res_count = ? WHERE host = ? AND board = ? AND key = ?',
    [resCount, ref.host, ref.board, ref.key]
  );
}

/** 既読位置を保存する。スレビューから離れるときに呼ぶ。 */
export async function updateReadState(
  db: SQLiteDatabase,
  ref: ThreadRef,
  readCount: number,
  scrollRes: number
): Promise<void> {
  await db.runAsync(
    `UPDATE thread SET read_count = MAX(read_count, ?), scroll_res = ?
     WHERE host = ? AND board = ? AND key = ?`,
    [readCount, scrollRes, ref.host, ref.board, ref.key]
  );
}

/** dat 落ちを記録する。 */
export async function markDead(db: SQLiteDatabase, ref: ThreadRef): Promise<void> {
  await db.runAsync(
    'UPDATE thread SET dat_dead = 1 WHERE host = ? AND board = ? AND key = ?',
    [ref.host, ref.board, ref.key]
  );
}

export async function setFavorite(
  db: SQLiteDatabase,
  ref: ThreadRef,
  favorite: boolean,
  title?: string | null
): Promise<void> {
  await ensureRow(db, ref, title);
  await db.runAsync(
    `UPDATE thread SET favorite = ?, fav_order = CASE WHEN ? = 1
       THEN COALESCE(fav_order, (SELECT COALESCE(MAX(fav_order), 0) + 1 FROM thread))
       ELSE NULL END
     WHERE host = ? AND board = ? AND key = ?`,
    [favorite ? 1 : 0, favorite ? 1 : 0, ref.host, ref.board, ref.key]
  );
}

/** 閲覧履歴。開いた順 (新しい順)。 */
export async function listOpenedHistory(
  db: SQLiteDatabase,
  limit = 200
): Promise<ThreadListItem[]> {
  const rows = await db.getAllAsync<RawListItem>(
    `${SELECT_LIST_ITEM} WHERE t.last_opened_at IS NOT NULL
     ORDER BY t.last_opened_at DESC LIMIT ?`,
    [limit]
  );
  return rows.map(toListItem);
}

/** 書き込み履歴。書いた順 (新しい順)。 */
export async function listPostedHistory(
  db: SQLiteDatabase,
  limit = 200
): Promise<ThreadListItem[]> {
  const rows = await db.getAllAsync<RawListItem>(
    `${SELECT_LIST_ITEM} WHERE t.last_posted_at IS NOT NULL
     ORDER BY t.last_posted_at DESC LIMIT ?`,
    [limit]
  );
  return rows.map(toListItem);
}

export async function listFavorites(db: SQLiteDatabase): Promise<ThreadListItem[]> {
  const rows = await db.getAllAsync<RawListItem>(
    `${SELECT_LIST_ITEM} WHERE t.favorite = 1 ORDER BY t.fav_order ASC`
  );
  return rows.map(toListItem);
}

/**
 * 閲覧履歴を刈り込む。
 *
 * 書き込んだスレとお気に入りは対象外 — 自分が書いたスレは価値が高く、
 * 件数もたかが知れているので無期限に残す。
 */
export async function pruneHistory(
  db: SQLiteDatabase,
  limit = OPENED_HISTORY_LIMIT
): Promise<number> {
  const victims = await db.getAllAsync<{ host: string; board: string; key: string }>(
    `SELECT host, board, key FROM thread
     WHERE favorite = 0 AND last_posted_at IS NULL AND last_opened_at IS NOT NULL
     ORDER BY last_opened_at DESC
     LIMIT -1 OFFSET ?`,
    [limit]
  );
  if (victims.length === 0) return 0;

  await db.withTransactionAsync(async () => {
    for (const v of victims) {
      await db.runAsync('DELETE FROM post WHERE host = ? AND board = ? AND key = ?', [
        v.host,
        v.board,
        v.key,
      ]);
      await db.runAsync('DELETE FROM thread WHERE host = ? AND board = ? AND key = ?', [
        v.host,
        v.board,
        v.key,
      ]);
    }
  });
  return victims.length;
}

/** 履歴から 1 件消す (スワイプ削除用)。 */
export async function removeFromHistory(
  db: SQLiteDatabase,
  ref: ThreadRef,
  kind: 'opened' | 'posted'
): Promise<void> {
  const column = kind === 'opened' ? 'last_opened_at' : 'last_posted_at';
  await db.runAsync(
    `UPDATE thread SET ${column} = NULL WHERE host = ? AND board = ? AND key = ?`,
    [ref.host, ref.board, ref.key]
  );
}
