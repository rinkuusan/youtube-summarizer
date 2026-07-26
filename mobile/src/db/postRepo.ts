import type { SQLiteDatabase } from 'expo-sqlite';

import type { Post } from '../parse/datLine';
import type { PostRow, ThreadRef } from './types';

/**
 * レスのキャッシュ。これがあることで、スレを開いた瞬間に前回の内容が出て、
 * 圏外でも取得済みのレスが読める。
 *
 * body は dat の生の文字列のまま入れる。トークン列 (parseBody の結果) を保存すると
 * パーサを直すたびに DB の作り直しが要るため。
 */

function toPost(r: PostRow): Post {
  return {
    res: r.res,
    name: r.name,
    wacchoi: r.wacchoi,
    trip: r.trip,
    isCap: r.is_cap === 1,
    mail: r.mail,
    dateText: r.date_str,
    timestamp: r.ts,
    uid: r.uid,
    be: null,
    body: r.body,
    isAbone: r.is_abone === 1,
  };
}

export async function load(db: SQLiteDatabase, ref: ThreadRef): Promise<Post[]> {
  const rows = await db.getAllAsync<PostRow>(
    'SELECT * FROM post WHERE host = ? AND board = ? AND key = ? ORDER BY res ASC',
    [ref.host, ref.board, ref.key]
  );
  return rows.map(toPost);
}

/** 追記または全置換。1 トランザクションでまとめて書く。 */
export async function save(db: SQLiteDatabase, ref: ThreadRef, posts: Post[]): Promise<void> {
  if (posts.length === 0) return;
  await db.withTransactionAsync(async () => {
    for (const p of posts) {
      await db.runAsync(
        `INSERT INTO post
           (host, board, key, res, name, mail, date_str, ts, uid, wacchoi, trip,
            is_cap, is_abone, body)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (host, board, key, res) DO UPDATE SET
           name = excluded.name, mail = excluded.mail, date_str = excluded.date_str,
           ts = excluded.ts, uid = excluded.uid, wacchoi = excluded.wacchoi,
           trip = excluded.trip, is_cap = excluded.is_cap,
           is_abone = excluded.is_abone, body = excluded.body`,
        [
          ref.host,
          ref.board,
          ref.key,
          p.res,
          p.name,
          p.mail,
          p.dateText,
          p.timestamp,
          p.uid,
          p.wacchoi,
          p.trip,
          p.isCap ? 1 : 0,
          p.isAbone ? 1 : 0,
          p.body,
        ]
      );
    }
  });
}

/** あぼーんで dat が書き換わったときは、番号がずれるので一度全部捨てる。 */
export async function clear(db: SQLiteDatabase, ref: ThreadRef): Promise<void> {
  await db.runAsync('DELETE FROM post WHERE host = ? AND board = ? AND key = ?', [
    ref.host,
    ref.board,
    ref.key,
  ]);
}

/** 自分が書いたレスに印を付ける。 */
export async function markMine(
  db: SQLiteDatabase,
  ref: ThreadRef,
  res: number
): Promise<void> {
  await db.runAsync(
    'UPDATE post SET is_mine = 1 WHERE host = ? AND board = ? AND key = ? AND res = ?',
    [ref.host, ref.board, ref.key, res]
  );
}

/** 同じ ID の投稿を数える (ID チップの「この ID の投稿 n 件」用)。 */
export async function countByUid(
  db: SQLiteDatabase,
  ref: ThreadRef,
  uid: string
): Promise<number> {
  const row = await db.getFirstAsync<{ n: number }>(
    'SELECT COUNT(*) AS n FROM post WHERE host = ? AND board = ? AND key = ? AND uid = ?',
    [ref.host, ref.board, ref.key, uid]
  );
  return row?.n ?? 0;
}
