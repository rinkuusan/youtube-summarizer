import type { SQLiteDatabase } from 'expo-sqlite';

/**
 * PRAGMA user_version によるマイグレーションの梯子。
 * SQLiteProvider の onInit から呼ぶ。
 *
 * 新しいバージョンを足すときは MIGRATIONS に追記するだけ。既存の要素は書き換えない
 * (既にそのバージョンまで上がった端末には二度と適用されないため)。
 */

type Migration = (db: SQLiteDatabase) => Promise<void>;

const MIGRATIONS: Migration[] = [
  // v1: 初版
  async (db) => {
    await db.execAsync(`
      CREATE TABLE thread (
        host TEXT NOT NULL,
        board TEXT NOT NULL,
        key TEXT NOT NULL,
        title TEXT,
        res_count INTEGER,
        cached_count INTEGER NOT NULL DEFAULT 0,
        dat_bytes INTEGER NOT NULL DEFAULT 0,
        etag TEXT,
        last_modified TEXT,
        read_count INTEGER NOT NULL DEFAULT 0,
        scroll_res INTEGER NOT NULL DEFAULT 0,
        favorite INTEGER NOT NULL DEFAULT 0,
        fav_order INTEGER,
        last_opened_at INTEGER,
        last_posted_at INTEGER,
        my_post_count INTEGER NOT NULL DEFAULT 0,
        last_fetched_at INTEGER,
        dat_dead INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (host, board, key)
      );
      CREATE INDEX idx_thread_opened ON thread(last_opened_at DESC);
      CREATE INDEX idx_thread_posted ON thread(last_posted_at DESC);
      CREATE INDEX idx_thread_fav ON thread(favorite, fav_order);

      CREATE TABLE post (
        host TEXT NOT NULL,
        board TEXT NOT NULL,
        key TEXT NOT NULL,
        res INTEGER NOT NULL,
        name TEXT NOT NULL DEFAULT '',
        mail TEXT NOT NULL DEFAULT '',
        date_str TEXT NOT NULL DEFAULT '',
        ts INTEGER,
        uid TEXT,
        wacchoi TEXT,
        trip TEXT,
        is_cap INTEGER NOT NULL DEFAULT 0,
        is_abone INTEGER NOT NULL DEFAULT 0,
        body TEXT NOT NULL DEFAULT '',
        is_mine INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (host, board, key, res)
      );
      CREATE INDEX idx_post_uid ON post(host, board, key, uid);

      CREATE TABLE board (
        host TEXT NOT NULL,
        id TEXT NOT NULL,
        name TEXT NOT NULL,
        category TEXT,
        category_order INTEGER,
        noname TEXT,
        max_message INTEGER,
        updated_at INTEGER,
        PRIMARY KEY (host, id)
      );

      CREATE TABLE fav_board (
        host TEXT NOT NULL,
        board TEXT NOT NULL,
        sort_order INTEGER,
        PRIMARY KEY (host, board)
      );

      CREATE TABLE ng_rule (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,
        pattern TEXT NOT NULL,
        scope_board TEXT,
        is_regex INTEGER NOT NULL DEFAULT 0,
        hide_mode TEXT NOT NULL DEFAULT 'abone',
        chain INTEGER NOT NULL DEFAULT 0,
        expires_at INTEGER,
        created_at INTEGER NOT NULL
      );

      CREATE TABLE kv (k TEXT PRIMARY KEY NOT NULL, v TEXT NOT NULL);
    `);
  },
];

export const LATEST_VERSION = MIGRATIONS.length;

export async function migrate(db: SQLiteDatabase): Promise<void> {
  // WAL は書き込み中の読み取りを妨げないので、スクロール中の保存が引っかからない。
  await db.execAsync('PRAGMA journal_mode = WAL;');

  const row = await db.getFirstAsync<{ user_version: number }>('PRAGMA user_version');
  let version = row?.user_version ?? 0;

  if (version >= LATEST_VERSION) return;

  for (let v = version; v < MIGRATIONS.length; v++) {
    await MIGRATIONS[v](db);
    version = v + 1;
    // PRAGMA はプレースホルダを受け付けないので直接埋め込む。値は自前の整数なので安全。
    await db.execAsync(`PRAGMA user_version = ${version}`);
  }
}
