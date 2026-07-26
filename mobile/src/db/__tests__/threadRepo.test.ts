/**
 * @jest-environment node
 *
 * マイグレーションと履歴まわりを、実物の SQLite に対して検証する。
 *
 * expo-sqlite はネイティブモジュールなので jest では動かないが、Node 22 の
 * node:sqlite で同じ SQL を実行できる。DDL の構文誤りや、履歴クエリの取り違えは
 * これで実機に載せる前に捕まえられる。
 */
import { DatabaseSync } from 'node:sqlite';

import { LATEST_VERSION, migrate } from '../migrations';
import * as threadRepo from '../threadRepo';
import type { ThreadRef } from '../types';

type Params = unknown[];

/** expo-sqlite の非同期 API を node:sqlite の同期 API に被せる。 */
function createDb() {
  const db = new DatabaseSync(':memory:');
  const bind = (params?: Params) => (params ?? []) as never[];

  return {
    raw: db,
    async execAsync(sql: string) {
      db.exec(sql);
    },
    async runAsync(sql: string, params?: Params) {
      const r = db.prepare(sql).run(...bind(params));
      return { changes: Number(r.changes), lastInsertRowId: Number(r.lastInsertRowid) };
    },
    async getFirstAsync(sql: string, params?: Params) {
      return db.prepare(sql).get(...bind(params)) ?? null;
    },
    async getAllAsync(sql: string, params?: Params) {
      return db.prepare(sql).all(...bind(params));
    },
    async withTransactionAsync(task: () => Promise<void>) {
      db.exec('BEGIN');
      try {
        await task();
        db.exec('COMMIT');
      } catch (e) {
        db.exec('ROLLBACK');
        throw e;
      }
    },
  };
}

// 型は expo-sqlite の SQLiteDatabase を要求するが、テストでは上のアダプタを渡す。
type AnyDb = Parameters<typeof threadRepo.touchOpened>[0];
const asDb = (d: ReturnType<typeof createDb>) => d as unknown as AnyDb;

const refA: ThreadRef = { host: 'mi.5ch.io', board: 'news4vip', key: '100' };
const refB: ThreadRef = { host: 'mi.5ch.io', board: 'news4vip', key: '200' };

async function setup() {
  const raw = createDb();
  const db = asDb(raw);
  await migrate(db);
  return { raw, db };
}

describe('migrations', () => {
  it('空の DB を最新まで上げる', async () => {
    const { raw } = await setup();
    const row = raw.raw.prepare('PRAGMA user_version').get() as { user_version: number };
    expect(row.user_version).toBe(LATEST_VERSION);
  });

  it('必要なテーブルが揃う', async () => {
    const { raw } = await setup();
    const names = (
      raw.raw.prepare("SELECT name FROM sqlite_master WHERE type='table'").all() as {
        name: string;
      }[]
    ).map((r) => r.name);
    expect(names).toEqual(
      expect.arrayContaining(['thread', 'post', 'board', 'fav_board', 'ng_rule', 'kv'])
    );
  });

  it('二度流しても壊れない', async () => {
    const { raw, db } = await setup();
    await expect(migrate(db)).resolves.toBeUndefined();
    const row = raw.raw.prepare('PRAGMA user_version').get() as { user_version: number };
    expect(row.user_version).toBe(LATEST_VERSION);
  });
});

describe('履歴', () => {
  it('スレを開くと閲覧履歴に自動で載る', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'テストスレ');

    const opened = await threadRepo.listOpenedHistory(db);
    expect(opened).toHaveLength(1);
    expect(opened[0]).toMatchObject({ key: '100', title: 'テストスレ' });
  });

  it('開いただけでは書き込み履歴に載らない', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'テストスレ');
    expect(await threadRepo.listPostedHistory(db)).toHaveLength(0);
  });

  it('書き込むと書き込み履歴に載り、自分のレス数が増える', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'テストスレ');
    await threadRepo.markPosted(db, refA);
    await threadRepo.markPosted(db, refA);

    const posted = await threadRepo.listPostedHistory(db);
    expect(posted).toHaveLength(1);
    expect(posted[0].myPostCount).toBe(2);
    // 書き込んだスレは閲覧履歴にも残る
    expect(await threadRepo.listOpenedHistory(db)).toHaveLength(1);
  });

  it('2 種類の履歴は独立している', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.touchOpened(db, refB, 'B');
    await threadRepo.markPosted(db, refB);

    expect((await threadRepo.listOpenedHistory(db)).map((t) => t.key).sort()).toEqual(['100', '200']);
    expect((await threadRepo.listPostedHistory(db)).map((t) => t.key)).toEqual(['200']);
  });

  it('閲覧履歴は新しく開いた順に並ぶ', async () => {
    const { db, raw } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.touchOpened(db, refB, 'B');
    // Date.now() の分解能では同時刻になりうるので、順序を明示的に作る
    raw.raw.prepare('UPDATE thread SET last_opened_at = ? WHERE key = ?').run(1000, '100');
    raw.raw.prepare('UPDATE thread SET last_opened_at = ? WHERE key = ?').run(2000, '200');

    expect((await threadRepo.listOpenedHistory(db)).map((t) => t.key)).toEqual(['200', '100']);
  });

  it('履歴から個別に消せる', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.markPosted(db, refA);

    await threadRepo.removeFromHistory(db, refA, 'opened');
    expect(await threadRepo.listOpenedHistory(db)).toHaveLength(0);
    // 書き込み履歴の方は残る
    expect(await threadRepo.listPostedHistory(db)).toHaveLength(1);
  });
});

describe('履歴の刈り込み', () => {
  it('上限を超えた古い閲覧履歴を捨てる', async () => {
    const { db, raw } = await setup();
    for (let i = 0; i < 5; i++) {
      const ref = { host: 'h', board: 'b', key: String(i) };
      await threadRepo.touchOpened(db, ref, `t${i}`);
      raw.raw.prepare('UPDATE thread SET last_opened_at = ? WHERE key = ?').run(1000 + i, String(i));
    }

    const removed = await threadRepo.pruneHistory(db, 3);
    expect(removed).toBe(2);
    expect((await threadRepo.listOpenedHistory(db)).map((t) => t.key)).toEqual(['4', '3', '2']);
  });

  it('お気に入りと書き込んだスレは刈らない', async () => {
    const { db, raw } = await setup();
    for (let i = 0; i < 5; i++) {
      const ref = { host: 'h', board: 'b', key: String(i) };
      await threadRepo.touchOpened(db, ref, `t${i}`);
      raw.raw.prepare('UPDATE thread SET last_opened_at = ? WHERE key = ?').run(1000 + i, String(i));
    }
    // 一番古い 0 をお気に入り、次に古い 1 を書き込み済みにする
    await threadRepo.setFavorite(db, { host: 'h', board: 'b', key: '0' }, true);
    await threadRepo.markPosted(db, { host: 'h', board: 'b', key: '1' });

    await threadRepo.pruneHistory(db, 2);
    const remaining = (await threadRepo.listOpenedHistory(db)).map((t) => t.key).sort();
    expect(remaining).toEqual(['0', '1', '3', '4']);
  });

  it('刈るときはレスのキャッシュも一緒に消す', async () => {
    const { db, raw } = await setup();
    const ref = { host: 'h', board: 'b', key: '1' };
    await threadRepo.touchOpened(db, ref, 't');
    raw.raw
      .prepare('INSERT INTO post (host, board, key, res, body) VALUES (?, ?, ?, ?, ?)')
      .run('h', 'b', '1', 1, 'body');

    await threadRepo.pruneHistory(db, 0);
    const left = raw.raw.prepare('SELECT COUNT(*) AS n FROM post').get() as { n: number };
    expect(left.n).toBe(0);
  });
});

describe('既読とお気に入り', () => {
  it('新着件数を総レス数と既読数から出す', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.updateResCount(db, refA, 100);
    await threadRepo.updateReadState(db, refA, 60, 60);

    const [item] = await threadRepo.listOpenedHistory(db);
    expect(item.resCount).toBe(100);
    expect(item.readCount).toBe(60);
    expect(item.unread).toBe(40);
  });

  it('既読数は後退しない', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.updateReadState(db, refA, 80, 80);
    // NG などで見えるレスが減っても、既読は巻き戻さない
    await threadRepo.updateReadState(db, refA, 40, 40);

    const [item] = await threadRepo.listOpenedHistory(db);
    expect(item.readCount).toBe(80);
  });

  it('新着が負にならない', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.updateResCount(db, refA, 10);
    await threadRepo.updateReadState(db, refA, 50, 50);

    expect((await threadRepo.listOpenedHistory(db))[0].unread).toBe(0);
  });

  it('お気に入りの登録と解除', async () => {
    const { db } = await setup();
    await threadRepo.setFavorite(db, refA, true, 'A');
    expect(await threadRepo.listFavorites(db)).toHaveLength(1);

    await threadRepo.setFavorite(db, refA, false);
    expect(await threadRepo.listFavorites(db)).toHaveLength(0);
  });

  it('同じスレを何度開いても行は増えない', async () => {
    const { db } = await setup();
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.touchOpened(db, refA, 'A');
    await threadRepo.touchOpened(db, refA, null);

    const opened = await threadRepo.listOpenedHistory(db);
    expect(opened).toHaveLength(1);
    // タイトルを null で呼んでも既存のタイトルを消さない
    expect(opened[0].title).toBe('A');
  });
});
