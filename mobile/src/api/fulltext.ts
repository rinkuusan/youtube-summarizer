import { Ch5Error } from '../net/errors';
import { log, logError } from '../net/log';
import { parseBody, segmentsToPlainText } from '../parse/body';
import { parseDat, type Post } from '../parse/datLine';
import { normalizeForSearch } from '../utils/normalize';
import { fetchArchivedThread } from './archived';
import { fetchDat } from './dat';
import { computeMomentum } from './momentum';
import { searchThreads } from './search';
import { fetchThreadList } from './subject';

/**
 * レス本文の全文検索。
 *
 * 5ch には本文を横断検索する公開 API が無い (公式の find.5ch はスレタイのみ。
 * logsoku はドメイン失効、mimizun はアクセス拒否、kakolog.jp は停止を実測で確認)。
 * そこで「スレを絞り込む → dat を落として端末側で本文を突き合わせる」で成立させる。
 *
 * - thread: 既に開いているスレの中だけ。通信ゼロ。
 * - board:  板の subject.txt から勢い上位 N 本を落として検索。
 * - all:    find.5ch のスレタイ検索で当たったスレを落として検索 = 実質 5ch 横断。
 *
 * dat を並列に落とすので、5ch に迷惑をかけない範囲 (同時 4 本) に絞る。
 */

export type SearchScope = 'thread' | 'board' | 'all';

/** 同時に落とす dat の本数。上げると規制を食らいやすくなる。 */
const CONCURRENCY = 4;

export interface SearchTarget {
  host: string;
  board: string;
  key: string;
  title: string;
}

export interface PostHit extends SearchTarget {
  post: Post;
  /** 一致箇所の周辺を切り出した表示用テキスト。 */
  snippet: string;
}

/** スレ単位にまとめた検索結果。同じ語が何度も出るスレを何行も出さないため。 */
export interface ThreadHit extends SearchTarget {
  /** そのスレ内で一致したレス数。 */
  count: number;
  /** 代表として見せる最初の一致レス。 */
  post: Post;
  snippet: string;
}

/**
 * レス単位の一致をスレ単位にまとめる。
 *
 * 実況板のように同じ語が連呼されるスレだと、レスの数だけ同じスレが並んでしまう。
 * 一覧では「どのスレに有るか」が知りたいので、スレ 1 行に畳んで件数を添える。
 * 並び順は最初に見つかった順を保つ (勢い順に走査しているため意味がある)。
 */
export function groupByThread(hits: PostHit[]): ThreadHit[] {
  const byThread = new Map<string, ThreadHit>();
  for (const h of hits) {
    const id = `${h.host}/${h.board}/${h.key}`;
    const found = byThread.get(id);
    if (found) {
      found.count++;
      // 代表は最も若いレス番号にする (スレの主題に近いことが多い)。
      if (h.post.res < found.post.res) {
        found.post = h.post;
        found.snippet = h.snippet;
      }
    } else {
      byThread.set(id, {
        host: h.host,
        board: h.board,
        key: h.key,
        title: h.title,
        count: 1,
        post: h.post,
        snippet: h.snippet,
      });
    }
  }
  return [...byThread.values()];
}

export interface SearchProgress {
  /** 走査し終えたスレ数。 */
  done: number;
  /** 走査対象のスレ総数。 */
  total: number;
  /** ここまでに見つかったレス数。 */
  hits: number;
  /** 取得に失敗したスレ数 (dat 落ち等)。 */
  failed: number;
}

/** 本文の検索用テキスト。タグとアンカーを剥がしてから正規化する。 */
function searchableText(post: Post): string {
  return normalizeForSearch(segmentsToPlainText(parseBody(post.body)));
}

/** 一致箇所の前後を切り出す。無ければ先頭から。 */
function makeSnippet(post: Post, normalizedQuery: string, span = 40): string {
  const plain = segmentsToPlainText(parseBody(post.body)).replace(/\s+/g, ' ').trim();
  const at = normalizeForSearch(plain).indexOf(normalizedQuery);
  if (at < 0) return plain.slice(0, span * 2);

  const from = Math.max(0, at - span);
  const to = Math.min(plain.length, at + normalizedQuery.length + span);
  return `${from > 0 ? '…' : ''}${plain.slice(from, to)}${to < plain.length ? '…' : ''}`;
}

/** 既に手元にあるレス配列から検索する (scope='thread')。通信しない。 */
export function searchLoadedPosts(posts: Post[], query: string, target: SearchTarget): PostHit[] {
  const q = normalizeForSearch(query.trim());
  if (!q) return [];

  return posts
    .filter((p) => !p.isAbone && searchableText(p).includes(q))
    .map((post) => ({ ...target, post, snippet: makeSnippet(post, q) }));
}

/** 板の中から、勢い上位 limit 本のスレを検索対象にする。 */
export async function collectBoardTargets(
  host: string,
  board: string,
  limit: number,
  signal?: AbortSignal
): Promise<SearchTarget[]> {
  const threads = await fetchThreadList(host, board, signal);
  const now = Date.now();
  return [...threads]
    .sort((a, b) => computeMomentum(b.key, b.resCount, now) - computeMomentum(a.key, a.resCount, now))
    .slice(0, limit)
    .map((t) => ({ host, board, key: t.key, title: t.title }));
}

/**
 * スレタイ検索の結果を検索対象にする = 5ch 横断。
 *
 * ここでは絞り込む前の all を使う。タイトルに語が無くても本文にはある、が普通だからで、
 * find.5ch が OR で拾ってきた分も含めて総当たりする方が本文検索としては当たる。
 * ただしタイトル一致したスレの方が本命なので、先に走査されるよう前に出す。
 */
export async function collectAllTargets(
  query: string,
  limit: number,
  signal?: AbortSignal
): Promise<SearchTarget[]> {
  const { all, matched } = await searchThreads(query, signal);

  const seen = new Set<string>();
  const ordered: SearchTarget[] = [];
  for (const h of [...matched, ...all]) {
    const id = `${h.host}/${h.board}/${h.key}`;
    if (seen.has(id)) continue;
    seen.add(id);
    ordered.push({ host: h.host, board: h.board, key: h.key, title: h.title });
  }

  log('info', 'fulltext', `5ch 横断の走査対象: ${Math.min(ordered.length, limit)} スレ (タイトル一致 ${matched.length} 件を優先)`);
  return ordered.slice(0, limit);
}

/**
 * 対象スレの dat を落として本文を検索する。
 *
 * 1 スレの失敗 (dat 落ち・規制) では全体を止めない。落ちた数は progress.failed に出す。
 */
export async function searchTargets(
  targets: SearchTarget[],
  query: string,
  onProgress: (p: SearchProgress) => void,
  signal?: AbortSignal
): Promise<PostHit[]> {
  const q = normalizeForSearch(query.trim());
  if (!q || targets.length === 0) return [];

  const results: PostHit[] = [];
  const progress: SearchProgress = { done: 0, total: targets.length, hits: 0, failed: 0 };
  let next = 0;

  log('info', 'fulltext', `本文検索 "${query}" 開始: ${targets.length} スレ (同時${CONCURRENCY})`);

  async function worker() {
    for (;;) {
      if (signal?.aborted) return;
      const i = next++;
      if (i >= targets.length) return;
      const t = targets[i];

      try {
        let posts: Post[];
        try {
          const r = await fetchDat(t.host, t.board, t.key, undefined, signal);
          posts = parseDat(r.lines).posts;
        } catch (e) {
          // スレタイ検索は板から落ちたスレも返してくる。実測ではその方が多数派なので、
          // dat 404 で諦めると本文検索がほとんど空振りする。read.cgi から拾い直す。
          if (!(e instanceof Ch5Error && e.kind === 'notFound')) throw e;
          posts = (await fetchArchivedThread(t.host, t.board, t.key, signal)).posts;
        }
        const found = searchLoadedPosts(posts, query, t);
        results.push(...found);
        progress.hits += found.length;
      } catch (e) {
        // 個別スレの失敗は握りつぶさずログに残しつつ、走査は続ける。
        progress.failed++;
        logError('fulltext', e, `スキップ ${t.host}/${t.board}/${t.key}`);
      } finally {
        progress.done++;
        onProgress({ ...progress });
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, targets.length) }, worker));

  // レス番号順に並べると読みやすい。スレ内 -> レス番の順。
  results.sort((a, b) => (a.key === b.key ? a.post.res - b.post.res : Number(a.key) - Number(b.key)));

  log(
    'info',
    'fulltext',
    `本文検索 "${query}" 完了: ${results.length} レス / ${progress.done} スレ走査 (失敗 ${progress.failed})`
  );

  return results;
}
