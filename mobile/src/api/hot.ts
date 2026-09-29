import { log, logError } from '../net/log';
import { computeMomentum } from './momentum';
import { fetchThreadList } from './subject';

/**
 * 板をまたいだ「今伸びてるスレ」。
 *
 * 5ch のヘッドライン板 (headline.5ch.io/ikioig 等) は 1 リクエストで全板の
 * 勢い順が取れるが、レス数が全件 `(2)` に潰れていて実数が無い (実体スレへの
 * ポインタなので)。順位を自前で組めず、スレを開くにも 1 本ずつ実 URL を
 * 引き直す必要がある。
 *
 * そこで主要板の subject.txt を並列に取って、実データで順位を作る。
 * 1121 板を舐めるのは 5ch に対して論外だし、真の全板順位の上位はどのみち
 * ここに挙げた板が占める。お気に入り板と最近見た板も対象に混ぜる。
 */

export interface HotThread {
  host: string;
  board: string;
  boardName: string;
  key: string;
  title: string;
  resCount: number;
  momentum: number;
  /** 並び替えに使った合成スコア。 */
  score: number;
}

/** 同時に取る subject.txt の本数。 */
const CONCURRENCY = 6;

/**
 * 既定で見る板。5ch で常時人が多いところ。
 * ここに無い板もお気に入り・最近見た板として自動的に対象に入る。
 */
export const DEFAULT_HOT_BOARDS = [
  'news4vip',
  'newsplus',
  'livegalileo',
  'poverty',
  'livejupiter',
  'news',
  'mnewsplus',
  'liveanb',
  'ghard',
  'gamefight',
  'livebase',
  'muscle',
  'esite',
  'operate',
];

/**
 * 運営のお知らせ・広告スレを落とす。
 *
 * 実体はスレだが読む対象ではなく、勢いが常に高いので放っておくと上位を占める。
 * 判定は「★で始まる」「運営キャップ付き」「お知らせ/PR 系の語」。
 */
const NOISE = [
  /^[★☆]/,
  /お知らせ/,
  /お報せ/,
  /運営情報/,
  /^PR[:：\s]/i,
  /プレミアム・?サービス/,
  /規約|利用案内/,
  /【お知らせ】/,
];

export function isNoiseThread(title: string): boolean {
  return NOISE.some((re) => re.test(title.trim()));
}

/**
 * 勢いとレス数を混ぜたスコア。
 *
 * 勢いだけだと「立った直後に数レス付いただけのスレ」が上に来る。
 * レス数だけだと「一日かけて伸びた終わりかけのスレ」が上に来る。
 * 両方の対数を掛け合わせて、伸びていて中身もあるスレを上げる。
 */
export function hotScore(momentum: number, resCount: number): number {
  return Math.log1p(Math.max(momentum, 0)) * Math.log1p(Math.max(resCount, 0));
}

export interface HotProgress {
  done: number;
  total: number;
  failed: number;
}

export async function fetchHotThreads(
  boards: { host: string; board: string; name: string }[],
  limit: number,
  onProgress?: (p: HotProgress) => void,
  signal?: AbortSignal
): Promise<HotThread[]> {
  const out: HotThread[] = [];
  const progress: HotProgress = { done: 0, total: boards.length, failed: 0 };
  const now = Date.now();
  let next = 0;

  async function worker() {
    for (;;) {
      if (signal?.aborted) return;
      const i = next++;
      if (i >= boards.length) return;
      const b = boards[i];
      try {
        const threads = await fetchThreadList(b.host, b.board, signal);
        for (const t of threads) {
          if (isNoiseThread(t.title)) continue;
          const momentum = computeMomentum(t.key, t.resCount, now);
          out.push({
            host: b.host,
            board: b.board,
            boardName: b.name,
            key: t.key,
            title: t.title,
            resCount: t.resCount,
            momentum,
            score: hotScore(momentum, t.resCount),
          });
        }
      } catch (e) {
        progress.failed++;
        logError('hot', e, `スキップ ${b.host}/${b.board}`);
      } finally {
        progress.done++;
        onProgress?.({ ...progress });
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(CONCURRENCY, boards.length) }, worker));

  out.sort((a, b) => b.score - a.score);
  log(
    'info',
    'hot',
    `新着: ${boards.length} 板から ${out.length} スレ (失敗 ${progress.failed}) → 上位 ${Math.min(limit, out.length)} 件`
  );
  return out.slice(0, limit);
}
