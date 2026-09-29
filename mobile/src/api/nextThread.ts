import { log } from '../net/log';
import { normalizeForSearch } from '../utils/normalize';
import { fetchThreadList, type ThreadSummary } from './subject';

/**
 * 次スレ候補。
 *
 * 5ch の続きスレは「同じ板に、同じ題名で、番号だけ増えた新しいスレが立つ」という
 * 慣習で回っている。題名から巻数の表記を剥がして突き合わせ、スレ作成時刻
 * (dat キー) が今のスレより新しいものを候補にする。
 *
 * 表記は板ごとにばらばらなので、判定は緩めに倒して候補を複数出す。
 * どれが本物かは人が見た方が早い。
 */

/** 巻数の表記。末尾に付く形をひととおり。 */
const VOLUME_PATTERNS: RegExp[] = [
  /\s*[★☆]\s*\d+\s*$/, //  ★3
  /\s*(?:part|pt)\.?\s*\d+\s*$/i, //  Part3
  /\s*(?:vol|no)\.?\s*\d+\s*$/i, //  vol.3
  /\s*[<＜(（【\[]\s*(?:第\s*)?\d+\s*(?:スレ|スレ目|部|章)?\s*[>＞)）】\]]\s*$/, //  【3】 (3スレ目)
  /\s*第?\s*\d+\s*(?:スレ目?|弾|章|部)\s*$/, //  3スレ目
  /\s*[#＃]\s*\d+\s*$/, //  #3
  /\s*\d+\s*$/, //  末尾の裸の数字 (最後の手段)
];

/** 巻末の番号や飾りを落とした「題名の芯」。 */
export function baseTitle(title: string): string {
  let s = title.trim();
  // ワッチョイ表記や [転載禁止] のような付帯は比較の邪魔になるので落とす。
  s = s.replace(/\s*\[[^\]]*\]\s*$/g, '').replace(/\s*[（(]ワッチョイ[^）)]*[）)]\s*/g, '');

  for (const re of VOLUME_PATTERNS) {
    const stripped = s.replace(re, '');
    // 何も剥がれなかったら次の書き方を試す。
    if (stripped === s) continue;
    // 題名が消し飛ぶほど剥がれたら行き過ぎ。剥がさずに打ち切る。
    if (stripped.trim().length < 4) break;
    s = stripped;
    break;
  }
  return s.trim();
}

export interface NextThreadCandidate extends ThreadSummary {
  host: string;
  board: string;
  /** 題名の一致度 (1 = 芯が完全一致)。 */
  similarity: number;
}

/** 芯どうしの一致度。前方一致の長さを短い方の長さで割る。 */
export function titleSimilarity(a: string, b: string): number {
  const x = normalizeForSearch(a);
  const y = normalizeForSearch(b);
  if (!x || !y) return 0;
  if (x === y) return 1;

  let i = 0;
  while (i < x.length && i < y.length && x[i] === y[i]) i++;
  return i / Math.min(x.length, y.length);
}

/**
 * 同じ板から次スレ候補を探す。
 *
 * 条件は「今のスレより後に立っている」+「題名の芯が十分似ている」。
 * 似ている順ではなく新しい順に返す。続きスレは普通いちばん新しいものだから。
 */
export async function findNextThreadCandidates(
  host: string,
  board: string,
  currentKey: string,
  currentTitle: string,
  signal?: AbortSignal,
  minSimilarity = 0.7
): Promise<NextThreadCandidate[]> {
  const threads = await fetchThreadList(host, board, signal);
  const base = baseTitle(currentTitle);
  const currentCreated = Number(currentKey);

  const out: NextThreadCandidate[] = [];
  for (const t of threads) {
    if (t.key === currentKey) continue;
    if (!(Number(t.key) > currentCreated)) continue;
    const similarity = titleSimilarity(base, baseTitle(t.title));
    if (similarity < minSimilarity) continue;
    out.push({ ...t, host, board, similarity });
  }

  out.sort((a, b) => Number(b.key) - Number(a.key));
  log('info', 'nextThread', `次スレ候補 "${base}": ${out.length} 件 / ${threads.length} スレ中`);
  return out;
}
