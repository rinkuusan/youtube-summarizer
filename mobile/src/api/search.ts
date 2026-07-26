import { Ch5Error, errorFromStatus } from '../net/errors';
import { fetchBytes } from '../net/http';
import { log } from '../net/log';
import { parseFindHtml, type SearchHit } from '../parse/findHtml';
import { normalizeForSearch } from '../utils/normalize';

/** スレタイの横断検索。find.5ch.net は UTF-8 の HTML を返す (JSON API は無い)。 */

/** find.5ch が 1 クエリで返す件数。offset/p/o を付けても増えないので、これが上限。 */
export const SEARCH_RESULT_LIMIT = 100;

export function searchUrl(query: string): string {
  return `https://find.5ch.net/search?q=${encodeURIComponent(query)}`;
}

export interface ThreadSearchResult {
  /** find.5ch が返した全件。クエリを含まないスレを大量に含む (下記参照)。 */
  all: SearchHit[];
  /** タイトルに実際にクエリを含むものだけ。画面に出すのはこっち。 */
  matched: SearchHit[];
}

/**
 * スレタイを 5ch 全体から検索する。
 *
 * find.5ch はクエリを形態素で割って OR 検索するため、返ってくる 100 件には
 * クエリを含まないスレが大量に混ざる。実測 (2026-07-26):
 *
 *   地震     100 件中 100 件が一致
 *   ラーメン 100 件中 100 件が一致
 *   ウマ娘   100 件中  21 件が一致  ← 「ウマ」「娘」に割れる
 *   からあげ 100 件中   0 件が一致  ← 「から」「あげ」に割れて全滅
 *
 * 「メスガキ」なら「百田尚樹「某政党のガキが」」のような無関係が 79 件混ざる。
 * つまり find.5ch の返却をそのまま出すと、複合語の検索は使い物にならない。
 * 信用せず、こちら側で必ず突き合わせる。
 *
 * 一方で本文検索の候補としては、タイトルに含まなくても本文には含む可能性があるので
 * 絞り込む前の all も返す。
 */
export async function searchThreads(
  query: string,
  signal?: AbortSignal
): Promise<ThreadSearchResult> {
  const q = query.trim();
  if (!q) return { all: [], matched: [] };

  const url = searchUrl(q);
  const res = await fetchBytes(url, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, url);

  const html = new TextDecoder().decode(res.bytes);
  const all = parseFindHtml(html);

  // 0 件は「該当なし」かもしれないし「HTML が変わって解析できていない」かもしれない。
  // 検索結果ページらしさを見て切り分ける。
  if (all.length === 0 && !/list_line|該当|見つかりません|0件/.test(html)) {
    throw new Ch5Error(
      'parse',
      '検索結果を解析できませんでした。5ch 側の仕様が変わった可能性があります。',
      { url }
    );
  }

  const matched = filterByTitle(all, q);
  log(
    'info',
    'search',
    `スレタイ検索 "${q}": find.5ch ${all.length} 件 → タイトル一致 ${matched.length} 件`,
    all.length >= SEARCH_RESULT_LIMIT ? `find.5ch の返却上限に到達` : undefined
  );

  return { all, matched };
}

/** タイトルにクエリを実際に含むものだけ残す。カタカナ/ひらがな/全角半角のゆれは吸収する。 */
export function filterByTitle(hits: SearchHit[], query: string): SearchHit[] {
  const nq = normalizeForSearch(query.trim());
  if (!nq) return hits;
  return hits.filter((h) => normalizeForSearch(h.title).includes(nq));
}

export type { SearchHit };
