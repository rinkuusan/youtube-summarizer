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
  const terms = splitTerms(query);
  if (terms.length === 0) return { all: [], matched: [] };

  // find.5ch は 1 リクエスト 1 語なので、OR の語は個別に引いて束ねる。
  const seen = new Set<string>();
  const all: SearchHit[] = [];
  for (const term of terms) {
    const url = searchUrl(term);
    const res = await fetchBytes(url, { signal });
    // find.5ch は本体とは別に落ちることがある。実測 2026-07-29: トップは 200 を
    // 返すのに /search だけ全クエリで 502。5ch 本体は生きているので、
    // 「アプリが壊れた」「その単語が弾かれた」と誤解されない文言にする。
    if (res.status >= 500) {
      throw new Ch5Error(
        'server',
        `5ch のスレタイ検索 (find.5ch) が応答しません (${res.status})。\n` +
          '5ch 本体は別サーバなので、板やスレの閲覧はそのまま使えます。\n' +
          '検索語の問題ではないので、時間をおいて試してください。',
        { status: res.status, url }
      );
    }
    if (res.status !== 200) throw errorFromStatus(res.status, url);

    const html = new TextDecoder().decode(res.bytes);
    const hits = parseFindHtml(html);

    // 0 件は「該当なし」かもしれないし「HTML が変わって解析できていない」かもしれない。
    // 検索結果ページらしさを見て切り分ける。
    if (hits.length === 0 && !/list_line|該当|見つかりません|0件/.test(html)) {
      throw new Ch5Error(
        'parse',
        '検索結果を解析できませんでした。5ch 側の仕様が変わった可能性があります。',
        { url }
      );
    }

    for (const h of hits) {
      const id = `${h.host}/${h.board}/${h.key}`;
      if (seen.has(id)) continue;
      seen.add(id);
      all.push(h);
    }
  }

  const q = terms.join('|');
  const matched = filterByTitle(all, q);
  log(
    'info',
    'search',
    `スレタイ検索 "${q}": find.5ch ${all.length} 件 → タイトル一致 ${matched.length} 件`,
    all.length >= SEARCH_RESULT_LIMIT ? `find.5ch の返却上限に到達` : undefined
  );

  return { all, matched };
}

/**
 * クエリを OR の語に割る。区切りは `|`。
 *
 * 「X」のように語そのものが一般的すぎる対象を探すために要る。
 * X (旧 Twitter) は "X" だけだと何にでも当たり、"Twitter" だけだと
 * 今の書かれ方を取りこぼす。`X|Twitter|ツイッター|ツイート` と並べて拾う。
 */
export function splitTerms(query: string): string[] {
  return query
    .split('|')
    .map((t) => t.trim())
    .filter(Boolean);
}

/** 英数字。短い英字語の「単語として」の一致を見るのに使う。 */
const ALNUM = /[0-9a-z]/;

/**
 * 語がテキストに「単語として」含まれるか。
 *
 * 2 文字以下の英数字の語 (X, AI, PC など) は、素の部分一致だと
 * Xperia・MAX・PCR のような無関係に当たり続ける。前後が英数字でないときだけ
 * 一致とみなす。日本語の語には語境界の概念が薄いので、この判定は掛けない。
 */
export function containsTerm(haystack: string, term: string): boolean {
  const at = haystack.indexOf(term);
  if (at < 0) return false;
  if (term.length > 2 || !/^[0-9a-z]+$/.test(term)) return true;

  // 出現位置を順に見て、前後が英数字でないものが 1 つでもあれば一致。
  for (let i = at; i >= 0; i = haystack.indexOf(term, i + 1)) {
    const before = i > 0 ? haystack[i - 1] : '';
    const after = haystack[i + term.length] ?? '';
    if (!ALNUM.test(before) && !ALNUM.test(after)) return true;
  }
  return false;
}

/**
 * タイトルにクエリを実際に含むものだけ残す。
 * カタカナ/ひらがな/全角半角のゆれは吸収し、`|` 区切りは OR として扱う。
 */
export function filterByTitle(hits: SearchHit[], query: string): SearchHit[] {
  const terms = splitTerms(query).map(normalizeForSearch).filter(Boolean);
  if (terms.length === 0) return hits;
  return hits.filter((h) => {
    const title = normalizeForSearch(h.title);
    return terms.some((t) => containsTerm(title, t));
  });
}

export type { SearchHit };
