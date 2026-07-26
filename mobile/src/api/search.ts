import { Ch5Error, errorFromStatus } from '../net/errors';
import { fetchBytes } from '../net/http';
import { parseFindHtml, type SearchHit } from '../parse/findHtml';

/** スレタイの横断検索。find.5ch.net は UTF-8 の HTML を返す (JSON API は無い)。 */

export function searchUrl(query: string): string {
  return `https://find.5ch.net/search?q=${encodeURIComponent(query)}`;
}

export async function searchThreads(query: string, signal?: AbortSignal): Promise<SearchHit[]> {
  const url = searchUrl(query);
  const res = await fetchBytes(url, { signal });
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

  return hits;
}

export type { SearchHit };
