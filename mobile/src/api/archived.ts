import { errorFromStatus } from '../net/errors';
import { fetchBytes } from '../net/http';
import { log } from '../net/log';
import { decodeSjis } from '../net/sjis';
import type { ParsedThread } from '../parse/datLine';
import { parseReadCgi } from '../parse/readCgi';

/**
 * dat 落ちしたスレを read.cgi から拾う。
 *
 * 板から落ちたスレは `/<board>/dat/<key>.dat` が 404 になるが、read.cgi は
 * しばらく中身を返し続ける。過去ログ倉庫 (`/kako/…`) の方は実測した限り
 * どのパスでも 404 だったので、read.cgi が現実的な唯一の経路。
 *
 * これはレス数の少ないスレが数時間で板から落ちる ニュー速VIP のような板で特に効く。
 * find.5ch のスレタイ検索はそういうスレも拾ってくるため、これが無いと
 * 検索結果の大半が「dat 落ち」で開けない。
 */

/** 範囲指定なしだと 5ch が省略表示にすることがあるので、明示的に全件を要求する。 */
export function readCgiUrl(host: string, board: string, key: string): string {
  return `https://${host}/test/read.cgi/${board}/${key}/1-1000`;
}

export async function fetchArchivedThread(
  host: string,
  board: string,
  key: string,
  signal?: AbortSignal
): Promise<ParsedThread> {
  const url = readCgiUrl(host, board, key);
  const res = await fetchBytes(url, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, url);

  const parsed = parseReadCgi(decodeSjis(res.bytes));
  log(
    parsed.posts.length > 0 ? 'info' : 'warn',
    'archived',
    `過去ログ ${key}: ${parsed.posts.length} レス (${res.bytes.length}B)`,
    parsed.posts.length === 0 ? `read.cgi の構造が変わった可能性がある url=${url}` : undefined
  );
  return parsed;
}
