import { errorFromStatus } from '../net/errors';
import { fetchBytes } from '../net/http';
import { decodeSjis } from '../net/sjis';

/** 板の設定 (SETTING.TXT)。Shift_JIS の `KEY=VALUE` が並ぶ。 */

export interface BoardSetting {
  title: string | null;
  /** デフォルトの名無し名。投稿フォームのプレースホルダに使う。 */
  noname: string | null;
  /** 本文の最大文字数。投稿フォームのカウンタに使う。 */
  maxMessage: number | null;
  maxLines: number | null;
}

export function parseSetting(text: string): BoardSetting {
  const map = new Map<string, string>();
  for (const line of text.split('\n')) {
    const eq = line.indexOf('=');
    if (eq <= 0) continue;
    map.set(line.slice(0, eq).trim(), line.slice(eq + 1).trim());
  }
  const num = (k: string): number | null => {
    const v = map.get(k);
    const n = v ? Number(v) : NaN;
    return Number.isFinite(n) ? n : null;
  };
  return {
    title: map.get('BBS_TITLE') ?? map.get('BBS_TITLE_ORIG') ?? null,
    noname: map.get('BBS_NONAME_NAME') ?? null,
    maxMessage: num('BBS_MESSAGE_COUNT'),
    maxLines: num('BBS_LINE_NUMBER'),
  };
}

export async function fetchSetting(
  host: string,
  board: string,
  signal?: AbortSignal
): Promise<BoardSetting> {
  const url = `https://${host}/${board}/SETTING.TXT`;
  const res = await fetchBytes(url, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, url);
  return parseSetting(decodeSjis(res.bytes));
}
