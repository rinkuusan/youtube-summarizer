import { errorFromStatus } from '../net/errors';
import { fetchBytes } from '../net/http';
import { decodeSjis } from '../net/sjis';
import { decodeEntities } from '../parse/entities';

/**
 * スレ一覧 (subject.txt)。Shift_JIS。
 * 1 行 = `<datkey>.dat<>タイトル (レス数)`
 */

export interface ThreadSummary {
  /** dat キー。スレ作成時刻の UNIX 秒でもある。 */
  key: string;
  title: string;
  resCount: number;
}

// タイトル自体に括弧が入りうるので、末尾の括弧をレス数として取る。
// 貪欲な .* が自然に「最後の括弧」に一致してくれる。
const LINE = /^(\d+)\.dat<>(.*)\s+\((\d+)\)\s*$/;
// 稀に `,` 区切りの古い形式があるので保険。
const LINE_LEGACY = /^(\d+)\.dat,(.*)\s+\((\d+)\)\s*$/;

export function parseSubject(lines: string[]): ThreadSummary[] {
  const out: ThreadSummary[] = [];
  for (const line of lines) {
    const m = LINE.exec(line) ?? LINE_LEGACY.exec(line);
    if (!m) continue;
    out.push({
      key: m[1],
      title: decodeEntities(m[2]).trim(),
      resCount: Number(m[3]),
    });
  }
  return out;
}

export function subjectUrl(host: string, board: string): string {
  return `https://${host}/${board}/subject.txt`;
}

export async function fetchThreadList(
  host: string,
  board: string,
  signal?: AbortSignal
): Promise<ThreadSummary[]> {
  const url = subjectUrl(host, board);
  const res = await fetchBytes(url, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, url);
  // subject.txt は差分取得しないので全体をデコードする。
  // 末尾に LF が無い場合に最終行を落とさないため、行境界の切り捨てはしない。
  const lines = decodeSjis(res.bytes).split('\n');
  return parseSubject(lines);
}
