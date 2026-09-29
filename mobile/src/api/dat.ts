import { Ch5Error, errorFromStatus } from '../net/errors';
import { fetchBytes, getHeader } from '../net/http';
import { decodeCompleteLines } from '../net/sjis';

/**
 * dat の取得。専ブラ定番の差分取得 (Range) を行う。
 *
 * 既に N バイト持っているとき、1 バイト手前 (N-1) から要求する。
 * 返ってきた先頭 1 バイトが既知の LF と一致すれば、その続きが素直な追記分。
 * 一致しなければ、あぼーん (レス削除) などで dat が書き換わっている
 * -> レス番号がずれるので全体を取り直す。
 *
 * If-Range も併用して、サーバ側でも検証させる。
 */

const LF = 0x0a;

export interface DatCursor {
  /** 保持している dat のバイト長。Range の起点。 */
  bytes: number;
  etag: string | null;
  lastModified: string | null;
  /** 保持している行数 = 最後のレス番号。 */
  lineCount: number;
}

export type DatFetchKind =
  /** 全体を取得した (初回、またはあぼーん検出による取り直し)。 */
  | 'full'
  /** 差分を追記した。 */
  | 'append'
  /** 変化なし。 */
  | 'unchanged';

export interface DatFetchResult {
  kind: DatFetchKind;
  /** kind='full' なら全行、'append' なら追加分の行のみ、'unchanged' なら空。 */
  lines: string[];
  /** append の場合、追加分の先頭レス番号。 */
  startRes: number;
  cursor: DatCursor;
  /** サーバ上の dat 全体のバイト数 (Content-Range から。取れなければ null)。 */
  totalBytes: number | null;
  /** あぼーん等で全体取り直しになった場合 true。 */
  refetched: boolean;
}

export function datUrl(host: string, board: string, key: string): string {
  return `https://${host}/${board}/dat/${key}.dat`;
}

export const emptyCursor: DatCursor = { bytes: 0, etag: null, lastModified: null, lineCount: 0 };

/** `bytes 100-200/33928` から全体サイズを取り出す。 */
function parseTotalBytes(contentRange: string | null): number | null {
  if (!contentRange) return null;
  const m = /\/(\d+)\s*$/.exec(contentRange);
  return m ? Number(m[1]) : null;
}

function cursorFrom(headers: [string, string][], bytes: number, lineCount: number): DatCursor {
  return {
    bytes,
    etag: getHeader(headers, 'etag'),
    lastModified: getHeader(headers, 'last-modified'),
    lineCount,
  };
}

/**
 * 200 なのに中身が空、を弾く。
 *
 * 5ch は不安定な瞬間に、実在するスレへ 200 + 0 バイトを返すことがある
 * (実測 2026-07-29: 同じ dat が 5 分前は 93579B、次は 0B で 8.2 秒かかった)。
 * これを「スレが空になった」と解釈すると、取得済みのキャッシュを消して
 * 0 レス表示にしてしまう。異常として扱い、呼び出し側にキャッシュを保たせる。
 */
function assertNotEmpty(bytes: Uint8Array, url: string): void {
  if (bytes.length > 0) return;
  throw new Ch5Error('server', '5ch が空の応答を返しました。時間をおいて再取得してください。', {
    status: 200,
    url,
  });
}

async function fetchFull(url: string, signal?: AbortSignal): Promise<DatFetchResult> {
  const res = await fetchBytes(url, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, url);
  assertNotEmpty(res.bytes, url);

  const { lines, consumed } = decodeCompleteLines(res.bytes);
  return {
    kind: 'full',
    lines,
    startRes: 1,
    cursor: cursorFrom(res.headers, consumed, lines.length),
    totalBytes: res.bytes.length,
    refetched: false,
  };
}

/**
 * dat を取得する。cursor を渡すと差分取得を試みる。
 */
export async function fetchDat(
  host: string,
  board: string,
  key: string,
  cursor: DatCursor = emptyCursor,
  signal?: AbortSignal
): Promise<DatFetchResult> {
  const url = datUrl(host, board, key);

  if (cursor.bytes <= 0) {
    return fetchFull(url, signal);
  }

  const headers: Record<string, string> = {
    // 1 バイト手前から要求する。先頭バイトが既知の LF かどうかで改変を検出する。
    Range: `bytes=${cursor.bytes - 1}-`,
  };
  // 検証子があればサーバ側にも判定させる。不一致なら 200 で全体が返ってくる。
  if (cursor.etag) headers['If-Range'] = cursor.etag;
  else if (cursor.lastModified) headers['If-Range'] = cursor.lastModified;

  const res = await fetchBytes(url, { headers, signal });

  // dat が縮んだ (カーソルがファイル末尾を越えた)
  if (res.status === 416) {
    return { ...(await fetchFull(url, signal)), refetched: true };
  }
  // If-Range 不一致。サーバが全体を返してきた。
  if (res.status === 200) {
    return { ...(await parseFullFrom(res, url)), refetched: true };
  }
  if (res.status === 304) {
    return {
      kind: 'unchanged',
      lines: [],
      startRes: cursor.lineCount + 1,
      cursor,
      totalBytes: null,
      refetched: false,
    };
  }
  if (res.status !== 206) {
    throw errorFromStatus(res.status, url);
  }

  // 206。先頭バイトが既知の LF でなければ、dat が書き換わっている。
  if (res.bytes.length === 0 || res.bytes[0] !== LF) {
    return { ...(await fetchFull(url, signal)), refetched: true };
  }

  // 先頭の 1 バイト (既知の LF) を捨てた残りが追記分。
  const fresh = res.bytes.subarray(1);
  const { lines, consumed } = decodeCompleteLines(fresh);

  return {
    kind: lines.length > 0 ? 'append' : 'unchanged',
    lines,
    startRes: cursor.lineCount + 1,
    cursor: {
      bytes: cursor.bytes + consumed,
      // Range 応答の検証子で更新する。無ければ元のを維持。
      etag: getHeader(res.headers, 'etag') ?? cursor.etag,
      lastModified: getHeader(res.headers, 'last-modified') ?? cursor.lastModified,
      lineCount: cursor.lineCount + lines.length,
    },
    totalBytes: parseTotalBytes(getHeader(res.headers, 'content-range')),
    refetched: false,
  };
}

/** 既に受け取った 200 応答から全体パース結果を作る (再取得を避ける)。 */
async function parseFullFrom(
  res: { bytes: Uint8Array; headers: [string, string][] },
  url: string
): Promise<DatFetchResult> {
  // ここは「あぼーん等で全体を取り直した」経路。空を通すとキャッシュを消してしまう。
  assertNotEmpty(res.bytes, url);
  const { lines, consumed } = decodeCompleteLines(res.bytes);
  return {
    kind: 'full',
    lines,
    startRes: 1,
    cursor: cursorFrom(res.headers, consumed, lines.length),
    totalBytes: res.bytes.length,
    refetched: false,
  };
}
