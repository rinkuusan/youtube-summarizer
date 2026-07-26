// RN 標準の fetch は .arrayBuffer() が未実装で例外を投げる (facebook/react-native#34402)。
// dat は Shift_JIS のバイト列で受け取る必要があるため、expo/fetch を明示的に import する。
// ネイティブでは expo/fetch がグローバルの fetch を置き換えるが、
// EXPO_PUBLIC_USE_RN_FETCH=1 で壊れた実装に戻りうるのでグローバルには依存しない。
import { fetch } from 'expo/fetch';

import { Ch5Error, errorFromThrown } from './errors';

/** 読み取りは専ブラの慣例に従って名乗る。5ch 側でも受理されることを実測で確認済み。 */
export const READ_UA = 'Monazilla/1.00 (GochViewer/0.1)';

const DEFAULT_TIMEOUT_MS = 20_000;

export interface BytesResponse {
  status: number;
  bytes: Uint8Array;
  /** 生のヘッダ対。標準 Headers と違い重複ヘッダ (Set-Cookie) が失われない。 */
  headers: [string, string][];
  /** リダイレクト後の最終 URL。5ch.net -> 5ch.io の追従先を知るのに使う。 */
  url: string;
  redirected: boolean;
}

/** ヘッダを大文字小文字を無視して引く。 */
export function getHeader(headers: [string, string][], name: string): string | null {
  const lower = name.toLowerCase();
  for (const [k, v] of headers) {
    if (k.toLowerCase() === lower) return v;
  }
  return null;
}

/** 同名ヘッダを全部集める (Set-Cookie 用)。 */
export function getHeaders(headers: [string, string][], name: string): string[] {
  const lower = name.toLowerCase();
  return headers.filter(([k]) => k.toLowerCase() === lower).map(([, v]) => v);
}

export interface FetchBytesOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  timeoutMs?: number;
  signal?: AbortSignal;
  /** 既定は 'omit'。ネイティブの Cookie ストアに黙って触られないようにする。 */
  credentials?: RequestCredentials;
}

/**
 * バイト列を返す fetch。5ch へのアクセスは全部ここを通す。
 *
 * ステータスの判定は呼び出し側に任せる（304 や 416 を「正常」として扱う経路があるため）。
 * ネットワーク層の失敗のみ Ch5Error にして throw する。
 */
export async function fetchBytes(url: string, opts: FetchBytesOptions = {}): Promise<BytesResponse> {
  const controller = new AbortController();
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  // 呼び出し側の signal と自前のタイムアウトを両方効かせる。
  const onExternalAbort = () => controller.abort();
  opts.signal?.addEventListener('abort', onExternalAbort);

  try {
    const res = await fetch(url, {
      method: opts.method ?? 'GET',
      headers: { 'User-Agent': READ_UA, ...opts.headers },
      body: opts.body,
      redirect: 'follow',
      credentials: opts.credentials ?? 'omit',
      signal: controller.signal,
    });

    // 304 / 416 は本文が無い。bytes() は空配列を返す。
    const bytes = await res.bytes();

    return {
      status: res.status,
      bytes,
      headers: res._rawHeaders,
      url: res.url,
      redirected: res.redirected,
    };
  } catch (e) {
    throw errorFromThrown(e, url);
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener('abort', onExternalAbort);
  }
}

/** URL からホスト名を取り出す。bbsmenu の url から板サーバを割り出すのに使う。 */
export function hostOf(url: string): string {
  const m = /^https?:\/\/([^/]+)/i.exec(url);
  if (!m) throw new Ch5Error('parse', `URL を解釈できません: ${url}`);
  return m[1];
}
