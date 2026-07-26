import Encoding from 'encoding-japanese';

/**
 * Shift_JIS の入出力。このアプリの土台。
 *
 * Hermes には TextDecoder('shift_jis') が存在せず、既存の TextDecoder polyfill も
 * UTF-8 系しか扱えない。なので SJIS は encoding-japanese (純 JS・依存ゼロ) に一本化する。
 */

/** 5ch の dat / subject.txt は Shift_JIS(CP932)。バイト列を文字列にする。 */
export function decodeSjis(bytes: Uint8Array): string {
  return Encoding.convert(bytes, { to: 'UNICODE', from: 'SJIS', type: 'string' });
}

/**
 * 投稿用。文字列を Shift_JIS にしてからパーセントエンコードする。
 *
 * encodeURIComponent は UTF-8 になるので使えない。
 * fallback:'html-entity' は必須 — 絵文字や SJIS に無い漢字が `&#12345;` になる。
 * これがまさに 5ch 側が期待する形で、指定しないと文字化けするか例外になる。
 */
export function encodeSjisPercent(s: string): string {
  const sjis = Encoding.convert(Encoding.stringToCode(s), {
    to: 'SJIS',
    from: 'UNICODE',
    fallback: 'html-entity',
    type: 'array',
  });
  return Encoding.urlEncode(sjis);
}

/** application/x-www-form-urlencoded なボディを SJIS で組み立てる。 */
export function buildSjisForm(fields: Record<string, string>): string {
  return Object.entries(fields)
    .map(([k, v]) => `${encodeSjisPercent(k)}=${encodeSjisPercent(v)}`)
    .join('&');
}

const LF = 0x0a;

/**
 * 完全な行（末尾が LF）までを切り出す。差分取得の要。
 *
 * Shift_JIS の 2 バイト目は 0x40-0x7E / 0x80-0xFC の範囲で、LF(0x0A) はそこに含まれない。
 * つまり LF が多バイト文字の一部になることは原理的に無い。
 * よって「LF の直後」で切っている限り、Range のバイト境界で文字が割れることは起こり得ない。
 *
 * 最後の LF より後ろの不完全な行は捨てる（バッファしない）。次回の Range で取り直す。
 * これにより保存するカーソルが常に行境界になる。
 */
export function splitCompleteLines(bytes: Uint8Array): { body: Uint8Array; consumed: number } {
  for (let i = bytes.length - 1; i >= 0; i--) {
    if (bytes[i] === LF) {
      return { body: bytes.subarray(0, i + 1), consumed: i + 1 };
    }
  }
  return { body: bytes.subarray(0, 0), consumed: 0 };
}

/** 完全な行だけをデコードして行配列にする。末尾の空要素は落とす。 */
export function decodeCompleteLines(bytes: Uint8Array): { lines: string[]; consumed: number } {
  const { body, consumed } = splitCompleteLines(bytes);
  if (consumed === 0) return { lines: [], consumed: 0 };
  const text = decodeSjis(body);
  const lines = text.split('\n');
  // 末尾は必ず LF なので最後は空文字列になる。それだけを落とす。
  if (lines.length > 0 && lines[lines.length - 1] === '') lines.pop();
  return { lines, consumed };
}
