/**
 * dat の本文に含まれる HTML エンティティを戻す。
 *
 * 重要: デコードは <a> タグを抽出した「後」に行う。先にデコードすると
 * 本文中の `&lt;a&gt;` (ユーザーが書いた文字列) が本物のタグに化けてしまう。
 */

const NAMED: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  hearts: '♥',
  diams: '♦',
  clubs: '♣',
  spades: '♠',
  copy: '©',
  reg: '®',
  trade: '™',
  middot: '·',
  hellip: '…',
};

export function decodeEntities(s: string): string {
  return s.replace(/&(#x[0-9a-fA-F]+|#\d+|[a-zA-Z]+);/g, (raw, body: string) => {
    if (body[0] === '#') {
      const code = body[1] === 'x' || body[1] === 'X' ? parseInt(body.slice(2), 16) : parseInt(body.slice(1), 10);
      if (!Number.isFinite(code) || code < 0 || code > 0x10ffff) return raw;
      try {
        return String.fromCodePoint(code);
      } catch {
        return raw;
      }
    }
    const named = NAMED[body.toLowerCase()];
    return named ?? raw;
  });
}
