import { resolveUrl } from '../../net/http';

/**
 * 書き込み確認ページは action="../test/bbs.cgi?guid=ON" を返す。
 * この ?guid=ON を落とすと、何度承諾しても確認ページが返り続ける
 * (実測 2026-07-28 kizuna.5ch.io/gamefight)。
 */
describe('resolveUrl', () => {
  const base = 'https://kizuna.5ch.io/test/bbs.cgi';

  it('実物の相対 action をクエリごと解決する', () => {
    expect(resolveUrl(base, '../test/bbs.cgi?guid=ON')).toBe(
      'https://kizuna.5ch.io/test/bbs.cgi?guid=ON'
    );
  });

  it('同階層の相対も解決する', () => {
    expect(resolveUrl(base, 'bbs.cgi?guid=ON')).toBe('https://kizuna.5ch.io/test/bbs.cgi?guid=ON');
  });

  it('ルート相対', () => {
    expect(resolveUrl(base, '/test/bbs.cgi?guid=ON')).toBe(
      'https://kizuna.5ch.io/test/bbs.cgi?guid=ON'
    );
  });

  it('絶対 URL はそのまま', () => {
    expect(resolveUrl(base, 'https://other.5ch.io/test/bbs.cgi')).toBe(
      'https://other.5ch.io/test/bbs.cgi'
    );
  });

  it('プロトコル相対は https を補う', () => {
    expect(resolveUrl(base, '//other.5ch.io/x')).toBe('https://other.5ch.io/x');
  });

  it('.. がルートを超えても壊れない', () => {
    expect(resolveUrl(base, '../../../a.cgi')).toBe('https://kizuna.5ch.io/a.cgi');
  });
});
