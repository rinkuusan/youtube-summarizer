import { readFileSync } from 'fs';
import { join } from 'path';

import { parseFindHtml } from '../findHtml';

/** find.5ch.net から実際に取得した検索結果 HTML を固定してある。 */
function loadFixture(): string {
  return readFileSync(join(__dirname, '..', '__fixtures__', 'find.html'), 'utf8');
}

describe('findHtml', () => {
  const hits = parseFindHtml(loadFixture());

  it('検索結果を全件取り出す', () => {
    expect(hits).toHaveLength(3);
  });

  it('href から host / 板 / datkey を分解する', () => {
    expect(hits[0]).toMatchObject({
      host: 'kizuna.5ch.io',
      board: 'morningcoffee',
      key: '1785029996',
    });
  });

  it('タイトル末尾の括弧をレス数として切り出す', () => {
    expect(hits[0].resCount).toBe(156);
    expect(hits[0].title).not.toMatch(/\(\d+\)$/);
    expect(hits[0].title).toContain('モーニング娘');
  });

  it('板の表示名を取る', () => {
    expect(hits[0].boardName).toBe('モ娘（狼）');
  });

  it('サーバ側が計算した勢いを取る', () => {
    expect(hits[0].momentumText).toMatch(/\/日$/);
  });

  it('タイトルからタグが除かれている', () => {
    for (const h of hits) {
      expect(h.title).not.toContain('<');
      expect(h.title).not.toContain('&amp;');
    }
  });

  it('HTML が想定と違っても例外を投げずに空を返す', () => {
    expect(parseFindHtml('<html><body>まったく別の中身</body></html>')).toEqual([]);
    expect(parseFindHtml('')).toEqual([]);
  });

  it('壊れたブロックは読み飛ばす', () => {
    const broken = '<div class="list_line"><a class="list_line_link" href="/bogus"></a></div></div></div>';
    expect(parseFindHtml(broken)).toEqual([]);
  });
});
