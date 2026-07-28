import { readFileSync } from 'fs';
import { join } from 'path';

import { parseReadCgi, toDatNameField } from '../readCgi';

/**
 * 実際の read.cgi の応答から先頭 3 レスを切り出したもの。
 * dat が 404 になったスレ (mi.5ch.io/news4vip/1785186106) から取得した。
 */
function load(): string {
  return readFileSync(join(__dirname, '..', '__fixtures__', 'readcgi.html'), 'utf8');
}

describe('parseReadCgi', () => {
  const { title, posts } = parseReadCgi(load());

  it('スレタイを取る', () => {
    expect(title).toContain('クリトリス');
  });

  it('レスを漏れなく取り、番号が連番になる', () => {
    expect(posts).toHaveLength(3);
    expect(posts.map((p) => p.res)).toEqual([1, 2, 3]);
  });

  it('名前・ID・日付を dat 由来と同じ形で埋める', () => {
    const p = posts[0];
    expect(p.name).toBe('以下、5ちゃんねるからVIPがお送りします');
    // dat 由来と同じく ID: の接頭辞は外した形で入る (NG の ID 照合がこれ前提)。
    expect(p.uid).toBe('6LPNKnMU0');
    expect(p.dateText).toMatch(/^\d{4}\/\d{2}\/\d{2}/);
    // 日付が epoch に変換できていること (履歴や勢いの計算で使う)
    expect(p.timestamp).toBeGreaterThan(0);
  });

  it('本文を <br> ごと残す (描画時に body.ts がトークン化する)', () => {
    expect(posts[0].body).toContain('<br>');
    expect(posts[0].body.length).toBeGreaterThan(0);
  });

  it('レスが 1 件も無い HTML でも落ちない', () => {
    expect(parseReadCgi('<html><body>なにもない</body></html>')).toEqual({
      title: null,
      posts: [],
    });
  });
});

describe('toDatNameField', () => {
  // read.cgi は <b>名前</b>(付加情報)、dat は 名前 </b>(付加情報)<b> と書く。
  // NG やワッチョイ抽出は dat 側の書き方を前提にしているので、そちらに寄せる。
  it('名前だけなら <b> を落とすだけ', () => {
    expect(toDatNameField('<b>名無しさん</b>')).toBe('名無しさん');
  });

  it('ワッチョイ付きは dat の並びに直す', () => {
    expect(toDatNameField('<b>名無し</b> (ﾜｯﾁｮｲW a587-tyyI)')).toBe(
      '名無し</b>(ﾜｯﾁｮｲW a587-tyyI)<b>'
    );
  });

  it('<b> が無い形でも壊れない', () => {
    expect(toDatNameField('名無し')).toBe('名無し');
  });
});

describe('文字参照をスレタイで戻す', () => {
  // 実機で「そこの君&#129781;やめとけ」と生のまま出ていた。
  // subject.txt 側はデコード済みなので、揃えないと同じスレのタイトルが食い違う。
  it('h1 の数値文字参照が実体に戻る', () => {
    const html = '<html><body><h1>そこの君&#129781;やめとけ</h1></body></html>';
    const { title } = parseReadCgi(html);
    expect(title).toBe(`そこの君${String.fromCodePoint(129781)}やめとけ`);
    expect(title).not.toContain('&#');
  });
});
