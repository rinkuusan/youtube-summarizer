import { readFileSync } from 'fs';
import { join } from 'path';

import { parseSubject } from '../../api/subject';
import { decodeCompleteLines, decodeSjis, encodeSjisPercent, splitCompleteLines } from '../../net/sjis';
import { anchorTargets, parseBody } from '../body';
import { parseDat, parseDatLine } from '../datLine';
import { decodeEntities } from '../entities';

/** 実際の 5ch から取得した dat (Shift_JIS) をそのまま固定してある。 */
function loadFixture(): Uint8Array {
  const b64 = readFileSync(join(__dirname, '..', '__fixtures__', 'sample.dat.b64'), 'utf8');
  return new Uint8Array(Buffer.from(b64, 'base64'));
}

describe('sjis', () => {
  it('Shift_JIS のバイト列を復号できる', () => {
    const text = decodeSjis(loadFixture());
    expect(text).toContain('以下、5ちゃんねるからVIPがお送りします');
    expect(text).toContain('VIPでウマ娘');
  });

  it('splitCompleteLines は最後の LF までを返し、不完全な行を捨てる', () => {
    const bytes = new Uint8Array([0x61, 0x0a, 0x62, 0x0a, 0x63]);
    const { body, consumed } = splitCompleteLines(bytes);
    expect(consumed).toBe(4);
    expect(Array.from(body)).toEqual([0x61, 0x0a, 0x62, 0x0a]);
  });

  it('LF が無ければ何も消費しない', () => {
    const { body, consumed } = splitCompleteLines(new Uint8Array([0x61, 0x62]));
    expect(consumed).toBe(0);
    expect(body.length).toBe(0);
  });

  it('多バイト文字の途中で切らない (LF は SJIS の 2 バイト目に現れない)', () => {
    // 「あい」= 0x82A0 0x82A2。LF を挟んで途中で切っても文字が壊れないことを確認する。
    const full = loadFixture();
    const { body } = splitCompleteLines(full.subarray(0, 300));
    expect(() => decodeSjis(body)).not.toThrow();
    // 復号結果に置換文字が出ない = 境界で割れていない
    expect(decodeSjis(body)).not.toContain('�');
  });

  it('decodeCompleteLines は末尾の空行を落とす', () => {
    const { lines } = decodeCompleteLines(loadFixture());
    expect(lines.length).toBe(14);
    expect(lines[lines.length - 1]).not.toBe('');
  });

  it('投稿用に Shift_JIS でパーセントエンコードする', () => {
    // 「書き込む」の SJIS パーセントエンコード
    expect(encodeSjisPercent('書き込む')).toBe('%8F%91%82%AB%8D%9E%82%DE');
    expect(encodeSjisPercent('sage')).toBe('sage');
  });

  it('SJIS に無い文字は数値文字参照にフォールバックする', () => {
    // 絵文字は SJIS に無いので &#...; になる (5ch が期待する形)
    const encoded = encodeSjisPercent('🍣');
    expect(decodeURIComponent(encoded)).toMatch(/&#\d+;/);
  });
});

describe('entities', () => {
  it('名前付き・数値の実体参照を戻す', () => {
    expect(decodeEntities('a&amp;b&lt;c&gt;d')).toBe('a&b<c>d');
    expect(decodeEntities('&#12354;')).toBe('あ');
    expect(decodeEntities('&#x3042;')).toBe('あ');
  });

  it('未知の実体はそのまま残す', () => {
    expect(decodeEntities('&unknownentity;')).toBe('&unknownentity;');
  });
});

describe('datLine', () => {
  const lines = decodeCompleteLines(loadFixture()).lines;

  it('1 レス目からスレタイを取る', () => {
    const { title, posts } = parseDat(lines);
    expect(title).toBe('VIPでウマ娘');
    expect(posts.length).toBe(lines.length);
  });

  it('レス番号は行インデックス + 1', () => {
    const { posts } = parseDat(lines);
    expect(posts[0].res).toBe(1);
    expect(posts[5].res).toBe(6);
  });

  it('ワッチョイを名前から切り離す', () => {
    const post = parseDatLine(lines[1], 2)!;
    expect(post.wacchoi).toBe('ﾜｯﾁｮｲW 3787-tyyI');
    expect(post.name).not.toContain('</b>');
    expect(post.name).not.toContain('<b>');
    expect(post.name).not.toContain('ﾜｯﾁｮｲ');
  });

  it('ID と日時を切り出す', () => {
    const post = parseDatLine(lines[0], 1)!;
    expect(post.uid).toBe('mMwNdz2O0');
    // 2026/07/25 19:09:14.630 JST = 10:09:14.630 UTC
    expect(new Date(post.timestamp!).toISOString()).toBe('2026-07-25T10:09:14.630Z');
  });

  it('ID の無い行でも null で通る', () => {
    const post = parseDatLine('名無し<><>2026/07/25(土) 19:09:14.630<>本文<>', 1)!;
    expect(post.uid).toBeNull();
    expect(post.name).toBe('名無し');
  });

  it('あぼーん行を検出する', () => {
    const post = parseDatLine('あぼーん<>あぼーん<>あぼーん<>あぼーん<>', 5)!;
    expect(post.isAbone).toBe(true);
  });

  it('フィールドが足りない行は null を返す', () => {
    expect(parseDatLine('壊れた行', 1)).toBeNull();
    expect(parseDatLine('', 1)).toBeNull();
  });
});

describe('body', () => {
  const lines = decodeCompleteLines(loadFixture()).lines;

  it('<br> を改行にする', () => {
    const post = parseDatLine(lines[0], 1)!;
    const text = parseBody(post.body)
      .map((s) => s.text)
      .join('');
    expect(text).toContain('\n');
    expect(text).not.toContain('<br>');
  });

  it('dat 内で HTML 化済みのアンカーをタップ可能にする', () => {
    const line = lines.find((l) => l.includes('read.cgi'))!;
    const post = parseDatLine(line, 10)!;
    const segs = parseBody(post.body);
    const anchors = segs.filter((s) => s.type === 'anchor');
    expect(anchors.length).toBeGreaterThan(0);
    expect(anchors[0]).toMatchObject({ type: 'anchor', from: 4, to: 4 });
  });

  it('タグ化されていない継続リスト (>>4,8-10,13) も拾う', () => {
    const line = lines.find((l) => l.includes('read.cgi') && l.includes(',8-10'))!;
    const post = parseDatLine(line, 10)!;
    const segs = parseBody(post.body);
    const anchors = segs.filter((s) => s.type === 'anchor') as {
      type: 'anchor';
      from: number;
      to: number;
    }[];
    // >>4 に続く 8-10, 13, 16, 21, 26 が個別のアンカーになっている
    expect(anchors.map((a) => a.from)).toEqual(expect.arrayContaining([4, 8, 13, 16, 21, 26]));
    expect(anchors.find((a) => a.from === 8)?.to).toBe(10);
  });

  it('生の >>N もアンカーになる', () => {
    const segs = parseBody(' &gt;&gt;123 テスト ');
    expect(segs.filter((s) => s.type === 'anchor')).toEqual([
      { type: 'anchor', from: 123, to: 123, text: '>>123' },
    ]);
  });

  it('本文中の &lt;a&gt; を偽のタグとして解釈しない', () => {
    // ユーザーが書いた「<a href=...>click</a>」という文字列がタグとして食われず、
    // 文字どおり残ることを確認する (エンティティ復号をタグ抽出より後にしている理由)。
    // 中の素の URL がリンクになるのは意図した挙動なので、そこは咎めない。
    const segs = parseBody(' &lt;a href="http://example.com"&gt;click&lt;/a&gt; ');
    const text = segs.map((s) => s.text).join('');
    expect(text).toBe('<a href="http://example.com">click</a>');
    expect(segs.some((s) => s.type === 'anchor')).toBe(false);
  });

  it('画像 URL と通常の URL を区別する', () => {
    const segs = parseBody(' https://i.imgur.com/RswR874.jpeg と https://example.com/page ');
    expect(segs.find((s) => s.type === 'image')).toMatchObject({
      url: 'https://i.imgur.com/RswR874.jpeg',
    });
    expect(segs.find((s) => s.type === 'link')).toMatchObject({ url: 'https://example.com/page' });
  });

  it('URL 末尾の句読点を URL に含めない', () => {
    const segs = parseBody(' https://example.com/a。 ');
    expect(segs.find((s) => s.type === 'link')).toMatchObject({ url: 'https://example.com/a' });
  });

  it('anchorTargets が広すぎる範囲を展開しない', () => {
    const segs = parseBody(' &gt;&gt;1-1000 ');
    expect(anchorTargets(segs)).toEqual([1]);
  });

  it('anchorTargets が通常の範囲を展開する', () => {
    const segs = parseBody(' &gt;&gt;8-10 ');
    expect(anchorTargets(segs)).toEqual([8, 9, 10]);
  });
});

describe('subject', () => {
  it('レス数を末尾の括弧から取る', () => {
    const rows = parseSubject(['1785028012.dat<>ここに大相撲見てる奴ゼロ説www  (5)']);
    expect(rows).toEqual([{ key: '1785028012', title: 'ここに大相撲見てる奴ゼロ説www', resCount: 5 }]);
  });

  it('タイトル自体に括弧が入っていても壊れない', () => {
    const rows = parseSubject(['1785028012.dat<>ニュース速報(VIP)の話 (123)']);
    expect(rows[0].title).toBe('ニュース速報(VIP)の話');
    expect(rows[0].resCount).toBe(123);
  });

  it('壊れた行は読み飛ばす', () => {
    expect(parseSubject(['', 'ゴミ', '1785028012.dat<>正常 (1)'])).toHaveLength(1);
  });
});
