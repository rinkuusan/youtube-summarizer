import { readFileSync } from 'fs';
import { join } from 'path';

import { parseFindHtml } from '../../parse/findHtml';
import { filterByTitle } from '../search';

/**
 * find.5ch はクエリを形態素で割って OR 検索するため、返却をそのまま出すと
 * クエリを含まないスレが大量に混ざる。実測 (2026-07-26):
 *
 *   地震     100 件中 100 件が一致
 *   ウマ娘   100 件中  21 件が一致
 *   からあげ 100 件中   0 件が一致  ← 「から」「あげ」に割れて全滅
 *
 * このフィクスチャは「娘」で検索した実データで、まさにその状況になっている。
 */
function loadHits() {
  return parseFindHtml(readFileSync(join(__dirname, '..', '..', 'parse', '__fixtures__', 'find.html'), 'utf8'));
}

describe('filterByTitle', () => {
  const hits = loadHits();

  it('前提: フィクスチャは「娘」を含む 3 件', () => {
    expect(hits).toHaveLength(3);
    expect(hits.every((h) => h.title.includes('娘'))).toBe(true);
  });

  it('複合語では、部分一致しかしないスレを落とす', () => {
    // 3 件とも「娘」を含むが、「ウマ娘」を含むのは 1 件だけ。
    // 絞らないとモーニング娘。のスレが「ウマ娘」の検索結果として出てしまう。
    const matched = filterByTitle(hits, 'ウマ娘');
    expect(matched).toHaveLength(1);
    expect(matched[0].title).toContain('VIPでウマ娘');
  });

  it('単一の語では全件残る (絞りすぎない)', () => {
    expect(filterByTitle(hits, '娘')).toHaveLength(3);
  });

  it('カタカナ・ひらがなのゆれを吸収する', () => {
    // 「うま娘」で「ウマ娘」を引けること。normalizeForSearch がカタカナをひらがなに寄せる。
    expect(filterByTitle(hits, 'うま娘')).toHaveLength(1);
  });

  it('一致が無ければ空になる (find.5ch の返却をそのまま出さない)', () => {
    expect(filterByTitle(hits, 'からあげ')).toHaveLength(0);
  });

  it('空クエリでは絞らない', () => {
    expect(filterByTitle(hits, '   ')).toHaveLength(3);
  });
});

describe('2 重エスケープの復元', () => {
  // find.5ch は dat 側の文字参照 (&#128563;) を更に HTML エスケープして
  // &amp;#128563; として吐く。1 回しか剥がさないと画面に生の &#128563; が出る。
  const hits = parseFindHtml(
    readFileSync(join(__dirname, '..', '..', 'parse', '__fixtures__', 'find-escaped.html'), 'utf8')
  );

  it('文字参照が実体に戻り、生の &# が残らない', () => {
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].title).toContain('\u{1F633}');
    for (const h of hits) {
      expect(h.title).not.toContain('&#');
      expect(h.title).not.toContain('&amp;');
    }
  });

  it('レス数の切り出しは 2 重デコード後も効く', () => {
    expect(hits[0].resCount).toBe(3);
    expect(hits[0].title).not.toMatch(/\(3\)$/);
  });
});
