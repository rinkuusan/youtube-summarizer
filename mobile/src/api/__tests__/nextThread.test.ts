import { baseTitle, titleSimilarity } from '../nextThread';

/**
 * 5ch の続きスレは「同じ題名 + 巻数」で立つ。表記は板ごとにばらばらなので、
 * 実際に見かける書き方を並べて固定する。
 */
describe('baseTitle', () => {
  it('よくある巻数表記を落とす', () => {
    expect(baseTitle('イオンモール熊本で爆発 ★16')).toBe('イオンモール熊本で爆発');
    expect(baseTitle('雑談スレ Part3')).toBe('雑談スレ');
    expect(baseTitle('なんJ深夜の部 vol.12')).toBe('なんJ深夜の部');
    expect(baseTitle('鞘師里保雑談スレ★91')).toBe('鞘師里保雑談スレ');
    expect(baseTitle('ウマ娘スレ #7')).toBe('ウマ娘スレ');
    expect(baseTitle('あつまれ雑談民 3スレ目')).toBe('あつまれ雑談民');
  });

  it('[記者名] やワッチョイ表記を落とす', () => {
    expect(baseTitle('速報です ★3  [nita★]')).toBe('速報です');
  });

  it('剥がしすぎない (題名が消えるなら元を採る)', () => {
    // 「2026」を巻数と誤認して題名を消し飛ばさないこと
    expect(baseTitle('2026')).toBe('2026');
  });

  it('巻数が無ければそのまま', () => {
    expect(baseTitle('ぼっちだけど質問ある？')).toBe('ぼっちだけど質問ある？');
  });
});

describe('titleSimilarity', () => {
  it('芯が一致すれば 1', () => {
    expect(titleSimilarity('イオンモール熊本で爆発', 'イオンモール熊本で爆発')).toBe(1);
  });

  it('全角半角・カタカナのゆれを吸収する', () => {
    expect(titleSimilarity('ウマ娘スレ', 'うま娘スレ')).toBe(1);
  });

  it('無関係な題名は低い', () => {
    expect(titleSimilarity('イオンモール熊本で爆発', 'ラーメン食いたい')).toBeLessThan(0.3);
  });

  it('途中まで同じなら中間の値', () => {
    const s = titleSimilarity('雑談スレ', '雑談したい');
    expect(s).toBeGreaterThan(0.4);
    expect(s).toBeLessThan(1);
  });
});
