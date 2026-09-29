import { containsTerm, filterByTitle, splitTerms } from '../search';
import type { SearchHit } from '../../parse/findHtml';

function hit(title: string): SearchHit {
  return {
    host: 'mi.5ch.io',
    board: 'news4vip',
    key: '1',
    title,
    resCount: 1,
    boardName: null,
    momentumText: null,
    updatedText: null,
  };
}

describe('splitTerms', () => {
  it('| で OR に割る', () => {
    expect(splitTerms('X|Twitter|ツイッター')).toEqual(['X', 'Twitter', 'ツイッター']);
  });
  it('前後の空白と空要素を落とす', () => {
    expect(splitTerms(' X | | ツイート ')).toEqual(['X', 'ツイート']);
  });
});

describe('containsTerm', () => {
  // 「X」のような短い英字語は素の部分一致だと無関係に当たり続ける。
  it('短い英字語は単語として一致したときだけ拾う', () => {
    expect(containsTerm('xの投稿が伸びた', 'x')).toBe(true);
    expect(containsTerm('旧twitter改めx', 'x')).toBe(true);
    expect(containsTerm('xperiaの新型', 'x')).toBe(false);
    expect(containsTerm('音量max', 'x')).toBe(false);
  });

  it('3 文字以上や日本語は素の部分一致のまま', () => {
    expect(containsTerm('ツイッターやってる', 'ツイッター')).toBe(true);
    expect(containsTerm('twitterの話', 'twitter')).toBe(true);
  });
});

describe('filterByTitle (OR)', () => {
  const hits = [
    hit('Xの投稿が伸びない'),
    hit('Xperia 1 VII 買った'),
    hit('ツイッター民さん、また炎上'),
    hit('ラーメン食いたい'),
  ];

  it('どれかの語に一致したものを残す', () => {
    const r = filterByTitle(hits, 'X|ツイッター');
    expect(r.map((h) => h.title)).toEqual(['Xの投稿が伸びない', 'ツイッター民さん、また炎上']);
  });

  it('Xperia を X の一致として拾わない', () => {
    expect(filterByTitle(hits, 'X').map((h) => h.title)).toEqual(['Xの投稿が伸びない']);
  });
});
