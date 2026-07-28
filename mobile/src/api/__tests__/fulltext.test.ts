import { groupByThread, type PostHit } from '../fulltext';
import type { Post } from '../../parse/datLine';

function post(res: number): Post {
  return {
    res,
    name: '名無し',
    wacchoi: null,
    trip: null,
    isCap: false,
    mail: '',
    dateText: `2026/07/28(火) 0${res}:00:00.000`,
    timestamp: res,
    uid: 'abc',
    be: null,
    body: `本文${res}`,
    isAbone: false,
  };
}

function hit(key: string, res: number): PostHit {
  return {
    host: 'mi.5ch.io',
    board: 'news4vip',
    key,
    title: `スレ${key}`,
    post: post(res),
    snippet: `snippet${res}`,
  };
}

describe('groupByThread', () => {
  // 実況板のように同じ語が連呼されるスレだと、畳まないとレスの数だけ同じスレが並ぶ。
  it('同じスレの複数一致を 1 行に畳んで件数を持つ', () => {
    const grouped = groupByThread([hit('111', 5), hit('111', 9), hit('111', 3)]);
    expect(grouped).toHaveLength(1);
    expect(grouped[0].count).toBe(3);
  });

  it('代表は最も若いレス番号になる', () => {
    const grouped = groupByThread([hit('111', 5), hit('111', 9), hit('111', 3)]);
    expect(grouped[0].post.res).toBe(3);
    expect(grouped[0].snippet).toBe('snippet3');
  });

  it('別スレは別行のまま、最初に見つかった順を保つ', () => {
    const grouped = groupByThread([hit('222', 1), hit('111', 1), hit('222', 4)]);
    expect(grouped.map((g) => g.key)).toEqual(['222', '111']);
    expect(grouped[0].count).toBe(2);
    expect(grouped[1].count).toBe(1);
  });

  it('空なら空', () => {
    expect(groupByThread([])).toEqual([]);
  });
});
