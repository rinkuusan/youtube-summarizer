import type { NgRule } from '../../db/types';
import { parseBody, type Segment } from '../../parse/body';
import type { Post } from '../../parse/datLine';
import { applyNg } from '../applyNg';

function post(res: number, body: string, extra: Partial<Post> = {}): Post {
  return {
    res,
    name: '名無し',
    wacchoi: null,
    trip: null,
    isCap: false,
    mail: '',
    dateText: '',
    timestamp: null,
    uid: `ID${res}`,
    be: null,
    body,
    isAbone: false,
    ...extra,
  };
}

function segmentsOf(posts: Post[]): Map<number, Segment[]> {
  return new Map(posts.map((p) => [p.res, parseBody(p.body)]));
}

function rule(over: Partial<NgRule>): NgRule {
  return {
    id: 1,
    kind: 'word',
    pattern: '',
    scope_board: null,
    is_regex: 0,
    hide_mode: 'abone',
    chain: 0,
    expires_at: null,
    created_at: 0,
    ...over,
  };
}

describe('applyNg', () => {
  it('ルールが無ければ元の配列をそのまま返す', () => {
    const posts = [post(1, ' あ '), post(2, ' い ')];
    const r = applyNg(posts, segmentsOf(posts), []);
    expect(r.visible).toBe(posts);
    expect(r.hiddenCount).toBe(0);
  });

  it('NG ワードでレスを消す', () => {
    const posts = [post(1, ' 普通のレス '), post(2, ' 荒らしです ')];
    const r = applyNg(posts, segmentsOf(posts), [rule({ kind: 'word', pattern: '荒らし' })]);
    expect(r.visible.map((p) => p.res)).toEqual([1]);
    expect(r.hiddenCount).toBe(1);
  });

  it('NG ワードは表記ゆれを吸収する（カタカナ/全角）', () => {
    const posts = [post(1, ' アラシ ')];
    const r = applyNg(posts, segmentsOf(posts), [rule({ kind: 'word', pattern: 'あらし' })]);
    expect(r.visible).toHaveLength(0);
  });

  it('NGID でレスを消す', () => {
    const posts = [post(1, ' a '), post(2, ' b ', { uid: 'BAD123' })];
    const r = applyNg(posts, segmentsOf(posts), [rule({ kind: 'id', pattern: 'BAD123' })]);
    expect(r.visible.map((p) => p.res)).toEqual([1]);
  });

  it('ID 非表示板 (uid が null) でも落ちない', () => {
    const posts = [post(1, ' a ', { uid: null }), post(2, ' b ', { uid: null })];
    const r = applyNg(posts, segmentsOf(posts), [rule({ kind: 'id', pattern: 'BAD' })]);
    expect(r.visible).toHaveLength(2);
    expect(r.hiddenCount).toBe(0);
  });

  it('名前とワッチョイで消せる', () => {
    const posts = [
      post(1, ' a ', { name: '普通' }),
      post(2, ' b ', { name: 'コテハン' }),
      post(3, ' c ', { wacchoi: 'ﾜｯﾁｮｲ aaaa-bbbb' }),
    ];
    const byName = applyNg(posts, segmentsOf(posts), [rule({ kind: 'name', pattern: 'コテハン' })]);
    expect(byName.visible.map((p) => p.res)).toEqual([1, 3]);

    const byWacchoi = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'wacchoi', pattern: 'aaaa-bbbb' }),
    ]);
    expect(byWacchoi.visible.map((p) => p.res)).toEqual([1, 2]);
  });

  it('正規表現ルールが使える', () => {
    const posts = [post(1, ' hello '), post(2, ' spam1234 ')];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'word', pattern: 'spam\\d+', is_regex: 1 }),
    ]);
    expect(r.visible.map((p) => p.res)).toEqual([1]);
  });

  it('壊れた正規表現で落ちず、無効なルールとして報告する', () => {
    const posts = [post(1, ' hello ')];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ id: 42, kind: 'word', pattern: '[unclosed', is_regex: 1 }),
    ]);
    expect(r.visible).toHaveLength(1);
    expect(r.invalidRuleIds).toEqual([42]);
  });

  it('mask 指定はあぼーん表示として残す', () => {
    const posts = [post(1, ' 荒らし ')];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'word', pattern: '荒らし', hide_mode: 'mask' }),
    ]);
    expect(r.visible).toHaveLength(1);
    expect(r.visible[0].isAbone).toBe(true);
    expect(r.maskedCount).toBe(1);
    expect(r.hiddenCount).toBe(0);
  });

  it('連鎖 NG が返信を消す', () => {
    const posts = [
      post(1, ' 荒らし '),
      post(2, ' &gt;&gt;1 そうだね '),
      post(3, ' 無関係 '),
    ];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'word', pattern: '荒らし', chain: 1 }),
    ]);
    expect(r.visible.map((p) => p.res)).toEqual([3]);
  });

  it('連鎖 NG が多段でも収束する', () => {
    const posts = [
      post(1, ' 荒らし '),
      post(2, ' &gt;&gt;1 '),
      post(3, ' &gt;&gt;2 '),
      post(4, ' &gt;&gt;3 '),
      post(5, ' 無関係 '),
    ];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'word', pattern: '荒らし', chain: 1 }),
    ]);
    expect(r.visible.map((p) => p.res)).toEqual([5]);
  });

  it('連鎖が無効なら返信は消さない', () => {
    const posts = [post(1, ' 荒らし '), post(2, ' &gt;&gt;1 そうだね ')];
    const r = applyNg(posts, segmentsOf(posts), [rule({ kind: 'word', pattern: '荒らし' })]);
    expect(r.visible.map((p) => p.res)).toEqual([2]);
  });

  it('相互参照でも無限ループしない', () => {
    const posts = [
      post(1, ' 荒らし '),
      post(2, ' &gt;&gt;3 '),
      post(3, ' &gt;&gt;2 '),
    ];
    const r = applyNg(posts, segmentsOf(posts), [
      rule({ kind: 'word', pattern: '荒らし', chain: 1 }),
    ]);
    // 2 と 3 は互いを指すだけで NG レスには到達しないので残る
    expect(r.visible.map((p) => p.res)).toEqual([2, 3]);
  });
});
