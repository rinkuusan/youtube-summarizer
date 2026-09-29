import type { NgRule } from '../db/types';
import { anchorTargets, segmentsToPlainText, type Segment } from '../parse/body';
import type { Post } from '../parse/datLine';
import { normalizeForSearch } from '../utils/normalize';

/**
 * NG の適用。
 *
 * 保存時ではなく描画時に掛ける。ルールを足したり消したりしても再取得が要らず、
 * レス番号 (= 行インデックス + 1) も崩れない。
 */

export interface NgResult {
  /** 透明 NG を除いた表示対象。mask 指定のレスは isAbone を立てた形で含まれる。 */
  visible: Post[];
  /** 透明 NG で消した件数。 */
  hiddenCount: number;
  /** あぼーん表示にした件数。 */
  maskedCount: number;
  /** 連鎖判定に使う「隠れたレス番号」。 */
  hiddenRes: Set<number>;
  /** 正規表現が壊れているルールの id。設定画面で赤く出す。 */
  invalidRuleIds: number[];
}

/** 連鎖 NG の反復上限。相互参照で無限に回らないための保険。 */
const MAX_CHAIN_PASSES = 5;

interface Compiled {
  rule: NgRule;
  regex: RegExp | null;
  needle: string;
}

function compile(rules: NgRule[]): { compiled: Compiled[]; invalidRuleIds: number[] } {
  const compiled: Compiled[] = [];
  const invalidRuleIds: number[] = [];

  for (const rule of rules) {
    if (rule.is_regex === 1) {
      try {
        compiled.push({ rule, regex: new RegExp(rule.pattern, 'i'), needle: '' });
      } catch {
        // 壊れた正規表現でアプリを落とさない。無効扱いにして呼び出し側に伝える。
        invalidRuleIds.push(rule.id);
      }
    } else {
      compiled.push({ rule, regex: null, needle: normalizeForSearch(rule.pattern) });
    }
  }
  return { compiled, invalidRuleIds };
}

function matches(c: Compiled, value: string | null): boolean {
  if (value === null || value === '') return false;
  if (c.regex) return c.regex.test(value);
  return normalizeForSearch(value).includes(c.needle);
}

/** 1 レスがどのルールに引っかかるか。引っかからなければ null。 */
function firstMatch(post: Post, bodyText: string, compiled: Compiled[]): NgRule | null {
  for (const c of compiled) {
    switch (c.rule.kind) {
      case 'word':
        if (matches(c, bodyText)) return c.rule;
        break;
      case 'id':
        // ID 非表示板では uid が null。matches が null を弾くので特別扱いは要らない。
        if (matches(c, post.uid)) return c.rule;
        break;
      case 'name':
        if (matches(c, post.name)) return c.rule;
        break;
      case 'wacchoi':
        if (matches(c, post.wacchoi)) return c.rule;
        break;
      case 'thread':
        // スレ単位の NG はスレ一覧側で使う。レスには効かない。
        break;
    }
  }
  return null;
}

export function applyNg(
  posts: Post[],
  segmentsByRes: Map<number, Segment[]>,
  rules: NgRule[]
): NgResult {
  if (rules.length === 0) {
    return {
      visible: posts,
      hiddenCount: 0,
      maskedCount: 0,
      hiddenRes: new Set(),
      invalidRuleIds: [],
    };
  }

  const { compiled, invalidRuleIds } = compile(rules);
  const hidden = new Set<number>();
  const masked = new Set<number>();

  const bodyTextCache = new Map<number, string>();
  const bodyTextOf = (post: Post): string => {
    const cached = bodyTextCache.get(post.res);
    if (cached !== undefined) return cached;
    const segs = segmentsByRes.get(post.res);
    const text = segs ? segmentsToPlainText(segs) : post.body;
    bodyTextCache.set(post.res, text);
    return text;
  };

  for (const post of posts) {
    const rule = firstMatch(post, bodyTextOf(post), compiled);
    if (!rule) continue;
    if (rule.hide_mode === 'mask') masked.add(post.res);
    else hidden.add(post.res);
  }

  // 連鎖 NG: 隠れたレスに返信しているレスも隠す。
  // どのルール由来かまでは追わず、連鎖が 1 つでも有効なら全体に効かせる
  // (規則ごとに来歴を持つ複雑さに見合う利得が無い)。
  if (rules.some((r) => r.chain === 1)) {
    for (let pass = 0; pass < MAX_CHAIN_PASSES; pass++) {
      let grew = false;
      for (const post of posts) {
        if (hidden.has(post.res) || masked.has(post.res)) continue;
        const segs = segmentsByRes.get(post.res);
        if (!segs) continue;
        if (anchorTargets(segs).some((t) => hidden.has(t) || masked.has(t))) {
          hidden.add(post.res);
          grew = true;
        }
      }
      if (!grew) break;
    }
  }

  const visible: Post[] = [];
  for (const post of posts) {
    if (hidden.has(post.res)) continue;
    visible.push(masked.has(post.res) ? { ...post, isAbone: true } : post);
  }

  return {
    visible,
    hiddenCount: hidden.size,
    maskedCount: masked.size,
    hiddenRes: new Set([...hidden, ...masked]),
    invalidRuleIds,
  };
}
