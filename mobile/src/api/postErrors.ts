/**
 * bbs.cgi の応答の分類。
 *
 * 5ch は応答を常に HTTP 200 で返し、成否は HTML の <title> に出る。
 * 5ch 側は予告なく仕様を変える (MonaTicket が yuki=akari を置き換えた、
 * X-* ヘッダ群が不要になった、など) ので、判定はコードではなくデータとして持つ。
 * 変更が来たらこの表を差し替えるだけで済む。
 */

export type PostOutcome =
  | 'success'
  | 'confirm' // 書き込み確認 (中間ページ)
  | 'cooldown' // SAMBA24 連投規制
  | 'banned' // IP 規制
  | 'duplicate'
  | 'tooLong'
  | 'tooManyThreads'
  | 'donguri'
  | 'blocked' // Cloudflare などエッジでの遮断
  | 'unknown';

interface Rule {
  outcome: PostOutcome;
  patterns: RegExp[];
  title: string;
  message: string;
  /** UI 側で追加の動作を出すための目印。 */
  action?: 'countdown' | 'openBrowser' | 'switchNetwork' | 'confirm';
}

const RULES: Rule[] = [
  {
    outcome: 'success',
    patterns: [/書きこみました/, /書き込みました/],
    title: '書き込みました',
    message: '',
  },
  {
    outcome: 'confirm',
    patterns: [/書き?込み確認/, /投稿確認/],
    title: '書き込み確認',
    message: '内容を確認してから送信してください。',
    action: 'confirm',
  },
  {
    outcome: 'cooldown',
    patterns: [/短時間に書き?込みすぎ/, /SAMBA/i, /連続投稿/],
    title: '連投規制 (SAMBA24)',
    message: 'しばらく待ってから、もう一度お試しください。',
    action: 'countdown',
  },
  {
    outcome: 'banned',
    patterns: [/規制中です/, /ホストは?規制されています/, /アクセス規制/, /書き?込み規制/],
    title: '書き込み規制中',
    message:
      '現在この回線からの書き込みが規制されています。Wi-Fi とモバイル回線を切り替えると解消する場合があります。',
    action: 'switchNetwork',
  },
  {
    outcome: 'duplicate',
    patterns: [/二重書き?込み/, /同じ内容/],
    title: '二重書き込み',
    message: '同じ内容が既に投稿されています。',
  },
  {
    outcome: 'tooLong',
    patterns: [/本文が長すぎ/, /長すぎます/],
    title: '本文が長すぎます',
    message: 'この板の上限を超えています。短くしてください。',
  },
  {
    outcome: 'tooManyThreads',
    patterns: [/スレッドを立てすぎ/, /たてすぎ/],
    title: 'スレッドを立てすぎです',
    message: 'この板でのスレッド作成が制限されています。',
  },
  {
    outcome: 'donguri',
    patterns: [/どんぐり/, /broken_acorn/i, /acorn/i, /レベルが足りません/],
    title: 'どんぐりの認証が必要です',
    message:
      'この板は「どんぐり」が必要です。ブラウザで一度書き込むと解消する場合があります。',
    action: 'openBrowser',
  },
  {
    outcome: 'blocked',
    patterns: [/cf-browser-verification/i, /Just a moment/i, /Attention Required/i, /cloudflare/i],
    title: 'アクセスが遮断されました',
    message: 'ブラウザでの確認が必要です。',
    action: 'openBrowser',
  },
];

export interface PostResult {
  outcome: PostOutcome;
  title: string;
  message: string;
  action?: Rule['action'];
  /** 応答 HTML から抜いた本文。確認画面はこれをそのまま表示する。 */
  bodyText: string;
  /** 元の HTML。切り分け用に保持する。 */
  html: string;
}

/** <title> を取り出す。 */
export function extractTitle(html: string): string {
  const m = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html);
  return m ? m[1].trim() : '';
}

/** タグを落として本文だけにする。確認画面をそのまま見せるのに使う。 */
export function extractText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(p|div|tr|li|h\d)>/gi, '\n')
    .replace(/<[^>]*>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * 応答 HTML を分類する。
 * 判定は <title> を優先し、見つからなければ本文全体を見る。
 */
export function classifyPostResponse(html: string, httpStatus = 200): PostResult {
  const title = extractTitle(html);
  const bodyText = extractText(html);
  const haystackTitle = title;
  const haystackAll = `${title}\n${bodyText}`;

  if (httpStatus === 403 || httpStatus === 451) {
    return {
      outcome: 'blocked',
      title: 'アクセスが遮断されました',
      message: `サーバに拒否されました (${httpStatus})。ブラウザから書き込んでみてください。`,
      action: 'openBrowser',
      bodyText,
      html,
    };
  }

  const build = ({ patterns: _patterns, ...rest }: Rule): PostResult => ({
    ...rest,
    bodyText,
    html,
  });

  // まず <title> だけで判定する。本文には注意書きとして無関係な語が入りがちなため。
  for (const rule of RULES) {
    if (rule.patterns.some((p) => p.test(haystackTitle))) return build(rule);
  }
  for (const rule of RULES) {
    if (rule.patterns.some((p) => p.test(haystackAll))) return build(rule);
  }

  return {
    outcome: 'unknown',
    title: title || '書き込みに失敗しました',
    message: '5ch からの応答を判別できませんでした。以下がそのままの内容です。',
    bodyText,
    html,
  };
}

/** SAMBA24 の待ち秒数を応答から拾う。取れなければ null。 */
export function extractCooldownSeconds(text: string): number | null {
  const m = /(\d+)\s*秒/.exec(text);
  return m ? Number(m[1]) : null;
}
