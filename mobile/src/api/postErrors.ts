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
  | 'wrongReferer' // Referer が板と噛み合っていない
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
    // 助詞は「は」「が」どちらもあるので、間を緩く取る。
    // 「ホストは」だけを見ていると「ホストが規制されています」を取りこぼす。
    patterns: [/規制中です/, /ホスト.{0,4}規制されて/, /アクセス規制/, /書き?込み規制/],
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
    // bbs.cgi が Referer を見て「この投稿は余所から来た」と判断したときに返す。
    // 5ch は判定条件を公開していないので断定はできないが、Referer が板と
    // 噛み合っていないケースで出るのが知られている。こちらの送り方の問題で
    // 起きうるエラーなので、規制と混同させずに分けておく。
    outcome: 'wrongReferer',
    patterns: [/余所でやって/, /よそでやって/, /ヨソでやって/],
    title: '5ch に「余所でやってくれ」と返されました',
    message:
      'bbs.cgi が、この書き込みを板と結び付いていないものとして拒否しています。' +
      '規制ではなく送り方の問題である可能性があります。ログを見せてもらえれば切り分けます。',
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
  /**
   * 応答のフォームに入っていた hidden 等の値。承諾して送り直すときに
   * そのまま積み直す。確認ページは `feature` のような使い捨てトークンを
   * 要求してくるので、これを返さないと何度承諾しても確認ページが返り続ける
   * (実測 2026-07-28: kizuna.5ch.io/gamefight は feature,submit の 2 つ)。
   */
  formFields: Record<string, string>;
  /** 応答フォームの送信先 (相対のことがある)。承諾時はここへ送り直す。 */
  formAction: string | null;
}

/**
 * 属性値を取り出す。5ch の確認ページは同じタグの中でクォートの有無が混在する。
 * 実物: `<input type=hidden name=FROM value="">` (name は裸、value は二重引用符)
 * 裸の値だけを見ていると FROM/mail/MESSAGE を丸ごと取りこぼす。
 */
function attr(attrs: string, key: string): string | null {
  const re = new RegExp(`\\b${key}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'>]+))`, 'i');
  const m = re.exec(attrs);
  if (!m) return null;
  return m[1] ?? m[2] ?? m[3] ?? '';
}

/**
 * 応答のフォームから name/value を集める。
 * value が無い input は空文字で持つ。
 */
export function extractFormFields(html: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of html.matchAll(/<input\b([^>]*)>/gi)) {
    const name = attr(m[1], 'name');
    if (!name) continue;
    out[name] = decodeAttr(attr(m[1], 'value') ?? '');
  }
  return out;
}

/**
 * 応答のフォームの送信先。
 *
 * 確認ページは `../test/bbs.cgi?guid=ON` を指しており、クエリ付きでないと
 * 5ch は承諾と認めない。こちらが素の bbs.cgi に投げ続けると確認ページが
 * 返り続ける (実測 2026-07-28)。
 */
export function extractFormAction(html: string): string | null {
  const m = /<form\b([^>]*)>/i.exec(html);
  return m ? attr(m[1], 'action') : null;
}

function decodeAttr(s: string): string {
  return s
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
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
  const formFields = extractFormFields(html);
  const formAction = extractFormAction(html);
  const haystackTitle = title;
  const haystackAll = `${title}\n${bodyText}`;

  if (httpStatus === 403 || httpStatus === 451) {
    return {
      outcome: 'blocked',
      title: 'アクセスが遮断されました',
      message: `サーバに拒否されました (${httpStatus})。ブラウザから書き込んでみてください。`,
      action: 'openBrowser',
      bodyText,
      formFields,
      formAction,
      html,
    };
  }

  const build = ({ patterns: _patterns, ...rest }: Rule): PostResult => ({
    ...rest,
    bodyText,
    formFields,
    formAction,
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
    formFields,
    formAction,
    html,
  };
}

/** SAMBA24 の待ち秒数を応答から拾う。取れなければ null。 */
export function extractCooldownSeconds(text: string): number | null {
  const m = /(\d+)\s*秒/.exec(text);
  return m ? Number(m[1]) : null;
}
