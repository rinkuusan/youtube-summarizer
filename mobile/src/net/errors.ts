/**
 * 5ch アクセスで起きるエラーを、ユーザーに見せられる日本語に正規化する。
 *
 * 方針: エラーを握りつぶさない。生 dat の直読みは 5ch の裁量でいつでも塞がれうるので、
 * 塞がれたときに「何が起きたか」がユーザーに見えることを優先する。
 */
import { logError } from './log';

export type Ch5ErrorKind =
  | 'network' // 回線・DNS・タイムアウト
  | 'notFound' // 404 (板/スレが無い、dat 落ち)
  | 'forbidden' // 403 / 451 (アクセス制限)
  | 'server' // 5xx
  | 'parse' // 取得はできたが解釈できない
  | 'unknown';

export class Ch5Error extends Error {
  readonly kind: Ch5ErrorKind;
  readonly status?: number;
  readonly url?: string;

  constructor(kind: Ch5ErrorKind, message: string, opts?: { status?: number; url?: string; cause?: unknown }) {
    super(message, opts?.cause !== undefined ? { cause: opts.cause } : undefined);
    this.name = 'Ch5Error';
    this.kind = kind;
    this.status = opts?.status;
    this.url = opts?.url;
  }
}

/** HTTP ステータスから Ch5Error を作る。 */
export function errorFromStatus(status: number, url: string): Ch5Error {
  if (status === 404) {
    return new Ch5Error('notFound', 'スレッドまたは板が見つかりません。dat 落ちしている可能性があります。', {
      status,
      url,
    });
  }
  if (status === 403 || status === 451) {
    return new Ch5Error(
      'forbidden',
      `アクセスが制限されています (${status})。回線を変えると解消する場合があります。`,
      { status, url }
    );
  }
  if (status >= 500) {
    return new Ch5Error('server', `5ch のサーバがエラーを返しました (${status})。`, { status, url });
  }
  return new Ch5Error('unknown', `予期しない応答です (${status})。`, { status, url });
}

/** fetch が throw した例外を Ch5Error に包む。 */
export function errorFromThrown(e: unknown, url: string): Ch5Error {
  if (e instanceof Ch5Error) return e;
  const msg = e instanceof Error ? e.message : String(e);
  if (/abort/i.test(msg)) {
    return new Ch5Error('network', '通信がタイムアウトしました。', { url, cause: e });
  }
  return new Ch5Error('network', `5ch に接続できません: ${msg}`, { url, cause: e });
}

/**
 * 画面にそのまま出せる一行メッセージ。
 *
 * 画面に出るエラーは必ずここを通るので、ログ記録もここでまとめて行う。
 * 各画面の catch に個別に仕込むと、画面を足したときに漏れる。
 */
export function toDisplayMessage(e: unknown): string {
  logError('ui', e, '画面にエラー表示');
  if (e instanceof Ch5Error) return e.message;
  return e instanceof Error ? e.message : String(e);
}
