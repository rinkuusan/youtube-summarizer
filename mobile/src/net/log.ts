/**
 * アプリ内ログ。
 *
 * release APK では Metro のコンソールも adb も使えない前提なので、
 * 端末の中だけでログを見て丸ごとコピーできるようにする。
 * リングバッファなので長く使ってもメモリは伸びない。
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  id: number;
  at: number;
  level: LogLevel;
  tag: string;
  msg: string;
  /** スタックトレースや JSON など、折りたたんで出す長い付随情報。 */
  detail?: string;
}

const MAX_ENTRIES = 500;

let seq = 0;
let entries: LogEntry[] = [];
const listeners = new Set<() => void>();

function emit() {
  for (const fn of listeners) fn();
}

export function log(level: LogLevel, tag: string, msg: string, detail?: unknown): void {
  // logcat にも流す。端末が施錠されていて画面を見られない状況でも
  // adb logcat -s ReactNativeJS で追えるようにするため。
  // console.log は下のフックで捕まえていないので二重記録にはならない。
  console.log(`[gv][${level}] ${tag}: ${msg}`);

  entries = [
    ...entries.slice(entries.length >= MAX_ENTRIES ? entries.length - MAX_ENTRIES + 1 : 0),
    { id: ++seq, at: Date.now(), level, tag, msg, detail: stringifyDetail(detail) },
  ];
  emit();
}

/** 例外を、種類・ステータス・URL・スタックまで残す形で記録する。 */
export function logError(tag: string, e: unknown, context?: string): void {
  const err = e as { name?: string; message?: string; kind?: string; status?: number; url?: string; stack?: string };
  const head = err?.message ?? String(e);
  const bits: string[] = [];
  if (err?.name) bits.push(`name=${err.name}`);
  if (err?.kind) bits.push(`kind=${err.kind}`);
  if (typeof err?.status === 'number') bits.push(`status=${err.status}`);
  if (err?.url) bits.push(`url=${err.url}`);
  if (err?.stack) bits.push(`\n${err.stack}`);

  const cause = (e as { cause?: unknown })?.cause;
  if (cause !== undefined) bits.push(`cause=${(cause as Error)?.message ?? String(cause)}`);

  log('error', tag, context ? `${context}: ${head}` : head, bits.join(' '));
}

function stringifyDetail(detail: unknown): string | undefined {
  if (detail === undefined || detail === null) return undefined;
  if (typeof detail === 'string') return detail.length > 0 ? detail : undefined;
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

export function getLogs(): LogEntry[] {
  return entries;
}

export function clearLogs(): void {
  entries = [];
  seq = 0;
  emit();
}

export function subscribeLogs(fn: () => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

function pad(n: number, w = 2): string {
  return String(n).padStart(w, '0');
}

/** ログ行の時刻表示 (HH:MM:SS.mmm)。 */
export function formatTime(at: number): string {
  const d = new Date(at);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

/** クリップボードに貼れる全文テキスト。 */
export function formatLogsForCopy(header: string): string {
  const body = entries
    .map((e) => {
      const line = `${formatTime(e.at)} [${e.level.toUpperCase()}] ${e.tag}: ${e.msg}`;
      return e.detail ? `${line}\n    ${e.detail.replace(/\n/g, '\n    ')}` : line;
    })
    .join('\n');
  return `${header}\n${'-'.repeat(40)}\n${body || '(ログなし)'}`;
}

/**
 * JS の取りこぼしを全部ここに集める。
 * - ErrorUtils: RN の未捕捉例外 (クラッシュ直前)
 * - console.error/warn: unhandled promise rejection もここに落ちてくる
 */
let installed = false;

export function installGlobalLogHandlers(): void {
  if (installed) return;
  installed = true;

  const errorUtils = (globalThis as { ErrorUtils?: {
    getGlobalHandler(): (e: unknown, isFatal?: boolean) => void;
    setGlobalHandler(h: (e: unknown, isFatal?: boolean) => void): void;
  } }).ErrorUtils;

  if (errorUtils) {
    const prev = errorUtils.getGlobalHandler();
    errorUtils.setGlobalHandler((e, isFatal) => {
      logError('crash', e, isFatal ? '致命的な未捕捉例外' : '未捕捉例外');
      prev?.(e, isFatal);
    });
  }

  for (const level of ['error', 'warn'] as const) {
    const orig = console[level].bind(console);
    console[level] = (...args: unknown[]) => {
      log(level, 'console', args.map(oneLine).join(' '));
      orig(...args);
    };
  }
}

function oneLine(v: unknown): string {
  if (typeof v === 'string') return v;
  if (v instanceof Error) return `${v.name}: ${v.message}`;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
