/**
 * 勢い。dat キーはスレ作成時刻の UNIX 秒そのものなので、そこから経過日数を出す。
 */
export function computeMomentum(datKey: string, resCount: number, nowMs = Date.now()): number {
  const createdSec = Number(datKey);
  if (!Number.isFinite(createdSec) || createdSec <= 0) return 0;
  const days = (nowMs / 1000 - createdSec) / 86400;
  // 立った直後のスレが無限大にならないよう下限を置く (1 時間)。
  const safeDays = Math.max(days, 1 / 24);
  return resCount / safeDays;
}

export function formatMomentum(v: number): string {
  if (v >= 10000) return `${Math.round(v / 1000)}k`;
  if (v >= 100) return String(Math.round(v));
  return v.toFixed(1);
}

/** スレ作成時刻 (epoch ミリ秒)。 */
export function threadCreatedAt(datKey: string): number | null {
  const sec = Number(datKey);
  return Number.isFinite(sec) && sec > 0 ? sec * 1000 : null;
}
