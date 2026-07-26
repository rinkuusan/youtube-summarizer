import { Ch5Error, errorFromStatus } from '../net/errors';
import { fetchBytes, hostOf } from '../net/http';

/**
 * 板一覧。bbsmenu.json は UTF-8 で、各要素が板キー (directory_name) と
 * サーバ URL の両方を持っているので、板 -> サーバの対応表を別途持つ必要がない。
 */

const BBSMENU_URL = 'https://menu.5ch.net/bbsmenu.json';

export interface Board {
  /** 板が乗っているサーバのホスト名 (例: mi.5ch.io)。 */
  host: string;
  /** 板キー (例: news4vip)。 */
  id: string;
  /** 表示名 (例: ニュース速報(VIP))。 */
  name: string;
  category: string;
  categoryOrder: number;
}

export interface BoardCategory {
  name: string;
  boards: Board[];
}

interface RawEntry {
  directory_name?: string;
  board_name?: string;
  category_name?: string;
  category_order?: number;
  url?: string;
}

/** bbsmenu.json の生 JSON を Board[] にする。 */
export function parseBbsmenu(json: unknown): Board[] {
  const root = json as { menu_list?: { category_content?: RawEntry[] }[] };
  if (!root?.menu_list || !Array.isArray(root.menu_list)) {
    throw new Ch5Error('parse', '板一覧の形式を解釈できませんでした。');
  }

  const boards: Board[] = [];
  for (const cat of root.menu_list) {
    for (const e of cat.category_content ?? []) {
      if (!e.url || !e.directory_name || !e.board_name) continue;
      let host: string;
      try {
        host = hostOf(e.url);
      } catch {
        continue;
      }
      boards.push({
        host,
        id: e.directory_name,
        name: e.board_name,
        category: e.category_name ?? 'その他',
        categoryOrder: e.category_order ?? 0,
      });
    }
  }
  if (boards.length === 0) {
    throw new Ch5Error('parse', '板一覧が空でした。5ch 側の仕様が変わった可能性があります。');
  }
  return boards;
}

/** カテゴリごとにまとめる。同一カテゴリ内の重複は取り除く。 */
export function groupByCategory(boards: Board[]): BoardCategory[] {
  const order: string[] = [];
  const map = new Map<string, Map<string, Board>>();

  for (const b of boards) {
    let bucket = map.get(b.category);
    if (!bucket) {
      bucket = new Map();
      map.set(b.category, bucket);
      order.push(b.category);
    }
    const dedupeKey = `${b.host}/${b.id}`;
    if (!bucket.has(dedupeKey)) bucket.set(dedupeKey, b);
  }

  return order.map((name) => ({
    name,
    boards: [...map.get(name)!.values()].sort((a, b) => a.categoryOrder - b.categoryOrder),
  }));
}

export async function fetchBoards(signal?: AbortSignal): Promise<Board[]> {
  const res = await fetchBytes(BBSMENU_URL, { signal });
  if (res.status !== 200) throw errorFromStatus(res.status, BBSMENU_URL);

  // bbsmenu.json だけは UTF-8。
  const text = new TextDecoder().decode(res.bytes);
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (e) {
    throw new Ch5Error('parse', '板一覧の JSON を解析できませんでした。', { url: BBSMENU_URL, cause: e });
  }
  return parseBbsmenu(json);
}
