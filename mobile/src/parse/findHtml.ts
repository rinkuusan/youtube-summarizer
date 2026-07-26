import { decodeEntities } from './entities';

/**
 * find.5ch.net の検索結果 HTML を解析する。
 *
 * JSON API は無いのでスクレイプするしかない。1 ページ 100 件。
 * 実際の構造 (取得して確認):
 *
 *   <div class="list_line">
 *     <a class="list_line_link" href="//mi.5ch.io/test/read.cgi/news4vip/1784974154">
 *       <div class="list_line_link_title">VIPでウマ娘  (399)</div>
 *     </a>
 *     <div class="list_line_info">
 *       <div class="... list_line_info_container-board"><a href="...">ニュー速VIP</a></div>
 *       <div class="list_line_info_container">2026年07月26日 11:55</div>
 *       <div class="... list_line_info_container-danger">2945.5/日</div>
 *     </div>
 *   </div>
 *
 * href から host / 板 / datkey がそのまま取れるので、結果から直接スレを開ける。
 *
 * 5ch 側の HTML 変更で壊れる前提で書く。壊れたときは例外ではなく空配列を返し、
 * 呼び出し側が「解析できませんでした」を出せるようにする。
 */

export interface SearchHit {
  host: string;
  board: string;
  key: string;
  title: string;
  resCount: number;
  /** 板の表示名。取れなければ null。 */
  boardName: string | null;
  /** サーバが計算済みの勢い (例: "2945.5/日")。 */
  momentumText: string | null;
  /** 最終更新の表示文字列。 */
  updatedText: string | null;
}

// 閉じタグの入れ子は正規表現で数えられないので、次のブロックの開始 (か文末) までを
// 1 件分とみなす。閉じタグの構造が変わっても壊れないぶん、こちらの方が頑丈。
const BLOCK = /<div\s+class="list_line">([\s\S]*?)(?=<div\s+class="list_line">|$)/g;
const LINK = /class="list_line_link"\s+href="([^"]+)"/;
const READ_CGI = /^(?:https?:)?\/\/([^/]+)\/test\/read\.cgi\/([^/]+)\/(\d+)/;
const TITLE = /class="list_line_link_title"[^>]*>([\s\S]*?)<\/div>/;
const BOARD = /list_line_info_container-board"[^>]*>\s*<a[^>]*>([\s\S]*?)<\/a>/;
const MOMENTUM = /list_line_info_container-danger"[^>]*>([\s\S]*?)<\/div>/;
const UPDATED = /class="list_line_info_container"[^>]*>([\s\S]*?)<\/div>/;

// タイトル末尾の (レス数)。subject.txt と同じで、貪欲な .* が最後の括弧に一致する。
const TITLE_RES = /^([\s\S]*)\s+\((\d+)\)\s*$/;

function stripTags(s: string): string {
  return decodeEntities(s.replace(/<[^>]*>/g, '')).trim();
}

export function parseFindHtml(html: string): SearchHit[] {
  const out: SearchHit[] = [];
  BLOCK.lastIndex = 0;
  let m: RegExpExecArray | null;

  while ((m = BLOCK.exec(html)) !== null) {
    const block = m[1];

    const link = LINK.exec(block);
    if (!link) continue;
    const rc = READ_CGI.exec(link[1]);
    if (!rc) continue;

    const rawTitle = TITLE.exec(block);
    if (!rawTitle) continue;
    const titleText = stripTags(rawTitle[1]);

    const tr = TITLE_RES.exec(titleText);
    const title = tr ? tr[1].trim() : titleText;
    const resCount = tr ? Number(tr[2]) : 0;

    const board = BOARD.exec(block);
    const momentum = MOMENTUM.exec(block);
    const updated = UPDATED.exec(block);

    out.push({
      host: rc[1],
      board: rc[2],
      key: rc[3],
      title,
      resCount,
      boardName: board ? stripTags(board[1]) : null,
      momentumText: momentum ? stripTags(momentum[1]) : null,
      updatedText: updated ? stripTags(updated[1]) : null,
    });
  }

  return out;
}
