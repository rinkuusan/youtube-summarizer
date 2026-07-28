import { parseDatLine, type ParsedThread, type Post } from './datLine';
import { decodeEntities } from './entities';

/**
 * read.cgi の HTML からレスを取り出す。
 *
 * dat 落ちしたスレは `/<board>/dat/<key>.dat` が 404 になるが、read.cgi は
 * しばらく中身を返し続ける。実測 (2026-07-28): find.5ch のスレタイ検索で
 * 返る ニュー速VIP のスレは、レス数が一桁のものが多く数時間で板から落ちるため、
 * 上位 12 件すべてが dat 404 だった。同じスレを read.cgi で引くと全レス取れる。
 *
 * 生成する Post は dat 由来のものと同じでなければ、NG やアンカーの処理が
 * 二重実装になる。そこで HTML から dat の 1 行を組み立て直し、
 * 実績のある parseDatLine にそのまま食わせる。
 *
 * 構造 (実物、改行なしで詰まっている):
 *   <div id="2" data-date="1785218625" data-userid="ID:xxx" data-id="2" class="clear post">
 *     <div class="post-header">
 *       <div><span class="postid">2</span><span class="postusername"><b>名前</b></span>…</div>
 *       <span><span class="date">2026/07/28(火) 06:01:46.981</span><span class="uid">ID:xxx</span></span>
 *     </div>
 *     <div class="post-content"> 本文 <br> 本文 </div>
 *   </div>
 */

/** レスブロックの開始タグ。属性の順は 5ch の実装依存なので data-id と class だけを見る。 */
const POST_OPEN = /<div\s[^>]*\bdata-id="(\d+)"[^>]*\bclass="[^"]*\bpost\b[^"]*"[^>]*>/gi;

const NAME = /<span class="postusername">([\s\S]*?)<\/span>/i;
const DATE = /<span class="date">([\s\S]*?)<\/span>/i;
const UID = /<span class="uid">([\s\S]*?)<\/span>/i;
const CONTENT = /<div class="post-content">([\s\S]*?)<\/div>/i;
const TITLE = /<h1[^>]*>([\s\S]*?)<\/h1>/i;
/** メール欄はリンクになる。sage 等を拾う。 */
const MAIL = /<a[^>]*href="mailto:([^"]*)"/i;

function firstGroup(re: RegExp, s: string): string {
  const m = re.exec(s);
  return m ? m[1].trim() : '';
}

/** タグを落として素のテキストにする (日付欄・スレタイ用)。 */
function plain(s: string): string {
  // 文字参照を戻す。戻さないと画面に &#129781; が生で出る。
  return decodeEntities(s.replace(/<[^>]*>/g, '')).trim();
}

/**
 * read.cgi の名前欄を dat の名前欄の書き方に直す。
 *
 * dat は `名前 </b>(ﾜｯﾁｮｲ)<b>` と書く (</b> と <b> の間が付加情報)。
 * read.cgi は逆に `<b>名前</b>(ﾜｯﾁｮｲ)` と、名前の方を <b> で囲って出す。
 * parseDatLine に食わせるため、dat 側の書き方に戻す。
 */
export function toDatNameField(inner: string): string {
  const m = /^\s*<b>([\s\S]*?)<\/b>([\s\S]*)$/i.exec(inner);
  if (!m) return inner.replace(/<\/?b>/gi, '').trim();
  const name = m[1].trim();
  const rest = m[2].trim();
  return rest ? `${name}</b>${rest}<b>` : name;
}

export function parseReadCgi(html: string): ParsedThread {
  const title = plain(firstGroup(TITLE, html)) || null;

  // 開始タグの位置を全部拾い、隣り合う開始タグの間を 1 レスとして切る。
  const opens: { res: number; from: number; to: number }[] = [];
  POST_OPEN.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = POST_OPEN.exec(html)) !== null) {
    opens.push({ res: Number(m[1]), from: POST_OPEN.lastIndex, to: m.index });
  }

  const posts: Post[] = [];
  for (let i = 0; i < opens.length; i++) {
    const block = html.slice(opens[i].from, opens[i + 1]?.to ?? html.length);

    const name = toDatNameField(firstGroup(NAME, block));
    const mail = firstGroup(MAIL, block);
    const dateText = plain(firstGroup(DATE, block));
    const uid = plain(firstGroup(UID, block));
    const body = firstGroup(CONTENT, block);

    // dat の 1 行 (名前<>メール<>日付 ID<>本文<>スレタイ) に組み直す。
    // スレタイは dat と同じく 1 レス目にだけ入れる。
    const dateField = [dateText, uid].filter(Boolean).join(' ');
    const line = [name, mail, dateField, body, opens[i].res === 1 ? (title ?? '') : ''].join('<>');

    const post = parseDatLine(line, opens[i].res);
    if (post) posts.push(post);
  }

  return { title, posts };
}
