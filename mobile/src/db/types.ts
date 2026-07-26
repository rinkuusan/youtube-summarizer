/** スレッドを一意に指す 3 つ組。DB のキーであり、画面の URL パラメータでもある。 */
export interface ThreadRef {
  host: string;
  board: string;
  key: string;
}

export interface ThreadRow {
  host: string;
  board: string;
  key: string;
  title: string | null;
  /** subject.txt 上の総レス数。 */
  res_count: number | null;
  /** 保持している行数。 */
  cached_count: number;
  /** 差分取得のバイトカーソル。 */
  dat_bytes: number;
  etag: string | null;
  last_modified: string | null;
  /** どこまで読んだか。新着 = res_count - read_count。 */
  read_count: number;
  /** 「ここまで読んだ」位置のレス番号。 */
  scroll_res: number;
  favorite: number;
  fav_order: number | null;
  /** 閲覧履歴。開くたびに更新される。 */
  last_opened_at: number | null;
  /** 書き込み履歴。投稿成功時に更新される。 */
  last_posted_at: number | null;
  my_post_count: number;
  last_fetched_at: number | null;
  dat_dead: number;
}

export interface PostRow {
  host: string;
  board: string;
  key: string;
  res: number;
  name: string;
  mail: string;
  date_str: string;
  ts: number | null;
  uid: string | null;
  wacchoi: string | null;
  trip: string | null;
  is_cap: number;
  is_abone: number;
  /** dat の生の本文フィールド。描画時に parseBody を掛ける。 */
  body: string;
  is_mine: number;
}

export interface BoardRow {
  host: string;
  id: string;
  name: string;
  category: string | null;
  category_order: number | null;
  noname: string | null;
  max_message: number | null;
  updated_at: number | null;
}

export type NgKind = 'word' | 'id' | 'name' | 'wacchoi' | 'thread';
export type NgHideMode = 'abone' | 'mask';

export interface NgRule {
  id: number;
  kind: NgKind;
  pattern: string;
  /** `host/board`。null なら全板共通。 */
  scope_board: string | null;
  is_regex: number;
  hide_mode: NgHideMode;
  /** 連鎖 NG: この規則で隠れたレスへの返信も隠す。 */
  chain: number;
  /** 期限切れ (epoch ms)。null なら無期限。 */
  expires_at: number | null;
  created_at: number;
}

/** 履歴・お気に入りの一覧行。スレ一覧と同じ形で描画できるようにしてある。 */
export interface ThreadListItem {
  host: string;
  board: string;
  key: string;
  title: string;
  boardName: string | null;
  resCount: number;
  readCount: number;
  /** res_count - read_count。負にはしない。 */
  unread: number;
  myPostCount: number;
  lastOpenedAt: number | null;
  lastPostedAt: number | null;
  favorite: boolean;
}
