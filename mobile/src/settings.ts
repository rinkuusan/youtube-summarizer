/** kv テーブルに JSON で入れるアプリ設定。 */

export const SETTINGS_KEY = 'app.settings';

export interface AppSettings {
  /** 画像を自動でサムネイル表示する。既定は off (モバイル回線の通信量への配慮)。 */
  autoShowImages: boolean;
  fontSize: number;
  /** 投稿時にメール欄へ既定で sage を入れる。 */
  defaultSage: boolean;
  /** 投稿時の既定の名前。 */
  defaultName: string;
}

export const DEFAULT_SETTINGS: AppSettings = {
  autoShowImages: false,
  fontSize: 15,
  defaultSage: true,
  defaultName: '',
};

/**
 * 自動更新の最短間隔。設定の初期値ではなくコード側の下限として持つ。
 * 専ブラとして容認される作法の一部なので、ユーザーに 0 にさせない。
 */
export const MIN_RELOAD_INTERVAL_MS = 30_000;
