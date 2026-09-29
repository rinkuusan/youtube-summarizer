import Encoding from 'encoding-japanese';

/**
 * 表記ゆれを吸収するための正規化。板の絞り込み、検索、NG 照合で共通に使う。
 *
 * 順番に意味がある。
 *  1. toHankakuCase  全角の英数記号を半角へ (カタカナは触らない)
 *  2. toZenkanaCase  半角カタカナを全角カタカナへ
 *  3. toHiraganaCase 全角カタカナをひらがなへ
 *
 * 2 を挟まないと半角カタカナがひらがなに落ちない。「ｸﾞﾛ」と「ぐろ」が
 * 別物として扱われ、NG も検索も半角カナで書かれた分を取りこぼす。
 */
export function normalizeForSearch(s: string): string {
  return Encoding.toHiraganaCase(Encoding.toZenkanaCase(Encoding.toHankakuCase(s))).toLowerCase();
}
