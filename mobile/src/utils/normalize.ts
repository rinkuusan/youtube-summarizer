import Encoding from 'encoding-japanese';

/**
 * 板名の絞り込み用の正規化。
 * カタカナ->ひらがな、全角->半角、小文字化を掛けて、表記ゆれを吸収する。
 */
export function normalizeForSearch(s: string): string {
  return Encoding.toHiraganaCase(Encoding.toHankakuCase(s)).toLowerCase();
}
