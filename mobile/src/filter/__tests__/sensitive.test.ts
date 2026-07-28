import { looksSensitive } from '../sensitive';

/**
 * 画像の中身は見ていない。5ch 側が書く警告を拾う方式なので、
 * 「警告を拾えること」と「普通のレスをぼかさないこと」の両方を固定する。
 */
describe('looksSensitive', () => {
  it('本文の警告を拾う', () => {
    expect(looksSensitive('閲覧注意 https://i.imgur.com/x.jpg')).toBe(true);
    expect(looksSensitive('グロ注意な')).toBe(true);
    expect(looksSensitive('心臓の弱い人は見るな')).toBe(true);
  });

  it('カタカナ・ひらがな・全角半角のゆれを吸収する', () => {
    expect(looksSensitive('ぐろちゅうい')).toBe(true);
    expect(looksSensitive('ｸﾞﾛ画像')).toBe(true);
  });

  it('スレタイ側の警告でも効く (以降のレスには書かれないことが多いため)', () => {
    expect(looksSensitive('ほい', '【閲覧注意】事故現場の画像')).toBe(true);
  });

  it('普通のレスはぼかさない', () => {
    expect(looksSensitive('猫かわいい https://i.imgur.com/x.jpg')).toBe(false);
    expect(looksSensitive('ラーメン食いたい', '今日の晩飯スレ')).toBe(false);
  });

  it('話題語だけでは反応しない (ニュース系が軒並みぼけないように)', () => {
    expect(looksSensitive('事故で死亡したらしい', 'ニュース速報')).toBe(false);
  });
});
