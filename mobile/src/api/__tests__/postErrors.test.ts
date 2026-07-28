import { extractFormAction, extractFormFields, classifyPostResponse, extractCooldownSeconds, extractText } from '../postErrors';

/** bbs.cgi の応答は常に HTTP 200 で、成否は <title> に出る。 */
function page(title: string, body = ''): string {
  return `<html><head><title>${title}</title></head><body>${body}</body></html>`;
}

describe('postErrors', () => {
  it('成功を判別する', () => {
    expect(classifyPostResponse(page('書きこみました。')).outcome).toBe('success');
    expect(classifyPostResponse(page('書き込みました')).outcome).toBe('success');
  });

  it('書き込み確認を判別し、本文をそのまま保持する', () => {
    const r = classifyPostResponse(page('書き込み確認', '<div>ここに注意書き</div>'));
    expect(r.outcome).toBe('confirm');
    expect(r.action).toBe('confirm');
    expect(r.bodyText).toContain('ここに注意書き');
  });

  it('SAMBA24 を判別する', () => {
    const r = classifyPostResponse(page('ERROR: 短時間に書き込みすぎです'));
    expect(r.outcome).toBe('cooldown');
    expect(r.action).toBe('countdown');
  });

  it('IP 規制を判別し、回線切替を案内する', () => {
    const r = classifyPostResponse(page('ERROR: 規制中です'));
    expect(r.outcome).toBe('banned');
    expect(r.action).toBe('switchNetwork');
    expect(r.message).toContain('回線');
  });

  it('どんぐりを判別してブラウザへ誘導する', () => {
    const r = classifyPostResponse(page('ERROR', 'どんぐりを植えてください'));
    expect(r.outcome).toBe('donguri');
    expect(r.action).toBe('openBrowser');
  });

  it('二重書き込み・長すぎを判別する', () => {
    expect(classifyPostResponse(page('ERROR: 二重書き込みです')).outcome).toBe('duplicate');
    expect(classifyPostResponse(page('ERROR: 本文が長すぎます')).outcome).toBe('tooLong');
  });

  it('HTTP 451/403 はエッジ遮断として扱う', () => {
    const r = classifyPostResponse('<html></html>', 451);
    expect(r.outcome).toBe('blocked');
    expect(r.action).toBe('openBrowser');
  });

  it('Cloudflare のチャレンジを判別する', () => {
    const r = classifyPostResponse(page('Just a moment...'));
    expect(r.outcome).toBe('blocked');
  });

  it('判別できない応答は unknown にして中身をそのまま見せる', () => {
    const r = classifyPostResponse(page('なにか未知のタイトル', '未知の本文'));
    expect(r.outcome).toBe('unknown');
    expect(r.bodyText).toContain('未知の本文');
    expect(r.html).toContain('未知のタイトル');
  });

  it('本文から待ち秒数を拾う', () => {
    expect(extractCooldownSeconds('あと 42 秒お待ちください')).toBe(42);
    expect(extractCooldownSeconds('秒数の記載なし')).toBeNull();
  });

  it('extractText がタグを落として改行にする', () => {
    expect(extractText('<p>あ</p><p>い</p>')).toBe('あ\nい');
    expect(extractText('a<br>b')).toBe('a\nb');
    expect(extractText('<script>var x=1</script>本文')).toBe('本文');
  });
});

describe('extractFormFields', () => {
  // 確認ページは feature のような使い捨てトークンを返させる。これを送り返さないと
  // 何度承諾しても確認ページが返り続ける (実測 2026-07-28 kizuna.5ch.io/gamefight)。
  const confirm = `<html><head><title>■ 書き込み確認 ■</title></head><body>
    <form method="POST" action="../test/bbs.cgi">
    <input type="hidden" name="feature" value="a1b2&amp;c3">
    <input type="submit" name="submit" value="上記全てを承諾して書き込む">
    </form></body></html>`;

  it('name/value を集め、エンティティを戻す', () => {
    expect(extractFormFields(confirm)).toEqual({
      feature: 'a1b2&c3',
      submit: '上記全てを承諾して書き込む',
    });
  });

  it('属性の順が逆でも拾う', () => {
    expect(extractFormFields('<input value="x" name="feature">')).toEqual({ feature: 'x' });
  });

  it('name の無い input は無視する', () => {
    expect(extractFormFields('<input type="text"><input name="a" value="1">')).toEqual({ a: '1' });
  });

  it('分類結果に formFields が乗る', () => {
    const r = classifyPostResponse(confirm);
    expect(r.outcome).toBe('confirm');
    expect(r.formFields.feature).toBe('a1b2&c3');
    // submit 文字列はこちらで推測せず、ページのものを使う
    expect(r.formFields.submit).toBe('上記全てを承諾して書き込む');
  });
});

describe('確認ページの実物の書き方', () => {
  // 5ch はクォート有無を混在させる。name= が裸なので、二重引用符だけを見ていると
  // FROM/mail/MESSAGE を丸ごと取りこぼす (実測 2026-07-28)。
  const real =
    '<html><head><title>■ 書き込み確認 ■</title></head><body>' +
    '<form method=POST action="../test/bbs.cgi?guid=ON">' +
    '<input type=hidden name=FROM value="">' +
    '<input type=hidden name=mail value="sage">' +
    '<input type=hidden name=MESSAGE value="本文">' +
    '<input type=hidden name=bbs value=gamefight>' +
    '<input type=submit value="上記全てを承諾して書き込む" name=submit>' +
    '</form></body></html>';

  it('クォート無しの name/value も拾う', () => {
    expect(extractFormFields(real)).toEqual({
      FROM: '',
      mail: 'sage',
      MESSAGE: '本文',
      bbs: 'gamefight',
      submit: '上記全てを承諾して書き込む',
    });
  });

  it('送信先をクエリごと取る', () => {
    expect(extractFormAction(real)).toBe('../test/bbs.cgi?guid=ON');
  });

  it('分類結果に両方乗る', () => {
    const r = classifyPostResponse(real);
    expect(r.outcome).toBe('confirm');
    expect(r.formAction).toContain('guid=ON');
    expect(r.formFields.MESSAGE).toBe('本文');
  });
});

describe('余所でやってくれ', () => {
  // 規制と紛らわしいが、こちらの送り方 (Referer) の問題でも出るので分けて扱う。
  it('規制ではなく wrongReferer として分類する', () => {
    const r = classifyPostResponse(page('ERROR', 'ERROR:余所でやってくれ'));
    expect(r.outcome).toBe('wrongReferer');
  });

  it('本物の規制は banned のまま', () => {
    expect(classifyPostResponse(page('ERROR', 'ホストが規制されています')).outcome).toBe('banned');
  });
});
