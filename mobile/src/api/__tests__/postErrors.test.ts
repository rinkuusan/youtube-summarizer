import { classifyPostResponse, extractCooldownSeconds, extractText } from '../postErrors';

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
