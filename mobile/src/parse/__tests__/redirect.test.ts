import { unwrapRedirect } from '../body';

/**
 * 5ch は外部リンクをリダイレクタで包む。href をそのまま使うと画像が
 * リダイレクトページを読みに行って失敗する (実機のログで確認)。
 */
describe('unwrapRedirect', () => {
  it('jump5.ch の ?URL 形式を剥がす', () => {
    expect(unwrapRedirect('http://jump5.ch/?https://i.imgur.com/HErkcMN.jpeg')).toBe(
      'https://i.imgur.com/HErkcMN.jpeg'
    );
  });

  it('jump.5ch.net でも剥がす', () => {
    expect(unwrapRedirect('https://jump.5ch.net/?https://example.com/a.png')).toBe(
      'https://example.com/a.png'
    );
  });

  it('ime.nu の古い形式は scheme を補う', () => {
    expect(unwrapRedirect('http://ime.nu/example.com/a.jpg')).toBe('http://example.com/a.jpg');
  });

  it('二重に包まれていても剥がし切る', () => {
    expect(
      unwrapRedirect('http://jump5.ch/?http://jump5.ch/?https://i.imgur.com/x.jpeg')
    ).toBe('https://i.imgur.com/x.jpeg');
  });

  it('包まれていない URL はそのまま', () => {
    expect(unwrapRedirect('https://i.imgur.com/x.jpeg')).toBe('https://i.imgur.com/x.jpeg');
    expect(unwrapRedirect('https://mi.5ch.io/test/read.cgi/news4vip/123/')).toBe(
      'https://mi.5ch.io/test/read.cgi/news4vip/123/'
    );
  });
});
