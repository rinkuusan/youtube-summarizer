import { fetchDat } from '../dat';
import { Ch5Error } from '../../net/errors';
import { fetchBytes } from '../../net/http';

jest.mock('../../net/http', () => ({
  fetchBytes: jest.fn(),
  getHeader: jest.requireActual('../../net/http').getHeader,
}));

const mocked = fetchBytes as jest.MockedFunction<typeof fetchBytes>;

function res(bytes: Uint8Array, status = 200) {
  return { status, bytes, headers: [] as [string, string][], url: 'u', redirected: false };
}

const LINE = '名無し<><>2026/07/29(水) 03:00:00.000 ID:abc<>本文<>スレタイ\n';

describe('fetchDat: 200 なのに 0 バイト', () => {
  beforeEach(() => mocked.mockReset());

  // 5ch は不安定な瞬間に、実在するスレへ 200 + 0 バイトを返すことがある。
  // これを「スレが空になった」と解釈すると取得済みキャッシュを消してしまう。
  it('初回取得では異常として投げる (空スレとして扱わない)', async () => {
    mocked.mockResolvedValue(res(new Uint8Array(0)));
    await expect(fetchDat('h', 'b', 'k')).rejects.toMatchObject({ kind: 'server' });
    await expect(fetchDat('h', 'b', 'k')).rejects.toBeInstanceOf(Ch5Error);
  });

  it('差分取得中に 200 で空が返っても投げる (再取得扱いにしない)', async () => {
    // カーソルがある状態 = Range 要求。If-Range 不一致で 200 が返る経路。
    mocked.mockResolvedValue(res(new Uint8Array(0), 200));
    const cursor = { bytes: 100, etag: null, lastModified: null, lineCount: 3 };
    await expect(fetchDat('h', 'b', 'k', cursor)).rejects.toMatchObject({ kind: 'server' });
  });

  it('中身があれば通常どおり全体取得になる', async () => {
    mocked.mockResolvedValue(res(new TextEncoder().encode(LINE)));
    const r = await fetchDat('h', 'b', 'k');
    expect(r.kind).toBe('full');
    expect(r.lines).toHaveLength(1);
  });
});
