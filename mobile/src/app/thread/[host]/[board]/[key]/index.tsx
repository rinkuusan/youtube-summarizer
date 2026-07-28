import { router, Stack, useLocalSearchParams } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { fetchArchivedThread } from '@/api/archived';
import { emptyCursor, fetchDat, type DatCursor } from '@/api/dat';
import { bodyMatchesMine, takePendingMine } from '@/api/post';
import { PostItem } from '@/components/PostItem';
import * as kvRepo from '@/db/kvRepo';
import * as ngRepo from '@/db/ngRepo';
import * as postRepo from '@/db/postRepo';
import * as threadRepo from '@/db/threadRepo';
import type { NgRule, ThreadRef } from '@/db/types';
import { applyNg } from '@/filter/applyNg';
import { Ch5Error, toDisplayMessage } from '@/net/errors';
import { logError } from '@/net/log';
import { Image } from 'expo-image';

import { ImageViewer } from '@/components/ImageViewer';
import { looksSensitive } from '@/filter/sensitive';
import { anchorTargets, parseBody, segmentsToPlainText, type Segment } from '@/parse/body';
import { parseDatLine, type Post } from '@/parse/datLine';
import { decodeEntities } from '@/parse/entities';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type AppSettings } from '@/settings';
import { colors, radius, spacing } from '@/theme/colors';

interface AnchorRef {
  from: number;
  to: number;
  /** ID 抽出や逆参照のように、範囲ではなく明示的な集合を出すとき。 */
  explicit?: number[];
  label: string;
}

/** 「ここまで読んだ」の区切りを混ぜた描画用リスト。 */
type ListItem = { kind: 'post'; post: Post } | { kind: 'divider'; unread: number };

export default function ThreadScreen() {
  const db = useSQLiteContext();
  const params = useLocalSearchParams<{
    host: string;
    board: string;
    key: string;
    title?: string;
    /** 履歴から「自分のレス」へ飛ぶときのレス番号。 */
    focusRes?: string;
  }>();
  const { host, board, key } = params;
  const threadRef = useMemo<ThreadRef>(() => ({ host, board, key }), [host, board, key]);

  const [posts, setPosts] = useState<Post[]>([]);
  const [threadTitle, setThreadTitle] = useState<string | null>(params.title ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<string | null>(null);
  const [anchorStack, setAnchorStack] = useState<AnchorRef[]>([]);
  const [ngRules, setNgRules] = useState<NgRule[]>([]);
  const [showNg, setShowNg] = useState(false);
  const [favorite, setFavorite] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  /** 開いた時点の既読数。区切り線が動かないよう、描画中は変えない。 */
  const [readAtOpen, setReadAtOpen] = useState(0);
  /** ポップアップ表示中の画像 URL。 */
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  /** 画像だけを並べるモード。 */
  const [imagesOnly, setImagesOnly] = useState(false);

  const cursorRef = useRef<DatCursor>(emptyCursor);
  const listRef = useRef<FlatList<ListItem>>(null);
  const postsRef = useRef<Post[]>([]);
  postsRef.current = posts;

  /**
   * 自分のレスに印を付ける。bbs.cgi はレス番号を返さないので、
   * 投稿時に控えた本文と一致するレスを後から探す。
   */
  const markMyPosts = useCallback(
    async (list: Post[]) => {
      const pending = await takePendingMine(db, threadRef);
      for (const p of pending) {
        const hit = [...list].reverse().find((post) => bodyMatchesMine(post.body, p.message));
        if (hit) await postRepo.markMine(db, threadRef, hit.res);
      }
    },
    [db, threadRef]
  );

  const fetchLatest = useCallback(
    async (incremental: boolean) => {
      setError(null);
      setLoading(true);
      try {
        const cursor = incremental ? cursorRef.current : emptyCursor;
        const result = await fetchDat(host, board, key, cursor);
        cursorRef.current = result.cursor;

        let nextPosts = postsRef.current;

        if (result.kind === 'full') {
          const parsed = result.lines
            .map((line, i) => parseDatLine(line, i + 1))
            .filter((p): p is Post => p !== null);
          // あぼーんで番号がずれているので、キャッシュは一度捨てる
          if (result.refetched) await postRepo.clear(db, threadRef);
          await postRepo.save(db, threadRef, parsed);
          nextPosts = parsed;
          setPosts(parsed);

          const first = result.lines[0]?.split('<>');
          // dat の生のスレタイには &#129781; のような文字参照が入る。
          // subject.txt 側はデコード済みなので、ここで戻さないと
          // 板一覧と履歴で同じスレのタイトルが食い違う。
          const title =
            first && first.length >= 5 && first[4] ? decodeEntities(first[4]).trim() : null;
          if (title) setThreadTitle(title);
          await threadRepo.saveCursor(db, threadRef, result.cursor, { title });
          setStatus(
            result.refetched
              ? `あぼーんを検出したため再取得 (${parsed.length}レス)`
              : `${parsed.length}レス`
          );
        } else if (result.kind === 'append') {
          const parsed = result.lines
            .map((line, i) => parseDatLine(line, result.startRes + i))
            .filter((p): p is Post => p !== null);
          await postRepo.save(db, threadRef, parsed);
          nextPosts = [...postsRef.current, ...parsed];
          setPosts(nextPosts);
          await threadRepo.saveCursor(db, threadRef, result.cursor);
          setStatus(`新着 ${parsed.length}レス (差分取得)`);
        } else {
          setStatus('新着なし');
        }

        await markMyPosts(nextPosts);
      } catch (e) {
        // dat 落ち (404) なら read.cgi から拾い直す。板から落ちただけで
        // 中身はしばらく読めるので、ここで諦めると検索結果の大半が開けなくなる。
        if (e instanceof Ch5Error && e.kind === 'notFound') {
          try {
            const archived = await fetchArchivedThread(host, board, key);
            if (archived.posts.length > 0) {
              await postRepo.save(db, threadRef, archived.posts);
              setPosts(archived.posts);
              if (archived.title) setThreadTitle(archived.title);
              setStatus(`${archived.posts.length}レス (dat 落ち: 過去ログから取得)`);
              await markMyPosts(archived.posts);
              return;
            }
          } catch (fallbackError) {
            logError('archived', fallbackError, '過去ログの取得も失敗');
          }
        }
        setError(toDisplayMessage(e));
      } finally {
        setLoading(false);
      }
    },
    [db, host, board, key, threadRef, markMyPosts]
  );

  // 初期化: キャッシュを先に出してから差分を取りに行く
  useEffect(() => {
    let cancelled = false;

    (async () => {
      setLoading(true);
      try {
        // 履歴への記録。これだけで閲覧履歴が自動的に積まれる。
        await threadRepo.touchOpened(db, threadRef, params.title ?? null);

        const [row, cached, rules, appSettings] = await Promise.all([
          threadRepo.get(db, threadRef),
          postRepo.load(db, threadRef),
          ngRepo.listForBoard(db, host, board),
          kvRepo.getJson<AppSettings>(db, SETTINGS_KEY, DEFAULT_SETTINGS),
        ]);
        if (cancelled) return;

        setNgRules(rules);
        setSettings(appSettings);
        setFavorite(row?.favorite === 1);
        setReadAtOpen(row?.read_count ?? 0);
        if (row?.title) setThreadTitle(row.title);

        if (cached.length > 0) {
          // ここまでは圏外でも読める。
          setPosts(cached);
          setStatus(`キャッシュ ${cached.length}レス`);
          cursorRef.current = {
            bytes: row?.dat_bytes ?? 0,
            etag: row?.etag ?? null,
            lastModified: row?.last_modified ?? null,
            lineCount: row?.cached_count ?? cached.length,
          };
        }
      } catch (e) {
        if (!cancelled) setError(toDisplayMessage(e));
      }

      if (!cancelled) await fetchLatest(cursorRef.current.bytes > 0);
    })();

    return () => {
      cancelled = true;
    };
    // params.title は初回の表示名にしか使わないので依存に入れない
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [db, threadRef, host, board, fetchLatest]);

  // 離脱時に既読位置を保存する
  useEffect(() => {
    return () => {
      const count = postsRef.current.length;
      if (count > 0) {
        threadRepo.updateReadState(db, threadRef, count, count).catch(() => {
          // 保存に失敗してもアプリを止める理由にはならない
        });
      }
    };
  }, [db, threadRef]);

  // --- 派生データ ---

  // 本文のトークン化はここで 1 回だけ。NG 判定・逆参照・描画で共有する。
  const segmentsByRes = useMemo(() => {
    const map = new Map<number, Segment[]>();
    for (const p of posts) map.set(p.res, parseBody(p.body));
    return map;
  }, [posts]);

  const ng = useMemo(
    () => applyNg(posts, segmentsByRes, showNg ? [] : ngRules),
    [posts, segmentsByRes, ngRules, showNg]
  );

  const replyMap = useMemo(() => {
    const map = new Map<number, number[]>();
    for (const p of posts) {
      const segs = segmentsByRes.get(p.res);
      if (!segs) continue;
      for (const target of anchorTargets(segs)) {
        const list = map.get(target);
        if (list) list.push(p.res);
        else map.set(target, [p.res]);
      }
    }
    return map;
  }, [posts, segmentsByRes]);

  const postsByRes = useMemo(() => new Map(posts.map((p) => [p.res, p])), [posts]);

  /** スレ中の画像を、出てきた順に集める (画像だけ表示モード用)。 */
  const imageList = useMemo(() => {
    const out: { url: string; res: number }[] = [];
    for (const p of ng.visible) {
      for (const seg of segmentsByRes.get(p.res) ?? []) {
        if (seg.type === 'image') out.push({ url: seg.url, res: p.res });
      }
    }
    return out;
  }, [ng.visible, segmentsByRes]);

  /**
   * ぼかすかどうかをレスごとに決める。
   * 画像の中身は見ておらず、本文やスレタイの警告語で判断している (filter/sensitive.ts)。
   */
  const shouldBlur = useCallback(
    (post: Post) => {
      if (settings.blurImages === 'never') return false;
      if (settings.blurImages === 'always') return true;
      return looksSensitive(segmentsToPlainText(segmentsByRes.get(post.res) ?? []), threadTitle);
    },
    [settings.blurImages, segmentsByRes, threadTitle]
  );

  const listData = useMemo(() => {
    const items: ListItem[] = [];
    let dividerPlaced = readAtOpen <= 0 || readAtOpen >= posts.length;
    for (const p of ng.visible) {
      if (!dividerPlaced && p.res > readAtOpen) {
        items.push({ kind: 'divider', unread: posts.length - readAtOpen });
        dividerPlaced = true;
      }
      items.push({ kind: 'post', post: p });
    }
    return items;
  }, [ng.visible, readAtOpen, posts.length]);

  /** 履歴から「自分のレス」を指定して開かれたとき、その行まで飛ぶ。 */
  const focusIndex = useMemo(() => {
    const target = Number(params.focusRes);
    if (!Number.isFinite(target) || target <= 0) return -1;
    return listData.findIndex((it) => it.kind === 'post' && it.post.res === target);
  }, [params.focusRes, listData]);

  const dividerIndex = useMemo(() => listData.findIndex((i) => i.kind === 'divider'), [listData]);

  // --- 操作 ---

  const openAnchor = useCallback((from: number, to: number) => {
    setAnchorStack((s) => [
      ...s,
      { from, to, label: from === to ? `>>${from}` : `>>${from}-${to}` },
    ]);
  }, []);

  const openReplies = useCallback(
    (res: number) => {
      const list = replyMap.get(res);
      if (!list || list.length === 0) return;
      setAnchorStack((s) => [
        ...s,
        {
          from: Math.min(...list),
          to: Math.max(...list),
          explicit: list,
          label: `>>${res} への返信`,
        },
      ]);
    },
    [replyMap]
  );

  const onIdPress = useCallback(
    (uid: string) => {
      const sameId = posts.filter((p) => p.uid === uid).map((p) => p.res);
      if (sameId.length === 0) return;
      Alert.alert(
        `ID:${uid}`,
        `このスレに ${sameId.length} 件`,
        [
          {
            text: '投稿を表示',
            onPress: () =>
              setAnchorStack((s) => [
                ...s,
                {
                  from: Math.min(...sameId),
                  to: Math.max(...sameId),
                  explicit: sameId,
                  label: `ID:${uid}`,
                },
              ]),
          },
          {
            text: 'NGID に追加',
            style: 'destructive',
            onPress: async () => {
              await ngRepo.addNgId(db, uid, `${host}/${board}`);
              setNgRules(await ngRepo.listForBoard(db, host, board));
            },
          },
          { text: 'キャンセル', style: 'cancel' },
        ],
        { cancelable: true }
      );
    },
    [db, posts, host, board]
  );

  const toggleFavorite = useCallback(async () => {
    const next = !favorite;
    setFavorite(next);
    await threadRepo.setFavorite(db, threadRef, next, threadTitle);
  }, [db, favorite, threadTitle, threadRef]);

  const current = anchorStack[anchorStack.length - 1];
  const popupPosts = useMemo(() => {
    if (!current) return [];
    if (current.explicit) {
      return current.explicit.map((n) => postsByRes.get(n)).filter((p): p is Post => p !== undefined);
    }
    const out: Post[] = [];
    for (let n = current.from; n <= current.to && n - current.from < 100; n++) {
      const p = postsByRes.get(n);
      if (p) out.push(p);
    }
    return out;
  }, [current, postsByRes]);

  const showEmpty = posts.length === 0;

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: threadTitle ?? 'スレッド',
          headerRight: () => (
            <View style={styles.headerActions}>
              {/* 画像だけ並べるモード。スレを画像ビューアとして使いたいとき用。 */}
              <Pressable onPress={() => setImagesOnly((v) => !v)} hitSlop={10}>
                <Text style={[styles.headerIcon, imagesOnly && styles.headerIconOn]}>
                  画像{imageList.length > 0 ? ` ${imageList.length}` : ''}
                </Text>
              </Pressable>
              <Pressable onPress={toggleFavorite} hitSlop={10}>
                <Text style={[styles.favStar, favorite && styles.favStarOn]}>★</Text>
              </Pressable>
            </View>
          ),
        }}
      />

      {ng.hiddenCount > 0 || ng.maskedCount > 0 || showNg ? (
        <Pressable style={styles.ngBar} onPress={() => setShowNg((v) => !v)}>
          <Text style={styles.ngBarText}>
            {showNg
              ? 'NG を一時的に解除中 — タップで戻す'
              : `NG で ${ng.hiddenCount + ng.maskedCount} 件非表示 — タップで表示`}
          </Text>
        </Pressable>
      ) : null}

      {error && showEmpty ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.button} onPress={() => fetchLatest(false)}>
            <Text style={styles.buttonText}>再試行</Text>
          </Pressable>
        </View>
      ) : showEmpty && loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : imagesOnly ? null : (
        <FlatList
          ref={listRef}
          data={listData}
          keyExtractor={(item, i) => (item.kind === 'post' ? `p${item.post.res}` : `d${i}`)}
          initialNumToRender={20}
          windowSize={11}
          removeClippedSubviews
          initialScrollIndex={focusIndex > 0 ? focusIndex : dividerIndex > 0 ? dividerIndex : undefined}
          onScrollToIndexFailed={(info) => {
            // 行の高さが可変なので一度で飛べないことがある。描画が進んでから再試行する。
            setTimeout(() => {
              listRef.current?.scrollToIndex({ index: info.index, animated: false });
            }, 200);
          }}
          renderItem={({ item }) =>
            item.kind === 'divider' ? (
              <View style={styles.divider}>
                <Text style={styles.dividerText}>ここまで読んだ（新着 {item.unread}）</Text>
              </View>
            ) : (
              <PostItem
                post={item.post}
                segments={segmentsByRes.get(item.post.res) ?? []}
                replies={replyMap.get(item.post.res)}
                onAnchorPress={openAnchor}
                onRepliesPress={openReplies}
                onIdPress={onIdPress}
                showImages={settings.autoShowImages}
                blurImages={shouldBlur(item.post)}
                onImagePress={setViewerUrl}
                fontSize={settings.fontSize}
              />
            )
          }
        />
      )}

      {imagesOnly ? (
        <FlatList
          data={imageList}
          keyExtractor={(it, i) => `${it.res}-${i}`}
          numColumns={2}
          contentContainerStyle={styles.grid}
          ListEmptyComponent={
            <Text style={styles.gridEmpty}>このスレに画像はありません。</Text>
          }
          renderItem={({ item }) => (
            <Pressable style={styles.gridCell} onPress={() => setViewerUrl(item.url)}>
              <Image
                source={{ uri: item.url }}
                style={styles.gridImage}
                contentFit="cover"
                transition={120}
                blurRadius={
                  settings.blurImages === 'always' ||
                  (settings.blurImages === 'sensitive' &&
                    looksSensitive(
                      segmentsToPlainText(segmentsByRes.get(item.res) ?? []),
                      threadTitle
                    ))
                    ? 60
                    : 0
                }
              />
              <Text style={styles.gridRes}>{item.res}</Text>
            </Pressable>
          )}
        />
      ) : null}

      <ImageViewer url={viewerUrl} onClose={() => setViewerUrl(null)} />

      {error && !showEmpty ? <Text style={styles.errorBanner}>{error}</Text> : null}

      <View style={styles.footer}>
        <Text style={styles.status} numberOfLines={1}>
          {loading ? '取得中...' : (status ?? '')}
        </Text>
        <Pressable style={styles.buttonGhost} disabled={loading} onPress={() => fetchLatest(true)}>
          <Text style={styles.buttonGhostText}>新着</Text>
        </Pressable>
        <Pressable
          style={styles.buttonGhost}
          onPress={() => listRef.current?.scrollToEnd({ animated: true })}>
          <Text style={styles.buttonGhostText}>最新へ</Text>
        </Pressable>
        <Pressable
          style={styles.button}
          onPress={() =>
            router.push({
              pathname: '/thread/[host]/[board]/[key]/post',
              params: { host, board, key, title: threadTitle ?? '' },
            })
          }>
          <Text style={styles.buttonText}>書き込む</Text>
        </Pressable>
      </View>

      <Modal
        visible={anchorStack.length > 0}
        transparent
        animationType="fade"
        onRequestClose={() => setAnchorStack((s) => s.slice(0, -1))}>
        <Pressable style={styles.backdrop} onPress={() => setAnchorStack([])}>
          <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
            <View style={styles.sheetHeader}>
              <Pressable onPress={() => setAnchorStack((s) => s.slice(0, -1))} hitSlop={8}>
                <Text style={styles.sheetBack}>{anchorStack.length > 1 ? '‹ 戻る' : '閉じる'}</Text>
              </Pressable>
              <Text style={styles.sheetTitle}>{current?.label ?? ''}</Text>
            </View>
            <ScrollView>
              {popupPosts.length === 0 ? (
                <Text style={styles.sheetEmpty}>まだ取得していないレスです</Text>
              ) : (
                popupPosts.map((p) => (
                  <PostItem
                    key={p.res}
                    post={p}
                    segments={segmentsByRes.get(p.res) ?? []}
                    replies={replyMap.get(p.res)}
                    onAnchorPress={openAnchor}
                    onRepliesPress={openReplies}
                    onIdPress={onIdPress}
                    showImages={settings.autoShowImages}
                    blurImages={shouldBlur(p)}
                    onImagePress={setViewerUrl}
                    fontSize={settings.fontSize}
                  />
                ))
              )}
            </ScrollView>
          </Pressable>
        </Pressable>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  headerActions: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  headerIcon: { color: colors.textDim, fontSize: 12 },
  headerIconOn: { color: colors.accentHover, fontWeight: '700' },
  grid: { padding: spacing.sm },
  gridCell: { flex: 1 / 2, margin: spacing.xs, aspectRatio: 1 },
  gridImage: { width: '100%', height: '100%', borderRadius: radius / 2, backgroundColor: colors.surface2 },
  gridRes: {
    position: 'absolute',
    left: spacing.xs,
    bottom: spacing.xs,
    color: colors.text,
    fontSize: 11,
    backgroundColor: '#000000aa',
    paddingHorizontal: 4,
    borderRadius: 4,
  },
  gridEmpty: { color: colors.textDim, textAlign: 'center', padding: spacing.xl },
  container: { flex: 1, backgroundColor: colors.bg },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
    padding: spacing.xl,
  },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
  errorBanner: {
    color: colors.error,
    fontSize: 11,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.xs,
    backgroundColor: colors.surface2,
  },
  favStar: { color: colors.textDim, fontSize: 20 },
  favStarOn: { color: colors.accent },
  ngBar: {
    backgroundColor: colors.surface2,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  ngBarText: { color: colors.textDim, fontSize: 11 },
  divider: { backgroundColor: colors.surface2, paddingVertical: spacing.xs, alignItems: 'center' },
  dividerText: { color: colors.success, fontSize: 11, fontWeight: '700' },
  footer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
    backgroundColor: colors.surface,
  },
  status: { color: colors.textDim, fontSize: 11, flex: 1 },
  button: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius,
    backgroundColor: colors.accent,
  },
  buttonText: { color: '#fff', fontSize: 13, fontWeight: '700' },
  buttonGhost: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius,
    backgroundColor: colors.surface2,
  },
  buttonGhostText: { color: colors.text, fontSize: 13 },
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.6)', justifyContent: 'flex-end' },
  sheet: {
    maxHeight: '75%',
    backgroundColor: colors.surface,
    borderTopLeftRadius: radius,
    borderTopRightRadius: radius,
    paddingBottom: spacing.xl,
  },
  sheetHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  sheetBack: { color: colors.accent, fontSize: 14 },
  sheetTitle: { color: colors.textDim, fontSize: 13 },
  sheetEmpty: { color: colors.textDim, fontSize: 13, padding: spacing.xl, textAlign: 'center' },
});
