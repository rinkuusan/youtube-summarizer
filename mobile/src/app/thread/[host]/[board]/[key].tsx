import { Stack, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { emptyCursor, fetchDat, type DatCursor } from '@/api/dat';
import { PostItem } from '@/components/PostItem';
import { toDisplayMessage } from '@/net/errors';
import { anchorTargets, parseBody, type Segment } from '@/parse/body';
import { parseDatLine, type Post } from '@/parse/datLine';
import { colors, radius, spacing } from '@/theme/colors';

interface AnchorRef {
  from: number;
  to: number;
}

export default function ThreadScreen() {
  const { host, board, key, title } = useLocalSearchParams<{
    host: string;
    board: string;
    key: string;
    title?: string;
  }>();

  const [posts, setPosts] = useState<Post[]>([]);
  const [threadTitle, setThreadTitle] = useState<string | null>(title ?? null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<string | null>(null);
  const [anchorStack, setAnchorStack] = useState<AnchorRef[]>([]);

  const cursorRef = useRef<DatCursor>(emptyCursor);
  const listRef = useRef<FlatList<Post>>(null);

  const load = useCallback(
    async (incremental: boolean) => {
      setError(null);
      setLoading(true);
      try {
        const cursor = incremental ? cursorRef.current : emptyCursor;
        const result = await fetchDat(host, board, key, cursor);
        cursorRef.current = result.cursor;

        if (result.kind === 'full') {
          const parsed = result.lines
            .map((line, i) => parseDatLine(line, i + 1))
            .filter((p): p is Post => p !== null);
          setPosts(parsed);
          const first = result.lines[0]?.split('<>');
          if (first && first.length >= 5 && first[4]) setThreadTitle(first[4].trim());
          setStatus(
            result.refetched
              ? `あぼーんを検出したため全体を再取得しました (${parsed.length}レス)`
              : `${parsed.length}レス`
          );
        } else if (result.kind === 'append') {
          const parsed = result.lines
            .map((line, i) => parseDatLine(line, result.startRes + i))
            .filter((p): p is Post => p !== null);
          setPosts((prev) => [...prev, ...parsed]);
          setStatus(`新着 ${parsed.length}レス (差分取得)`);
        } else {
          setStatus('新着なし');
        }
      } catch (e) {
        setError(toDisplayMessage(e));
      } finally {
        setLoading(false);
      }
    },
    [host, board, key]
  );

  useEffect(() => {
    load(false);
  }, [load]);

  // 本文のトークン化はここで 1 回だけ行い、逆参照マップと描画の両方で使い回す。
  const segmentsByRes = useMemo(() => {
    const map = new Map<number, Segment[]>();
    for (const p of posts) map.set(p.res, parseBody(p.body));
    return map;
  }, [posts]);

  /** レス番号 -> そのレスに返信しているレス番号の一覧。 */
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

  const postsByRes = useMemo(() => {
    const map = new Map<number, Post>();
    for (const p of posts) map.set(p.res, p);
    return map;
  }, [posts]);

  const openAnchor = useCallback((from: number, to: number) => {
    setAnchorStack((s) => [...s, { from, to }]);
  }, []);

  const openReplies = useCallback(
    (res: number) => {
      const list = replyMap.get(res);
      if (!list || list.length === 0) return;
      // 逆参照は連続範囲とは限らないので、先頭と末尾で挟んだ範囲として開く。
      setAnchorStack((s) => [...s, { from: Math.min(...list), to: Math.max(...list) }]);
    },
    [replyMap]
  );

  const current = anchorStack[anchorStack.length - 1];
  const popupPosts = useMemo(() => {
    if (!current) return [];
    const out: Post[] = [];
    for (let n = current.from; n <= current.to && n - current.from < 100; n++) {
      const p = postsByRes.get(n);
      if (p) out.push(p);
    }
    return out;
  }, [current, postsByRes]);

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: threadTitle ?? 'スレッド' }} />

      {error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.button} onPress={() => load(false)}>
            <Text style={styles.buttonText}>再試行</Text>
          </Pressable>
        </View>
      ) : posts.length === 0 && loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <FlatList
          ref={listRef}
          data={posts}
          keyExtractor={(p) => String(p.res)}
          initialNumToRender={20}
          windowSize={11}
          removeClippedSubviews
          renderItem={({ item }) => (
            <PostItem
              post={item}
              segments={segmentsByRes.get(item.res) ?? []}
              replies={replyMap.get(item.res)}
              onAnchorPress={openAnchor}
              onRepliesPress={openReplies}
            />
          )}
        />
      )}

      <View style={styles.footer}>
        <Text style={styles.status} numberOfLines={1}>
          {loading ? '取得中...' : (status ?? '')}
        </Text>
        <Pressable
          style={styles.button}
          disabled={loading}
          onPress={() => {
            load(true);
          }}>
          <Text style={styles.buttonText}>新着取得</Text>
        </Pressable>
        <Pressable
          style={styles.buttonGhost}
          onPress={() => listRef.current?.scrollToEnd({ animated: true })}>
          <Text style={styles.buttonGhostText}>最新へ</Text>
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
              <Text style={styles.sheetTitle}>
                {current ? (current.from === current.to ? `>>${current.from}` : `>>${current.from}-${current.to}`) : ''}
              </Text>
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
  container: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: spacing.md, padding: spacing.xl },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
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
    paddingHorizontal: spacing.lg,
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
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'flex-end',
  },
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
