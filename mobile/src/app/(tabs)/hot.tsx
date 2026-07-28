import { router, useFocusEffect } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { fetchHotThreads, DEFAULT_HOT_BOARDS, type HotProgress, type HotThread } from '@/api/hot';
import { formatMomentum } from '@/api/momentum';
import * as boardRepo from '@/db/boardRepo';
import * as recentBoards from '@/db/recentBoards';
import { toDisplayMessage } from '@/net/errors';
import { colors, spacing } from '@/theme/colors';

/** 一覧に出す件数。 */
const LIMIT = 150;

/**
 * 新着。板をまたいで「今伸びてるスレ」を出す。
 * 対象は 主要板 + お気に入り板 + 最近見た板 (api/hot.ts 参照)。
 */
export default function HotScreen() {
  const db = useSQLiteContext();
  const [threads, setThreads] = useState<HotThread[] | null>(null);
  const [progress, setProgress] = useState<HotProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const load = useCallback(
    async (isRefresh = false) => {
      abortRef.current?.abort();
      const ac = new AbortController();
      abortRef.current = ac;

      setError(null);
      if (isRefresh) setRefreshing(true);

      try {
        // 既定の主要板に、お気に入りと最近見た板を足す。板一覧から名前も引く。
        const [favs, recents, allBoards] = await Promise.all([
          boardRepo.listFavoriteBoards(db),
          recentBoards.list(db),
          boardRepo.loadAll(db),
        ]);
        const byId = new Map(allBoards.map((b) => [`${b.host}/${b.id}`, b.name]));

        const seen = new Set<string>();
        const targets: { host: string; board: string; name: string }[] = [];
        const add = (host: string, board: string, name?: string) => {
          const id = `${host}/${board}`;
          if (seen.has(id)) return;
          seen.add(id);
          targets.push({ host, board, name: name ?? byId.get(id) ?? board });
        };

        for (const f of favs) add(f.host, f.id, f.name);
        for (const r of recents) add(r.host, r.board, r.name);
        // 既定の板は板一覧から host を引く。見つからないものは飛ばす。
        for (const b of allBoards) {
          if (DEFAULT_HOT_BOARDS.includes(b.id)) add(b.host, b.id, b.name);
        }

        if (targets.length === 0) {
          setError('板一覧をまだ取得していません。板一覧タブを一度開いてください。');
          return;
        }

        setProgress({ done: 0, total: targets.length, failed: 0 });
        setThreads(await fetchHotThreads(targets, LIMIT, setProgress, ac.signal));
      } catch (e) {
        setError(toDisplayMessage(e));
      } finally {
        setRefreshing(false);
      }
    },
    [db]
  );

  useFocusEffect(
    useCallback(() => {
      if (threads === null) load();
    }, [threads, load])
  );

  if (error && !threads) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
        <Pressable style={styles.retry} onPress={() => load()}>
          <Text style={styles.retryText}>再試行</Text>
        </Pressable>
      </View>
    );
  }

  if (!threads) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.dim}>
          {progress ? `${progress.done}/${progress.total} 板を確認中...` : '取得中...'}
        </Text>
      </View>
    );
  }

  return (
    <FlatList
      style={styles.container}
      data={threads}
      keyExtractor={(t) => `${t.host}/${t.board}/${t.key}`}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => load(true)}
          tintColor={colors.accent}
        />
      }
      ListHeaderComponent={
        <Text style={styles.count}>
          {threads.length} スレ
          {progress ? ` ・ ${progress.total} 板から` : ''}
          {progress && progress.failed > 0 ? `（${progress.failed} 板取得失敗）` : ''}
        </Text>
      }
      renderItem={({ item, index }) => (
        <Pressable
          style={styles.row}
          onPress={() =>
            router.push({
              pathname: '/thread/[host]/[board]/[key]',
              params: {
                host: item.host,
                board: item.board,
                key: item.key,
                title: item.title,
              },
            })
          }>
          <Text style={styles.rank}>{index + 1}</Text>
          <View style={styles.body}>
            <Text style={styles.title} numberOfLines={2}>
              {item.title}
            </Text>
            <View style={styles.metaRow}>
              <Text style={styles.board}>{item.boardName}</Text>
              <Text style={styles.meta}>{item.resCount}レス</Text>
              <Text style={styles.momentum}>勢い {formatMomentum(item.momentum)}</Text>
            </View>
          </View>
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.md,
    padding: spacing.xl,
    backgroundColor: colors.bg,
  },
  dim: { color: colors.textDim, fontSize: 13 },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
  retry: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface2,
    borderRadius: 12,
  },
  retryText: { color: colors.text },
  count: {
    color: colors.textDim,
    fontSize: 11,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  rank: {
    color: colors.textDim,
    fontSize: 12,
    fontVariant: ['tabular-nums'],
    minWidth: 24,
    textAlign: 'right',
  },
  body: { flex: 1, gap: spacing.xs },
  title: { color: colors.text, fontSize: 15, lineHeight: 21 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md },
  board: { color: colors.accentHover, fontSize: 11 },
  meta: { color: colors.textDim, fontSize: 11 },
  momentum: { color: colors.success, fontSize: 11 },
});
