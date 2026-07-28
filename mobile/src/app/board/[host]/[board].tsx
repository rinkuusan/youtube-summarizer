import { router, Stack, useFocusEffect, useLocalSearchParams } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  FlatList,
  Pressable,
  RefreshControl,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { computeMomentum, formatMomentum } from '@/api/momentum';
import { fetchThreadList, type ThreadSummary } from '@/api/subject';
import * as threadRepo from '@/db/threadRepo';
import type { ThreadListItem } from '@/db/types';
import { toDisplayMessage } from '@/net/errors';
import { colors, radius, spacing } from '@/theme/colors';
import { normalizeForSearch } from '@/utils/normalize';

type SortKey = 'momentum' | 'new' | 'res' | 'created';

const SORTS: { key: SortKey; label: string }[] = [
  { key: 'momentum', label: '勢い' },
  { key: 'new', label: '新着' },
  { key: 'res', label: 'レス数' },
  { key: 'created', label: '新スレ順' },
];

export default function ThreadListScreen() {
  const db = useSQLiteContext();
  const { host, board, name } = useLocalSearchParams<{ host: string; board: string; name?: string }>();
  const [threads, setThreads] = useState<ThreadSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [sort, setSort] = useState<SortKey>('momentum');
  const [query, setQuery] = useState('');
  /** 既読・お気に入りの状態。スレ一覧に新着バッジを出すのに使う。 */
  const [known, setKnown] = useState<Map<string, ThreadListItem>>(new Map());

  const loadKnown = useCallback(async () => {
    const [opened, favs] = await Promise.all([
      threadRepo.listOpenedHistory(db, 1000),
      threadRepo.listFavorites(db),
    ]);
    const map = new Map<string, ThreadListItem>();
    for (const t of [...opened, ...favs]) {
      if (t.host === host && t.board === board) map.set(t.key, t);
    }
    setKnown(map);
  }, [db, host, board]);

  // スレを読んで戻ってきたら既読状態を反映する
  useFocusEffect(
    useCallback(() => {
      loadKnown();
    }, [loadKnown])
  );

  const toggleFavorite = useCallback(
    async (t: ThreadSummary) => {
      const ref = { host, board, key: t.key };
      const isFav = known.get(t.key)?.favorite ?? false;
      await threadRepo.setFavorite(db, ref, !isFav, t.title);
      await loadKnown();
      Alert.alert(t.title, isFav ? 'お気に入りから外しました' : 'お気に入りに追加しました');
    },
    [db, host, board, known, loadKnown]
  );

  const load = useCallback(
    async (isRefresh = false) => {
      setError(null);
      if (isRefresh) setRefreshing(true);
      else setThreads(null);
      try {
        setThreads(await fetchThreadList(host, board));
      } catch (e) {
        setError(toDisplayMessage(e));
      } finally {
        setRefreshing(false);
      }
    },
    [host, board]
  );

  useEffect(() => {
    load();
  }, [load]);

  const visible = useMemo(() => {
    if (!threads) return [];
    const now = Date.now();
    const q = normalizeForSearch(query.trim());
    const filtered = q ? threads.filter((t) => normalizeForSearch(t.title).includes(q)) : threads;

    const sorted = [...filtered];
    switch (sort) {
      case 'momentum':
        sorted.sort((a, b) => computeMomentum(b.key, b.resCount, now) - computeMomentum(a.key, a.resCount, now));
        break;
      case 'res':
        sorted.sort((a, b) => b.resCount - a.resCount);
        break;
      case 'created':
        sorted.sort((a, b) => Number(b.key) - Number(a.key));
        break;
      case 'new':
        // subject.txt の並び順が板の「新着順」そのもの
        break;
    }
    return sorted;
  }, [threads, sort, query]);

  return (
    <View style={styles.container}>
      <Stack.Screen
        options={{
          title: name ?? board,
          headerRight: () => (
            <Pressable
              hitSlop={12}
              onPress={() =>
                router.push({
                  pathname: '/newthread',
                  params: { host, board, name: name ?? board },
                })
              }>
              <Text style={styles.newThread}>＋ スレ立て</Text>
            </Pressable>
          ),
        }}
      />

      <View style={styles.toolbar}>
        <View style={styles.sortRow}>
          {SORTS.map((s) => (
            <Pressable
              key={s.key}
              onPress={() => setSort(s.key)}
              style={[styles.sortChip, sort === s.key && styles.sortChipActive]}>
              <Text style={[styles.sortText, sort === s.key && styles.sortTextActive]}>{s.label}</Text>
            </Pressable>
          ))}
        </View>
        <TextInput
          style={styles.search}
          placeholder="スレタイを絞り込む"
          placeholderTextColor={colors.textDim}
          value={query}
          onChangeText={setQuery}
          autoCorrect={false}
          autoCapitalize="none"
          clearButtonMode="while-editing"
        />
      </View>

      {error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retry} onPress={() => load()}>
            <Text style={styles.retryText}>再試行</Text>
          </Pressable>
        </View>
      ) : !threads ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : (
        <FlatList
          data={visible}
          keyExtractor={(t) => t.key}
          refreshControl={
            <RefreshControl refreshing={refreshing} onRefresh={() => load(true)} tintColor={colors.accent} />
          }
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.dim}>該当するスレッドがありません</Text>
            </View>
          }
          renderItem={({ item, index }) => {
            const state = known.get(item.key);
            const unread = state ? Math.max(0, item.resCount - state.readCount) : 0;
            return (
              <Pressable
                style={styles.row}
                onPress={() =>
                  router.push({
                    pathname: '/thread/[host]/[board]/[key]',
                    params: { host, board, key: item.key, title: item.title },
                  })
                }
                onLongPress={() => toggleFavorite(item)}>
                <Text style={styles.index}>{index + 1}</Text>
                <View style={styles.rowBody}>
                  <Text style={[styles.title, state && styles.titleRead]} numberOfLines={2}>
                    {item.title}
                  </Text>
                  <View style={styles.metaRow}>
                    <Text style={styles.res}>{item.resCount}レス</Text>
                    <Text style={styles.momentum}>
                      勢い {formatMomentum(computeMomentum(item.key, item.resCount))}
                    </Text>
                    {state && unread > 0 ? <Text style={styles.unread}>新着 {unread}</Text> : null}
                    {state?.favorite ? <Text style={styles.fav}>★</Text> : null}
                  </View>
                </View>
              </Pressable>
            );
          }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  newThread: { color: colors.accentHover, fontSize: 13, fontWeight: '700' },
  container: { flex: 1, backgroundColor: colors.bg },
  center: { padding: spacing.xl, alignItems: 'center', justifyContent: 'center', gap: spacing.md },
  dim: { color: colors.textDim, fontSize: 13 },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
  retry: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius,
    backgroundColor: colors.surface2,
  },
  retryText: { color: colors.text },
  toolbar: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.sm,
    gap: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  sortRow: { flexDirection: 'row', gap: spacing.sm },
  sortChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: 999,
    backgroundColor: colors.surface,
  },
  sortChipActive: { backgroundColor: colors.accent },
  sortText: { color: colors.textDim, fontSize: 12 },
  sortTextActive: { color: '#fff', fontWeight: '700' },
  search: {
    backgroundColor: colors.surface,
    borderRadius: radius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 14,
  },
  row: {
    flexDirection: 'row',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  index: { color: colors.textDim, fontSize: 12, minWidth: 24, fontVariant: ['tabular-nums'] },
  rowBody: { flex: 1, gap: spacing.xs },
  title: { color: colors.text, fontSize: 15, lineHeight: 21 },
  /** 一度開いたスレは少し落とす。未読との区別が付くように。 */
  titleRead: { color: colors.textDim },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md },
  res: { color: colors.textDim, fontSize: 11 },
  momentum: { color: colors.accentHover, fontSize: 11 },
  unread: { color: colors.success, fontSize: 11, fontWeight: '700' },
  fav: { color: colors.accent, fontSize: 11 },
});
