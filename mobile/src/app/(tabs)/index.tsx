import { router, useFocusEffect } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  SectionList,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { fetchBoards, groupByCategory, type Board } from '@/api/bbsmenu';
import * as boardRepo from '@/db/boardRepo';
import * as recentBoards from '@/db/recentBoards';
import { toDisplayMessage } from '@/net/errors';
import { colors, radius, spacing } from '@/theme/colors';
import { normalizeForSearch } from '@/utils/normalize';

export default function BoardListScreen() {
  const db = useSQLiteContext();
  const [boards, setBoards] = useState<Board[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [recent, setRecent] = useState<recentBoards.RecentBoard[]>([]);

  /**
   * bbsmenu.json は 180KB あるので毎回は取りに行かない。
   * キャッシュがあれば先に出し、古ければ裏で取り直す。圏外でも板一覧は出る。
   */
  const load = useCallback(
    async (force = false) => {
      setError(null);
      try {
        const cached = await boardRepo.loadAll(db);
        if (cached.length > 0) setBoards(cached);
        else setBoards(null);

        if (force || cached.length === 0 || !(await boardRepo.isFresh(db))) {
          const fresh = await fetchBoards();
          await boardRepo.saveAll(db, fresh);
          setBoards(fresh);
        }
      } catch (e) {
        // キャッシュが出せているならエラーで潰さない
        setBoards((prev) => {
          if (!prev || prev.length === 0) setError(toDisplayMessage(e));
          return prev;
        });
      }
    },
    [db]
  );

  useEffect(() => {
    load();
  }, [load]);

  // 「最近見た板」は板を開くたびに変わるので、画面に戻るたびに読み直す。
  useFocusEffect(
    useCallback(() => {
      recentBoards.list(db).then(setRecent).catch(() => undefined);
    }, [db])
  );

  const toggleFavoriteBoard = useCallback(
    async (b: Board) => {
      const isFav = await boardRepo.isFavoriteBoard(db, b.host, b.id);
      await boardRepo.setFavoriteBoard(db, b.host, b.id, !isFav);
      Alert.alert(b.name, isFav ? 'お気に入りから外しました' : 'お気に入りに追加しました');
    },
    [db]
  );

  /** 最近見た板を Board の形に戻す。板一覧に無い板は落とす。 */
  const recentSection = useMemo(() => {
    if (!boards || recent.length === 0) return null;
    const byId = new Map(boards.map((b) => [`${b.host}/${b.id}`, b]));
    const data = recent
      .map((r) => byId.get(`${r.host}/${r.board}`))
      .filter((b): b is Board => b !== undefined);
    if (data.length === 0) return null;
    return { title: '最近見た板', data, count: data.length };
  }, [boards, recent]);

  const sections = useMemo(() => {
    if (!boards) return [];
    const q = normalizeForSearch(query.trim());
    const grouped = groupByCategory(boards);

    if (!q) {
      // 最近見た板は常に先頭。カテゴリの開閉とは独立に扱う。
      const head = recentSection && !collapsed[recentSection.title] ? [recentSection] : [];
      const headCollapsed =
        recentSection && collapsed[recentSection.title]
          ? [{ ...recentSection, data: [] as Board[] }]
          : [];
      return [
        ...head,
        ...headCollapsed,
        ...grouped.map((g) => ({
          title: g.name,
          data: collapsed[g.name] ? [] : g.boards,
          count: g.boards.length,
        })),
      ];
    }

    // 絞り込み中はカテゴリの開閉を無視して、一致した板だけ出す。
    return grouped
      .map((g) => ({
        title: g.name,
        data: g.boards.filter((b) => normalizeForSearch(b.name).includes(q) || b.id.includes(q)),
      }))
      .filter((g) => g.data.length > 0)
      .map((g) => ({ ...g, count: g.data.length }));
  }, [boards, query, collapsed, recentSection]);

  if (error) {
    return (
      <View style={styles.center}>
        <Text style={styles.errorText}>{error}</Text>
        <Pressable style={styles.retry} onPress={() => load(true)}>
          <Text style={styles.retryText}>再試行</Text>
        </Pressable>
      </View>
    );
  }

  if (!boards) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color={colors.accent} />
        <Text style={styles.dim}>板一覧を取得中...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.searchWrap}>
        <TextInput
          style={styles.search}
          placeholder="板を絞り込む"
          placeholderTextColor={colors.textDim}
          value={query}
          onChangeText={setQuery}
          autoCorrect={false}
          autoCapitalize="none"
          clearButtonMode="while-editing"
        />
      </View>

      <SectionList
        sections={sections}
        keyExtractor={(item) => `${item.host}/${item.id}`}
        stickySectionHeadersEnabled
        renderSectionHeader={({ section }) => (
          <Pressable
            onPress={() => setCollapsed((c) => ({ ...c, [section.title]: !c[section.title] }))}
            style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>{section.title}</Text>
            <Text style={styles.sectionCount}>{section.count}</Text>
          </Pressable>
        )}
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() =>
              router.push({
                pathname: '/board/[host]/[board]',
                params: { host: item.host, board: item.id, name: item.name },
              })
            }
            onLongPress={() => toggleFavoriteBoard(item)}>
            <Text style={styles.boardName}>{item.name}</Text>
            <Text style={styles.boardId}>{item.id}</Text>
          </Pressable>
        )}
      />
    </View>
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
    borderRadius: radius,
    backgroundColor: colors.surface2,
  },
  retryText: { color: colors.text },
  searchWrap: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.bg,
  },
  search: {
    backgroundColor: colors.surface,
    borderRadius: radius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 15,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.surface2,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  sectionTitle: { color: colors.accentHover, fontSize: 13, fontWeight: '700' },
  sectionCount: { color: colors.textDim, fontSize: 12 },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  boardName: { color: colors.text, fontSize: 15, flexShrink: 1 },
  boardId: { color: colors.textDim, fontSize: 11 },
});
