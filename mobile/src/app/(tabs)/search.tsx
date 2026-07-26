import { router, useLocalSearchParams } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  collectAllTargets,
  searchTargets,
  type PostHit,
  type SearchProgress,
} from '@/api/fulltext';
import { searchThreads, SEARCH_RESULT_LIMIT, type SearchHit } from '@/api/search';
import { toDisplayMessage } from '@/net/errors';
import { colors, radius, spacing } from '@/theme/colors';

/**
 * 検索タブ。
 *
 * - title: find.5ch のスレタイ検索。返却をそのまま出さず、タイトルを突き合わせて絞る
 *   (find.5ch は語を分割して OR 検索するので、素のままだと複合語が壊滅する。api/search.ts 参照)
 * - body:  候補スレの dat を落として本文を検索する。5ch には本文の横断検索 API が無いので、
 *   スレタイ検索の結果を種にして総当たりする
 */

type Mode = 'title' | 'body';

/** 本文検索で dat を落とすスレ数。find.5ch が返す 100 件を全部掘る。 */
const BODY_SCAN_LIMIT = 100;

export default function SearchScreen() {
  const { q } = useLocalSearchParams<{ q?: string }>();

  const [mode, setMode] = useState<Mode>('title');
  const [query, setQuery] = useState(q ?? '');
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  /** find.5ch が返した総数。何件を無関係として捨てたか出すために持つ。 */
  const [rawCount, setRawCount] = useState(0);
  const [postHits, setPostHits] = useState<PostHit[] | null>(null);
  const [progress, setProgress] = useState<SearchProgress | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    const trimmed = query.trim();
    if (!trimmed || loading) return;

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setLoading(true);
    setError(null);
    setHits(null);
    setPostHits(null);
    setProgress(null);

    try {
      if (mode === 'title') {
        const { all, matched } = await searchThreads(trimmed, ac.signal);
        setRawCount(all.length);
        setHits(matched);
      } else {
        const targets = await collectAllTargets(trimmed, BODY_SCAN_LIMIT, ac.signal);
        setProgress({ done: 0, total: targets.length, hits: 0, failed: 0 });
        setPostHits(await searchTargets(targets, trimmed, setProgress, ac.signal));
      }
    } catch (e) {
      setError(toDisplayMessage(e));
    } finally {
      setLoading(false);
    }
  }, [query, loading, mode]);

  // 初期クエリ付きで開かれたら (gochviewer:///search?q=...) 1 回だけ走らせる。
  const autoRan = useRef(false);
  useEffect(() => {
    if (!q || autoRan.current) return;
    autoRan.current = true;
    run();
  }, [q, run]);

  const openThread = useCallback(
    (h: { host: string; board: string; key: string; title: string }) => {
      router.push({
        pathname: '/thread/[host]/[board]/[key]',
        params: { host: h.host, board: h.board, key: h.key, title: h.title },
      });
    },
    []
  );

  const header = useMemo(() => {
    if (hits) {
      const dropped = rawCount - hits.length;
      return (
        <Text style={styles.count}>
          {hits.length} 件
          {dropped > 0 ? `（find.5ch の ${rawCount} 件から、タイトルに含まない ${dropped} 件を除外）` : ''}
          {rawCount >= SEARCH_RESULT_LIMIT ? ' ・ find.5ch の返却上限に到達' : ''}
        </Text>
      );
    }
    if (postHits && progress) {
      return (
        <Text style={styles.count}>
          {postHits.length} レス / {progress.done} スレ走査
          {progress.failed > 0 ? `（${progress.failed} スレ取得失敗）` : ''}
        </Text>
      );
    }
    return null;
  }, [hits, rawCount, postHits, progress]);

  return (
    <View style={styles.container}>
      <View style={styles.modeRow}>
        {(
          [
            { key: 'title', label: 'スレタイ' },
            { key: 'body', label: 'レス本文' },
          ] as const
        ).map((m) => (
          <Pressable
            key={m.key}
            onPress={() => setMode(m.key)}
            style={[styles.chip, mode === m.key && styles.chipOn]}>
            <Text style={[styles.chipText, mode === m.key && styles.chipTextOn]}>{m.label}</Text>
          </Pressable>
        ))}
      </View>

      <View style={styles.searchWrap}>
        <TextInput
          style={styles.search}
          placeholder={mode === 'title' ? 'スレタイを検索（全板）' : 'レス本文を検索（全板）'}
          placeholderTextColor={colors.textDim}
          value={query}
          onChangeText={setQuery}
          onSubmitEditing={run}
          returnKeyType="search"
          autoCorrect={false}
          autoCapitalize="none"
          clearButtonMode="while-editing"
        />
        <Pressable
          style={styles.button}
          onPress={loading ? () => abortRef.current?.abort() : run}>
          <Text style={styles.buttonText}>{loading ? '中止' : '検索'}</Text>
        </Pressable>
      </View>

      {mode === 'body' ? (
        <Text style={styles.note}>
          スレタイ検索で当たった上位 {BODY_SCAN_LIMIT} スレの dat を落として本文を探します。
          5ch には本文の横断検索がないので、時間がかかります。
        </Text>
      ) : null}

      {loading ? (
        <View style={styles.progress}>
          <ActivityIndicator color={colors.accent} />
          <Text style={styles.dim}>
            {progress
              ? `${progress.done}/${progress.total} スレ走査 ・ ${progress.hits} 件ヒット${
                  progress.failed > 0 ? ` ・ ${progress.failed} 件取得失敗` : ''
                }`
              : '検索中...'}
          </Text>
        </View>
      ) : null}

      {error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : null}

      {hits ? (
        <FlatList
          data={hits}
          keyExtractor={(h) => `${h.host}/${h.board}/${h.key}`}
          ListHeaderComponent={header}
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.dim}>
                {rawCount > 0
                  ? `find.5ch は ${rawCount} 件返しましたが、どれもタイトルに「${query.trim()}」を含みません。\nfind.5ch は語を分割して OR 検索するため、複合語だとこうなります。\n語を短く切るか、「レス本文」で探してみてください。`
                  : '該当するスレッドがありません'}
              </Text>
            </View>
          }
          renderItem={({ item }) => (
            <Pressable style={styles.row} onPress={() => openThread(item)}>
              <Text style={styles.title} numberOfLines={2}>
                {item.title}
              </Text>
              <View style={styles.metaRow}>
                {item.boardName ? <Text style={styles.board}>{item.boardName}</Text> : null}
                <Text style={styles.meta}>{item.resCount}レス</Text>
                {item.momentumText ? <Text style={styles.momentum}>{item.momentumText}</Text> : null}
                {item.updatedText ? <Text style={styles.meta}>{item.updatedText}</Text> : null}
              </View>
            </Pressable>
          )}
        />
      ) : postHits ? (
        <FlatList
          data={postHits}
          keyExtractor={(h) => `${h.key}:${h.post.res}`}
          ListHeaderComponent={header}
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.dim}>一致するレスがありません</Text>
            </View>
          }
          renderItem={({ item }) => (
            <Pressable style={styles.row} onPress={() => openThread(item)}>
              <Text style={styles.title} numberOfLines={2}>
                {item.title}
              </Text>
              <Text style={styles.snippet}>{item.snippet}</Text>
              <View style={styles.metaRow}>
                <Text style={styles.meta}>{item.post.res}レス目</Text>
                <Text style={styles.meta}>{item.post.name}</Text>
                <Text style={styles.meta}>{item.post.dateText}</Text>
              </View>
            </Pressable>
          )}
        />
      ) : !loading && !error ? (
        <View style={styles.center}>
          <Text style={styles.dim}>キーワードを入力して検索</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  modeRow: {
    flexDirection: 'row',
    gap: spacing.xs,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
  },
  chip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: radius,
    backgroundColor: colors.surface,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  chipOn: { backgroundColor: colors.surface2, borderColor: colors.accent },
  chipText: { color: colors.textDim, fontSize: 12 },
  chipTextOn: { color: colors.accentHover },
  searchWrap: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  search: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 15,
  },
  button: {
    paddingHorizontal: spacing.lg,
    justifyContent: 'center',
    borderRadius: radius,
    backgroundColor: colors.accent,
  },
  buttonText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  note: {
    color: colors.textDim,
    fontSize: 11,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  progress: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  center: { padding: spacing.xl * 2, alignItems: 'center', gap: spacing.md },
  dim: { color: colors.textDim, fontSize: 13, textAlign: 'center', lineHeight: 20 },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
  count: {
    color: colors.textDim,
    fontSize: 11,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  row: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.xs,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  title: { color: colors.text, fontSize: 15, lineHeight: 21 },
  snippet: { color: colors.accentHover, fontSize: 12, lineHeight: 18 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md },
  board: { color: colors.accentHover, fontSize: 11 },
  meta: { color: colors.textDim, fontSize: 11 },
  momentum: { color: colors.success, fontSize: 11 },
});
