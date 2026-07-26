import { router } from 'expo-router';
import { useCallback, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { searchThreads, type SearchHit } from '@/api/search';
import { toDisplayMessage } from '@/net/errors';
import { colors, radius, spacing } from '@/theme/colors';

export default function SearchScreen() {
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    const q = query.trim();
    if (!q) return;
    setLoading(true);
    setError(null);
    try {
      setHits(await searchThreads(q));
    } catch (e) {
      setError(toDisplayMessage(e));
      setHits(null);
    } finally {
      setLoading(false);
    }
  }, [query]);

  return (
    <View style={styles.container}>
      <View style={styles.searchWrap}>
        <TextInput
          style={styles.search}
          placeholder="スレタイを検索（全板）"
          placeholderTextColor={colors.textDim}
          value={query}
          onChangeText={setQuery}
          onSubmitEditing={run}
          returnKeyType="search"
          autoCorrect={false}
          autoCapitalize="none"
          clearButtonMode="while-editing"
        />
        <Pressable style={styles.button} onPress={run} disabled={loading}>
          <Text style={styles.buttonText}>検索</Text>
        </Pressable>
      </View>

      {loading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} />
        </View>
      ) : error ? (
        <View style={styles.center}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : (
        <FlatList
          data={hits ?? []}
          keyExtractor={(h) => `${h.host}/${h.board}/${h.key}`}
          ListEmptyComponent={
            <View style={styles.center}>
              <Text style={styles.dim}>
                {hits === null ? 'スレタイを入力して検索' : '該当するスレッドがありません'}
              </Text>
            </View>
          }
          renderItem={({ item }) => (
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
              <Text style={styles.title} numberOfLines={2}>
                {item.title}
              </Text>
              <View style={styles.metaRow}>
                {item.boardName ? <Text style={styles.board}>{item.boardName}</Text> : null}
                <Text style={styles.meta}>{item.resCount}レス</Text>
                {item.momentumText ? (
                  <Text style={styles.momentum}>{item.momentumText}</Text>
                ) : null}
                {item.updatedText ? <Text style={styles.meta}>{item.updatedText}</Text> : null}
              </View>
            </Pressable>
          )}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
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
  center: { padding: spacing.xl * 2, alignItems: 'center', gap: spacing.md },
  dim: { color: colors.textDim, fontSize: 13 },
  errorText: { color: colors.error, fontSize: 14, textAlign: 'center' },
  row: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.xs,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  title: { color: colors.text, fontSize: 15, lineHeight: 21 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.md },
  board: { color: colors.accentHover, fontSize: 11 },
  meta: { color: colors.textDim, fontSize: 11 },
  momentum: { color: colors.success, fontSize: 11 },
});
