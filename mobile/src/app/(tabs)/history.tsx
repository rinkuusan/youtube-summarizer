import { router, useFocusEffect } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Alert, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';

import { ThreadRow } from '@/components/ThreadRow';
import * as threadRepo from '@/db/threadRepo';
import type { ThreadListItem } from '@/db/types';
import { colors, spacing } from '@/theme/colors';

/**
 * 履歴。開いたスレと書き込んだスレを分けて出す。
 *
 * どちらも自動で記録される:
 *   閲覧   ... スレビューの mount で threadRepo.touchOpened()
 *   書き込み ... 投稿成功時に threadRepo.markPosted()
 * ユーザーが手で登録する操作は無い。
 */

type Segment = 'opened' | 'posted';

const SEGMENTS: { key: Segment; label: string }[] = [
  { key: 'opened', label: '閲覧' },
  { key: 'posted', label: '書き込み' },
];

export default function HistoryScreen() {
  const db = useSQLiteContext();
  const [segment, setSegment] = useState<Segment>('opened');
  const [items, setItems] = useState<ThreadListItem[]>([]);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(
    async (seg: Segment) => {
      const rows =
        seg === 'opened'
          ? await threadRepo.listOpenedHistory(db)
          : await threadRepo.listPostedHistory(db);
      setItems(rows);
      setLoaded(true);
    },
    [db]
  );

  // タブに戻るたびに読み直す。他の画面でスレを開いた結果を反映させるため。
  useFocusEffect(
    useCallback(() => {
      load(segment);
    }, [load, segment])
  );

  const openThread = useCallback((item: ThreadListItem) => {
    router.push({
      pathname: '/thread/[host]/[board]/[key]',
      params: { host: item.host, board: item.board, key: item.key, title: item.title },
    });
  }, []);

  const confirmRemove = useCallback(
    (item: ThreadListItem) => {
      Alert.alert(
        '履歴から削除',
        item.title,
        [
          { text: 'キャンセル', style: 'cancel' },
          {
            text: '削除',
            style: 'destructive',
            onPress: async () => {
              await threadRepo.removeFromHistory(db, item, segment);
              load(segment);
            },
          },
        ],
        { cancelable: true }
      );
    },
    [db, segment, load]
  );

  const switchTo = useCallback(
    (seg: Segment) => {
      setSegment(seg);
      setLoaded(false);
      load(seg);
    },
    [load]
  );

  return (
    <View style={styles.container}>
      <View style={styles.segmentRow}>
        {SEGMENTS.map((s) => (
          <Pressable
            key={s.key}
            onPress={() => switchTo(s.key)}
            style={[styles.segment, segment === s.key && styles.segmentActive]}>
            <Text style={[styles.segmentText, segment === s.key && styles.segmentTextActive]}>
              {s.label}
            </Text>
          </Pressable>
        ))}
      </View>

      <FlatList
        data={items}
        keyExtractor={(i) => `${i.host}/${i.board}/${i.key}`}
        renderItem={({ item }) => (
          <ThreadRow
            item={item}
            onPress={openThread}
            onLongPress={confirmRemove}
            timestamp={segment === 'opened' ? item.lastOpenedAt : item.lastPostedAt}
          />
        )}
        ListEmptyComponent={
          loaded ? (
            <View style={styles.empty}>
              <Text style={styles.emptyText}>
                {segment === 'opened'
                  ? 'まだスレッドを開いていません'
                  : 'まだ書き込んだスレッドはありません'}
              </Text>
              {segment === 'posted' ? (
                <Text style={styles.emptyHint}>書き込みに成功したスレッドがここに残ります</Text>
              ) : null}
            </View>
          ) : null
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  segmentRow: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  segment: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderRadius: 999,
    backgroundColor: colors.surface,
  },
  segmentActive: { backgroundColor: colors.accent },
  segmentText: { color: colors.textDim, fontSize: 13 },
  segmentTextActive: { color: '#fff', fontWeight: '700' },
  empty: { padding: spacing.xl * 2, alignItems: 'center', gap: spacing.sm },
  emptyText: { color: colors.textDim, fontSize: 14 },
  emptyHint: { color: colors.textDim, fontSize: 12, opacity: 0.7 },
});
