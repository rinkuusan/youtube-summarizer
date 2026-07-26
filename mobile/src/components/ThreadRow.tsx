import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { ThreadListItem } from '../db/types';
import { colors, spacing } from '../theme/colors';

interface Props {
  item: ThreadListItem;
  onPress: (item: ThreadListItem) => void;
  onLongPress?: (item: ThreadListItem) => void;
  /** 右下に出す時刻。履歴の種類によって開いた時刻か書いた時刻かが変わる。 */
  timestamp?: number | null;
}

function formatWhen(ms: number | null | undefined): string {
  if (!ms) return '';
  const d = new Date(ms);
  const now = new Date();
  const sameDay =
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate();
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  if (sameDay) return `${hh}:${mm}`;
  return `${d.getMonth() + 1}/${d.getDate()} ${hh}:${mm}`;
}

function ThreadRowImpl({ item, onPress, onLongPress, timestamp }: Props) {
  return (
    <Pressable
      style={styles.row}
      onPress={() => onPress(item)}
      onLongPress={onLongPress ? () => onLongPress(item) : undefined}>
      <View style={styles.body}>
        <Text style={styles.title} numberOfLines={2}>
          {item.title || '(無題)'}
        </Text>
        <View style={styles.metaRow}>
          {item.boardName ? <Text style={styles.board}>{item.boardName}</Text> : null}
          <Text style={styles.meta}>{item.resCount}レス</Text>
          {item.unread > 0 ? <Text style={styles.unread}>新着 {item.unread}</Text> : null}
          {item.myPostCount > 0 ? (
            <Text style={styles.mine}>自分 {item.myPostCount}</Text>
          ) : null}
          {item.favorite ? <Text style={styles.fav}>★</Text> : null}
        </View>
      </View>
      <Text style={styles.when}>{formatWhen(timestamp)}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  body: { flex: 1, gap: spacing.xs },
  title: { color: colors.text, fontSize: 15, lineHeight: 21 },
  metaRow: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: spacing.sm },
  board: { color: colors.accentHover, fontSize: 11 },
  meta: { color: colors.textDim, fontSize: 11 },
  unread: { color: colors.success, fontSize: 11, fontWeight: '700' },
  mine: { color: '#fbbf24', fontSize: 11 },
  fav: { color: colors.accent, fontSize: 11 },
  when: { color: colors.textDim, fontSize: 10, fontVariant: ['tabular-nums'] },
});

export const ThreadRow = memo(ThreadRowImpl);
