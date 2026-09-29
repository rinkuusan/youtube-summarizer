import * as Clipboard from 'expo-clipboard';
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import { Stack } from 'expo-router';
import { useCallback, useMemo, useState, useSyncExternalStore } from 'react';
import { FlatList, Platform, Pressable, StyleSheet, Text, View } from 'react-native';

import {
  clearLogs,
  formatLogsForCopy,
  formatTime,
  getLogs,
  subscribeLogs,
  type LogEntry,
  type LogLevel,
} from '@/net/log';
import { colors, radius, spacing } from '@/theme/colors';

/**
 * アプリ内ログビューア。
 * adb も Metro も無い実機の release APK で、通信とエラーをそのまま読むための画面。
 */

const LEVEL_COLOR: Record<LogLevel, string> = {
  debug: colors.textDim,
  info: colors.meta,
  warn: '#fbbf24',
  error: colors.error,
};

type Filter = 'all' | 'error' | 'http';

const FILTERS: { key: Filter; label: string }[] = [
  { key: 'all', label: 'すべて' },
  { key: 'error', label: 'エラーのみ' },
  { key: 'http', label: '通信のみ' },
];

/** 端末とビルドの素性。バグ報告のときに一番効く情報なので必ず先頭に付ける。 */
export function environmentSummary(): string {
  const version = Constants.expoConfig?.version ?? '?';
  return [
    `5chビューアー v${version}`,
    `${Platform.OS} ${Device.osVersion ?? '?'} / ${Device.modelName ?? '?'} (${Device.manufacturer ?? '?'})`,
    `__DEV__=${__DEV__}`,
    new Date().toISOString(),
  ].join('\n');
}

export default function LogsScreen() {
  const entries = useSyncExternalStore(subscribeLogs, getLogs, getLogs);
  const [filter, setFilter] = useState<Filter>('all');
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [copied, setCopied] = useState(false);

  const visible = useMemo(() => {
    const list =
      filter === 'error'
        ? entries.filter((e) => e.level === 'error' || e.level === 'warn')
        : filter === 'http'
          ? entries.filter((e) => e.tag === 'http')
          : entries;
    // 新しいものを上に。
    return [...list].reverse();
  }, [entries, filter]);

  const copy = useCallback(async () => {
    await Clipboard.setStringAsync(formatLogsForCopy(environmentSummary()));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, []);

  const renderItem = useCallback(
    ({ item }: { item: LogEntry }) => {
      const open = expanded[item.id];
      return (
        <Pressable
          onPress={() => item.detail && setExpanded((s) => ({ ...s, [item.id]: !s[item.id] }))}
          style={styles.row}>
          <View style={styles.rowHead}>
            <Text style={styles.time}>{formatTime(item.at)}</Text>
            <Text style={[styles.tag, { color: LEVEL_COLOR[item.level] }]}>{item.tag}</Text>
            {item.detail ? <Text style={styles.chevron}>{open ? '▾' : '▸'}</Text> : null}
          </View>
          <Text style={[styles.msg, { color: LEVEL_COLOR[item.level] }]}>{item.msg}</Text>
          {open && item.detail ? <Text style={styles.detail}>{item.detail}</Text> : null}
        </Pressable>
      );
    },
    [expanded]
  );

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: `ログ (${entries.length})` }} />

      <View style={styles.toolbar}>
        {FILTERS.map((f) => (
          <Pressable
            key={f.key}
            onPress={() => setFilter(f.key)}
            style={[styles.chip, filter === f.key && styles.chipOn]}>
            <Text style={[styles.chipText, filter === f.key && styles.chipTextOn]}>{f.label}</Text>
          </Pressable>
        ))}
        <View style={styles.spacer} />
        <Pressable onPress={copy} style={styles.chip}>
          <Text style={styles.chipText}>{copied ? 'コピー済' : '全文コピー'}</Text>
        </Pressable>
        <Pressable onPress={clearLogs} style={styles.chip}>
          <Text style={styles.chipText}>消去</Text>
        </Pressable>
      </View>

      <Text style={styles.env}>{environmentSummary()}</Text>

      <FlatList
        data={visible}
        keyExtractor={(e) => String(e.id)}
        renderItem={renderItem}
        ListEmptyComponent={<Text style={styles.empty}>ログはまだありません。</Text>}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  toolbar: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    alignItems: 'center',
  },
  spacer: { flex: 1 },
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
  env: {
    color: colors.textDim,
    fontSize: 10,
    fontFamily: 'monospace',
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.sm,
  },
  row: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  rowHead: { flexDirection: 'row', alignItems: 'center', gap: spacing.sm },
  time: { color: colors.textDim, fontSize: 10, fontFamily: 'monospace' },
  tag: { fontSize: 10, fontWeight: '700' },
  chevron: { color: colors.textDim, fontSize: 10 },
  msg: { fontSize: 12, fontFamily: 'monospace', marginTop: 2 },
  detail: {
    color: colors.textDim,
    fontSize: 10,
    fontFamily: 'monospace',
    marginTop: spacing.xs,
    paddingLeft: spacing.sm,
    borderLeftWidth: 2,
    borderLeftColor: colors.border,
  },
  empty: { color: colors.textDim, textAlign: 'center', marginTop: spacing.xl },
});
