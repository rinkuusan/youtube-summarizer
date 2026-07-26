import { Stack } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  FlatList,
  Pressable,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import * as ngRepo from '@/db/ngRepo';
import type { NgKind, NgRule } from '@/db/types';
import { colors, radius, spacing } from '@/theme/colors';

const KINDS: { key: NgKind; label: string }[] = [
  { key: 'word', label: 'ワード' },
  { key: 'id', label: 'ID' },
  { key: 'name', label: '名前' },
  { key: 'wacchoi', label: 'ワッチョイ' },
];

const KIND_LABEL: Record<string, string> = Object.fromEntries(KINDS.map((k) => [k.key, k.label]));

export default function NgSettingsScreen() {
  const db = useSQLiteContext();
  const [rules, setRules] = useState<NgRule[]>([]);
  const [kind, setKind] = useState<NgKind>('word');
  const [pattern, setPattern] = useState('');
  const [isRegex, setIsRegex] = useState(false);
  const [chain, setChain] = useState(false);

  const load = useCallback(async () => {
    setRules(await ngRepo.listAll(db));
  }, [db]);

  useEffect(() => {
    load();
  }, [load]);

  const add = useCallback(async () => {
    const p = pattern.trim();
    if (!p) return;
    if (isRegex) {
      // 壊れた正規表現は登録前に弾く。描画側でも try/catch しているが、
      // ここで教えた方が親切。
      try {
        new RegExp(p);
      } catch (e) {
        Alert.alert('正規表現が不正です', e instanceof Error ? e.message : String(e));
        return;
      }
    }
    await ngRepo.add(db, { kind, pattern: p, isRegex, chain });
    setPattern('');
    load();
  }, [db, kind, pattern, isRegex, chain, load]);

  const remove = useCallback(
    (rule: NgRule) => {
      Alert.alert('NG を削除', rule.pattern, [
        { text: 'キャンセル', style: 'cancel' },
        {
          text: '削除',
          style: 'destructive',
          onPress: async () => {
            await ngRepo.remove(db, rule.id);
            load();
          },
        },
      ]);
    },
    [db, load]
  );

  return (
    <View style={styles.container}>
      <Stack.Screen options={{ title: 'NG の管理' }} />

      <View style={styles.form}>
        <View style={styles.kindRow}>
          {KINDS.map((k) => (
            <Pressable
              key={k.key}
              onPress={() => setKind(k.key)}
              style={[styles.kindChip, kind === k.key && styles.kindChipActive]}>
              <Text style={[styles.kindText, kind === k.key && styles.kindTextActive]}>
                {k.label}
              </Text>
            </Pressable>
          ))}
        </View>

        <TextInput
          style={styles.input}
          placeholder={isRegex ? '正規表現' : 'NG にする文字列'}
          placeholderTextColor={colors.textDim}
          value={pattern}
          onChangeText={setPattern}
          onSubmitEditing={add}
          autoCorrect={false}
          autoCapitalize="none"
        />

        <View style={styles.toggleRow}>
          <View style={styles.toggle}>
            <Text style={styles.toggleLabel}>正規表現</Text>
            <Switch
              value={isRegex}
              onValueChange={setIsRegex}
              trackColor={{ true: colors.accent, false: colors.border }}
            />
          </View>
          <View style={styles.toggle}>
            <Text style={styles.toggleLabel}>連鎖</Text>
            <Switch
              value={chain}
              onValueChange={setChain}
              trackColor={{ true: colors.accent, false: colors.border }}
            />
          </View>
          <Pressable style={styles.addButton} onPress={add}>
            <Text style={styles.addButtonText}>追加</Text>
          </Pressable>
        </View>
        <Text style={styles.hint}>
          「連鎖」を有効にすると、NG になったレスに返信しているレスも隠れる。
        </Text>
      </View>

      <FlatList
        data={rules}
        keyExtractor={(r) => String(r.id)}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>NG はまだありません</Text>
            <Text style={styles.emptyHint}>
              スレッドのレス番号やIDを長押しからも登録できます
            </Text>
          </View>
        }
        renderItem={({ item }) => (
          <Pressable style={styles.ruleRow} onPress={() => remove(item)}>
            <View style={styles.ruleBody}>
              <Text style={styles.rulePattern} numberOfLines={2}>
                {item.pattern}
              </Text>
              <View style={styles.ruleMeta}>
                <Text style={styles.ruleKind}>{KIND_LABEL[item.kind] ?? item.kind}</Text>
                {item.is_regex === 1 ? <Text style={styles.ruleFlag}>正規表現</Text> : null}
                {item.chain === 1 ? <Text style={styles.ruleFlag}>連鎖</Text> : null}
                {item.scope_board ? (
                  <Text style={styles.ruleFlag}>{item.scope_board}</Text>
                ) : (
                  <Text style={styles.ruleFlag}>全板</Text>
                )}
                {item.expires_at ? (
                  <Text style={styles.ruleExpiry}>
                    {new Date(item.expires_at).getMonth() + 1}/
                    {new Date(item.expires_at).getDate()} まで
                  </Text>
                ) : null}
              </View>
            </View>
            <Text style={styles.delete}>削除</Text>
          </Pressable>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  form: {
    padding: spacing.lg,
    gap: spacing.sm,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  kindRow: { flexDirection: 'row', gap: spacing.sm },
  kindChip: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: 999,
    backgroundColor: colors.surface,
  },
  kindChipActive: { backgroundColor: colors.accent },
  kindText: { color: colors.textDim, fontSize: 12 },
  kindTextActive: { color: '#fff', fontWeight: '700' },
  input: {
    backgroundColor: colors.surface,
    borderRadius: radius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    color: colors.text,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    fontSize: 15,
  },
  toggleRow: { flexDirection: 'row', alignItems: 'center', gap: spacing.lg },
  toggle: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  toggleLabel: { color: colors.textDim, fontSize: 12 },
  addButton: {
    marginLeft: 'auto',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius,
    backgroundColor: colors.accent,
  },
  addButtonText: { color: '#fff', fontSize: 13, fontWeight: '700' },
  hint: { color: colors.textDim, fontSize: 11 },
  ruleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  ruleBody: { flex: 1, gap: spacing.xs },
  rulePattern: { color: colors.text, fontSize: 14 },
  ruleMeta: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm },
  ruleKind: { color: colors.accentHover, fontSize: 11 },
  ruleFlag: { color: colors.textDim, fontSize: 11 },
  ruleExpiry: { color: colors.textDim, fontSize: 11 },
  delete: { color: colors.error, fontSize: 12 },
  empty: { padding: spacing.xl * 2, alignItems: 'center', gap: spacing.sm },
  emptyText: { color: colors.textDim, fontSize: 14 },
  emptyHint: { color: colors.textDim, fontSize: 12, opacity: 0.7 },
});
