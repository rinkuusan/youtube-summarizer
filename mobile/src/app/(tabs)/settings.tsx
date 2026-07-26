import { router, useFocusEffect } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useState } from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Switch, Text, View } from 'react-native';

import * as kvRepo from '@/db/kvRepo';
import * as ngRepo from '@/db/ngRepo';
import * as threadRepo from '@/db/threadRepo';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type AppSettings } from '@/settings';
import { colors, radius, spacing } from '@/theme/colors';

export default function SettingsScreen() {
  const db = useSQLiteContext();
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [ngCount, setNgCount] = useState(0);

  const load = useCallback(async () => {
    setSettings(await kvRepo.getJson(db, SETTINGS_KEY, DEFAULT_SETTINGS));
    setNgCount((await ngRepo.listAll(db)).length);
  }, [db]);

  useFocusEffect(
    useCallback(() => {
      load();
    }, [load])
  );

  const update = useCallback(
    async (patch: Partial<AppSettings>) => {
      const next = { ...settings, ...patch };
      setSettings(next);
      await kvRepo.setJson(db, SETTINGS_KEY, next);
    },
    [db, settings]
  );

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>表示</Text>

      <View style={styles.row}>
        <View style={styles.rowBody}>
          <Text style={styles.label}>画像を自動で表示</Text>
          <Text style={styles.hint}>モバイル回線では通信量に注意</Text>
        </View>
        <Switch
          value={settings.autoShowImages}
          onValueChange={(v) => update({ autoShowImages: v })}
          trackColor={{ true: colors.accent, false: colors.border }}
        />
      </View>

      <View style={styles.row}>
        <View style={styles.rowBody}>
          <Text style={styles.label}>文字サイズ</Text>
        </View>
        <View style={styles.stepper}>
          {[13, 15, 17, 19].map((size) => (
            <Pressable
              key={size}
              onPress={() => update({ fontSize: size })}
              style={[styles.sizeChip, settings.fontSize === size && styles.sizeChipActive]}>
              <Text
                style={[
                  styles.sizeText,
                  settings.fontSize === size && styles.sizeTextActive,
                ]}>
                {size}
              </Text>
            </Pressable>
          ))}
        </View>
      </View>

      <Text style={styles.sectionTitle}>投稿</Text>

      <View style={styles.row}>
        <View style={styles.rowBody}>
          <Text style={styles.label}>既定でメール欄に sage</Text>
        </View>
        <Switch
          value={settings.defaultSage}
          onValueChange={(v) => update({ defaultSage: v })}
          trackColor={{ true: colors.accent, false: colors.border }}
        />
      </View>

      <Text style={styles.sectionTitle}>NG</Text>

      <Pressable style={styles.linkRow} onPress={() => router.push('/settings/ng')}>
        <Text style={styles.label}>NG の管理</Text>
        <Text style={styles.linkValue}>{ngCount}件 ›</Text>
      </Pressable>

      <Text style={styles.sectionTitle}>診断</Text>

      {/* release APK には Metro も adb も無いので、ここが不具合を追う唯一の経路になる。 */}
      <Pressable style={styles.linkRow} onPress={() => router.push('/logs')}>
        <Text style={styles.label}>ログを見る</Text>
        <Text style={styles.linkValue}>通信・エラー・クラッシュ ›</Text>
      </Pressable>

      <Text style={styles.sectionTitle}>データ</Text>

      <Pressable
        style={styles.linkRow}
        onPress={() => {
          Alert.alert(
            '閲覧履歴を消す',
            'お気に入りと書き込んだスレッドは残ります。',
            [
              { text: 'キャンセル', style: 'cancel' },
              {
                text: '消す',
                style: 'destructive',
                onPress: async () => {
                  const n = await threadRepo.pruneHistory(db, 0);
                  Alert.alert('完了', `${n}件を削除しました`);
                },
              },
            ]
          );
        }}>
        <Text style={styles.label}>閲覧履歴を消す</Text>
        <Text style={styles.linkValue}>›</Text>
      </Pressable>

      <Text style={styles.footnote}>
        生 dat の直読みは 5ch の裁量でいつでも塞がれうる。読み取りの User-Agent は
        専ブラの慣例に従って Monazilla を名乗り、自動更新は 30 秒以上の間隔を空けている。
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { paddingBottom: spacing.xl * 2 },
  sectionTitle: {
    color: colors.accentHover,
    fontSize: 12,
    fontWeight: '700',
    backgroundColor: colors.surface2,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  linkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  rowBody: { flex: 1, gap: 2 },
  label: { color: colors.text, fontSize: 15 },
  hint: { color: colors.textDim, fontSize: 11 },
  linkValue: { color: colors.textDim, fontSize: 13 },
  stepper: { flexDirection: 'row', gap: spacing.xs },
  sizeChip: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius / 2,
    backgroundColor: colors.surface,
  },
  sizeChipActive: { backgroundColor: colors.accent },
  sizeText: { color: colors.textDim, fontSize: 12 },
  sizeTextActive: { color: '#fff', fontWeight: '700' },
  footnote: {
    color: colors.textDim,
    fontSize: 11,
    lineHeight: 17,
    padding: spacing.lg,
    opacity: 0.8,
  },
});
