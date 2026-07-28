import { Stack, useLocalSearchParams } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import * as WebBrowser from 'expo-web-browser';
import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import { checkLocalCooldown, submitPost } from '@/api/post';
import { extractCooldownSeconds, type PostResult } from '@/api/postErrors';
import { fetchSetting, type BoardSetting } from '@/api/setting';
import * as kvRepo from '@/db/kvRepo';
import type { ThreadRef } from '@/db/types';
import { toDisplayMessage } from '@/net/errors';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type AppSettings } from '@/settings';
import { colors, radius, spacing } from '@/theme/colors';

/**
 * スレ立て。
 *
 * bbs.cgi へ送る中身はレス投稿とほぼ同じで、key の代わりに subject を積み、
 * submit を「新規スレッド作成」にするだけ。確認ページの承諾も同じ経路を通る
 * ので、api/post.ts の submitPost をそのまま使う。
 */
export default function NewThreadScreen() {
  const db = useSQLiteContext();
  const { host, board, name: boardName } = useLocalSearchParams<{
    host: string;
    board: string;
    name?: string;
  }>();

  // スレ立てはまだ key が無い。ThreadRef の key は空で通す。
  const ref = useMemo<ThreadRef>(() => ({ host, board, key: '' }), [host, board]);

  const [subject, setSubject] = useState('');
  const [name, setName] = useState('');
  const [mail, setMail] = useState('');
  const [message, setMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState<PostResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [boardSetting, setBoardSetting] = useState<BoardSetting | null>(null);
  const [cooldown, setCooldown] = useState(0);

  useEffect(() => {
    (async () => {
      const s = await kvRepo.getJson<AppSettings>(db, SETTINGS_KEY, DEFAULT_SETTINGS);
      setName(s.defaultName);
      setCooldown(await checkLocalCooldown(db, host, board));
      fetchSetting(host, board)
        .then(setBoardSetting)
        .catch(() => undefined);
    })();
  }, [db, host, board]);

  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  const send = useCallback(
    async (accepted: boolean) => {
      if (!subject.trim() || !message.trim()) return;
      setSending(true);
      setError(null);
      try {
        const r = await submitPost(
          db,
          ref,
          { name, mail, message, subject: subject.trim() },
          {
            accepted,
            confirmFields: accepted ? result?.formFields : undefined,
            confirmAction: accepted ? result?.formAction : undefined,
          }
        );
        setResult(r);
        if (r.outcome === 'success') {
          setSubject('');
          setMessage('');
        } else if (r.outcome === 'cooldown') {
          setCooldown(extractCooldownSeconds(r.bodyText) ?? 30);
        }
      } catch (e) {
        setError(toDisplayMessage(e));
      } finally {
        setSending(false);
      }
    },
    [db, ref, name, mail, message, subject, result]
  );

  const maxMessage = boardSetting?.maxMessage ?? null;
  const overLimit = maxMessage !== null && message.length > maxMessage;
  const blocked = sending || cooldown > 0 || overLimit || !subject.trim() || !message.trim();

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <Stack.Screen options={{ title: `スレ立て${boardName ? ` — ${boardName}` : ''}` }} />

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        <TextInput
          style={styles.input}
          placeholder="スレッドタイトル"
          placeholderTextColor={colors.textDim}
          value={subject}
          onChangeText={setSubject}
          autoCorrect={false}
        />

        <View style={styles.fieldRow}>
          <TextInput
            style={[styles.input, styles.flex]}
            placeholder={boardSetting?.noname ?? '名前（省略可）'}
            placeholderTextColor={colors.textDim}
            value={name}
            onChangeText={setName}
            autoCorrect={false}
          />
          <View style={styles.sageToggle}>
            <Text style={styles.sageLabel}>sage</Text>
            <Switch
              value={mail.includes('sage')}
              onValueChange={(v) => setMail(v ? 'sage' : '')}
              trackColor={{ true: colors.accent, false: colors.border }}
            />
          </View>
        </View>

        <TextInput
          style={[styles.input, styles.message]}
          placeholder="本文（1レス目）"
          placeholderTextColor={colors.textDim}
          value={message}
          onChangeText={setMessage}
          multiline
          textAlignVertical="top"
        />

        <View style={styles.counterRow}>
          <Text style={[styles.counter, overLimit && styles.counterOver]}>
            {message.length}
            {maxMessage !== null ? ` / ${maxMessage}` : ''}
          </Text>
          {cooldown > 0 ? <Text style={styles.cooldown}>あと {cooldown} 秒</Text> : null}
        </View>

        <Pressable
          style={[styles.submit, blocked && styles.submitDisabled]}
          disabled={blocked}
          onPress={() => send(false)}>
          {sending ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitText}>スレッドを立てる</Text>
          )}
        </Pressable>

        {error ? <Text style={styles.error}>{error}</Text> : null}

        {result ? (
          <View
            style={[
              styles.resultBox,
              result.outcome === 'success' ? styles.resultOk : styles.resultNg,
            ]}>
            <Text style={styles.resultTitle}>{result.title}</Text>
            {result.message ? <Text style={styles.resultMessage}>{result.message}</Text> : null}

            {/* 5ch からの応答は改変せずそのまま見せる */}
            {result.outcome !== 'success' ? (
              <ScrollView style={styles.responseBox} nestedScrollEnabled>
                <Text style={styles.responseText}>{result.bodyText}</Text>
              </ScrollView>
            ) : null}

            {result.action === 'confirm' ? (
              <Pressable style={styles.submit} onPress={() => send(true)} disabled={sending}>
                <Text style={styles.submitText}>上記に同意してスレッドを立てる</Text>
              </Pressable>
            ) : null}

            {result.action === 'openBrowser' ? (
              <Pressable
                style={styles.secondary}
                onPress={() => WebBrowser.openBrowserAsync(`https://${host}/${board}/`)}>
                <Text style={styles.secondaryText}>ブラウザで開く</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}

        <Text style={styles.note}>
          スレ立ては 5ch 側の制限（どんぐり・スレ立て規制・IP 規制）で弾かれることがある。
          その場合は上に 5ch からの応答をそのまま表示するので、原因の切り分けに使ってほしい。
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, gap: spacing.md },
  fieldRow: { flexDirection: 'row', gap: spacing.md, alignItems: 'center' },
  flex: { flex: 1 },
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
  message: { minHeight: 160, lineHeight: 22 },
  sageToggle: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  sageLabel: { color: colors.textDim, fontSize: 12 },
  counterRow: { flexDirection: 'row', justifyContent: 'space-between' },
  counter: { color: colors.textDim, fontSize: 11 },
  counterOver: { color: colors.error },
  cooldown: { color: colors.error, fontSize: 11 },
  submit: {
    backgroundColor: colors.accent,
    borderRadius: radius,
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  submitDisabled: { backgroundColor: colors.surface2 },
  submitText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  secondary: {
    backgroundColor: colors.surface2,
    borderRadius: radius,
    paddingVertical: spacing.sm,
    alignItems: 'center',
  },
  secondaryText: { color: colors.text, fontSize: 14 },
  error: { color: colors.error, fontSize: 13 },
  resultBox: {
    borderRadius: radius,
    borderWidth: 1,
    padding: spacing.md,
    gap: spacing.sm,
  },
  resultOk: { borderColor: colors.success },
  resultNg: { borderColor: colors.error },
  resultTitle: { color: colors.text, fontSize: 15, fontWeight: '700' },
  resultMessage: { color: colors.textDim, fontSize: 13 },
  responseBox: {
    maxHeight: 260,
    backgroundColor: colors.surface,
    borderRadius: radius / 2,
    padding: spacing.sm,
  },
  responseText: { color: colors.textDim, fontSize: 12, lineHeight: 18 },
  note: { color: colors.textDim, fontSize: 11, lineHeight: 17 },
});
