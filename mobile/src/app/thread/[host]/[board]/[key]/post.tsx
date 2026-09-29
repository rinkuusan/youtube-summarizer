import { router, Stack, useLocalSearchParams } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import * as WebBrowser from 'expo-web-browser';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
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

import { fetchSetting, type BoardSetting } from '@/api/setting';
import {
  checkLocalCooldown,
  clearDraft,
  loadDraft,
  rememberMyPost,
  saveDraft,
  submitPost,
  type PostDraft,
} from '@/api/post';
import { extractCooldownSeconds, type PostResult } from '@/api/postErrors';
import * as kvRepo from '@/db/kvRepo';
import * as threadRepo from '@/db/threadRepo';
import type { ThreadRef } from '@/db/types';
import { toDisplayMessage } from '@/net/errors';
import { DEFAULT_SETTINGS, SETTINGS_KEY, type AppSettings } from '@/settings';
import { colors, radius, spacing } from '@/theme/colors';

export default function PostFormScreen() {
  const db = useSQLiteContext();
  const params = useLocalSearchParams<{
    host: string;
    board: string;
    key: string;
    title?: string;
  }>();
  const { host, board, key } = params;
  const threadRef = useMemo<ThreadRef>(() => ({ host, board, key }), [host, board, key]);

  const [name, setName] = useState('');
  const [mail, setMail] = useState('sage');
  const [message, setMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState<PostResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [boardSetting, setBoardSetting] = useState<BoardSetting | null>(null);
  const [cooldown, setCooldown] = useState(0);

  const draftTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // 初期化
  useEffect(() => {
    (async () => {
      const [draft, appSettings] = await Promise.all([
        loadDraft(db, threadRef),
        kvRepo.getJson<AppSettings>(db, SETTINGS_KEY, DEFAULT_SETTINGS),
      ]);
      if (draft) {
        setName(draft.name);
        setMail(draft.mail);
        setMessage(draft.message);
      } else {
        setName(appSettings.defaultName);
        setMail(appSettings.defaultSage ? 'sage' : '');
      }
      setCooldown(await checkLocalCooldown(db, host, board));
      // 文字数の上限が知りたいだけなので、取れなくても投稿はできる
      fetchSetting(host, board)
        .then(setBoardSetting)
        .catch(() => undefined);
    })();
  }, [db, threadRef, host, board]);

  // 連投規制のカウントダウン
  useEffect(() => {
    if (cooldown <= 0) return;
    const t = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [cooldown]);

  /** 下書きの自動保存。規制エラーで長文を失わせないため。 */
  const persistDraft = useCallback(
    (draft: PostDraft) => {
      if (draftTimer.current) clearTimeout(draftTimer.current);
      draftTimer.current = setTimeout(() => {
        saveDraft(db, threadRef, draft).catch(() => undefined);
      }, 400);
    },
    [db, threadRef]
  );

  useEffect(() => {
    persistDraft({ name, mail, message });
  }, [name, mail, message, persistDraft]);

  const send = useCallback(
    async (accepted: boolean) => {
      if (!message.trim()) return;
      setSending(true);
      setError(null);
      try {
        // 承諾時は、いま表示している確認ページのフォームをそのまま送り返す。
        // feature のような使い捨てトークンが入っており、これが無いと
        // 5ch は承諾と認めず確認ページを返し続ける。
        const r = await submitPost(
          db,
          threadRef,
          { name, mail, message },
          {
            accepted,
            confirmFields: accepted ? result?.formFields : undefined,
            confirmAction: accepted ? result?.formAction : undefined,
          }
        );
        setResult(r);

        if (r.outcome === 'success') {
          // 書き込み履歴はこれだけで自動的に積まれる
          await threadRepo.markPosted(db, threadRef);
          // レス番号は返ってこないので、本文を控えて後から照合する
          await rememberMyPost(db, threadRef, message);
          await clearDraft(db, threadRef);
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
    // result は承諾時に確認ページのフォームを送り返すのに要る。
    // 依存に入れないと古い確認ページのトークンを送ってしまう。
    [db, threadRef, name, mail, message, result]
  );

  const openInBrowser = useCallback(() => {
    WebBrowser.openBrowserAsync(`https://${host}/test/read.cgi/${board}/${key}/`);
  }, [host, board, key]);

  const maxMessage = boardSetting?.maxMessage ?? null;
  const overLimit = maxMessage !== null && message.length > maxMessage;
  const blocked = sending || cooldown > 0 || overLimit || !message.trim();

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <Stack.Screen options={{ title: '書き込み', presentation: 'modal' }} />

      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
        {params.title ? (
          <Text style={styles.threadTitle} numberOfLines={2}>
            {params.title}
          </Text>
        ) : null}

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
          placeholder="本文"
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
          {cooldown > 0 ? (
            <Text style={styles.cooldown}>連投規制まであと {cooldown} 秒</Text>
          ) : null}
        </View>

        <Pressable
          style={[styles.submit, blocked && styles.submitDisabled]}
          disabled={blocked}
          onPress={() => send(false)}>
          {sending ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitText}>書き込む</Text>
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

            {/*
              5ch からの応答は改変せずそのまま見せる。書き込み確認画面を勝手に承諾したり
              内容を書き換えたりしないことは、専ブラに対して 5ch が明示的に求めている作法。
            */}
            {result.outcome !== 'success' && result.bodyText ? (
              <ScrollView style={styles.rawBox} nestedScrollEnabled>
                <Text style={styles.rawText}>{result.bodyText}</Text>
              </ScrollView>
            ) : null}

            {result.action === 'confirm' ? (
              <Pressable style={styles.confirmButton} onPress={() => send(true)}>
                <Text style={styles.confirmText}>上記に同意して書き込む</Text>
              </Pressable>
            ) : null}

            {result.action === 'openBrowser' || result.outcome === 'unknown' ? (
              <Pressable style={styles.secondaryButton} onPress={openInBrowser}>
                <Text style={styles.secondaryText}>ブラウザで書き込む</Text>
              </Pressable>
            ) : null}

            {result.outcome === 'success' ? (
              <Pressable style={styles.secondaryButton} onPress={() => router.back()}>
                <Text style={styles.secondaryText}>スレッドに戻る</Text>
              </Pressable>
            ) : null}
          </View>
        ) : null}

        <Text style={styles.footnote}>
          5ch 側の規制（どんぐり・連投規制・IP 規制）により、内容が正しくても書き込めない
          ことがある。その場合は上に 5ch からの応答をそのまま表示するので、
          原因の切り分けに使ってほしい。ブラウザからなら書けることも多い。
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg },
  content: { padding: spacing.lg, gap: spacing.md },
  threadTitle: { color: colors.textDim, fontSize: 12 },
  fieldRow: { flexDirection: 'row', gap: spacing.md, alignItems: 'center' },
  flex: { flex: 1 },
  sageToggle: { flexDirection: 'row', alignItems: 'center', gap: spacing.xs },
  sageLabel: { color: colors.textDim, fontSize: 12 },
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
  counterRow: { flexDirection: 'row', justifyContent: 'space-between' },
  counter: { color: colors.textDim, fontSize: 11 },
  counterOver: { color: colors.error, fontWeight: '700' },
  cooldown: { color: colors.error, fontSize: 11 },
  submit: {
    backgroundColor: colors.accent,
    borderRadius: radius,
    paddingVertical: spacing.md,
    alignItems: 'center',
  },
  submitDisabled: { opacity: 0.4 },
  submitText: { color: '#fff', fontSize: 15, fontWeight: '700' },
  error: { color: colors.error, fontSize: 13 },
  resultBox: {
    borderRadius: radius,
    borderWidth: 1,
    padding: spacing.md,
    gap: spacing.sm,
    backgroundColor: colors.surface,
  },
  resultOk: { borderColor: colors.success },
  resultNg: { borderColor: colors.error },
  resultTitle: { color: colors.text, fontSize: 15, fontWeight: '700' },
  resultMessage: { color: colors.textDim, fontSize: 13, lineHeight: 19 },
  rawBox: {
    maxHeight: 220,
    backgroundColor: colors.bg,
    borderRadius: radius / 2,
    padding: spacing.sm,
  },
  rawText: { color: colors.textDim, fontSize: 11, lineHeight: 17 },
  confirmButton: {
    backgroundColor: colors.accent,
    borderRadius: radius,
    paddingVertical: spacing.sm,
    alignItems: 'center',
  },
  confirmText: { color: '#fff', fontSize: 14, fontWeight: '700' },
  secondaryButton: {
    backgroundColor: colors.surface2,
    borderRadius: radius,
    paddingVertical: spacing.sm,
    alignItems: 'center',
  },
  secondaryText: { color: colors.text, fontSize: 14 },
  footnote: { color: colors.textDim, fontSize: 11, lineHeight: 17, opacity: 0.8 },
});
