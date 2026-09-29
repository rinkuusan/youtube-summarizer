import { DarkTheme, router, Stack, ThemeProvider } from 'expo-router';
import { SQLiteProvider, type SQLiteDatabase } from 'expo-sqlite';
import { StatusBar } from 'expo-status-bar';
import { GestureHandlerRootView } from 'react-native-gesture-handler';

import { ErrorBoundary } from '@/components/ErrorBoundary';
import { migrate } from '@/db/migrations';
import * as ngRepo from '@/db/ngRepo';
import * as threadRepo from '@/db/threadRepo';
import { installGlobalLogHandlers, log } from '@/net/log';
import { colors } from '@/theme/colors';

// 未捕捉例外と console 出力の吸い上げは、どの画面より先に有効にする。
// release APK には Metro も adb も無いので、これがログを見る唯一の経路になる。
installGlobalLogHandlers();
log('info', 'app', 'アプリ起動');

const theme = {
  ...DarkTheme,
  colors: {
    ...DarkTheme.colors,
    background: colors.bg,
    card: colors.surface,
    text: colors.text,
    border: colors.border,
    primary: colors.accent,
  },
};

async function initDatabase(db: SQLiteDatabase) {
  await migrate(db);
  // 起動のたびに軽く掃除する。どちらも失敗してもアプリは動くべきなので握る。
  try {
    await ngRepo.purgeExpired(db);
    await threadRepo.pruneHistory(db);
  } catch (e) {
    console.warn('[db] 起動時の掃除に失敗しました', e);
  }
}

const openLogs = () => router.push('/logs');

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
    <SQLiteProvider databaseName="gochviewer.db" onInit={initDatabase}>
      <ThemeProvider value={theme}>
        <StatusBar style="light" />
        <ErrorBoundary onShowLogs={openLogs}>
          <Stack
            screenOptions={{
              headerStyle: { backgroundColor: colors.surface },
              headerTintColor: colors.text,
              headerTitleStyle: { fontSize: 16 },
              contentStyle: { backgroundColor: colors.bg },
            }}>
            <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
            <Stack.Screen name="logs" options={{ title: 'ログ' }} />
          </Stack>
        </ErrorBoundary>
      </ThemeProvider>
    </SQLiteProvider>
    </GestureHandlerRootView>
  );
}
