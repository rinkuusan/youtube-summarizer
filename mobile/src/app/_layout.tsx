import { DarkTheme, Stack, ThemeProvider } from 'expo-router';
import { SQLiteProvider, type SQLiteDatabase } from 'expo-sqlite';
import { StatusBar } from 'expo-status-bar';

import { migrate } from '@/db/migrations';
import * as ngRepo from '@/db/ngRepo';
import * as threadRepo from '@/db/threadRepo';
import { colors } from '@/theme/colors';

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

export default function RootLayout() {
  return (
    <SQLiteProvider databaseName="gochviewer.db" onInit={initDatabase}>
      <ThemeProvider value={theme}>
        <StatusBar style="light" />
        <Stack
          screenOptions={{
            headerStyle: { backgroundColor: colors.surface },
            headerTintColor: colors.text,
            headerTitleStyle: { fontSize: 16 },
            contentStyle: { backgroundColor: colors.bg },
          }}>
          <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
        </Stack>
      </ThemeProvider>
    </SQLiteProvider>
  );
}
