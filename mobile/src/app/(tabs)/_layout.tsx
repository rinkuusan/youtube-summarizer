import { Tabs } from 'expo-router';
import { Text, type ColorValue } from 'react-native';

import { colors } from '@/theme/colors';

/** アイコンフォントを足さずに済ませる。絵文字で十分見分けが付く。 */
function icon(glyph: string) {
  return ({ color }: { color: ColorValue }) => (
    <Text style={{ fontSize: 18, color, lineHeight: 22 }}>{glyph}</Text>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerStyle: { backgroundColor: colors.surface },
        headerTintColor: colors.text,
        headerTitleStyle: { fontSize: 16 },
        tabBarStyle: { backgroundColor: colors.surface, borderTopColor: colors.border },
        tabBarActiveTintColor: colors.accent,
        tabBarInactiveTintColor: colors.textDim,
        tabBarLabelStyle: { fontSize: 10 },
        sceneStyle: { backgroundColor: colors.bg },
      }}>
      <Tabs.Screen name="index" options={{ title: '板一覧', tabBarIcon: icon('☰') }} />
      <Tabs.Screen name="favorites" options={{ title: '履歴・お気に入り', tabBarIcon: icon('★') }} />
      <Tabs.Screen name="hot" options={{ title: '新着', tabBarIcon: icon('🔥') }} />
      {/* 履歴はお気に入りタブに統合したので、タブとしては出さない */}
      <Tabs.Screen name="history" options={{ href: null }} />
      <Tabs.Screen name="search" options={{ title: '検索', tabBarIcon: icon('🔍') }} />
      <Tabs.Screen name="settings" options={{ title: '設定', tabBarIcon: icon('⚙') }} />
    </Tabs>
  );
}
