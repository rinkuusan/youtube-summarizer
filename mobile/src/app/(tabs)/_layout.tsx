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
      <Tabs.Screen name="favorites" options={{ title: 'お気に入り', tabBarIcon: icon('★') }} />
      <Tabs.Screen name="history" options={{ title: '履歴', tabBarIcon: icon('🕘') }} />
      <Tabs.Screen name="search" options={{ title: '検索', tabBarIcon: icon('🔍') }} />
      <Tabs.Screen name="settings" options={{ title: '設定', tabBarIcon: icon('⚙') }} />
    </Tabs>
  );
}
