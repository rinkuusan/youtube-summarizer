import { Image } from 'expo-image';
import * as WebBrowser from 'expo-web-browser';
import { useState } from 'react';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';

import { colors, radius, spacing } from '../theme/colors';

/**
 * 画像のポップアップ表示。
 *
 * 段階を 2 つに分ける:
 *  1. ポップアップ  … 画面内に収めて全体を見せる (contain)
 *  2. 全画面        … 余白を捨てて画面いっぱいに広げる (cover)
 * もう一度タップで 1 に戻る。閉じるのは端末の戻るか、右上の×。
 *
 * リンク先をブラウザで開く導線も残す。動画や、拡張子で画像と判定できない
 * URL はこちらに逃がす必要があるため。
 */

interface Props {
  url: string | null;
  onClose: () => void;
}

export function ImageViewer({ url, onClose }: Props) {
  const [full, setFull] = useState(false);

  if (!url) return null;

  return (
    <Modal visible transparent animationType="fade" onRequestClose={onClose}>
      <View style={styles.backdrop}>
        <Pressable style={styles.imageArea} onPress={() => setFull((v) => !v)}>
          <Image
            source={{ uri: url }}
            style={styles.image}
            contentFit={full ? 'cover' : 'contain'}
            transition={120}
          />
        </Pressable>

        <View style={styles.bar}>
          <Text style={styles.hint} numberOfLines={1}>
            {full ? 'タップで全体表示に戻る' : 'タップで全画面'}
          </Text>
          <Pressable onPress={() => WebBrowser.openBrowserAsync(url)} hitSlop={10}>
            <Text style={styles.action}>ブラウザで開く</Text>
          </Pressable>
          <Pressable onPress={onClose} hitSlop={10}>
            <Text style={styles.action}>閉じる</Text>
          </Pressable>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: '#000000ee' },
  imageArea: { flex: 1 },
  image: { flex: 1, width: '100%' },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.lg,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    paddingBottom: spacing.xl,
    backgroundColor: colors.surface,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  hint: { color: colors.textDim, fontSize: 12, flex: 1 },
  action: {
    color: colors.accentHover,
    fontSize: 13,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius / 2,
  },
});
