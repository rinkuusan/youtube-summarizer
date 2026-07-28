import { Image } from 'expo-image';
import { File, Paths } from 'expo-file-system';
import * as MediaLibrary from 'expo-media-library';
import * as WebBrowser from 'expo-web-browser';
import { useCallback, useState } from 'react';
import { Alert, Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, {
  runOnJS,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';

import { logError } from '../net/log';
import { colors, radius, spacing } from '../theme/colors';

/**
 * 画像のポップアップ表示。
 *
 * - 1 タップ目でここが開く (画面に収めた状態)
 * - もう一度タップで等倍相当まで拡大。ピンチでも拡縮できる
 * - 拡大中は指でドラッグして動かせる
 * - 長押しで端末のギャラリーに保存
 */

const AnimatedImage = Animated.createAnimatedComponent(Image);

/** タップで拡大したときの倍率。 */
const TAP_SCALE = 2.5;
const MAX_SCALE = 6;

interface Props {
  url: string | null;
  onClose: () => void;
}

export function ImageViewer({ url, onClose }: Props) {
  const [saving, setSaving] = useState(false);

  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const x = useSharedValue(0);
  const y = useSharedValue(0);
  const savedX = useSharedValue(0);
  const savedY = useSharedValue(0);

  const reset = useCallback(() => {
    scale.value = withTiming(1);
    savedScale.value = 1;
    x.value = withTiming(0);
    y.value = withTiming(0);
    savedX.value = 0;
    savedY.value = 0;
  }, [scale, savedScale, x, y, savedX, savedY]);

  const close = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  /** ギャラリーへ保存。拡張子はURLから拾い、無ければ jpg にしておく。 */
  const save = useCallback(async () => {
    if (!url || saving) return;
    setSaving(true);
    try {
      const granted = await MediaLibrary.requestPermissionsAsync(true);
      if (!granted.granted) {
        Alert.alert('保存できません', '写真へのアクセスが許可されていません。');
        return;
      }
      const ext = /\.([a-z0-9]{3,4})(?:[?#]|$)/i.exec(url)?.[1] ?? 'jpg';
      const dest = new File(Paths.cache, `gv_${Date.now()}.${ext}`);
      const file = await File.downloadFileAsync(url, dest);
      await MediaLibrary.Asset.create(file.uri);
      Alert.alert('保存しました', 'ギャラリーに追加しました。');
    } catch (e) {
      logError('image', e, '画像の保存に失敗');
      Alert.alert('保存に失敗しました', String((e as Error)?.message ?? e));
    } finally {
      setSaving(false);
    }
  }, [url, saving]);

  const pinch = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = Math.min(Math.max(savedScale.value * e.scale, 0.5), MAX_SCALE);
    })
    .onEnd(() => {
      // 等倍より小さくなったら戻す。指を離した所で固定すると迷子になる。
      if (scale.value < 1) {
        scale.value = withTiming(1);
        x.value = withTiming(0);
        y.value = withTiming(0);
        savedX.value = 0;
        savedY.value = 0;
      }
      savedScale.value = Math.max(scale.value, 1);
    });

  const pan = Gesture.Pan()
    .onUpdate((e) => {
      // 等倍のときは動かさない。画面に収まっているので動かす意味が無い。
      if (savedScale.value <= 1) return;
      x.value = savedX.value + e.translationX;
      y.value = savedY.value + e.translationY;
    })
    .onEnd(() => {
      savedX.value = x.value;
      savedY.value = y.value;
    });

  const doubleTapLike = Gesture.Tap().onEnd(() => {
    if (savedScale.value > 1) {
      scale.value = withTiming(1);
      x.value = withTiming(0);
      y.value = withTiming(0);
      savedScale.value = 1;
      savedX.value = 0;
      savedY.value = 0;
    } else {
      scale.value = withTiming(TAP_SCALE);
      savedScale.value = TAP_SCALE;
    }
  });

  const longPress = Gesture.LongPress()
    .minDuration(450)
    .onStart(() => {
      runOnJS(save)();
    });

  // 離散的な操作 (長押し / タップ) はどちらか一方だけ。
  // 拡縮と移動は同時に効かせる。Exclusive で全部を包むと、先頭の長押しが
  // 後続を塞いでタップもピンチも効かなくなる。
  const gesture = Gesture.Simultaneous(pinch, pan, Gesture.Exclusive(longPress, doubleTapLike));

  const imageStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: x.value }, { translateY: y.value }, { scale: scale.value }],
  }));

  if (!url) return null;

  return (
    <Modal visible transparent animationType="fade" onRequestClose={close}>
      {/*
        Modal の中身はネイティブの別ビュー階層に出るため、アプリ直下の
        GestureHandlerRootView の外側になる。ここに置き直さないと
        タップもピンチも一切届かない。
      */}
      <GestureHandlerRootView style={styles.backdrop}>
        <GestureDetector gesture={gesture}>
          <View style={styles.imageArea}>
            <AnimatedImage
              source={{ uri: url }}
              style={[styles.image, imageStyle]}
              contentFit="contain"
              transition={120}
            />
          </View>
        </GestureDetector>

        <View style={styles.bar}>
          <Text style={styles.hint} numberOfLines={1}>
            {saving ? '保存中…' : 'タップで拡大 ・ ピンチで拡縮 ・ 長押しで保存'}
          </Text>
          <Pressable onPress={save} hitSlop={10} disabled={saving}>
            <Text style={styles.action}>保存</Text>
          </Pressable>
          <Pressable onPress={() => WebBrowser.openBrowserAsync(url)} hitSlop={10}>
            <Text style={styles.action}>ブラウザ</Text>
          </Pressable>
          <Pressable onPress={close} hitSlop={10}>
            <Text style={styles.action}>閉じる</Text>
          </Pressable>
        </View>
      </GestureHandlerRootView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: '#000000ee' },
  imageArea: { flex: 1, overflow: 'hidden' },
  image: { flex: 1, width: '100%' },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    paddingBottom: spacing.xl,
    backgroundColor: colors.surface,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.border,
  },
  hint: { color: colors.textDim, fontSize: 11, flex: 1 },
  action: {
    color: colors.accentHover,
    fontSize: 13,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: radius / 2,
  },
});
