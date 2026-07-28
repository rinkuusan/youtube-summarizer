import { Image } from 'expo-image';
import * as WebBrowser from 'expo-web-browser';
import { memo, useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { Segment } from '../parse/body';
import { colors, radius, spacing } from '../theme/colors';

interface Props {
  segments: Segment[];
  /** アンカーがタップされた。参照先をポップアップ表示するのに使う。 */
  onAnchorPress?: (from: number, to: number) => void;
  /** 画像を展開するか (モバイル回線では既定オフにする想定)。 */
  showImages?: boolean;
  /** サムネイルにぼかしを掛ける。タップで解除する。 */
  blurImages?: boolean;
  /** 画像をタップしたとき。ポップアップ表示は画面側が持つ。 */
  onImagePress?: (url: string) => void;
  fontSize?: number;
}

function PostBodyImpl({
  segments,
  onAnchorPress,
  showImages = false,
  blurImages = false,
  onImagePress,
  fontSize = 15,
}: Props) {
  const images = showImages ? segments.filter((s) => s.type === 'image') : [];

  return (
    <>
      <Text style={[styles.body, { fontSize, lineHeight: fontSize * 1.6 }]} selectable>
        {segments.map((seg, i) => {
          switch (seg.type) {
            case 'anchor':
              return (
                <Text key={i} style={styles.anchor} onPress={() => onAnchorPress?.(seg.from, seg.to)}>
                  {seg.text}
                </Text>
              );
            case 'image':
              // 画像リンクはブラウザに飛ばさず、その場でポップアップする。
              return (
                <Text key={i} style={styles.link} onPress={() => onImagePress?.(seg.url)}>
                  {seg.text}
                </Text>
              );
            case 'link':
              return (
                <Text key={i} style={styles.link} onPress={() => WebBrowser.openBrowserAsync(seg.url)}>
                  {seg.text}
                </Text>
              );
            default:
              return <Text key={i}>{seg.text}</Text>;
          }
        })}
      </Text>

      {images.map((seg, i) =>
        seg.type === 'image' ? (
          <Thumb key={`img-${i}`} url={seg.url} blurred={blurImages} onPress={onImagePress} />
        ) : null
      )}
    </>
  );
}

/**
 * サムネイル 1 枚。
 *
 * ぼかしが掛かっている間は、1 回目のタップでぼかしを外すだけにする。
 * ぼかしたまま拡大表示に飛ぶと、見たくないものを避ける意味が無くなる。
 */
function Thumb({
  url,
  blurred,
  onPress,
}: {
  url: string;
  blurred: boolean;
  onPress?: (url: string) => void;
}) {
  const [revealed, setRevealed] = useState(false);
  const hidden = blurred && !revealed;

  return (
    <Pressable onPress={() => (hidden ? setRevealed(true) : onPress?.(url))}>
      <Image
        source={{ uri: url }}
        style={styles.thumb}
        contentFit="cover"
        transition={120}
        blurRadius={hidden ? 60 : 0}
      />
      {hidden ? (
        <View style={styles.veil} pointerEvents="none">
          <Text style={styles.veilText}>閲覧注意 — タップで表示</Text>
        </View>
      ) : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  body: {
    color: colors.text,
  },
  anchor: {
    color: colors.accent,
    textDecorationLine: 'underline',
  },
  link: {
    color: colors.accentHover,
    textDecorationLine: 'underline',
  },
  thumb: {
    width: '100%',
    height: 220,
    borderRadius: radius / 2,
    marginTop: spacing.sm,
    backgroundColor: colors.surface2,
  },
  veil: {
    ...StyleSheet.absoluteFill,
    marginTop: spacing.sm,
    borderRadius: radius / 2,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#00000055',
  },
  veilText: {
    color: colors.text,
    fontSize: 13,
    fontWeight: '700',
  },
});

export const PostBody = memo(PostBodyImpl);
