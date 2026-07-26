import { Image } from 'expo-image';
import * as WebBrowser from 'expo-web-browser';
import { memo } from 'react';
import { Pressable, StyleSheet, Text } from 'react-native';

import type { Segment } from '../parse/body';
import { colors, radius, spacing } from '../theme/colors';

interface Props {
  segments: Segment[];
  /** アンカーがタップされた。参照先をポップアップ表示するのに使う。 */
  onAnchorPress?: (from: number, to: number) => void;
  /** 画像を展開するか (モバイル回線では既定オフにする想定)。 */
  showImages?: boolean;
  fontSize?: number;
}

function PostBodyImpl({ segments, onAnchorPress, showImages = false, fontSize = 15 }: Props) {
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
            case 'link':
            case 'image':
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
          <Pressable key={`img-${i}`} onPress={() => WebBrowser.openBrowserAsync(seg.url)}>
            <Image source={{ uri: seg.url }} style={styles.thumb} contentFit="cover" transition={120} />
          </Pressable>
        ) : null
      )}
    </>
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
});

export const PostBody = memo(PostBodyImpl);
