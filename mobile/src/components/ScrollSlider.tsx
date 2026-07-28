import { useCallback, useRef, useState } from 'react';
import { PanResponder, StyleSheet, Text, View } from 'react-native';

import { colors } from '../theme/colors';

/**
 * 右端の高速スクローラ。
 *
 * 長いスレを指でなぞって一気に上下するためのもの。つまみを掴んでいる間だけ
 * 濃く出し、離すと薄く戻す。位置は割合で渡すだけにして、実際のスクロールは
 * 呼び出し側 (FlatList を持っている側) に任せる。
 *
 * PanResponder を使っているのは、これが「掴んで動かす」だけの単純な操作で、
 * reanimated を挟むほどの滑らかさを必要としないため。
 */

interface Props {
  /** 0..1。現在のスクロール位置。 */
  progress: number;
  /** つまみが動いたとき。0..1 を渡す。 */
  onScrub: (ratio: number) => void;
  /** つまみの横に出す文字 (レス番号など)。 */
  label?: string;
  /** トラックの高さ。 */
  height: number;
}

const THUMB = 44;

export function ScrollSlider({ progress, onScrub, label, height }: Props) {
  const [dragging, setDragging] = useState(false);
  const trackHeight = Math.max(height - THUMB, 1);
  // PanResponder のコールバックは生成時の値を捕まえるので、ref で最新を渡す。
  const scrubRef = useRef(onScrub);
  scrubRef.current = onScrub;
  const trackRef = useRef(trackHeight);
  trackRef.current = trackHeight;

  const toRatio = useCallback((y: number) => {
    return Math.min(Math.max(y / trackRef.current, 0), 1);
  }, []);

  const responder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,
      onPanResponderGrant: (e) => {
        setDragging(true);
        scrubRef.current(toRatio(e.nativeEvent.locationY));
      },
      onPanResponderMove: (e) => {
        scrubRef.current(toRatio(e.nativeEvent.locationY));
      },
      onPanResponderRelease: () => setDragging(false),
      onPanResponderTerminate: () => setDragging(false),
    })
  ).current;

  const top = Math.min(Math.max(progress, 0), 1) * trackHeight;

  return (
    <View style={[styles.track, { height }]} {...responder.panHandlers}>
      <View style={[styles.thumb, dragging && styles.thumbOn, { top }]}>
        <Text style={styles.grip}>≡</Text>
      </View>
      {dragging && label ? (
        <View style={[styles.bubble, { top: Math.max(top - 6, 0) }]}>
          <Text style={styles.bubbleText}>{label}</Text>
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  track: {
    position: 'absolute',
    right: 0,
    top: 0,
    width: 28,
    justifyContent: 'flex-start',
  },
  thumb: {
    position: 'absolute',
    right: 2,
    width: 24,
    height: THUMB,
    borderRadius: 12,
    backgroundColor: colors.surface2,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.55,
  },
  thumbOn: { opacity: 1, backgroundColor: colors.accent, borderColor: colors.accent },
  grip: { color: colors.text, fontSize: 14, lineHeight: 16 },
  bubble: {
    position: 'absolute',
    right: 34,
    backgroundColor: colors.accent,
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  bubbleText: { color: '#fff', fontSize: 13, fontWeight: '700' },
});
