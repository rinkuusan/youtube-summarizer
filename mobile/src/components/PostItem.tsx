import { memo } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { Segment } from '../parse/body';
import type { Post } from '../parse/datLine';
import { colors, spacing } from '../theme/colors';
import { PostBody } from './PostBody';

interface Props {
  post: Post;
  /** 本文のトークン列。画面側で 1 回だけパースして渡す (二重パースを避ける)。 */
  segments: Segment[];
  /** このレスに返信しているレス番号 (逆参照)。 */
  replies?: number[];
  onAnchorPress?: (from: number, to: number) => void;
  onRepliesPress?: (res: number) => void;
  onIdPress?: (uid: string) => void;
  showImages?: boolean;
  fontSize?: number;
}

function PostItemImpl({
  post,
  segments,
  replies,
  onAnchorPress,
  onRepliesPress,
  onIdPress,
  showImages,
  fontSize,
}: Props) {
  if (post.isAbone) {
    return (
      <View style={styles.container}>
        <Text style={styles.abone}>{post.res} あぼーん</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.res}>{post.res}</Text>
        <Text style={styles.name} numberOfLines={1}>
          {post.name || '名無し'}
          {post.trip ? <Text style={styles.trip}> {post.trip}</Text> : null}
          {post.isCap ? <Text style={styles.cap}> ★</Text> : null}
        </Text>
      </View>

      <View style={styles.metaRow}>
        <Text style={styles.meta}>{post.dateText.replace(/\s*ID:.*$/, '').replace(/\s*BE:.*$/, '')}</Text>
        {post.uid ? (
          <Pressable onPress={() => onIdPress?.(post.uid!)} hitSlop={6}>
            <Text style={styles.uid}>ID:{post.uid}</Text>
          </Pressable>
        ) : null}
        {post.wacchoi ? <Text style={styles.wacchoi}>({post.wacchoi})</Text> : null}
      </View>

      <PostBody
        segments={segments}
        onAnchorPress={onAnchorPress}
        showImages={showImages}
        fontSize={fontSize}
      />

      {replies && replies.length > 0 ? (
        <Pressable onPress={() => onRepliesPress?.(post.res)} hitSlop={6} style={styles.repliesChip}>
          <Text style={styles.repliesText}>返信 {replies.length}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: spacing.sm,
  },
  res: {
    color: colors.accent,
    fontVariant: ['tabular-nums'],
    fontWeight: '700',
    fontSize: 13,
    minWidth: 28,
  },
  name: {
    color: colors.success,
    fontSize: 13,
    fontWeight: '600',
    flexShrink: 1,
  },
  trip: { color: colors.textDim, fontWeight: '400' },
  cap: { color: '#fbbf24' },
  metaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: spacing.sm,
    marginTop: 2,
    marginBottom: spacing.sm,
  },
  meta: { color: colors.meta, fontSize: 11 },
  uid: { color: colors.meta, fontSize: 11, textDecorationLine: 'underline' },
  wacchoi: { color: colors.textDim, fontSize: 11 },
  abone: { color: colors.textDim, fontSize: 13, fontStyle: 'italic' },
  repliesChip: {
    alignSelf: 'flex-start',
    marginTop: spacing.sm,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: 999,
    backgroundColor: colors.surface2,
  },
  repliesText: { color: colors.accentHover, fontSize: 11 },
});

export const PostItem = memo(PostItemImpl);
