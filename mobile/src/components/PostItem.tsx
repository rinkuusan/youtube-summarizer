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
  /** サムネイルにぼかしを掛ける。 */
  blurImages?: boolean;
  /** 画像タップ。ポップアップ表示は画面側が持つ。 */
  onImagePress?: (url: string) => void;
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
  blurImages,
  onImagePress,
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
    <View style={[styles.container, post.isMine && styles.mine]}>
      <View style={styles.header}>
        <Text style={[styles.res, post.isMine && styles.resMine]}>{post.res}</Text>
        {post.isMine ? <Text style={styles.mineBadge}>自分</Text> : null}
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
        blurImages={blurImages}
        onImagePress={onImagePress}
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
  /**
   * 自分のレス。スレを流し読みしていて目に留まる程度に留める。
   * 左の縦線と薄い下地だけで、本文の可読性は変えない。
   */
  mine: {
    backgroundColor: '#7c6cff14',
    borderLeftWidth: 3,
    borderLeftColor: colors.accent,
    paddingLeft: spacing.lg - 3,
  },
  resMine: { fontWeight: '700' },
  mineBadge: {
    color: colors.accent,
    fontSize: 10,
    fontWeight: '700',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.accent,
    borderRadius: 4,
    paddingHorizontal: 4,
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
