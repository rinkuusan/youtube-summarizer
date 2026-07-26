import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { logError, log } from '@/net/log';
import { colors, radius, spacing } from '@/theme/colors';

/**
 * 描画中の例外で画面が真っ白になるのを防ぐ。
 * release APK では白画面になると何も分からないので、その場でエラーを出す。
 */

interface Props {
  children: ReactNode;
  /** 「ログを見る」を押したときの遷移。 */
  onShowLogs?: () => void;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    logError('render', error, '描画中の例外');
    if (info.componentStack) log('error', 'render', 'componentStack', info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <View style={styles.root}>
        <Text style={styles.title}>画面の描画に失敗しました</Text>
        <ScrollView style={styles.box}>
          <Text style={styles.message}>{error.message}</Text>
          {error.stack ? <Text style={styles.stack}>{error.stack}</Text> : null}
        </ScrollView>
        <View style={styles.actions}>
          <Pressable style={styles.btn} onPress={() => this.setState({ error: null })}>
            <Text style={styles.btnText}>再描画</Text>
          </Pressable>
          {this.props.onShowLogs ? (
            <Pressable style={styles.btn} onPress={this.props.onShowLogs}>
              <Text style={styles.btnText}>ログを見る</Text>
            </Pressable>
          ) : null}
        </View>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg, padding: spacing.lg, gap: spacing.md },
  title: { color: colors.error, fontSize: 16, fontWeight: '700', marginTop: spacing.xl },
  box: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radius,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
    padding: spacing.md,
  },
  message: { color: colors.text, fontSize: 14, marginBottom: spacing.md },
  stack: { color: colors.textDim, fontSize: 11, fontFamily: 'monospace' },
  actions: { flexDirection: 'row', gap: spacing.md },
  btn: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius,
    backgroundColor: colors.surface2,
  },
  btnText: { color: colors.text, fontSize: 14 },
});
