/**
 * 配色は既存の Web 版 (static/index.html) と揃えている。
 * 同じ作者の 2 つのアプリが兄弟に見えるようにするため。
 */
export const colors = {
  bg: '#0b0b10',
  surface: '#16161e',
  surface2: '#1e1e2a',
  border: '#2a2a3a',
  accent: '#7c6cff',
  accentHover: '#9b8fff',
  text: '#eaeaf0',
  textDim: '#7a7a8e',
  success: '#4ade80',
  error: '#f87171',
  /** レス番号・ID などの補助情報 */
  meta: '#8a8aa0',
} as const;

export const radius = 16;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
} as const;
