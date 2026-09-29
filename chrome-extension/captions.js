// This self-contained function runs in the YouTube MAIN world. No extension secrets enter it.
async function extractCaptions(expectedId, language) {
  function currentId() { return new URL(location.href).searchParams.get('v'); }
  if (currentId() !== expectedId) throw new Error('動画が切り替わりました。');
  const player = document.getElementById('movie_player');
  let response = player?.getPlayerResponse?.();
  if (response?.videoDetails?.videoId !== expectedId) response = window.ytInitialPlayerResponse;
  if (response?.videoDetails?.videoId !== expectedId) throw new Error('現在の動画情報を取得できません。ページを再読み込みしてください。');
  if (response.playabilityStatus?.liveStreamability || response.videoDetails.isLive || (response.videoDetails.isLiveContent && !response.videoDetails.lengthSeconds)) throw new Error('配信中の動画の全文はまだ確定していません。');
  const title = response.videoDetails.title || expectedId;
  const tracks = response.captions?.playerCaptionsTracklistRenderer?.captionTracks || [];
  const track = tracks.find(t => t.languageCode === language && t.kind !== 'asr') || tracks.find(t => t.languageCode === language) || tracks.find(t => t.kind !== 'asr') || tracks[0];
  let actualLanguage = track?.languageCode || language;
  if (track) {
    const url = new URL(track.baseUrl);
    if (url.protocol !== 'https:' || !['www.youtube.com', 'youtube.com'].includes(url.hostname) || url.pathname !== '/api/timedtext') throw new Error('字幕URLを確認できません。');
    url.searchParams.set('fmt', 'json3');
    try {
      const result = await fetch(url.href, {credentials:'include', signal:AbortSignal.timeout(18000)});
      if (result.ok) {
        const data = await result.json();
        const text = (data.events || []).filter(e => e.segs).map(e => e.segs.map(s => s.utf8 || '').join('')).join('\n').trim();
        if (text && currentId() === expectedId) return {text, lang:actualLanguage, title, source:'YouTube字幕'};
      }
    } catch { /* Some YouTube players require the transcript endpoint instead. */ }
  }
  const initial = document.querySelector('ytd-watch-flexy')?.data || window.ytInitialData;
  let endpoint;
  function findEndpoint(node) {
    if (!node || typeof node !== 'object' || endpoint) return;
    if (node.getTranscriptEndpoint?.params) { endpoint = node.getTranscriptEndpoint; return; }
    for (const value of Object.values(node)) findEndpoint(value);
  }
  findEndpoint(initial);
  if (!endpoint) throw new Error('全文字幕の取得口が見つかりません。字幕なし、またはYouTube側の仕様変更の可能性があります。');
  const context = window.ytcfg?.get?.('INNERTUBE_CONTEXT');
  if (!context) throw new Error('YouTubeの字幕取得情報を読み込めませんでした。');
  const result = await fetch('https://www.youtube.com/youtubei/v1/get_transcript?prettyPrint=false', {
    method:'POST', credentials:'include', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({context, params:endpoint.params}), signal:AbortSignal.timeout(18000)
  });
  if (!result.ok) throw new Error('YouTube字幕の取得に失敗しました（HTTP ' + result.status + '）。');
  const data = await result.json(), segments = [];
  let continuation = false;
  function walk(node) {
    if (!node || typeof node !== 'object') return;
    if (node.continuationItemRenderer || node.continuationCommand || node.nextContinuationData) continuation = true;
    if (node.transcriptSegmentRenderer) {
      const n = node.transcriptSegmentRenderer;
      segments.push({start:Number(n.startMs || 0), text:n.snippet?.runs?.map(r=>r.text || '').join('') || n.snippet?.simpleText || ''}); return;
    }
    if (node.transcriptSegmentViewModel) {
      const n = node.transcriptSegmentViewModel;
      segments.push({start:Number(n.startMs || 0), text:n.snippet?.content || ''}); return;
    }
    for (const value of Object.values(node)) walk(value);
  }
  walk(data);
  if (continuation) throw new Error('字幕に未取得の続きがあります。全文として保存せず停止しました。');
  const text = segments.sort((a,b)=>a.start-b.start).map(s=>s.text).join('\n').trim();
  if (!text) throw new Error('全文字幕を取得できませんでした。');
  if (currentId() !== expectedId) throw new Error('動画が切り替わりました。');
  // Endpoint language may differ from the preferred track; do not claim it was verified.
  return {text, lang:'auto', title, source:'YouTube全文字幕'};
}
