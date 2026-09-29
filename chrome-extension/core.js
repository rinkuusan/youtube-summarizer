(function(root) {
  const videoId = url => {
    try { const u = new URL(url); return u.hostname === 'www.youtube.com' && u.pathname === '/watch' && /^[\w-]{11}$/.test(u.searchParams.get('v') || '') ? u.searchParams.get('v') : null; } catch { return null; }
  };
  function watchDelta(previous, current) {
    if (!previous || !previous.active || !current.active) return 0;
    const wall = (current.now - previous.now) / 1000;
    const media = current.time - previous.time;
    const expected = wall * current.rate;
    if (wall <= 0 || wall > 2.5 || media <= 0 || Math.abs(media - expected) > Math.max(.4, expected * .4)) return 0;
    return wall; // actual elapsed viewing time, not skipped media time or playback speed
  }
  root.VideoNotes = {videoId, watchDelta};
  if (typeof module !== 'undefined') module.exports = root.VideoNotes;
})(typeof globalThis === 'undefined' ? this : globalThis);
