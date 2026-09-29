(() => {
  const host = document.createElement('div'); host.id='video-notes-panel';
  host.style.cssText='position:fixed;right:16px;bottom:18px;z-index:2147483000';
  const root = host.attachShadow({mode:'closed'});
  root.innerHTML = `<style>
    :host{font:13px/1.5 system-ui;color:#f3f0e8}details{width:340px;background:#242722;border:1px solid #8a9974;box-shadow:0 3px 15px #0005}
    summary{cursor:pointer;padding:8px 12px;background:#39442e;color:#f1ebdc;font-weight:600}.body{padding:10px}
    button{font:inherit;border:1px solid #82906b;background:#424d37;color:#fff;padding:5px 8px;cursor:pointer;margin:2px}button:disabled{opacity:.4;cursor:default}
    textarea{box-sizing:border-box;width:100%;height:180px;background:#171b16;color:#eee;border:1px solid #657357;padding:7px;margin-top:8px;resize:vertical}
    .status{margin:8px 0;overflow-wrap:anywhere}small{color:#c5cbbb} [hidden]{display:none!important}
    @media(max-width:600px){details{width:280px}}
  </style><details><summary>動画ノート · 全文字幕</summary><div class="body">
    <button id="get">全文を取得</button><button id="settings">設定</button>
    <div class="status" id="status" role="status" aria-live="polite">準備中</div><small id="timer"></small>
    <div><button id="fallback" hidden>Supadataで既存字幕を取得（クレジット消費）</button></div>
    <textarea id="text" aria-label="取得した全文" readonly hidden></textarea>
    <div><button id="copy" disabled>コピー</button><button id="save" disabled>TXT保存</button></div>
  </div></details>`;
  document.documentElement.append(host);
  const $ = id=>root.getElementById(id);
  let id=null, version=0, elapsed=0, previous=null, triggered=false, busy=false, record=null, config={auto:true,language:'auto'}, tick=0;
  const message = value=>chrome.runtime.sendMessage(value);
  function setRecord(value) {
    record=value; $('text').value=value.text; $('text').hidden=false;
    $('copy').disabled=false; $('save').disabled=false; $('fallback').hidden=true;
    $('status').textContent='全文トランスクリプト完了！ ' + value.text.length.toLocaleString() + '文字 / ' + value.source;
    busy=false; $('get').disabled=false; triggered=true;
  }
  async function obtain(fallback=false) {
    if (busy || !id) return;
    if (fallback && !confirm('Supadataの既存字幕取得を1回実行します。クレジットを消費します。AI文字起こしは行いません。')) return;
    const requestVersion=version;
    busy=true; $('get').disabled=true; $('status').textContent='全文取得中…'; $('fallback').hidden=true;
    try {
      const result=await message({type:'obtain',id,fallback});
      if(requestVersion!==version) return;
      if(result.error) throw new Error(result.error);
      if(result.record) setRecord(result.record);
      else $('status').textContent='全文取得中…保存完了を待っています。';
    } catch(error) {
      if(requestVersion!==version) return;
      busy=false; $('get').disabled=false; $('status').textContent=error.message; $('fallback').hidden=false;
    }
  }
  async function refreshStatus() {
    const requestVersion=version;
    try {
      config=await message({type:'settings'});
      const result=await message({type:'status',id});
      if(requestVersion!==version) return;
      if(result.record) setRecord(result.record);
      else if(result.error) { busy=false; $('get').disabled=false; $('status').textContent=result.error; }
      else if(busy && !result.pending) { busy=false; $('get').disabled=false; $('status').textContent='処理が中断しました。もう一度取得できます。'; }
    } catch { if(requestVersion===version) $('status').textContent='拡張との接続が切れました。ページを再読み込みしてください。'; }
  }
  $('get').onclick=()=>obtain(); $('fallback').onclick=()=>obtain(true);
  $('settings').onclick=()=>message({type:'options'});
  $('copy').onclick=async()=>{try {await navigator.clipboard.writeText(record.text); $('status').textContent='全文をコピーしました。';} catch {$('status').textContent='コピーできませんでした。本文を選択するかTXT保存を使ってください。';}};
  $('save').onclick=()=>{const url=URL.createObjectURL(new Blob([record.text],{type:'text/plain;charset=utf-8'})); const a=document.createElement('a'); a.href=url; a.download=id+'-'+record.lang+'.txt'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);};
  function reset(next) {
    version++; id=next; elapsed=0; previous=null; triggered=false; busy=false; record=null;
    host.hidden=!id; $('text').value=''; $('text').hidden=true; $('copy').disabled=true; $('save').disabled=true; $('get').disabled=false; $('fallback').hidden=true;
    $('status').textContent='手動取得、または実視聴3分で自動取得します。';
    if(id) refreshStatus();
  }
  setInterval(()=>{
    const next=VideoNotes.videoId(location.href); if(next!==id) reset(next);
    if(!id) {host.hidden=true;return;}
    const video=document.querySelector('#movie_player video'), player=document.getElementById('movie_player');
    const sample={now:performance.now(),time:video?.currentTime || 0,rate:video?.playbackRate || 1,
      active:!!video && !document.hidden && document.hasFocus() && !video.paused && !video.ended && !video.seeking && video.readyState>=3 && !player?.classList.contains('ad-showing') && !player?.classList.contains('ad-interrupting')};
    if(!triggered) elapsed+=VideoNotes.watchDelta(previous,sample); previous=sample;
    $('timer').textContent=config.auto ? '実視聴 '+Math.min(180,Math.floor(elapsed))+' / 180秒（広告・停止・非表示を除外）' : '自動取得 OFF';
    if(config.auto && elapsed>=180 && !triggered) {triggered=true;obtain();}
    if(++tick%6===0 && !record) refreshStatus();
  },1000);
})();
