importScripts('captions.js');
const defaults = {auto:true, language:'auto', supadataKey:''};
const running = new Map();
let apiQueue = Promise.resolve();
const boot = chrome.storage.local.setAccessLevel({accessLevel:'TRUSTED_CONTEXTS'});
const keyOf = (id, lang) => 'transcript:' + id + ':' + lang;
async function settings() { return {...defaults, ...(await chrome.storage.local.get('settings')).settings}; }
async function read(key) { return (await chrome.storage.local.get(key))[key]; }
async function notify(record) {
  try { await chrome.notifications.create('video:' + record.id, {type:'basic', iconUrl:'icon.png', title:'全文トランスクリプト完了！', message:record.title + '（' + record.text.length.toLocaleString() + '文字）'}); } catch { /* The page also displays completion. */ }
}
function validateSender(sender) { return sender.tab && sender.url?.startsWith('https://www.youtube.com/'); }
async function apiFetch(url, key) {
  const task = apiQueue.catch(()=>{}).then(async () => {
    const last = await read('lastApiCall') || 0;
    await new Promise(resolve => setTimeout(resolve, Math.max(0, 1100 - (Date.now() - last))));
    await chrome.storage.local.set({lastApiCall:Date.now()});
    const response = await fetch(url, {headers:{'x-api-key':key}, signal:AbortSignal.timeout(22000)});
    if (![200,202].includes(response.status)) throw new Error('Supadata HTTP ' + response.status + '。自動再試行はしません。キー・残高・利用上限を確認してください。');
    return {status:response.status, data:await response.json()};
  });
  apiQueue = task; return task;
}
async function storeResult(key, job, value) {
  const text = typeof value.content === 'string' ? value.content : value.text;
  if (!text?.trim()) throw new Error('字幕が空です。完了として保存しませんでした。');
  const record = {id:job.id, title:value.title || job.id, text, lang:value.lang || job.language, source:value.source || 'Supadata既存字幕', savedAt:Date.now()};
  await chrome.storage.local.set({[key]:record}); // success only after persistent storage succeeds
  await chrome.storage.local.remove('job:' + key);
  await notify(record); return {record};
}
async function obtain(message, sender) {
  const config = await settings(), id = message.id, language = config.language;
  if (!/^[\w-]{11}$/.test(id || '')) throw new Error('動画IDが不正です。');
  const key = keyOf(id, language), cached = await read(key);
  if (cached) return {record:cached};
  if (running.has(key)) return running.get(key);
  const task = (async () => {
    const previous = await read('job:' + key);
    if (previous?.kind === 'supadata') {
      if (previous.jobId) return {pending:true, message:'取得中です。結果を待っています。'};
      throw new Error('前回のAPI処理結果が未確定です。重複課金を避けて自動再送を止めています。設定で確認後に解除できます。');
    }
    if (previous && Date.now() - previous.startedAt < 60000) return {pending:true};
    const job = {id, language, startedAt:Date.now(), kind:message.fallback ? 'supadata' : 'direct'};
    await chrome.storage.local.set({['job:' + key]:job});
    try {
      if (!message.fallback) {
        const result = await chrome.scripting.executeScript({target:{tabId:sender.tab.id}, world:'MAIN', func:extractCaptions, args:[id, language]});
        const value = result[0]?.result;
        if (!value?.text) throw new Error('YouTubeから全文を取得できませんでした。字幕なし、またはページの再読み込みが必要です。');
        return await storeResult(key, job, value);
      }
      if (!config.supadataKey) throw new Error('設定画面にSupadata APIキーを入力してください。');
      const url = new URL('https://api.supadata.ai/v1/transcript');
      url.searchParams.set('url', 'https://www.youtube.com/watch?v=' + id);
      url.searchParams.set('text','true'); url.searchParams.set('mode','native');
      if (language !== 'auto') url.searchParams.set('lang',language);
      const value = await apiFetch(url.href, config.supadataKey);
      if (value.status === 202) {
        if (!value.data.jobId) throw new Error('APIの受付結果を確認できませんでした。');
        job.jobId = value.data.jobId; await chrome.storage.local.set({['job:' + key]:job});
        await chrome.alarms.create('poll-transcripts',{periodInMinutes:.5});
        return {pending:true};
      }
      return await storeResult(key, job, value.data);
    } catch (error) {
      if (job.kind === 'direct' || !config.supadataKey) await chrome.storage.local.remove('job:' + key);
      else { job.error = error.message; await chrome.storage.local.set({['job:' + key]:job}); }
      throw error;
    }
  })().finally(()=>running.delete(key));
  running.set(key,task); return task;
}
async function pollJobs() {
  await boot;
  const all = await chrome.storage.local.get(null), config = await settings();
  let pending = false;
  for (const [name, job] of Object.entries(all)) {
    if (!name.startsWith('job:') || !job.jobId || job.error) continue;
    if (Date.now() - job.startedAt > 15*60000) { job.error='取得が時間切れになりました。自動再送はしていません。'; await chrome.storage.local.set({[name]:job}); continue; }
    pending = true;
    try {
      const result = await apiFetch('https://api.supadata.ai/v1/transcript/' + encodeURIComponent(job.jobId), config.supadataKey);
      if (result.data.content) await storeResult(name.slice(4), job, result.data);
      else if (result.data.status === 'failed' || result.data.error) { job.error='字幕取得ジョブが失敗しました。'; await chrome.storage.local.set({[name]:job}); }
    } catch (error) { job.error=error.message; await chrome.storage.local.set({[name]:job}); }
  }
  if (!pending) await chrome.alarms.clear('poll-transcripts');
}
chrome.alarms.onAlarm.addListener(alarm => { if (alarm.name === 'poll-transcripts') pollJobs().catch(()=>{}); });
chrome.runtime.onStartup.addListener(() => chrome.alarms.create('poll-transcripts',{periodInMinutes:.5}));
chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (!validateSender(sender)) return;
  (async () => {
    await boot;
    const config = await settings();
    if (message.type === 'settings') return {auto:config.auto, language:config.language};
    if (message.type === 'obtain') return obtain(message,sender);
    if (message.type === 'status') {
      const key = keyOf(message.id,config.language), record = await read(key), job=await read('job:' + key);
      if (!record && job?.kind === 'direct' && Date.now()-job.startedAt > 60000 && !running.has(key)) {
        await chrome.storage.local.remove('job:' + key);
        return {error:'字幕取得が中断しました。もう一度取得してください。', pending:false};
      }
      return {record, error:job?.error, pending:!!job};
    }
    if (message.type === 'options') { await chrome.runtime.openOptionsPage(); return {}; }
    throw new Error('未対応の操作です。');
  })().then(reply, error=>reply({error:error.message})); return true;
});
