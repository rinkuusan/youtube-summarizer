const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
class Element {
  constructor(){ this.listeners={}; this.value=''; this.children=[]; this.dataset={}; this.disabled=false; this.hidden=false; this.style={}; this.classList={add(){},remove(){},toggle(){}}; }
  dispatchEvent(){} addEventListener(name,fn){this.listeners[name]=fn;} setAttribute(){} focus(){} append(...items){this.children.push(...items)} replaceChildren(){this.children=[]}
}
const elements=new Map(); const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const context=vm.createContext({console,URL,Set,TextDecoder,TextEncoder,Response,ReadableStream,AbortSignal,FormData,Blob,setTimeout,clearTimeout,document:{getElementById:element,querySelector:selector=>({value:selector.includes('mode')?'transcript':'auto'}),querySelectorAll:()=>[],createElement:()=>new Element()},localStorage:{getItem:()=>'',setItem:()=>{}},navigator:{clipboard:{writeText:async()=>{}}},fetch:async()=>new Response('{}')});
context.window=context;context.AndroidClipboard={paste(){},copy(){}};context.Event=class {};
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../assets/index.html'),'utf8').split('<script>')[1].split('</script>')[0],context);
const run=code=>vm.runInContext(code,context);
function stream(events,tail=false){ const bytes=new TextEncoder().encode(': keepalive\n\n'+events.map(e=>'data: '+JSON.stringify(e)).join('\n\n')+(tail?'':'\n\n'));return new Response(new ReadableStream({start(c){for(let i=0;i<bytes.length;i+=3)c.enqueue(bytes.slice(i,i+3));c.close()}})); }
(async()=>{
  assert.equal(run("parseVideoUrls('https://youtu.be/jNQXAC9IVRw https://www.youtube.com/watch?v=jNQXAC9IVRw&t=1').length"),1);
  assert.equal(run("parseVideoUrls('https://youtube.com/shorts/jNQXAC9IVRw\\n'.trim()).length"),1);
  assert.throws(()=>run("parseVideoUrls('https://youtube.com.evil.test/watch?v=jNQXAC9IVRw')"));
  assert.throws(()=>run("parseVideoUrls('')"));
  assert.throws(()=>run("parseVideoUrls(Array.from({length:21},(_,i)=>'https://youtu.be/'+String(i).padStart(11,'0')).join(' '))"));
  context.res=stream([{type:'status',message:'日本語'},{type:'progress',value:90},{type:'result',text:'全文テスト🙂'}],true);
  assert.equal(await run('readSSE(res, {status(){},progress(){}})'),'全文テスト🙂');
  context.res=stream([{type:'status',message:'unfinished'}]); await assert.rejects(run('readSSE(res, {status(){},progress(){}})'),/接続が終了/);
  const calls=[];
  context.fetch=async(url,options)=>{const body=JSON.parse(options.body);calls.push(body);return calls.length===2?stream([{type:'error',message:'字幕なし'}]):stream([{type:'result',text:'全文 <script>alert(1)</script> '+calls.length}]);};
  await run("processBatch(['https://youtu.be/aaaaaaaaaaa','https://youtu.be/bbbbbbbbbbb','https://youtu.be/ccccccccccc'],'transcript','ja','test-key')");
  assert.equal(calls.length,3);assert.equal(run("batchItems.map(x=>x.state).join(',')"),'done,error,done');assert.equal(run('batchItems[0].body.textContent'),'全文 <script>alert(1)</script> 1');assert.equal(run('batchItems[2].text'),'全文 <script>alert(1)</script> 3');assert.equal(calls[0].language,'ja');assert.equal(calls[0].groq_api_key,'test-key');
  assert.match(run("renderMarkdown('<img src=x onerror=alert(1)>')"),/&lt;img/);
  let stoppedCalls=0;context.fetch=async()=>{stoppedCalls++;run('stopRequested=true');return stream([{type:'result',text:'first'}]);};
  await run("processBatch(['https://youtu.be/aaaaaaaaaaa','https://youtu.be/bbbbbbbbbbb'],'prompt','auto','')");assert.equal(stoppedCalls,1);assert.equal(run("batchItems.map(x=>x.state).join(',')"),'done,stopped');
  context.fetch=async(url,options)=>{assert.equal(url,'/process-file');assert.equal(options.body.get('mode'),'transcript');return stream([{type:'result',text:'音声全文'}]);};
  run("activeTab='file'; selectedFile=new Blob(['audio'],{type:'audio/wav'}); selectedFile.name='test.wav'");await run('startProcess()');assert.equal(run('currentRawText'),'音声全文');assert.equal(run('processing'),false);
  run("processing=false; urlInput.value='https://www.youtube.com/watch?v=aaaaaaaaaaa'");
  assert.equal(run("window.importYouTubeUrls(['https://youtu.be/aaaaaaaaaaa','https://youtu.be/bbbbbbbbbbb'])"),true);
  assert.equal(run("urlInput.value.split('\\n').length"),2);
  assert.equal(run("activeTab"),'url');
  run("processing=true");
  assert.equal(run("window.importYouTubeUrls(['https://youtu.be/ccccccccccc'])"),false);
  assert.equal(run("urlInput.value.split('\\n').length"),2);
  run("processing=false; urlInput.value=Array.from({length:20},(_,i)=>'https://youtu.be/'+String(i).padStart(11,'0')).join('\\n')");
  assert.equal(run("window.importYouTubeUrls(['https://youtu.be/ccccccccccc'])"),false);
  const shared=[]; context.AndroidClipboard.share=text=>shared.push(text);
  context.fetch=async()=>stream([{type:'result',text:'共有する本文🙂'}]);
  await run("processBatch(['https://youtu.be/aaaaaaaaaaa','https://youtu.be/bbbbbbbbbbb'],'transcript','ja','')");
  assert.equal(run('shareAllBtn.disabled'),false);
  run("shareAllBtn.listeners.click()");
  assert.equal(shared[0],'https://youtu.be/aaaaaaaaaaa\n\n共有する本文🙂\n\n---\n\nhttps://youtu.be/bbbbbbbbbbb\n\n共有する本文🙂');
  run('batchItems[1].share.listeners.click()'); assert.equal(shared[1],'https://youtu.be/bbbbbbbbbbb\n\n共有する本文🙂');
  run("batchItems[1].state='error'; shareAllBtn.listeners.click()"); assert.ok(!shared[2].includes('bbbbbbbbbbb'));
  run("currentRawText='音声結果'; document.getElementById('shareResultBtn').listeners.click()"); assert.equal(shared[3],'音声結果');
  run("shareText('')"); assert.equal(shared.length,4);
  const longText='長文🙂'.repeat(80000); context.longText=longText; run('shareText(longText)'); assert.equal(shared[4],longText);
  console.log('PASS: sharing all/single/audio, exclude failed, preserve long text.');
  console.log('PASS: native import append/dedup/busy deferral/overflow; URL normalization/deduplication/limit; fragmented SSE and missing result; 3-video batch continues after failure; plain-text safety; stop after current; audio upload regression.');
})().catch(e=>{console.error(e);process.exitCode=1});
