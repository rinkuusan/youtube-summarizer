const vm = require('node:vm');
const fs = require('node:fs');
const assert = require('node:assert/strict');
class Element {
  constructor(){ this.value=''; this.children=[]; this.dataset={}; this.disabled=false; this.hidden=false; this.style={}; this.classList={add(){},remove(){},toggle(){}}; }
  addEventListener(){} setAttribute(){} focus(){} append(...items){this.children.push(...items)} replaceChildren(){this.children=[]}
}
const elements=new Map(); const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id)};
const context=vm.createContext({console,URL,Set,TextDecoder,TextEncoder,Response,ReadableStream,AbortSignal,FormData,Blob,setTimeout,clearTimeout,document:{getElementById:element,querySelector:selector=>({value:selector.includes('mode')?'transcript':'auto'}),querySelectorAll:()=>[],createElement:()=>new Element()},localStorage:{getItem:()=>'',setItem:()=>{}},navigator:{clipboard:{writeText:async()=>{}}},fetch:async()=>new Response('{}')});
context.window=context;
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'../static/index.html'),'utf8').split('<script>')[1].split('</script>')[0],context);
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
  console.log('PASS: URL normalization/deduplication/limit; fragmented SSE and missing result; 3-video batch continues after failure; plain-text safety; stop after current; audio upload regression.');
})().catch(e=>{console.error(e);process.exitCode=1});
