const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const {videoId,watchDelta}=require('../chrome-extension/core.js');
const base={active:true,now:1000,time:1,rate:1};
assert.equal(videoId('https://www.youtube.com/watch?v=jNQXAC9IVRw'),'jNQXAC9IVRw');
assert.equal(videoId('https://evil.example/watch?v=jNQXAC9IVRw'),null);
assert.equal(watchDelta(base,{...base,now:2000,time:2}),1);
assert.equal(watchDelta(base,{...base,now:2000,time:20}),0);
assert.equal(watchDelta(base,{...base,now:2000,time:2,active:false}),0);
assert.equal(watchDelta(base,{...base,now:9000,time:9}),0);
assert.equal(watchDelta({...base,rate:2},{...base,now:2000,time:3,rate:2}),1);
let elapsed=0;for(let n=1;n<=180;n++)elapsed+=watchDelta({...base,now:n*1000,time:n},{...base,now:(n+1)*1000,time:n+1});assert.equal(elapsed,180);
const ext=path.join(__dirname,'../chrome-extension');
function worker(store={},extract=async()=>({text:'全文',lang:'ja',source:'fixture'}),fetcher=async()=>{throw Error('unexpected paid API')}){
  let listener,extractions=0,notifications=0,alarm;
  const context=vm.createContext({console,URL,Map,Date,Promise,AbortSignal,setTimeout,fetch:fetcher,importScripts(){},extractCaptions(){},chrome:{
    storage:{local:{async setAccessLevel(){},async get(key){return key===null?{...store}:{[key]:store[key]}},async set(value){Object.assign(store,value)},async remove(key){delete store[key]}}},
    scripting:{async executeScript(){extractions++;return[{result:await extract()}]}},notifications:{async create(){notifications++}},
    alarms:{async create(){},async clear(){},onAlarm:{addListener(fn){alarm=fn}}},runtime:{onStartup:{addListener(){}},onMessage:{addListener(fn){listener=fn}},async openOptionsPage(){}}
  }});
  vm.runInContext(fs.readFileSync(path.join(ext,'worker.js'),'utf8'),context);
  const send=msg=>new Promise(resolve=>listener(msg,{url:'https://www.youtube.com/watch?v=jNQXAC9IVRw',tab:{id:1}},resolve));
  return {send,store,context,counts:()=>({extractions,notifications})};
}
(async()=>{
  const w=worker(),request={type:'obtain',id:'jNQXAC9IVRw'};
  const results=await Promise.all([w.send(request),w.send(request)]);
  assert.equal(results[0].record.text,'全文');assert.equal(w.counts().extractions,1);assert.equal(w.counts().notifications,1);
  await w.send(request);assert.equal(w.counts().extractions,1);
  const restarted=worker(w.store);assert.equal((await restarted.send(request)).record.text,'全文');assert.equal(restarted.counts().extractions,0);
  const failure=worker({},async()=>{throw Error('字幕なし')});assert.match((await failure.send(request)).error,/字幕なし/);assert.equal(failure.counts().notifications,0);
  const unresolved=worker({'job:transcript:jNQXAC9IVRw:auto':{kind:'supadata',startedAt:0}});assert.match((await unresolved.send({...request,fallback:true})).error,/未確定/);
  const paid=worker({settings:{auto:true,language:'ja',supadataKey:'secret'}},undefined,async(url,options)=>{assert.equal(new URL(url).searchParams.get('mode'),'native');assert.equal(options.headers['x-api-key'],'secret');return{status:200,async json(){return{content:'API全文',lang:'ja'}}}});
  assert.equal((await paid.send({...request,fallback:true})).record.text,'API全文');
  assert.equal('supadataKey' in await paid.send({type:'settings'}),false);
  const captionCode=fs.readFileSync(path.join(ext,'captions.js'),'utf8');
  const data={videoDetails:{videoId:'jNQXAC9IVRw',title:'fixture'},captions:{playerCaptionsTracklistRenderer:{captionTracks:[{baseUrl:'https://www.youtube.com/api/timedtext?v=jNQXAC9IVRw',languageCode:'ja'}]}}};
  let current='jNQXAC9IVRw';
  const captionContext=vm.createContext({URL,AbortSignal,location:{get href(){return 'https://www.youtube.com/watch?v='+current}},window:{},document:{getElementById(){return{getPlayerResponse:()=>data}}},fetch:async()=>({ok:true,json:async()=>({events:Array.from({length:3000},(_,i)=>({segs:[{utf8:'字幕'+i}]}))})})});
  vm.runInContext(captionCode,captionContext);
  const complete=await vm.runInContext("extractCaptions('jNQXAC9IVRw','ja')",captionContext);assert.ok(complete.text.endsWith('字幕2999'));
  current='aaaaaaaaaaa';await assert.rejects(vm.runInContext("extractCaptions('jNQXAC9IVRw','ja')",captionContext),/切り替わり/);
  console.log('PASS: 180 seconds, pause/hidden/ad eligibility, seek, playback speed; cache, restart, deduplication, failures, native-only API, key isolation, 3000 caption segments, stale video rejection.');
})().catch(error=>{console.error(error);process.exitCode=1});
