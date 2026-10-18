const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');const root=path.resolve(__dirname,'..');
(async()=>{
const b=await chromium.launch({channel:'msedge',headless:true,args:['--disable-gpu']});
try{
const p=await b.newPage({viewport:{width:1920,height:1080}});
const errs=[];p.on('pageerror',e=>errs.push(e.message));
async function shot(name){
 await p.getByRole('button',{name:'Stop',exact:true}).waitFor({state:'hidden',timeout:60000});
 await p.evaluate(()=>document.fonts.ready);
 await p.screenshot({path:path.join(root,'screenshots',name),fullPage:true});
}
await p.goto('http://127.0.0.1:8501/',{waitUntil:'networkidle'});
await p.getByRole('link',{name:/进入影像工作台/}).waitFor();
// Exercise the actual hover animation and restoration on all five cards.
for(const route of ['workspace','cases','compare','statistics','consultation']){
 const card=p.locator('.st-key-glass_card_'+route);
 await card.hover();
 await p.waitForTimeout(100);
 if(await card.evaluate(e=>getComputedStyle(e).animationName)!=='glass-jelly')throw Error(route+' missing jelly animation');
 await p.waitForTimeout(700);
 const enlarged=await card.evaluate(e=>new DOMMatrixReadOnly(getComputedStyle(e).transform).a);
 if(Math.abs(enlarged-1.035)>.005)throw Error(route+' hover scale incorrect');
 await p.mouse.move(10,100);await p.waitForTimeout(650);
 const restored=await card.evaluate(e=>new DOMMatrixReadOnly(getComputedStyle(e).transform).a);
 if(Math.abs(restored-1)>.005)throw Error(route+' did not restore');
}
await p.emulateMedia({reducedMotion:'reduce'});
await p.locator('.st-key-glass_card_workspace').hover();
if(await p.locator('.st-key-glass_card_workspace').evaluate(e=>getComputedStyle(e).animationName)!=='none')throw Error('Reduced motion not respected');
await p.mouse.move(10,100);await p.emulateMedia({reducedMotion:'no-preference'});
await shot('26_空间玻璃首页1920.png');
for(const [name,route,id] of [['影像工作台','workspace','27'],['案例库','cases','28'],['影像对比','compare','29'],['统计分析','statistics','30'],['健康问诊','consultation','31']]){
 await p.getByRole('link',{name:new RegExp('进入'+name)}).click();
 await p.waitForURL('**/'+route);
 const ready={workspace:/开始分析/,cases:/搜索案例/,compare:/影像对比/,statistics:/统计分析/,consultation:/清空对话/};
 if(route==='workspace')await p.getByRole('button',{name:ready[route]}).waitFor();
 else if(route==='cases')await p.getByRole('textbox',{name:ready[route]}).waitFor();
 else if(route==='consultation')await p.getByRole('button',{name:ready[route]}).waitFor();
 else await p.locator('[data-testid="stCaptionContainer"]').last().waitFor();
 await p.waitForTimeout(1500);
 await shot(id+'_空间玻璃'+name+'.png');
 const exception=await p.locator('[data-testid="stException"]').count();
 if(exception)throw Error(name+' page exception: '+await p.locator('[data-testid="stException"]').innerText());
 const bg=await p.locator('[data-testid="stAppViewContainer"]').evaluate(e=>getComputedStyle(e).backgroundImage);
 if(!bg.includes('data:image/png'))throw Error(name+' background missing');
 const glass=p.locator('.st-key-feature_surface');
 if(!await glass.count() || !await glass.evaluate(e=>getComputedStyle(e).backdropFilter.includes('blur')))throw Error(name+' shared glass surface missing');
 await p.getByRole('link',{name:/首页/}).click();
 await p.getByRole('link',{name:/进入影像工作台/}).waitFor();
}
await p.setViewportSize({width:1440,height:900});await shot('32_空间玻璃首页1440.png');
await p.setViewportSize({width:1280,height:800});await shot('33_空间玻璃首页1280.png');
if(await p.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth))throw Error('Horizontal overflow');
if(errs.length)throw Error(errs.join('\n'));
fs.writeFileSync(path.join(root,'outputs','spatial_ui_qa.json'),JSON.stringify({checked_at:new Date().toISOString(),errors:errs,routes:['workspace','cases','compare','statistics','consultation'],viewports:[1920,1440,1280]},null,2));
console.log('Five glass cards, return-home routing, all page backgrounds and three desktop widths passed');
}finally{await b.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
