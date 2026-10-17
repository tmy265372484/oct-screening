const {chromium}=require('playwright');
const path=require('path'), fs=require('fs');
const root=path.resolve(__dirname,'..');
(async()=>{
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--disable-gpu']});
try{
 const page=await browser.newPage({viewport:{width:1920,height:1180}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 async function settled(){
  await page.getByRole('button',{name:'Stop',exact:true}).waitFor({state:'hidden',timeout:60000});
  await page.evaluate(()=>document.fonts.ready);
  await page.locator('img').evaluateAll(xs=>Promise.all(xs.map(i=>i.complete?null:new Promise(r=>{i.onload=r;i.onerror=r}))));
 }
 async function shot(name){await settled();await page.screenshot({path:path.join(root,'screenshots',name),fullPage:true});}
 await page.goto(process.argv[2]||'http://127.0.0.1:8501/',{waitUntil:'networkidle'});
 await page.getByRole('button',{name:/开始分析/}).waitFor();
 await shot('17_用户端影像导入.png');
 // Exercise actual file upload, not only canned examples.
 await page.locator('input[type=file]').setInputFiles(path.join(root,'assets/demo/CNV.png'));
 await page.getByRole('button',{name:/开始分析/}).click();
 await page.getByRole('button',{name:/导出完整报告/}).waitFor({timeout:60000});
 await shot('18_用户端对照阅片.png');
 const text=await page.locator('body').innerText();
 for(const term of ['训练完成','最佳验证准确率','实验档案','数据与误判审查','检查点']){
  if(text.includes(term))throw Error('Research content exposed: '+term);
 }
 const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:/导出完整报告/}).click()]);
 const report=path.join(root,'outputs','consumer_report.html');await download.saveAs(report);
 if(fs.readFileSync(report,'utf8').includes('检查点与验证记录'))throw Error('Research metrics remain in consumer report');
 await page.getByText('原始影像',{exact:true}).click();
 await settled();
 await page.waitForFunction(()=>document.querySelectorAll('.image-viewport').length===1);
 await page.getByText('双图对照',{exact:true}).click();
 await page.waitForFunction(()=>document.querySelectorAll('.image-viewport').length===2);
 await page.setViewportSize({width:1440,height:1100});await shot('19_用户端1440宽屏.png');
 await page.getByRole('link',{name:/案例库/}).click();
 await page.getByRole('button',{name:/查看案例详情/}).first().click();
 await page.getByRole('button',{name:/导出完整报告/}).waitFor();await shot('20_用户端检查记录.png');
 const rp=await browser.newPage({viewport:{width:1440,height:1200}});
 await rp.goto(require('url').pathToFileURL(report).href);await rp.evaluate(()=>document.fonts.ready);
 await rp.screenshot({path:path.join(root,'screenshots','21_用户端筛查报告.png'),fullPage:true});
 fs.writeFileSync(path.join(root,'outputs','consumer_browser_qa.json'),JSON.stringify({page_errors:errors,checked_at:new Date().toISOString(),checks:['actual PNG upload and inference','two-pane and single-pane viewer','HTML export','records','no research navigation','1920 and 1440 desktop screenshots']},null,2));
 if(errors.length)throw Error(errors.join('\n'));
 console.log('Consumer UI QA passed. Screenshots 17–21.');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
