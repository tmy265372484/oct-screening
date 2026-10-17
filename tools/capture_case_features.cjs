const {chromium}=require('playwright');
const path=require('path'),fs=require('fs');
const root=path.resolve(__dirname,'..');
(async()=>{
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--disable-gpu']});
try{
 const p=await browser.newPage({viewport:{width:1920,height:1200}});
 async function shot(name){
  await p.getByRole('button',{name:'Stop',exact:true}).waitFor({state:'hidden'});
  await p.evaluate(()=>document.fonts.ready);
  await p.screenshot({path:path.join(root,'screenshots',name),fullPage:true});
 }
 await p.goto('http://127.0.0.1:8501/',{waitUntil:'networkidle'});
 await p.getByText('example 图库',{exact:true}).click();
 await p.getByText('选择本地图片',{exact:true}).waitFor();
 await p.getByRole('button',{name:/开始分析/}).click();
 await p.getByRole('button',{name:/导出完整报告/}).waitFor({timeout:60000});
 await shot('22_example图库分析.png');
 await p.getByRole('link',{name:/案例库/}).click();
 await p.getByRole('button',{name:/查看案例详情/}).first().waitFor();
 await shot('23_持久化案例库.png');
 await p.getByRole('button',{name:/查看案例详情/}).first().click();
 await p.getByRole('button',{name:/保存备注/}).waitFor();
 await shot('24_案例详细展示.png');
 // New session must see the archived image and report, not session memory.
 const fresh=await browser.newPage();
 await fresh.goto('http://127.0.0.1:8501/cases',{waitUntil:'networkidle'});
 await fresh.getByRole('button',{name:/查看案例详情/}).first().waitFor();
 await fresh.close();
 await p.getByRole('link',{name:/健康咨询/}).click();
 await p.getByRole('button',{name:'就诊需要准备什么',exact:true}).click();
 await p.getByText(/建议准备：症状开始时间/).waitFor();
 const input=p.getByPlaceholder('描述症状或询问报告含义…');
 await input.fill('突然视力下降怎么办');
 await input.press('Enter');
 await p.getByText(/你提到了可能需要及时处理的症状/).waitFor();
 await shot('25_健康咨询本地引导.png');
 console.log('Example analysis, persistent case detail, new-session archive and health chat passed.');
}finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
