/* Optional visual QA: npm install --no-save playwright, with Edge installed.
 * Start Streamlit separately. Capture actual rendered UI, never synthetic states.
 */
const {chromium} = require('playwright');
const path = require('path');
const fs = require('fs');
const root = path.resolve(__dirname, '..');
const base = process.argv[2] || 'http://127.0.0.1:8501';

(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true,args:['--disable-gpu']});
  try {
    const page = await browser.newPage({viewport:{width:1920,height:1180},deviceScaleFactor:1});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    async function shot(filename) {
      await page.waitForLoadState('networkidle');
      await page.getByRole('button',{name:'Stop',exact:true}).waitFor({state:'hidden',timeout:30000});
      await page.evaluate(() => document.fonts.ready);
      await page.locator('img').evaluateAll(images => Promise.all(images.map(i => i.complete ? null : new Promise(r => {i.onload=r;i.onerror=r;}))));
      await page.screenshot({path:path.join(root,'screenshots',filename),fullPage:true});
    }
    await page.goto(base,{waitUntil:'networkidle'});
    await page.getByRole('button',{name:/开始分析/}).click();
    await page.getByRole('button',{name:/下载 HTML 报告/}).waitFor({timeout:60000});
    await page.getByText('1 例',{exact:true}).waitFor();
    await shot('10_电脑端筛查工作台.png');
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('button',{name:/下载 HTML 报告/}).click(),
    ]);
    const report = path.join(root,'outputs','demo_report.html');
    await download.saveAs(report);
    await page.getByRole('link',{name:/会话报告/}).click();
    await page.getByRole('button',{name:/下载 HTML 报告/}).waitFor();
    await page.setViewportSize({width:1920,height:2200});
    await shot('11_会话与报告工作区.png');
    await page.getByRole('link',{name:/实验档案/}).click();
    await page.getByText('80.70%',{exact:true}).waitFor();
    await page.getByRole('button',{name:/下载实验摘要/}).waitFor();
    await page.locator('[data-testid="stVegaLiteChart"] svg').first().waitFor({timeout:30000});
    await page.waitForFunction(() => [...document.querySelectorAll('[data-testid="stVegaLiteChart"] .mark-line path')]
      .filter(p => (p.getAttribute('d') || '').length > 100).length >= 2);
    await shot('12_实验档案与测试评估.png');
    const reportPage = await browser.newPage({viewport:{width:1440,height:1200}});
    await reportPage.goto(require('url').pathToFileURL(report).href,{waitUntil:'networkidle'});
    await reportPage.evaluate(() => document.fonts.ready);
    await reportPage.screenshot({path:path.join(root,'screenshots','13_HTML报告实际预览.png'),fullPage:true});
    fs.writeFileSync(path.join(root,'outputs','browser_qa.json'), JSON.stringify({
      checked_at:new Date().toISOString(),page_errors:errors,
      tested:['real inference and CAM','HTML report download','session records navigation','training and test evidence navigation','offline HTML rendering'],
    },null,2));
    if(errors.length) throw new Error(errors.join('\n'));
    console.log('Browser visual QA passed; screenshots 10–13 saved.');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1});
