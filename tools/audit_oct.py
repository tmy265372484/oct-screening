"""Read-only split audit and fixed-model DRUSEN error review. Never trains."""
from __future__ import annotations
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import html
from io import BytesIO
import itertools
import json
from pathlib import Path
import re
import sys
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from config import CLASS_NAMES


def inspect_image(path: Path, root: Path) -> dict:
    relative = path.relative_to(root)
    match = re.fullmatch(r'(CNV|DME|DRUSEN|NORMAL)-(\d+)-(\d+)', path.stem)
    row = {'path': relative.as_posix(), 'split': relative.parts[0], 'class': relative.parts[1],
           'candidate_id': match.group(2) if match else None}
    try:
        payload = path.read_bytes()
        row['file_sha256'] = hashlib.sha256(payload).hexdigest()
        with Image.open(BytesIO(payload)) as source:
            source.load()
            gray = source.convert('L')
            row['size'] = list(gray.size)
            # Match native grayscale content, with dimensions, without resizing.
            row['pixel_sha256'] = hashlib.sha256(
                f'{gray.width}x{gray.height}:L:'.encode() + gray.tobytes()).hexdigest()
    except Exception as error:
        row['error'] = type(error).__name__ + ': ' + str(error)
    return row


def summarize(rows: list[dict]) -> dict:
    splits = ['train', 'val', 'test']
    result = {'counts': {}, 'unparsed_filenames': {}, 'decode_errors': sum('error' in r for r in rows),
              'candidate_id_counts': {}, 'pairs': {}, 'duplicate_groups': {}}
    for split in splits:
        subset = [r for r in rows if r['split'] == split]
        result['counts'][split] = dict(Counter(r['class'] for r in subset))
        result['unparsed_filenames'][split] = sum(r['candidate_id'] is None for r in subset)
        result['candidate_id_counts'][split] = len({r['candidate_id'] for r in subset if r['candidate_id']})
    for key in ['file_sha256', 'pixel_sha256']:
        groups = defaultdict(list)
        for row in rows:
            if key in row:
                groups[row[key]].append(row)
        duplicates = [g for g in groups.values() if len(g) > 1]
        result['duplicate_groups'][key] = {
            'groups': len(duplicates), 'images': sum(map(len, duplicates)),
            'cross_split_groups': sum(len({r['split'] for r in g}) > 1 for g in duplicates),
            'conflicting_label_groups': sum(len({r['class'] for r in g}) > 1 for g in duplicates)}
    for a, b in itertools.combinations(splits, 2):
        ra, rb = ([r for r in rows if r['split'] == s] for s in (a, b))
        pair = {}
        for key in ['candidate_id', 'file_sha256', 'pixel_sha256']:
            shared = {r[key] for r in ra if r.get(key)} & {r[key] for r in rb if r.get(key)}
            pair[key] = {'shared_values': len(shared),
                         a + '_images': sum(r.get(key) in shared for r in ra),
                         b + '_images': sum(r.get(key) in shared for r in rb)}
        pair['same_class_candidate_ids'] = {}
        for category in CLASS_NAMES:
            ca = {r['candidate_id'] for r in ra if r['class'] == category and r['candidate_id']}
            cb = {r['candidate_id'] for r in rb if r['class'] == category and r['candidate_id']}
            pair['same_class_candidate_ids'][category] = len(ca & cb)
        result['pairs'][a + '_' + b] = pair
    result['limitations'] = [
        '文件名中间数字仅作为候选患者编号；本地无患者映射表，未核实其官方语义。',
        '字节SHA256和原尺寸解码灰度像素SHA256检查完全相同内容；未穷举近似图、相邻B-scan或不同压缩/裁剪后的近重复。',
        'examples来自训练目录的复制，不属于独立评估；本次审计范围为train/val/test。',
        '未移动、删除、重分数据，未训练；不能据此声称已修复泄漏或获得独立临床性能。']
    return result


def encoded_image(image):
    image = image.copy()
    image.thumbnail((700, 500))
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def duplicate_review(rows, root, output):
    groups = defaultdict(list)
    for row in rows:
        if row.get('file_sha256'):
            groups[row['file_sha256']].append(row)
    conflicts = [g for g in groups.values() if len({r['class'] for r in g}) > 1]
    cross_split = [g for g in groups.values() if len({r['split'] for r in g}) > 1]
    (output / 'duplicate_evidence.json').write_text(json.dumps(
        {'cross_split_groups': cross_split, 'conflicting_labels': conflicts}, ensure_ascii=False, indent=2), encoding='utf-8')
    cards = []
    # Deterministic examples, selected by hash, without implying clinical relabeling.
    for index, group in enumerate(sorted(conflicts, key=lambda g: g[0]['file_sha256'])[:4]):
        with Image.open(root / group[0]['path']) as image:
            image_uri = encoded_image(image)
        counts = Counter(r['split'] + '/' + r['class'] for r in group)
        cards.append(f"<section><h3>重复内容组 DUP-{index+1:02}</h3><div class='images'><figure><img src='{image_uri}'></figure><div><p>同一文件内容出现在：</p>"
                     + ''.join(f'<p>{html.escape(k)}：{v} 张</p>' for k, v in sorted(counts.items()))
                     + '<p>以上文件SHA-256完全相同；需要核验来源和标签，不自动认定哪一个类别正确。</p></div></div></section>')
    return ''.join(cards)


def enrich_report(root, output):
    """Render additional evidence from the completed audit snapshot; no rescan."""
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    duplicate_cards = duplicate_review(rows, root, output)
    predictions = json.loads((output / 'drusen_predictions.json').read_text(encoding='utf-8'))
    summary['drusen']['score_statistics'] = {}
    total = len(predictions)
    counts = Counter(r['prediction'] for r in predictions)
    wrong = total - counts['DRUSEN']
    major = counts['CNV'] + counts['NORMAL']
    high_wrong = sum(r['prediction'] != 'DRUSEN' and r['score'] >= .9 for r in predictions)
    table = '<table><tr><th>DRUSEN被预测为</th><th>张数</th><th>占本类比例</th><th>模型分数中位数</th><th>分数≥90%的张数</th></tr>'
    for category in CLASS_NAMES:
        group = [r for r in predictions if r['prediction'] == category]
        stats = {'count':len(group), 'median_model_score':statistics.median(r['score'] for r in group) if group else None,
                 'score_at_least_90_percent':sum(r['score'] >= .9 for r in group)}
        summary['drusen']['score_statistics'][category] = stats
        median = f"{stats['median_model_score']:.2%}" if group else '无样本'
        table += f"<tr><td>{category}</td><td>{len(group)}</td><td>{len(group)/len(predictions):.1%}</td><td>{median}</td><td>{stats['score_at_least_90_percent']}</td></tr>"
    table += f'</table><p>正确{counts["DRUSEN"]}张、误判{wrong}张。CNV与NORMAL合计占误判的{major}/{wrong}。{high_wrong}张误判的模型分数仍≥90%，仅增加分数阈值无法排除所有错误。</p>'
    table += '<p>'+html.escape(summary['drusen']['selection_rule'])+'热图独立归一化，不能跨目标直接比较颜色强度，也不能仅凭热图确定误判原因。</p>'
    report = (output / 'report.html').read_text(encoding='utf-8')
    report = re.sub(r'<h2>已确认的数据质量问题</h2>.*?(?=<h2>DRUSEN)', '', report, flags=re.S)
    report = re.sub(r'<h2>相同内容、不同类别：人工复核样例</h2>.*?(?=<h2>下一步实验顺序)', '', report, flags=re.S)
    heading = f'<h2>DRUSEN 全部{total}张的预测分布</h2>'
    report = re.sub(r'<h2>DRUSEN 全部\d+张的预测分布</h2>.*?<h2>可复核案例</h2>', lambda m:heading+table+'<h2>可复核案例</h2>', report, flags=re.S)
    val_count = sum(summary['counts']['val'].values())
    overlap = summary['pairs']['train_val']
    dup = overlap['file_sha256']['val_images']
    ids = overlap['candidate_id']['val_images']
    conflicting = summary['duplicate_groups']['file_sha256']['conflicting_label_groups']
    train_count = sum(summary['counts']['train'].values())
    drusen_count = summary['counts']['train']['DRUSEN']
    normal_count = summary['counts']['train']['NORMAL']
    text = f'<h2>已确认的数据质量问题</h2><p><strong>验证集中{dup}/{val_count}张（{dup/val_count:.2%}）与训练图像字节完全相同。</strong>验证集{ids}张的候选患者编号也在训练集中出现；候选编号并非经过患者映射表核验的身份信息。</p>'
    text += f'<p>全数据有{conflicting}组完全相同文件跨类别出现，需要复核标签来源；测试交叉数量见上表，近似重复尚未穷举。</p>'
    text += f'<p>训练集DRUSEN为{drusen_count}/{train_count}（{drusen_count/train_count:.2%}），NORMAL为{normal_count}张；不平衡是待对照验证的因素，不能直接认定为全部误判的原因。</p>'
    if dup or ids:
        text += '<p><strong>结论：现有验证划分存在交叉，不宜把当前验证准确率当作独立验证证据。先整理数据与标签，再开展训练对照。</strong></p>'
    report = report.replace(heading, text+heading)
    report = report.replace('<h2>下一步实验顺序</h2>', '<h2>相同内容、不同类别：人工复核样例</h2>'+duplicate_cards+'<h2>下一步实验顺序</h2>')
    (output / 'report.html').write_text(report, encoding='utf-8')
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    (output / 'summary.json').write_text(encoded, encoding='utf-8')
    (ROOT / 'assets/data_audit_summary.json').write_text(encoded, encoding='utf-8')


def drusen_review(root, output):
    import numpy as np
    from predict_torch import OCTPredictor
    from grad_cam_torch import generate_gradcam, GradCAMError
    predictor = OCTPredictor(device='cpu')
    paths = sorted((root / 'test' / 'DRUSEN').glob('*.jpeg'))
    if not paths:
        raise ValueError('DRUSEN test directory is empty')
    results = []
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            prediction = predictor.predict(source)
        results.append({'case_id': f'DR-{index+1:03}', 'path': str(path.relative_to(root)),
                        'prediction': prediction['predicted_class'], 'score': prediction['model_score'],
                        'probabilities': prediction['probabilities']})
    selected = []
    selection_counts = {}
    for category in CLASS_NAMES:
        group = sorted([r for r in results if r['prediction'] == category], key=lambda r: (-r['score'], r['case_id']))
        selection_counts[category] = len(group)
        if group:
            # Predefined, reproducible: highest score + middle ranked example per group.
            selected.extend([group[0]] + ([group[len(group)//2]] if len(group) > 1 else []))
    cards = []
    for row in selected:
        with Image.open(root / row['path']) as source:
            original = source.convert('RGB')
        panels = [('原图', original)]
        notes = []
        targets = [row['prediction']] + (['DRUSEN'] if row['prediction'] != 'DRUSEN' else [])
        for target in targets:
            try:
                cam = generate_gradcam(predictor, original, target_class=target)
                panels.append((f'目标 {target} 的关注区域', cam['overlay']))
            except GradCAMError as error:
                notes.append(f'{target}: {error}')
        cards.append('<section><h3>' + row['case_id'] + ' · 数据标签 DRUSEN → 预测 ' + row['prediction']
                     + f" · 模型分数 {row['score']:.2%}</h3><div class='images'>"
                     + ''.join(f"<figure><img src='{encoded_image(im)}'><figcaption>{label}</figcaption></figure>" for label, im in panels)
                     + '</div><p>' + html.escape('；'.join(notes) or '每幅热图独立归一化；颜色强弱不能跨类别直接比较。') + '</p></section>')
    aggregate = {'total': len(results), 'predicted_counts': selection_counts,
                 'recall': selection_counts.get('DRUSEN', 0)/len(results),
                 'selected_case_ids': [r['case_id'] for r in selected],
                 'selection_rule': '每个预测类别按模型分数降序，取最高分及中间名次各1例；包含正确对照，不代表错误类型的随机抽样。',
                 'checkpoint_epoch': predictor.checkpoint_info['epoch'],
                 'limitations': '测试标签作为参考，未由眼科专家重新标注。热图无法证明病灶、标签错误或误判原因；这些案例仅供错误审查，不用于反复调测试集。'}
    (output / 'drusen_predictions.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    return aggregate, ''.join(cards)


def main():
    root = ROOT / 'data' / 'OCT'
    output = ROOT / 'outputs' / 'data_audit'
    output.mkdir(parents=True, exist_ok=True)
    models = [ROOT / p for p in ['models/torch_resume/best.pt', 'models/torch_resume/latest.pt', 'models/oct_classifier.keras']]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
    paths = sorted(p for split in ['train', 'val', 'test'] for p in (root / split).rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    rows = []
    print(f'Auditing {len(paths)} images', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool, (output / 'manifest.jsonl').open('w', encoding='utf-8') as file:
        for row in pool.map(lambda p: inspect_image(p, root), paths):
            rows.append(row)
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
            if len(rows) % 5000 == 0:
                print(f'Audited {len(rows)}/{len(paths)}', flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    summary['drusen'], cards = drusen_review(root, output)
    summary['generated_at'] = datetime.now().astimezone().isoformat(timespec='seconds')
    summary['model_hashes'] = hashes
    summary['model_files_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in models)
    if not summary['model_files_unchanged']:
        raise RuntimeError('Model changed during audit')
    encoded = json.dumps(summary, ensure_ascii=False, indent=2)
    (output / 'summary.json').write_text(encoded, encoding='utf-8')
    (ROOT / 'assets' / 'data_audit_summary.json').write_text(encoded, encoding='utf-8')
    table = '<table><tr><th>数据对</th><th>候选编号交集</th><th>字节相同组</th><th>灰度像素相同组</th></tr>'
    for name, pair in summary['pairs'].items():
        table += f"<tr><td>{name}</td><td>{pair['candidate_id']['shared_values']}</td><td>{pair['file_sha256']['shared_values']}</td><td>{pair['pixel_sha256']['shared_values']}</td></tr>"
    table += '</table>'
    body = f"<h1>OCT 数据审计与 DRUSEN 错误分析</h1><p>{summary['generated_at']} · 固定第{summary['drusen']['checkpoint_epoch']}轮检查点 · 只读审查</p>"
    body += '<h2>数据划分与重复内容</h2>' + table
    body += '<p>编号交集来自文件名中间数字，仅为候选患者编号。完全重复的像素内容是独立于编号的证据。</p>'
    body += '<p>现有 prepare_data.py 按图片随机抽取验证集，没有按患者编号分组。应先核验标识并建立按患者及重复内容分组的固定划分，再进行模型对照。</p>'
    body += f'<h2>DRUSEN 全部{summary["drusen"]["total"]}张的预测分布</h2><pre>' + html.escape(json.dumps(summary['drusen'], ensure_ascii=False, indent=2)) + '</pre>'
    body += '<h2>可复核案例</h2><p>每组取最高分及中位次序样本；高分误判与正确对照同时展示。以下不对形态或病因做自动临床解释。</p>' + cards
    body += '<h2>下一步实验顺序</h2><ol><li>确认患者标识；以患者/重复图像连接组隔离train、val、test，冻结划分清单。</li><li>在新划分上重新建立基线；旧模型已接触旧训练图像，不能用重分后测试图给旧模型声称独立成绩。</li><li>固定随机种子和训练预算，分别比较类别加权、最后一组网络层微调和增强强度；只用验证集选择。</li><li>最终方案锁定后再使用未参与错误分析的留出数据进行一次评估。</li></ol>'
    body += '<ul>' + ''.join('<li>'+html.escape(s)+'</li>' for s in summary['limitations']) + '</ul>'
    page = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; img-src data:; style-src 'unsafe-inline'\"><title>OCT数据与错误分析</title><style>body{font:16px/1.7 'Microsoft YaHei',sans-serif;background:#edf3f8;color:#17334a;margin:0;padding:36px}main{max-width:1500px;margin:auto}section{background:white;padding:20px;margin:20px 0;border:1px solid #ccdce7;border-radius:8px}.images{display:flex;gap:20px}figure{flex:1;margin:0;min-width:0}img{width:100%;max-height:420px;object-fit:contain;background:#111}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border:1px solid #ccdce7;text-align:left}pre{white-space:pre-wrap;background:white;padding:20px}h1,h2{color:#116c8e}</style><main>" + body + '</main></html>'
    (output / 'report.html').write_text(page, encoding='utf-8')
    enrich_report(root, output)
    print('Completed: outputs/data_audit/report.html', flush=True)


if __name__ == '__main__':
    main()
