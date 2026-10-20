"""Create candidate-patient/content grouped manifests; never move data or train.

Input is the local read-only audit manifest. IDs are filename-derived candidates,
not verified patient identities. The existing test split stays locked.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def prepare(rows, validation_fraction=.2):
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    if any(r.get('split') not in ('train','val','test') for r in rows):
        raise ValueError('Unknown split')
    parent=list(range(len(rows)))
    def find(i):
        while i!=parent[i]:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        for key in ('candidate_id','file_sha256','pixel_sha256'):
            if r.get(key):
                token=(key,r[key])
                if token in seen: parent[find(i)]=find(seen[token])
                else: seen[token]=i
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[find(i)].append(r)
    output={'train':[], 'val':[], 'test':[], 'quarantine':[]}
    for group in groups.values():
        if any(r['split']=='test' for r in group):
            output['test'] += [dict(r,proposed_split='test') for r in group if r['split']=='test']
            output['quarantine'] += [dict(r,reason='与锁定测试集共享候选编号或相同内容') for r in group if r['split']!='test']
            continue
        by_hash=defaultdict(set)
        for r in group:
            for k in ('file_sha256','pixel_sha256'):
                if r.get(k): by_hash[(k,r[k])].add(r['class'])
        if any(r.get('error') or not r.get('candidate_id') or not r.get('pixel_sha256') or not r.get('file_sha256') for r in group) or any(len(v)>1 for v in by_hash.values()):
            output['quarantine'] += [dict(r,reason='缺少身份/内容指纹、损坏或相同内容跨标签；需人工核查') for r in group]
            continue
        token='|'.join(sorted({str(r['candidate_id']) for r in group}))
        value=int(hashlib.sha256(('oct-group-v1:'+token).encode()).hexdigest()[:16],16)/16**16
        split='val' if value < validation_fraction else 'train'
        # Exact same content within a component contributes only once.
        kept=set()
        for r in sorted(group,key=lambda r:r['path']):
            if r['pixel_sha256'] in kept:
                output['quarantine'].append(dict(r,reason='同组完全相同内容的冗余副本'))
            else:
                kept.add(r['pixel_sha256']); output[split].append(dict(r,proposed_split=split))
    for field in ('candidate_id','file_sha256','pixel_sha256'):
        sets=[{r[field] for r in output[s] if r.get(field)} for s in ('train','val','test')]
        if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
            raise ValueError('Cross-split overlap remains: '+field)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('outputs/data_audit/manifest.jsonl'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/grouped_split'))
    args=p.parse_args()
    raw=args.manifest.read_bytes()
    rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    result=prepare(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for split,items in result.items():
        (args.output_dir/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    summary={'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),
             'counts':{s:dict(Counter(r['class'] for r in items)) for s,items in result.items()},
             'patient_identity_verified':False,'model_retrained':False,
             'note':'仅生成分组候选清单；未移动数据。须核实患者身份与隔离样本，之后重新训练、固定模型再评估。现有权重不能用于证明重划分后的独立性能。'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
