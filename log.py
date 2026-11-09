import json,sys,hashlib,re
n,fam,hyp,syms,note=sys.argv[1:6]
out=open(f'{n}.out').read()
ds=float(re.search(r'dev_score: ([\d.]+)',out).group(1))
folds=json.loads(re.search(r'folds: (\{.*?\})',out).group(1))
era=json.loads(re.search(r'\): (\{"era0.*?\})',out).group(1))
pyf=float(re.search(r'years ([\d.]+)',out).group(1)); kn=int(re.search(r'for (\d+) knobs',out).group(1))
fs=sorted(folds.values())
rec=dict(agent='v16n-5',experiment=n,hypothesis=hyp,family=fam,symbols=syms.split(','),source_sha256=hashlib.sha256(open('strategy.py','rb').read()).hexdigest(),dev_score=ds,fold_score=sum(fs[:2])/2,positive_year_fraction=pyf,knobs=kn,folds=folds,era_blocks=era,null_p=None,provisional=True,eligible=False,note=note)
open('experiments.jsonl','a').write(json.dumps(rec)+'\n'); print(ds,era,pyf)
