"""신인(hist_n=0) field-shift 배율이 학습 시즌 홀드아웃에서도 지지되는가 (LB 없이 검증)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.features import CLASSES, evaluate, load_train
CACHE=ROOT / 'experiments' / 'train_merged.pkl'
d=pd.read_pickle(CACHE) if CACHE.exists() else load_train(ROOT/'data')
V=np.array([100.,30.,0.]); DIR=(V-V.mean())/((V-V.mean())@V)

def move(p,delta):
    dp=np.outer(delta,DIR); lim=np.ones(len(p))
    for k in range(3):
        neg=dp[:,k]<0
        if neg.any(): lim[neg]=np.minimum(lim[neg],(p[neg,k]-1e-9)/-dp[neg,k])
    return p+dp*np.clip(lim,0,1)[:,None]

print(f"{'시즌':>6} {'신인수':>6} " + ' '.join(f'{m:>14}' for m in ('배율0.5','배율1.0','배율1.5','배율2.0','배율2.5','배율3.0')))
tot={m:[] for m in (0.5,1.0,1.5,2.0,2.5,3.0)}
for season in (2022,2023,2024):
    for tag in ('d12l200','d14l100'):
        r=pd.read_pickle(ROOT/f'experiments/decomp/s22rows_{tag}_{season}.pkl')
        tr=d[d.season.eq(season)&d.control_grade.isin(CLASSES)].loc[r.index]
        probs=r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
        base=(-3.0*(tr.balls_before.to_numpy(float)-0.9)+1.5*(tr.strikes_before.to_numpy(float)-0.87)
              -1.5*(tr.num_runners_on.to_numpy(float)-0.68)+0.919*(tr.outs_before.to_numpy(float)-0.982))
        hn=r.hist_n.to_numpy(float); is_rk=hn==0
        nrk=r.loc[is_rk].groupby('pitcher_id').size(); nrk=int((nrk>=50).sum())
        ref=evaluate(move(probs,base),r.control_grade,r.pitcher_id)
        line=[]
        for m in (0.5,1.0,1.5,2.0,2.5,3.0):
            e=evaluate(move(probs,base*np.where(is_rk,m,1.0)),r.control_grade,r.pitcher_id)
            line.append(f'{e["tau"]-ref["tau"]:+.4f}/{e["total"]-ref["total"]:+.3f}')
            tot[m].append(e['total']-ref['total'])
        print(f'{season} {tag[:7]:>8} {nrk:>4} ' + ' '.join(f'{x:>14}' for x in line))
print('\n배율별 6개 비교 평균 dTotal (배율1.0 = 기준):')
for m,v in tot.items(): print(f'  {m}: {np.mean(v):+.3f}  (양수 {sum(1 for x in v if x>0)}/6)')
