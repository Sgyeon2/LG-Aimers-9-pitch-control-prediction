"""같은 점수 이동 delta 를 만들 때 확률 질량 경로별 Brier 비용 비교.
점수 v=(100,30,0). 제약 sum(dp)=0, v.dp=delta 에서 ||dp||^2 최소화 → dp ∝ (v - mean(v)).
이론 비용: S<->H  delta^2/2450,  순수 S<->F  delta^2/5000,  최소노름  delta^2/5267."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features import CLASSES, evaluate, load_train
CACHE = ROOT / 'experiments' / 'train_merged.pkl'
d = pd.read_pickle(CACHE) if CACHE.exists() else load_train(ROOT / 'data')

V = np.array([100., 30., 0.]); DIR = (V - V.mean()) / ((V - V.mean()) @ V)   # v.DIR = 1
print('최소노름 방향 dp/delta =', DIR.round(6), ' 검산 v.dp =', V @ DIR)

def move(probs, delta, vec):
    """dp = delta*vec, 확률이 [1e-9,1] 밖으로 나가면 그 행만 축소."""
    dp = np.outer(delta, vec)
    p = probs + dp
    bad = (p < 1e-9).any(axis=1)
    if bad.any():
        lim = np.ones(len(p))
        for k in range(3):
            neg = dp[:, k] < 0
            lim[neg] = np.minimum(lim[neg], (probs[neg, k] - 1e-9) / -dp[neg, k])
        p = probs + dp * np.clip(lim, 0, 1)[:, None]
    return p

PATHS = {
    'S<->H (현재)': np.array([1., -1., 0.]) / 70.,
    '순수 S<->F':   np.array([1., 0., -1.]) / 100.,
    '최소노름':     DIR,
}
for season in (2022, 2023, 2024):
    r = pd.read_pickle(ROOT / f'experiments/decomp/s22rows_d12l200_{season}.pkl')
    tr = d[d.season.eq(season) & d.control_grade.isin(CLASSES)].loc[r.index]
    assert (tr.pitcher_id.to_numpy() == r.pitcher_id.to_numpy()).all()
    probs = r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
    base = evaluate(probs, r.control_grade, r.pitcher_id)
    delta1 = (-1.0*(tr.balls_before.to_numpy(float)-0.9) + 0.5*(tr.strikes_before.to_numpy(float)-0.87)
              - 0.5*(tr.num_runners_on.to_numpy(float)-0.68))
    print(f'\n== {season}  base pitch {base["pitch"]:.4f} tau {base["tau"]:.4f}')
    for name, vec in PATHS.items():
        for x in (3, 4, 5):
            e = evaluate(move(probs, x*delta1, vec), r.control_grade, r.pitcher_id)
            print(f'   {name:13s} x{x}: dTau {e["tau"]-base["tau"]:+.4f}  dPitch {e["pitch"]-base["pitch"]:+.4f}  dTotal {e["total"]-base["total"]:+.4f}')
