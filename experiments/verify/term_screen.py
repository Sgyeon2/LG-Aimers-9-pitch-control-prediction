"""현재 3항(x3, opt 경로) 위에 후보 항을 하나씩 더해 3시즌 부호 일관성을 본다. 적합 없음 = 선택 편향만."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features import CLASSES, evaluate, load_train
CACHE = ROOT / 'experiments' / 'train_merged.pkl'
d = pd.read_pickle(CACHE) if CACHE.exists() else load_train(ROOT / 'data')
V = np.array([100., 30., 0.]); DIR = (V - V.mean()) / ((V - V.mean()) @ V)

def move(probs, delta):
    dp = np.outer(delta, DIR); lim = np.ones(len(probs))
    for k in range(3):
        neg = dp[:, k] < 0
        if neg.any(): lim[neg] = np.minimum(lim[neg], (probs[neg, k] - 1e-9) / -dp[neg, k])
    return probs + dp * np.clip(lim, 0, 1)[:, None]

def cand(tr):
    b = tr.balls_before.to_numpy(float); s = tr.strikes_before.to_numpy(float)
    return {
        'three_ball':   ((b == 3).astype(float), 0.083),
        'two_strikes':  ((s == 2).astype(float), 0.288),
        'first_pitch':  (((b == 0) & (s == 0)).astype(float), 0.258),
        'outs_before':  (tr.outs_before.to_numpy(float), 0.982),
        'inning':       (tr.inning.to_numpy(float), 4.986),
        'li':           (tr.li.to_numpy(float), 0.989),
        'game_month':   (tr.game_month.to_numpy(float), 6.46),
        'runner_on_3b': (tr.runner_on_3b.to_numpy(float), 0.111),
        'run_total_before': (tr.run_total_before.to_numpy(float), 4.875),
        'score_diff_pitcher_team': (tr.score_diff_pitcher_team.to_numpy(float), 0.068),
        'home_win_expectancy': (tr.home_win_expectancy.to_numpy(float), 50.647),
        'top_bottom':   ((tr.top_bottom.astype(str) == 'T').astype(float), 0.5),
    }

S = {}
for season in (2022, 2023, 2024):
    r = pd.read_pickle(ROOT / f'experiments/decomp/s22rows_d12l200_{season}.pkl')
    tr = d[d.season.eq(season) & d.control_grade.isin(CLASSES)].loc[r.index]
    assert (tr.pitcher_id.to_numpy() == r.pitcher_id.to_numpy()).all()
    probs = r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
    base3 = (-3.0*(tr.balls_before.to_numpy(float)-0.9) + 1.5*(tr.strikes_before.to_numpy(float)-0.87)
             - 1.5*(tr.num_runners_on.to_numpy(float)-0.68))
    S[season] = dict(probs=probs, base3=base3, grade=r.control_grade, pid=r.pitcher_id.to_numpy(), C=cand(tr))

ref = {s: evaluate(move(S[s]['probs'], S[s]['base3']), S[s]['grade'], S[s]['pid']) for s in S}
print('x3 opt 기준선:', {s: f"tau {ref[s]['tau']:.4f} total {ref[s]['total']:.3f}" for s in ref})
rows = []
for name in S[2022]['C']:
    sd = np.std(np.concatenate([S[s]['C'][name][0] for s in S]))
    for mult in (-1.5, -0.75, -0.3, 0.3, 0.75, 1.5):
        coef = mult / max(sd, 1e-9)
        dt, dp_, dtot = [], [], []
        for s in S:
            v, c0 = S[s]['C'][name]
            e = evaluate(move(S[s]['probs'], S[s]['base3'] + coef * (v - c0)), S[s]['grade'], S[s]['pid'])
            dt.append(e['tau'] - ref[s]['tau']); dp_.append(e['pitch'] - ref[s]['pitch']); dtot.append(e['total'] - ref[s]['total'])
        if all(t > 0 for t in dtot) or all(t < 0 for t in dtot):
            rows.append((np.mean(dtot), name, round(coef, 3), mult, np.round(dt, 4), np.round(dtot, 3), np.mean(dp_)))
rows.sort(reverse=True)
print('\n3시즌 부호 일관 항 (dTotal 평균 내림차순):')
for m, name, coef, mult, dt, dtot, dpm in rows[:14]:
    print(f'  {name:24s} coef {coef:+8.3f} (mult{mult:+.2f})  dTau {dt}  dTotal {dtot}  평균 {m:+.3f}  dPitch평균 {dpm:+.4f}')
