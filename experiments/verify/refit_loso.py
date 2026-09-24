"""LOSO: 한 시즌을 빼고 계수를 적합해 그 시즌에서 평가(정직한 전이 추정)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
exec(open(ROOT/'experiments/verify/refit_fieldshift.py').read().split('# 현재 채택 계수')[0].replace(
    "TAG = sys.argv[1] if len(sys.argv) > 1 else 'd12l200'", "TAG = 'd12l200'"))

def solve_sub(seasons, lam):
    Ms_, rs_ = [], []
    for s in seasons:
        st = store[s]
        g = pd.DataFrame(st['F'][COLS].to_numpy() - np.array([centers[c] for c in COLS]), columns=COLS)
        g['pid'] = st['pid']; g['act'] = st['act']; g['pred'] = st['pred']
        agg = g.groupby('pid').mean(); n = g.groupby('pid').size(); agg = agg[n >= 50]
        res = (agg['act'] - agg['pred']).to_numpy(); res = res - res.mean()
        Ms_.append(agg[COLS].to_numpy()); rs_.append(res)
    M_ = np.vstack(Ms_); r_ = np.concatenate(rs_)
    return np.linalg.solve(M_.T @ M_ + lam * len(M_) * W, M_.T @ r_)

def sh(p, delta):
    p = p.copy(); mv = np.clip(delta / 70.0, -p[:, 0] + 1e-9, p[:, 1] - 1e-9)
    p[:, 0] += mv; p[:, 1] -= mv; return p

cur = np.zeros(len(COLS))
cur[COLS.index('balls_before')] = -1.0; cur[COLS.index('strikes_before')] = 0.5; cur[COLS.index('num_runners_on')] = -0.5
v_cur = cur @ W @ cur

print(f"{'lam':>7} {'scale':>5} | " + ' '.join(f'{s}: dTau  dTotal' for s in SEASONS) + '   mean dTotal')
for lam in (0.001, 0.003, 0.01, 0.03):
    for x in (2, 3, 4):
        line, tots = [], []
        for s in SEASONS:
            c = solve_sub([q for q in SEASONS if q != s], lam)
            c = c * np.sqrt(v_cur / (c @ W @ c)) * x
            st = store[s]
            X = st['F'][COLS].to_numpy() - np.array([centers[q] for q in COLS])
            base = evaluate(st['probs'], st['grade'], st['pid'])
            e = evaluate(sh(st['probs'], X @ c), st['grade'], st['pid'])
            line.append(f"{e['tau']-base['tau']:+.4f} {e['total']-base['total']:+.3f}"); tots.append(e['total'] - base['total'])
        print(f'{lam:>7} {x:>5} | ' + '  '.join(line) + f'   {np.mean(tots):+.3f}')
# 현재 계수 기준
for x in (2, 3, 4):
    tots = []
    for s in SEASONS:
        st = store[s]; X = st['F'][COLS].to_numpy() - np.array([centers[q] for q in COLS])
        base = evaluate(st['probs'], st['grade'], st['pid'])
        e = evaluate(sh(st['probs'], X @ (cur * x)), st['grade'], st['pid']); tots.append(e['total'] - base['total'])
    print(f'{"현재":>7} {x:>5} |' + ' ' * 46 + f'   {np.mean(tots):+.3f}')
