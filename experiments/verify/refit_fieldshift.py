"""투수 수준 잔차를 행 필드의 투수 평균으로 회귀해 field-shift 계수를 다시 적합한다.
행 단위 분산(= pitch 비용)으로 벌점을 준 일반화 능형: (M'M + lam*W) c = M'r
  M: 투수별 (필드 평균 - center),  r: 투수별 (실제 평균점수 - 예측 평균점수), 시즌별 중심화
  W: 행 단위 공분산 E[(f-c)(f'-c')]  → c'Wc = E[delta^2] ∝ pitch 손실
계수는 학습 시즌(2022~2024 홀드아웃) 에서만 추정한다. test 미사용.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features import CLASSES, evaluate, load_train

CACHE = ROOT / 'experiments' / 'train_merged.pkl'
d = pd.read_pickle(CACHE) if CACHE.exists() else load_train(ROOT / 'data')
SEASONS = (2022, 2023, 2024)
TAG = sys.argv[1] if len(sys.argv) > 1 else 'd12l200'

def feats(tr):
    """행별 파생: test.csv 29열에서만 만든다."""
    b = tr.balls_before.to_numpy(float); s = tr.strikes_before.to_numpy(float)
    return pd.DataFrame({
        'balls_before': b,
        'strikes_before': s,
        'three_ball': (b == 3).astype(float),
        'two_strikes': (s == 2).astype(float),
        'first_pitch': ((b == 0) & (s == 0)).astype(float),
        'num_runners_on': tr.num_runners_on.to_numpy(float),
        'runner_on_3b': tr.runner_on_3b.to_numpy(float),
        'outs_before': tr.outs_before.to_numpy(float),
        'inning': tr.inning.to_numpy(float),
        'li': tr.li.to_numpy(float),
        'game_month': tr.game_month.to_numpy(float),
        'run_total_before': tr.run_total_before.to_numpy(float),
        'score_diff_pitcher_team': tr.score_diff_pitcher_team.to_numpy(float),
        'home_win_expectancy': tr.home_win_expectancy.to_numpy(float),
    })

store = {}
for season in SEASONS:
    r = pd.read_pickle(ROOT / f'experiments/decomp/s22rows_{TAG}_{season}.pkl')
    tr = d[d.season.eq(season) & d.control_grade.isin(CLASSES)].loc[r.index]
    assert (tr.pitcher_id.to_numpy() == r.pitcher_id.to_numpy()).all()
    F = feats(tr)
    probs = r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
    act = r.control_grade.map(dict(zip(CLASSES, [100., 30., 0.]))).to_numpy(float)
    pred = probs @ np.array([100., 30., 0.])
    store[season] = dict(F=F, probs=probs, act=act, pred=pred, pid=r.pitcher_id.to_numpy(),
                         grade=r.control_grade)

COLS = list(store[SEASONS[0]]['F'].columns)
centers = {c: float(np.mean(np.concatenate([store[s]['F'][c].to_numpy() for s in SEASONS]))) for c in COLS}
print('centers:', {k: round(v, 3) for k, v in centers.items()})

# 행 단위 공분산 W (중심 기준)
allX = np.vstack([ (store[s]['F'][COLS].to_numpy() - np.array([centers[c] for c in COLS])) for s in SEASONS ])
W = allX.T @ allX / len(allX)

# 투수 수준 설계행렬
Ms, rs = [], []
for s in SEASONS:
    st = store[s]
    g = pd.DataFrame(st['F'][COLS].to_numpy() - np.array([centers[c] for c in COLS]), columns=COLS)
    g['pid'] = st['pid']; g['act'] = st['act']; g['pred'] = st['pred']
    agg = g.groupby('pid').agg(['mean'])
    n = g.groupby('pid').size()
    agg.columns = [a for a, _ in agg.columns]
    agg = agg[n >= 50]
    res = (agg['act'] - agg['pred']).to_numpy()
    res = res - res.mean()          # 시즌 수준(상수) 제거 — 순위에 무관
    Ms.append(agg[COLS].to_numpy()); rs.append(res)
M = np.vstack(Ms); rvec = np.concatenate(rs)
print('pitchers pooled:', M.shape)

def solve(lam):
    A = M.T @ M + lam * len(M) * W
    return np.linalg.solve(A, M.T @ rvec)

def sh(probs, delta):
    p = probs.copy()
    mv = np.clip(delta / 70.0, -p[:, 0] + 1e-9, p[:, 1] - 1e-9)
    p[:, 0] += mv; p[:, 1] -= mv
    return p

def score(coef, scale):
    out = []
    for s in SEASONS:
        st = store[s]
        X = st['F'][COLS].to_numpy() - np.array([centers[c] for c in COLS])
        delta = scale * (X @ coef)
        base = evaluate(st['probs'], st['grade'], st['pid'])
        e = evaluate(sh(st['probs'], delta), st['grade'], st['pid'])
        out.append((e['tau'] - base['tau'], e['pitch'] - base['pitch'], e['total'] - base['total']))
    return np.array(out)

# 현재 채택 계수(x1) 기준선
cur = np.zeros(len(COLS))
cur[COLS.index('balls_before')] = -1.0
cur[COLS.index('strikes_before')] = 0.5
cur[COLS.index('num_runners_on')] = -0.5
print('\n=== 현재 계수(S79 x1 방향) 스케일 스윕 ===')
for x in (1, 2, 3, 4):
    a = score(cur, x); print(f'  x{x}: dTau {a[:,0].round(4)}  dPitch {a[:,1].round(4)}  dTotal mean {a[:,2].mean():+.3f}')

print('\n=== 재적합 계수 ===')
best = None
for lam in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
    c = solve(lam)
    # 단위 스케일 정규화: E[delta^2] 를 현재 x1 과 같게
    v_cur = cur @ W @ cur; v_new = c @ W @ c
    c_n = c * np.sqrt(v_cur / v_new)
    print(f'\n lam={lam}  coef(E[d^2] 정규화):', {COLS[i]: round(c_n[i], 3) for i in range(len(COLS)) if abs(c_n[i]) > 0.02})
    for x in (1, 2, 3, 4):
        a = score(c_n, x)
        m = a[:, 2].mean()
        print(f'   x{x}: dTau {a[:,0].round(4)}  dPitch {a[:,1].round(4)}  dTotal mean {m:+.3f}')
        if best is None or m > best[0]: best = (m, lam, x, c_n.copy())
print('\nBEST local:', 'lam', best[1], 'scale', best[2], 'dTotal', round(best[0], 3))
print('coef:', {COLS[i]: round(best[3][i] * best[2], 4) for i in range(len(COLS)) if abs(best[3][i] * best[2]) > 0.01})
print('centers:', {COLS[i]: round(centers[COLS[i]], 4) for i in range(len(COLS)) if abs(best[3][i] * best[2]) > 0.01})
