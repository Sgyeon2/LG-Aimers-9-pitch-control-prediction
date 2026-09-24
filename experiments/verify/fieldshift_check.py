"""독립 검증: 행 필드 기반 상수 이동(field-shift)이 투수 순위 tau 를 올리는가.
덤프(행별 예측) + train.csv 상황 필드를 정렬해 확인. 재학습 없음."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features import CLASSES, evaluate, load_train

CACHE = ROOT / 'experiments' / 'train_merged.pkl'
d = pd.read_pickle(CACHE) if CACHE.exists() else load_train(ROOT / 'data')
print('train rows', len(d), 'cols', len(d.columns))

FIELDS = ['balls_before', 'strikes_before', 'num_runners_on', 'inning', 'li']
tag = sys.argv[1] if len(sys.argv) > 1 else 'd12l200'

def get(season):
    r = pd.read_pickle(ROOT / f'experiments/decomp/s22rows_{tag}_{season}.pkl')
    tr = d[d.season.eq(season) & d.control_grade.isin(CLASSES)]
    assert len(tr) == len(r), (len(tr), len(r))
    tr = tr.loc[r.index]  # 덤프 index = train.csv 행 index 라는 가정 검증
    assert (tr.pitcher_id.to_numpy() == r.pitcher_id.to_numpy()).all(), 'pitcher_id 불일치'
    assert (tr.game_type.to_numpy() == r.game_type.to_numpy()).all(), 'game_type 불일치'
    assert (tr.control_grade.to_numpy() == r.control_grade.to_numpy()).all(), 'label 불일치'
    return r, tr

def shift_probs(probs, delta, s=0.4):
    """점수(100pS+30pH)를 delta 만큼 이동: Failure 질량을 Shadow s / Heart (1-s) 로."""
    p = probs.copy()
    m = delta / (30.0 + 70.0 * s)
    lower = -np.minimum(p[:, 0] / s, p[:, 1] / (1 - s))
    m = np.clip(m, lower, p[:, 2])
    p[:, 0] += m * s; p[:, 1] += m * (1 - s); p[:, 2] -= m
    return p

def sh_probs(probs, delta):
    """pFailure 고정, Shadow<->Heart 만 이동 (점수 70 점당 질량 1)."""
    p = probs.copy()
    mv = np.clip(delta / 70.0, -p[:, 0] + 1e-9, p[:, 1] - 1e-9)
    p[:, 0] += mv; p[:, 1] -= mv
    return p

for season in (2022, 2023, 2024):
    r, tr = get(season)
    probs = r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
    base = evaluate(probs, r.control_grade, r.pitcher_id)
    # 투수 수준 기울기: 실제 평균 점수 vs 투수 평균 balls_before
    g = pd.DataFrame({'pid': r.pitcher_id.to_numpy(), 'act': (r.control_grade.map(dict(zip(CLASSES,[100.,30.,0.])))).to_numpy(),
                      'pred': probs @ np.array([100., 30., 0.]), 'balls': tr.balls_before.to_numpy(float)})
    a = g.groupby('pid').agg(n=('act','size'), act=('act','mean'), pred=('pred','mean'), balls=('balls','mean'))
    a = a[a.n >= 50]
    sa = np.polyfit(a.balls, a.act, 1)[0]; sp = np.polyfit(a.balls, a.pred, 1)[0]
    print(f'\n== {season} ({tag}) 투수 {len(a)}명  base tau {base["tau"]:.4f} player {base["player"]:.3f} pitch {base["pitch"]:.4f}')
    print(f'   기울기 actual~평균balls {sa:.2f} / pred~평균balls {sp:.2f} → 미반영 갭 {sa-sp:.2f}')
    for name, delta in [
        ('balls -1',        -1.0 * (tr.balls_before.to_numpy(float) - 0.9)),
        ('제안 3항 x1',     -1.0*(tr.balls_before.to_numpy(float)-0.9) + 0.5*(tr.strikes_before.to_numpy(float)-0.87) - 0.5*(tr.num_runners_on.to_numpy(float)-0.68)),
        ('제안 3항 x2',     2*(-1.0*(tr.balls_before.to_numpy(float)-0.9) + 0.5*(tr.strikes_before.to_numpy(float)-0.87) - 0.5*(tr.num_runners_on.to_numpy(float)-0.68))),
        ('제안 3항 x3',     3*(-1.0*(tr.balls_before.to_numpy(float)-0.9) + 0.5*(tr.strikes_before.to_numpy(float)-0.87) - 0.5*(tr.num_runners_on.to_numpy(float)-0.68))),
    ]:
        for mode, fn in (('S<->H', sh_probs), ('S<->F', shift_probs)):
            e = evaluate(fn(probs, delta), r.control_grade, r.pitcher_id)
            print(f'   {name:12s} {mode}: dTau {e["tau"]-base["tau"]:+.4f}  dPlayer {e["player"]-base["player"]:+.3f}  dPitch {e["pitch"]-base["pitch"]:+.4f}  dTotal {e["total"]-base["total"]:+.4f}')
