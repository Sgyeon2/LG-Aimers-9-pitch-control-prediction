"""규칙 5.2 기계적 검증: 추론이 '행 자신의 입력 + 학습 시점 고정 상수' 만 쓰는지 증명한다.
같은 행의 예측값이 (1) 다른 행의 존재/순서, (2) 배치 크기, (3) 단독 추론 에서 모두 동일해야 한다."""
import os, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = os.environ.get('LGA_PYTHON', sys.executable)
ZIP = ROOT / 'submit.zip'

def run(rows, work):
    d = work / 'data'; d.mkdir(parents=True, exist_ok=True)
    rows.to_csv(d / 'test.csv', index=False)
    pd.DataFrame({'row_id': rows.row_id, 'prob_shadow': 1/3, 'prob_heart': 1/3, 'prob_failure': 1/3}).to_csv(d / 'sample_submission.csv', index=False)
    subprocess.run([PY, 'script.py'], cwd=work, check=True, capture_output=True)
    return pd.read_csv(work / 'output' / 'submission.csv', dtype={'row_id': str}).set_index('row_id')

base = pd.read_csv(ROOT / 'data/train.csv', dtype={'row_id': str})
base = base[base.season == 2024].head(3000).copy(); base['season'] = 2025
print(f'검증용 가짜 test {len(base)} 행 (2024 행의 season 만 2025 로)')

tmp = Path(tempfile.mkdtemp())
try:
    W = tmp / 'w'; W.mkdir(); shutil.unpack_archive(ZIP, W, 'zip')
    full = run(base, W)

    # (1) 행 순서를 뒤집어도 같은가
    rev = run(base.iloc[::-1].copy(), W)
    d1 = np.abs(full.loc[rev.index].to_numpy() - rev.to_numpy()).max()

    # (2) 임의 부분집합(300행)만 넣어도 같은가 — 다른 행의 '존재'가 영향을 주는지
    sub = base.sample(300, random_state=0)
    d2 = np.abs(full.loc[sub.row_id].to_numpy() - run(sub, W).loc[sub.row_id].to_numpy()).max()

    # (3) 한 행씩 단독 추론해도 같은가 (10개 표본)
    d3 = 0.0
    for i in range(10):
        one = base.iloc[[i]]
        d3 = max(d3, float(np.abs(full.loc[one.row_id].to_numpy() - run(one, W).to_numpy()).max()))

    # (4) 같은 투수의 다른 행만 모두 제거해도 같은가 — 투수 단위 test 집계 여부의 직접 검사
    pid = base.pitcher_id.value_counts().index[0]
    keep = base[(base.pitcher_id != pid) | (base.index == base[base.pitcher_id == pid].index[0])]
    tgt = base[base.pitcher_id == pid].iloc[[0]].row_id
    d4 = float(np.abs(full.loc[tgt].to_numpy() - run(keep, W).loc[tgt].to_numpy()).max())

    print(f'(1) 행 순서 역전           최대 절대차 {d1:.3e}')
    print(f'(2) 300행 부분집합만 추론  최대 절대차 {d2:.3e}')
    print(f'(3) 1행 단독 추론 x10      최대 절대차 {d3:.3e}')
    print(f'(4) 같은 투수 다른 행 제거 최대 절대차 {d4:.3e}  (투수 {pid}, {int((base.pitcher_id==pid).sum())}행 → 1행)')
    worst = max(d1, d2, d3, d4)
    ok = worst < 1e-12   # 배정도 반올림 한계(확률 ~0.5 에서 eps ~1e-16)
    print(f'\n최대 절대차 {worst:.3e} — 배정도 반올림 한계 1e-12 기준')
    print('결과:', '행 단위 독립 추론 확인 (다른 행의 존재·순서·개수와 무관)' if ok else '차이 발견 — 행 간 의존성 존재')
finally:
    shutil.rmtree(tmp, ignore_errors=True)
