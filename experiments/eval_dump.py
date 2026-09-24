"""experiments/decomp/s22rows_<tag>_<val>.pkl 에서 ALL / R 한정 지표를 계산해 태그 간 비교."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.features import CLASSES, evaluate
tags = sys.argv[1:]
for tag in tags:
    for val in (2022, 2023, 2024):
        f = ROOT / 'experiments' / 'decomp' / f's22rows_{tag}_{val}.pkl'
        if not f.exists():
            print(f'{tag} {val}: (없음)'); continue
        r = pd.read_pickle(f)
        probs = r[[f'p_{g.lower()}' for g in CLASSES]].to_numpy(float)
        a = evaluate(probs, r.control_grade, r.pitcher_id)
        m = r.game_type.eq('R').to_numpy()
        b = evaluate(probs[m], r.control_grade[m], r.pitcher_id[m])
        print(f"{tag:14s} {val}  ALL total {a['total']:.3f} pitch {a['pitch']:.3f} player {a['player']:.3f} tau {a['tau']:.4f} | "
              f"R total {b['total']:.3f} pitch {b['pitch']:.3f} player {b['player']:.3f} tau {b['tau']:.4f}")
