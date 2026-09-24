"""S22 구성(RF, --keep-ids --f-seasons --train-from --hist-keep-f --era-gt) 로컬 검증 + 행/투수별 예측 dump.

train.py 의 데이터 선택 로직을 검증 시즌 기준으로 재현한다.
  python3 experiments/val_s22.py --val 2024 2023 2022 [--f-seasons auto|2022] [--tag x] [--extra-hist]
출력: experiments/decomp/s22rows_<tag>_<val>.pkl (행별 pitcher_id, game_type, actual, pred, hist_score, hist_n)
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from src.features import CLASSES, GRADE_SCORE, build_features, cat_cols, evaluate, make_history  # noqa
from run_val import get_data, RF_OVERRIDES, make_model  # noqa
OUT = ROOT / "experiments" / "decomp"; OUT.mkdir(exist_ok=True)


def run(data, val, args):
    t0 = time.time()
    pool = data.loc[data.season.lt(val)]
    f_keep = {val - 1} if args.f_seasons == "auto" else {int(x) for x in args.f_seasons.split(",")}
    f_keep = {s for s in f_keep if s < val}
    train_pool = pool.loc[pool.game_type.ne("F") | pool.season.isin(f_keep)]
    hist_src = pool if args.hist_keep_f else train_pool
    if args.hist_f_from:
        hist_src = hist_src.loc[~(hist_src.game_type.eq("F") & hist_src.season.lt(args.hist_f_from))]
    train_from = args.train_from if args.train_from else min(2021, val - 2)
    hk = dict(decay=1.0, alpha=args.alpha, extra=not args.no_extra, era_adjust=args.era_gt, era_by_gt=args.era_gt, f_share=args.f_share, ball_hist=args.ball_hist,
              rev_split=args.rev_split, alpha_rev=args.alpha_rev, alpha_zone=args.alpha_zone, soft_sigma=args.soft_sigma)
    feats, ys, gts = [], [], []
    for season in sorted(train_pool.season.unique()):
        if season < train_from:
            continue
        cur = train_pool.loc[train_pool.season.eq(season) & train_pool.control_grade.isin(CLASSES)]
        hist = make_history(hist_src.loc[hist_src.season.lt(season)], season, **hk)
        feats.append(build_features(cur, hist, drop_ids=False)); ys.append(cur.control_grade); gts.append(cur.game_type)
    X = pd.concat(feats, ignore_index=True); y = pd.concat(ys, ignore_index=True)
    cur_v = data.loc[data.season.eq(val) & data.control_grade.isin(CLASSES)]
    hist_v = make_history(hist_src.loc[hist_src.season.lt(val)], val, **hk)
    Xv = build_features(cur_v, hist_v, drop_ids=False)
    if args.drop:
        dc = [c for c in X.columns if any(c.startswith(d) for d in args.drop)]
        X, Xv = X.drop(columns=dc), Xv.drop(columns=dc)
    cats = [c for c in cat_cols(False) if c in X.columns]
    probs, n_models = None, 0
    for m_name in args.model:
        for i in range(args.seeds):
            model, fit_kw = make_model(m_name, X, cats)
            model.set_params(clf__random_state=args.seed + i)
            model.fit(X, y, **fit_kw)
            order = [list(model.classes_).index(g) for g in CLASSES]
            p = model.predict_proba(Xv)[:, order]
            probs = p if probs is None else probs + p; n_models += 1
    probs = probs / n_models
    rows = cur_v[["pitcher_id", "game_type", "control_grade", "inning"]].copy()
    rows["actual"] = (rows.control_grade.to_numpy()[:, None] == np.asarray(CLASSES)).astype(float) @ GRADE_SCORE
    rows["pred"] = probs @ GRADE_SCORE
    for k, g in enumerate(CLASSES):
        rows[f"p_{g.lower()}"] = probs[:, k]
    rows["hist_score"] = Xv["hist_pitcher_score"].to_numpy() if "hist_pitcher_score" in Xv else np.nan
    rows["hist_n"] = Xv["hist_pitcher_n"].to_numpy()
    rows["hist_score_plain"] = np.nan
    rows.to_pickle(OUT / f"s22rows_{args.tag}_{val}.pkl")
    res_all = evaluate(probs, cur_v.control_grade, cur_v.pitcher_id)
    keepR = cur_v.game_type.eq("R").to_numpy()
    res_R = evaluate(probs[keepR], cur_v.control_grade[keepR], cur_v.pitcher_id[keepR])
    print(f"[{args.tag} val={val} train_from={train_from} F={sorted(f_keep)} feat={X.shape[1]} n_train={len(X):,}] "
          f"ALL total {res_all['total']:.3f} pitch {res_all['pitch']:.3f} player {res_all['player']:.3f} tau {res_all['tau']:.4f} | "
          f"R total {res_R['total']:.3f} pitch {res_R['pitch']:.3f} player {res_R['player']:.3f} tau {res_R['tau']:.4f} | {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, nargs="+", default=[2024, 2023, 2022])
    ap.add_argument("--model", nargs="+", default=["rf"])
    ap.add_argument("--f-seasons", default="auto")
    ap.add_argument("--hist-keep-f", action="store_true", default=True)
    ap.add_argument("--era-gt", action="store_true", default=True)
    ap.add_argument("--no-extra", action="store_true")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--train-from", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--drop", nargs="*", default=())
    ap.add_argument("--rf-params", default="n_estimators=300")
    ap.add_argument("--f-share", action="store_true")
    ap.add_argument("--ball-hist", action="store_true")
    ap.add_argument("--rev-split", action="store_true", help="반대투구 비율·비반대 존 점수 구성요소별 이력 추가")
    ap.add_argument("--alpha-rev", type=float, default=50.0)
    ap.add_argument("--alpha-zone", type=float, default=400.0)
    ap.add_argument("--hist-f-from", type=int, default=None, help="이력에 쓰는 F 투구를 이 시즌 이후로 제한 (구체제 F 라벨 제외)")
    ap.add_argument("--soft-sigma", type=float, default=0.0, help="투수 이력을 좌표 커널 소프트 라벨(sd m)로 집계, 0 = 하드")
    ap.add_argument("--seed", type=int, default=42, help="단일 모델 random_state")
    ap.add_argument("--tag", default="s22")
    args = ap.parse_args()
    for kv in filter(None, args.rf_params.split(",")):
        k, v = kv.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        RF_OVERRIDES[k] = v
    data = get_data()
    for val in args.val:
        run(data, val, args)


if __name__ == "__main__":
    main()
