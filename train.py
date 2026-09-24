from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.features import (CLASSES, build_features, cat_cols, load_train,  # noqa: E402
                          make_history)

LGBM_PARAMS = {
    "lgbm": dict(objective="multiclass", n_estimators=500, learning_rate=0.05, num_leaves=63,
                 min_child_samples=500, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                 reg_lambda=5.0, n_jobs=-1, random_state=42, verbose=-1),
    "lgbm_reg": dict(objective="multiclass", n_estimators=300, learning_rate=0.05, num_leaves=31,
                     min_child_samples=2000, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                     reg_lambda=10.0, n_jobs=-1, random_state=42, verbose=-1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["lgbm"], choices=list(LGBM_PARAMS) + ["rf"],
                    help="여러 개면 확률 평균 (예: lgbm_reg rf)")
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--drop", nargs="*", default=(), help="제거할 피처 접두어")
    ap.add_argument("--out", default="model/model.pkl")
    ap.add_argument("--test-season", type=int, default=2025)
    ap.add_argument("--train-from", type=int, default=2019, help="학습 행으로 쓸 최소 시즌 (이력은 전체 사용)")
    ap.add_argument("--era", action="store_true", help="이력을 시즌 리그 평균 대비 편차로 계산")
    ap.add_argument("--era-gt", action="store_true", help="--era 를 시즌×경기유형(R/F) 평균 대비로 (퓨처스 수준 변화 제거)")
    ap.add_argument("--seeds", type=int, default=1, help="시드를 바꿔 N개 모델 학습, 추론 시 확률 평균")
    ap.add_argument("--tm-std", action="store_true", help="Trackman 이력을 시즌 내 z-score 로 표준화")
    ap.add_argument("--no-extra", action="store_true", help="baseline 과 동일한 49개 피처만 사용")
    ap.add_argument("--keep-ids", action="store_true", help="pitcher_id/batter_id 를 범주 피처로 유지 (baseline 방식)")
    ap.add_argument("--hist-only", action="store_true", help="모델 없이 투수 이력 비율을 그대로 확률로 출력 (진단용)")
    ap.add_argument("--offset", action="store_true",
                    help="상황 피처만 쓰는 LightGBM + init_score=log(투수 이력 비율). 투수 순위는 이력이 결정")
    ap.add_argument("--offset-keep-season", action="store_true", help="offset 모드에서 season 피처 유지")
    ap.add_argument("--offset-rounds", type=int, default=300)
    ap.add_argument("--hist-gt", default=None, help="이력을 특정 game_type 행으로만 계산 (예: R)")
    ap.add_argument("--train-gt", default=None, help="학습 행을 특정 game_type 으로 제한 (예: R)")
    ap.add_argument("--drop-f-from", type=int, default=None,
                    help="이 시즌 이후의 F(퓨처스) 행을 학습과 이력에서 제외 (예: 2023 → 2023~2024 F 제외, F 규칙은 2022 까지로 학습)")
    ap.add_argument("--hist-keep-f", action="store_true", help="--drop-f-from 을 학습 행에만 적용하고 이력에는 F 행을 유지")
    ap.add_argument("--f-seasons", default="", help="학습 행의 F 는 이 시즌들만 사용 (예: 2021,2022). 이력은 --hist-keep-f 와 동일하게 전체 유지")
    ap.add_argument("--rf-params", default="", help="RF 하이퍼파라미터 덮어쓰기, 예: n_estimators=500,max_features=0.3")
    ap.add_argument("--spread", action="store_true", help="투수별 릴리스·도달 위치 표준편차(제구 일관성) 이력 추가")
    ap.add_argument("--f-share", action="store_true", help="투수 이력에 퓨처스 투구 비중·1군 투구 수 피처 추가")
    ap.add_argument("--ball-hist", action="store_true", help="원천 볼 판정·반대투구 비율 이력 추가 (Unlabeled 행 포함)")
    ap.add_argument("--rev-split", action="store_true", help="이력에 반대투구 비율·비반대 존 점수를 따로 수축한 구성요소별 점수 추가 (라벨 정의 기반)")
    ap.add_argument("--alpha-rev", type=float, default=50.0, help="--rev-split 반대투구 비율 수축 강도")
    ap.add_argument("--alpha-zone", type=float, default=400.0, help="--rev-split 비반대 존 점수 수축 강도")
    ap.add_argument("--soft-sigma", type=float, default=0.0,
                    help="투수 이력을 하드 등급 대신 좌표 커널 소프트 라벨(sd m, 예 0.1)로 집계 (0 = 기존 하드 라벨)")
    ap.add_argument("--hist-f-from", type=int, default=None, help="이력에 쓰는 F(퓨처스) 투구를 이 시즌 이후로 제한 (2023 판정 체제 변경 이전 F 라벨 제외)")
    ap.add_argument("--f-submodel", default="", choices=["", "rf", "lgbm", "lgbm_reg"],
                    help="F(퓨처스) 행만으로 별도 모델을 학습해 추론 시 F 행 예측을 혼합 (F 규칙 구조를 온전히 표현)")
    ap.add_argument("--f-sub-seasons", default="2021,2022", help="F 서브모델 학습 시즌")
    ap.add_argument("--f-sub-w", type=float, default=0.5)
    ap.add_argument("--rookie-model", default="", choices=["", "lgbm_reg", "lgbm", "rf"],
                    help="학습 이력 없는 투수(hist_pitcher_n=0) 행만으로 별도 모델을 학습해 추론 시 신인 행 예측을 혼합")
    ap.add_argument("--rookie-w", type=float, default=0.5, help="신인 행에서 신인 모델 확률의 혼합 가중치")
    ap.add_argument("--rookie-from", type=int, default=2020, help="신인 모델 학습 시즌 시작 (2019 는 전원이 이력 없음이라 제외)")
    ap.add_argument("--r-model", default="", choices=["", "lgbm_reg", "lgbm"],
                    help="1군(R) 행 전용 LightGBM 을 따로 학습(R 행만, R 이력만, game_type·팀·타자이력·ID 제외)해 추론 시 R 행 확률을 혼합")
    ap.add_argument("--r-w", type=float, default=0.5, help="R 행에서 R 전용 모델 확률의 혼합 가중치")
    ap.add_argument("--r-from", type=int, default=2019, help="R 전용 모델 학습 시즌 시작")
    ap.add_argument("--f-weight", type=float, default=1.0,
                    help="학습 시 F(퓨처스) 행의 sample_weight (2022 F 만 남기면 F 비율이 2.6%%로 줄어 test 의 ~10%% 와 어긋남을 보정)")
    args = ap.parse_args()
    drop_ids = not args.keep_ids
    extra = not args.no_extra

    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    t0 = time.time()
    data = load_train(ROOT / "data")
    print(f"데이터 {len(data):,}행 로드 ({time.time() - t0:.0f}s)")

    feats, ys, gts = [], [], []
    full_data = data
    if args.drop_f_from:
        data = data.loc[~(data.game_type.eq("F") & data.season.ge(args.drop_f_from))]
        print(f"F 행 제외(season>={args.drop_f_from}) 후 {len(data):,}행" + (" (이력은 전체 유지)" if args.hist_keep_f else ""))
    if args.f_seasons:
        keep = {int(x) for x in args.f_seasons.split(",")}
        data = data.loc[data.game_type.ne("F") | data.season.isin(keep)]
        print(f"F 행은 {sorted(keep)} 만 학습에 사용 → {len(data):,}행" + (" (이력은 전체 유지)" if args.hist_keep_f else ""))
    hist_base = full_data if args.hist_keep_f else data
    hist_src = hist_base if not args.hist_gt else hist_base.loc[hist_base.game_type.eq(args.hist_gt)]
    if args.hist_f_from:
        hist_src = hist_src.loc[~(hist_src.game_type.eq("F") & hist_src.season.lt(args.hist_f_from))]
        print(f"이력의 F 투구는 season>={args.hist_f_from} 만 사용 → 이력 원천 {len(hist_src):,}행")
    for season in sorted(data.season.unique()):
        if season < args.train_from:
            continue
        current = data.loc[data.season.eq(season) & data.control_grade.isin(CLASSES)]
        if args.train_gt:
            current = current.loc[current.game_type.eq(args.train_gt)]
        history = make_history(hist_src.loc[hist_src.season.lt(season)], season, args.decay, args.alpha, extra=extra,
                               era_adjust=args.era or args.era_gt, tm_standardize=args.tm_std, era_by_gt=args.era_gt,
                               spread=args.spread, f_share=args.f_share, ball_hist=args.ball_hist,
                               rev_split=args.rev_split, alpha_rev=args.alpha_rev, alpha_zone=args.alpha_zone, soft_sigma=args.soft_sigma)
        feats.append(build_features(current, history, drop_ids))
        ys.append(current.control_grade)
        gts.append(current.game_type)
    X = pd.concat(feats, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    sample_weight = np.where(pd.concat(gts, ignore_index=True).eq("F").to_numpy(), args.f_weight, 1.0)
    if args.f_weight != 1.0:
        print(f"F 행 가중치 {args.f_weight} (F 행 {int((sample_weight != 1).sum()):,}개)")
    if args.drop:
        X = X.drop(columns=[c for c in X.columns if any(c.startswith(d) for d in args.drop)])
    RATE_COLS = [f"hist_pitcher_{g.lower()}_rate" for g in CLASSES]
    init_score = None
    if args.offset:
        init_score = np.log(np.clip(X[RATE_COLS].to_numpy(float), 1e-6, 1))
        keep = [c for c in X.columns if not c.startswith("hist_")
                and c not in ("pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id")]
        if not args.offset_keep_season:
            keep = [c for c in keep if c != "season"]
        X = X[keep]
    cats = [c for c in cat_cols(drop_ids) if c in X.columns]
    nums = [c for c in X.columns if c not in cats]
    print(f"학습 {len(X):,}행 / 피처 {X.shape[1]}개 (범주 {len(cats)})")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    pipelines = []
    if args.offset:
        # 라벨을 CLASSES 순서의 정수로 고정 (init_score 열 순서와 일치)
        y = y.map({g: i for i, g in enumerate(CLASSES)})
        params = dict(LGBM_PARAMS["lgbm_reg"], n_estimators=args.offset_rounds)
        for i in range(args.seeds):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
            pre = ColumnTransformer([("cat", enc, cats), ("num", "passthrough", nums)])
            pipeline = Pipeline([("pre", pre), ("clf", LGBMClassifier(**dict(params, random_state=42 + i)))])
            pipeline.fit(X, y, clf__init_score=init_score, clf__categorical_feature=list(range(len(cats))))
            pipelines.append(pipeline)
            print(f"학습 완료 offset seed {i + 1}/{args.seeds} ({time.time() - t0:.0f}s)")
    for m_name in ([] if (args.hist_only or args.offset) else args.model):
        for i in range(args.seeds):
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
            if m_name == "rf":
                pre = ColumnTransformer([("cat", enc, cats),
                                         ("num", SimpleImputer(strategy="median", keep_empty_features=True), nums)])
                rf_kw = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features="sqrt")
                for kv in filter(None, args.rf_params.split(",")):
                    k, v = kv.split("=")
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                    rf_kw[k] = v
                clf = RandomForestClassifier(n_jobs=-1, random_state=42 + i, **rf_kw)
                pipeline = Pipeline([("pre", pre), ("clf", clf)])
                pipeline.fit(X, y, clf__sample_weight=sample_weight)
            else:
                params = dict(LGBM_PARAMS[m_name], random_state=42 + i)
                pre = ColumnTransformer([("cat", enc, cats), ("num", "passthrough", nums)])
                pipeline = Pipeline([("pre", pre), ("clf", LGBMClassifier(**params))])
                pipeline.fit(X, y, clf__categorical_feature=list(range(len(cats))), clf__sample_weight=sample_weight)
            pipelines.append(pipeline)
            print(f"학습 완료 {m_name} seed {i + 1}/{args.seeds} ({time.time() - t0:.0f}s)")

    rookie = None
    if args.rookie_model:
        # 신인 행: 해당 시즌 이전 라벨 이력이 전혀 없는 투수의 행. 역할(이닝·li·경기유형·월)이 투수 수준을 대변하도록 별도 학습.
        rf_, ry_ = [], []
        for season in sorted(data.season.unique()):
            if season < args.rookie_from:
                continue
            current = data.loc[data.season.eq(season) & data.control_grade.isin(CLASSES)]
            history = make_history(hist_src.loc[hist_src.season.lt(season)], season, args.decay, args.alpha, extra=extra,
                                   era_adjust=args.era or args.era_gt, tm_standardize=args.tm_std, era_by_gt=args.era_gt,
                                   spread=args.spread, f_share=args.f_share, ball_hist=args.ball_hist,
                               rev_split=args.rev_split, alpha_rev=args.alpha_rev, alpha_zone=args.alpha_zone, soft_sigma=args.soft_sigma)
            Xs = build_features(current, history, drop_ids=True)
            m = Xs["hist_pitcher_n"].eq(0).to_numpy()
            rf_.append(Xs.loc[m]); ry_.append(current.control_grade.loc[m])
        Xr = pd.concat(rf_); yr = pd.concat(ry_)
        Xr = Xr.drop(columns=[c for c in Xr.columns if c.startswith("hist_pitcher_") or c.startswith("hist_tm_")])
        rcats = [c for c in cat_cols(True) if c in Xr.columns]
        rnums = [c for c in Xr.columns if c not in rcats]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        if args.rookie_model == "rf":
            pre = ColumnTransformer([("cat", enc, rcats), ("num", SimpleImputer(strategy="median", keep_empty_features=True), rnums)])
            rpipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=300, max_depth=12,
                                                                            min_samples_leaf=200, max_features="sqrt"))])
            rpipe.fit(Xr, yr)
        else:
            pre = ColumnTransformer([("cat", enc, rcats), ("num", "passthrough", rnums)])
            rpipe = Pipeline([("pre", pre), ("clf", LGBMClassifier(**LGBM_PARAMS[args.rookie_model]))])
            rpipe.fit(Xr, yr, clf__categorical_feature=list(range(len(rcats))))
        rookie = {"pipeline": rpipe, "feature_columns": list(Xr.columns), "w": args.rookie_w, "model": args.rookie_model,
                  "class_indices": [list(rpipe.classes_).index(g) for g in CLASSES]}
        print(f"신인 모델 학습 완료 ({args.rookie_model}, 행 {len(Xr):,}, 피처 {Xr.shape[1]}, w={args.rookie_w}) ({time.time() - t0:.0f}s)")

    r_model = None
    if args.r_model:
        # 1군(R) 행 전용 LightGBM: R 행만 학습, R 이력만(era 보정 없음), game_type·팀·타자 이력·ID 제외 (experiments/val_hybrid.py 구성).
        # RF 와 1군 순위 상관이 0.81 로 다른 모델이라 R 행 확률을 w 로 혼합하면 로컬 3시즌 R tau +0.014~+0.022, 전체행 +0.000~+0.033.
        R_DROP = ("game_type", "pitcher_team_id", "batter_team_id", "hist_batter_")
        r_src = full_data.loc[full_data.game_type.eq("R")]
        rx_, rl_ = [], []
        for season in sorted(r_src.season.unique()):
            if season < args.r_from:
                continue
            current = r_src.loc[r_src.season.eq(season) & r_src.control_grade.isin(CLASSES)]
            hist_r = make_history(r_src.loc[r_src.season.lt(season)], season, args.decay, args.alpha, extra=extra)
            rx_.append(build_features(current, hist_r, drop_ids=True)); rl_.append(current.control_grade)
        Xl = pd.concat(rx_, ignore_index=True); yl = pd.concat(rl_, ignore_index=True)
        Xl = Xl.drop(columns=[c for c in Xl.columns if c.startswith(R_DROP)])
        lcats = [c for c in cat_cols(True) if c in Xl.columns]
        lnums = [c for c in Xl.columns if c not in lcats]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        pre = ColumnTransformer([("cat", enc, lcats), ("num", "passthrough", lnums)])
        lpipe = Pipeline([("pre", pre), ("clf", LGBMClassifier(**LGBM_PARAMS[args.r_model]))])
        lpipe.fit(Xl, yl, clf__categorical_feature=list(range(len(lcats))))
        r_model = {"pipeline": lpipe, "feature_columns": list(Xl.columns), "w": args.r_w, "model": args.r_model, "drop_ids": True,
                   "class_indices": [list(lpipe.classes_).index(g) for g in CLASSES],
                   "history": make_history(r_src, args.test_season, args.decay, args.alpha, extra=extra)}
        print(f"R 전용 모델 학습 완료 ({args.r_model}, 행 {len(Xl):,}, 피처 {Xl.shape[1]}, w={args.r_w}) ({time.time() - t0:.0f}s)")

    f_sub = None
    if args.f_submodel:
        fs_seasons = {int(x) for x in args.f_sub_seasons.split(",")}
        ff, fy = [], []
        for season in sorted(fs_seasons):
            current = full_data.loc[full_data.season.eq(season) & full_data.game_type.eq("F") & full_data.control_grade.isin(CLASSES)]
            history = make_history(hist_src.loc[hist_src.season.lt(season)], season, args.decay, args.alpha, extra=extra,
                                   era_adjust=args.era or args.era_gt, tm_standardize=args.tm_std, era_by_gt=args.era_gt,
                                   spread=args.spread, f_share=args.f_share, ball_hist=args.ball_hist,
                               rev_split=args.rev_split, alpha_rev=args.alpha_rev, alpha_zone=args.alpha_zone, soft_sigma=args.soft_sigma)
            ff.append(build_features(current, history, drop_ids=True)); fy.append(current.control_grade)
        Xf = pd.concat(ff); yf = pd.concat(fy)
        Xf = Xf.drop(columns=[c for c in ("game_type", "season") if c in Xf.columns])
        fcats = [c for c in cat_cols(True) if c in Xf.columns]
        fnums = [c for c in Xf.columns if c not in fcats]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        if args.f_submodel == "rf":
            pre = ColumnTransformer([("cat", enc, fcats), ("num", SimpleImputer(strategy="median", keep_empty_features=True), fnums)])
            fpipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_jobs=-1, random_state=42, n_estimators=300, max_depth=10,
                                                                            min_samples_leaf=100, max_features="sqrt"))])
            fpipe.fit(Xf, yf)
        else:
            pre = ColumnTransformer([("cat", enc, fcats), ("num", "passthrough", fnums)])
            fpipe = Pipeline([("pre", pre), ("clf", LGBMClassifier(**LGBM_PARAMS[args.f_submodel]))])
            fpipe.fit(Xf, yf, clf__categorical_feature=list(range(len(fcats))))
        f_sub = {"pipeline": fpipe, "feature_columns": list(Xf.columns), "w": args.f_sub_w, "model": args.f_submodel,
                 "class_indices": [list(fpipe.classes_).index(g) for g in CLASSES]}
        print(f"F 서브모델 학습 완료 ({args.f_submodel}, 시즌 {sorted(fs_seasons)}, 행 {len(Xf):,}, 피처 {Xf.shape[1]}, w={args.f_sub_w}) ({time.time() - t0:.0f}s)")

    bundle = {
        "r_model": r_model,
        "f_sub": f_sub,
        "rookie": rookie,
        "mode": "hist_only" if args.hist_only else ("offset" if args.offset else "model"),
        "pipeline": pipelines[0] if pipelines else None,
        "pipelines": pipelines,
        "feature_columns": list(X.columns),
        "history": make_history(hist_src, args.test_season, args.decay, args.alpha, extra=extra,
                                era_adjust=args.era or args.era_gt, tm_standardize=args.tm_std, era_by_gt=args.era_gt,
                                spread=args.spread, f_share=args.f_share, ball_hist=args.ball_hist,
                               rev_split=args.rev_split, alpha_rev=args.alpha_rev, alpha_zone=args.alpha_zone, soft_sigma=args.soft_sigma),
        "drop_ids": drop_ids,
        "drop_prefixes": list(args.drop),
        "cat_cols": cats,
        "params": {"model": args.model, "decay": args.decay, "alpha": args.alpha,
                   "train_from": args.train_from, "era_adjust": args.era, "seeds": args.seeds,
                   "tm_std": args.tm_std, "drop": list(args.drop), "n_train": int(len(X)),
                   "extra": extra, "keep_ids": args.keep_ids, "hist_only": args.hist_only,
                   "offset": args.offset, "offset_keep_season": args.offset_keep_season,
                   "hist_gt": args.hist_gt, "train_gt": args.train_gt, "drop_f_from": args.drop_f_from, "era_gt": args.era_gt, "hist_keep_f": args.hist_keep_f, "rf_params": args.rf_params, "spread": args.spread, "f_seasons": args.f_seasons, "f_weight": args.f_weight, "f_share": args.f_share, "ball_hist": args.ball_hist, "rev_split": args.rev_split, "alpha_rev": args.alpha_rev, "alpha_zone": args.alpha_zone, "hist_f_from": args.hist_f_from, "soft_sigma": args.soft_sigma, "rookie_model": args.rookie_model, "f_submodel": args.f_submodel, "f_sub_seasons": args.f_sub_seasons, "f_sub_w": args.f_sub_w, "rookie_w": args.rookie_w, "rookie_from": args.rookie_from, "r_model": args.r_model, "r_w": args.r_w, "r_from": args.r_from,
                   "versions": {m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "lightgbm")},
                   "python": sys.version.split()[0]},
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out, compress=3)
    (out.parent / "train_meta.json").write_text(json.dumps(bundle["params"], ensure_ascii=False, indent=2))
    print(f"저장: {out} ({out.stat().st_size / 1e6:.1f} MB)  버전 {bundle['params']['versions']}")


if __name__ == "__main__":
    main()
