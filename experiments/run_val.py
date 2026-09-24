"""시즌 홀드아웃 검증 실험.

사용 예
  python experiments/run_val.py naive                      # 이력 파라미터(decay, alpha) 그리드 → 투수 순위 tau-b
  python experiments/run_val.py run base_rf hist_lgbm      # 모델 실험 (val 2024, 2023)
  python experiments/run_val.py run hist_lgbm --val 2024 --decay 0.7 --alpha 100
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import (CLASSES, GRADE_SCORE, build_features, cat_cols,  # noqa: E402
                          evaluate, load_train, make_history)

CACHE = ROOT / "experiments" / "train_merged.pkl"
RESULTS = ROOT / "experiments" / "results.tsv"

EXPS = {
    # name: (model, extra_history, drop_ids)
    "base_rf": ("rf", False, False),
    "hist_rf": ("rf", True, False),
    "base_lgbm": ("lgbm", False, True),
    "hist_lgbm": ("lgbm", True, True),
    "hist_lgbm_reg": ("lgbm_reg", True, True),
    "blend_rf_lgbmreg": (("rf", "lgbm_reg"), True, True),
    "naive": ("naive", True, True),
}


def get_data() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    data = load_train(ROOT / "data")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(CACHE)
    return data


def naive_tau(data, val_season, decay, alpha, col="hist_pitcher_score"):
    """모델 없이 이력 점수만으로 투수 순위를 매겼을 때의 tau-b."""
    from scipy.stats import kendalltau
    hist = make_history(data.loc[data.season.lt(val_season)], val_season, decay, alpha, extra=True)
    cur = data.loc[data.season.eq(val_season) & data.control_grade.isin(CLASSES)]
    n = cur.groupby("pitcher_id").size()
    elig = n[n >= 50].index
    actual = cur.control_grade.map(dict(zip(CLASSES, GRADE_SCORE))).groupby(cur.pitcher_id).mean()[elig]
    pred = hist["pitcher_table"][col].reindex(elig).fillna(hist["pitcher_fallback"][col])
    return float(kendalltau(actual.round(10), pred.round(10), variant="b").statistic)


def make_xy(data, val_season, decay, alpha, extra, drop_ids, train_from=2019, era=False, tm_std=False,
            hist_gt=None, train_gt=None):
    feats, rows = [], []
    hist_src = data if not hist_gt else data.loc[data.game_type.eq(hist_gt)]
    for season in sorted(data.season.unique()):
        if season > val_season or (season < train_from and season != val_season):
            continue
        current = data.loc[data.season.eq(season) & data.control_grade.isin(CLASSES)]
        if train_gt and season != val_season:
            current = current.loc[current.game_type.eq(train_gt)]
        history = make_history(hist_src.loc[hist_src.season.lt(season)], season, decay, alpha, extra,
                               era_adjust=era, tm_standardize=tm_std)
        feats.append(build_features(current, history, drop_ids))
        rows.append(current[["season", "pitcher_id", "control_grade", "game_type"]])
    X = pd.concat(feats, ignore_index=True)
    rows = pd.concat(rows, ignore_index=True)
    return X, rows


RF_OVERRIDES = {}


def make_model(name, X, cats):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    nums = [c for c in X.columns if c not in cats]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        pre = ColumnTransformer([("cat", enc, cats),
                                 ("num", SimpleImputer(strategy="median", keep_empty_features=True), nums)])
        rf_kw = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, max_features="sqrt")
        rf_kw.update(RF_OVERRIDES)
        clf = RandomForestClassifier(n_jobs=-1, random_state=42, **rf_kw)
        return Pipeline([("pre", pre), ("clf", clf)]), {}
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        pre = ColumnTransformer([("cat", enc, cats), ("num", "passthrough", nums)])
        clf = LGBMClassifier(objective="multiclass", n_estimators=500, learning_rate=0.05,
                             num_leaves=63, min_child_samples=500, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, n_jobs=20, random_state=42,
                             verbose=-1)
        return Pipeline([("pre", pre), ("clf", clf)]), {"clf__categorical_feature": list(range(len(cats)))}
    if name == "lgbm_reg":
        from lightgbm import LGBMClassifier
        pre = ColumnTransformer([("cat", enc, cats), ("num", "passthrough", nums)])
        clf = LGBMClassifier(objective="multiclass", n_estimators=300, learning_rate=0.05,
                             num_leaves=31, min_child_samples=2000, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=10.0, n_jobs=20, random_state=42,
                             verbose=-1)
        return Pipeline([("pre", pre), ("clf", clf)]), {"clf__categorical_feature": list(range(len(cats)))}
    raise ValueError(name)


def run_exp(data, name, val_season, decay, alpha, drop=(), tag="", dump=False, train_from=2019, era=False,
            seeds=1, tm_std=False, eval_gt=None, hist_gt=None, train_gt=None):
    model_name, extra, drop_ids = EXPS[name]
    t0 = time.time()
    X, rows = make_xy(data, val_season, decay, alpha, extra, drop_ids, train_from, era, tm_std, hist_gt, train_gt)
    hist_cols = X[["hist_pitcher_score", "hist_pitcher_n"]] if "hist_pitcher_score" in X else None
    if drop:
        X = X.drop(columns=[c for c in X.columns if any(c.startswith(d) for d in drop)])
    y = rows.control_grade
    is_valid = rows.season.eq(val_season)
    cats = [c for c in cat_cols(drop_ids) if c in X.columns]
    probs, n_models = None, 0
    if model_name == "naive":
        rate_cols = [f"hist_pitcher_{g.lower()}_rate" for g in CLASSES]
        probs, n_models = X.loc[is_valid, rate_cols].to_numpy(float), 1
    for m_name in (() if model_name == "naive" else (model_name if isinstance(model_name, tuple) else (model_name,))):
        for i in range(seeds):
            model, fit_kw = make_model(m_name, X, cats)
            if seeds > 1:
                model.set_params(clf__random_state=42 + i)
            model.fit(X.loc[~is_valid], y.loc[~is_valid], **fit_kw)
            order = [list(model.classes_).index(g) for g in CLASSES]
            p_i = model.predict_proba(X.loc[is_valid])[:, order]
            probs = p_i if probs is None else probs + p_i
            n_models += 1
    probs = probs / n_models
    if eval_gt:
        keep = (rows.loc[is_valid, "game_type"] == eval_gt).to_numpy()
        res = evaluate(probs[keep], y.loc[is_valid][keep], rows.loc[is_valid, "pitcher_id"][keep])
    else:
        res = evaluate(probs, y.loc[is_valid], rows.loc[is_valid, "pitcher_id"])
    res.update(exp=name + tag, val=val_season, decay=decay, alpha=alpha, n_feat=X.shape[1],
               n_train=int((~is_valid).sum()), sec=round(time.time() - t0))
    if dump:
        v = rows.loc[is_valid].copy()
        v["actual"] = (v.control_grade.to_numpy()[:, None] == np.asarray(CLASSES)).astype(float) @ GRADE_SCORE
        v["pred"] = probs @ GRADE_SCORE
        if hist_cols is not None:
            v["hist_score"] = hist_cols.loc[is_valid, "hist_pitcher_score"].to_numpy()
            v["hist_n"] = hist_cols.loc[is_valid, "hist_pitcher_n"].to_numpy()
        agg = {"n": ("actual", "size"), "actual": ("actual", "mean"), "pred": ("pred", "mean")}
        if hist_cols is not None:
            agg.update(hist_score=("hist_score", "first"), hist_n=("hist_n", "first"))
        v.groupby("pitcher_id").agg(**agg).to_csv(ROOT / "experiments" / f"dump_{name}{tag}_{val_season}.csv")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["naive", "run"])
    ap.add_argument("exps", nargs="*")
    ap.add_argument("--val", type=int, nargs="+", default=[2024, 2023])
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--drop", nargs="*", default=(), help="제거할 피처 접두어 목록 (ablation)")
    ap.add_argument("--tag", default="", help="results.tsv 에 붙일 실험 이름 접미어 (앞에 - 가 자동으로 붙음)")
    ap.add_argument("--dump", action="store_true", help="검증 시즌 투수별 예측 표 저장")
    ap.add_argument("--train-from", type=int, default=2019, help="학습 행으로 쓸 최소 시즌 (이력은 전체 사용)")
    ap.add_argument("--era", action="store_true", help="이력을 시즌 리그 평균 대비 편차로 계산")
    ap.add_argument("--seeds", type=int, default=1, help="시드를 바꿔 N개 모델 확률 평균")
    ap.add_argument("--tm-std", action="store_true", help="Trackman 이력을 시즌 내 z-score 로 표준화")
    ap.add_argument("--rf-params", default="", help="RF 하이퍼파라미터 덮어쓰기, 예: n_estimators=300,max_features=0.3")
    ap.add_argument("--eval-gt", default=None, help="검증 평가를 특정 game_type 행으로 제한 (예: R)")
    ap.add_argument("--hist-gt", default=None, help="이력을 특정 game_type 행으로만 계산 (예: R)")
    ap.add_argument("--train-gt", default=None, help="학습 행을 특정 game_type 으로 제한 (예: R)")
    args = ap.parse_args()
    args.tag = f"-{args.tag}" if args.tag else ""
    if args.rf_params:
        for kv in args.rf_params.split(","):
            k, v = kv.split("=")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            RF_OVERRIDES[k] = v

    t0 = time.time()
    data = get_data()
    print(f"데이터 {len(data):,}행 로드 ({time.time() - t0:.0f}s)")

    if args.cmd == "naive":
        out = []
        for val in args.val:
            for decay in (1.0, 0.85, 0.7, 0.5, 0.3):
                for alpha in (30, 100, 300, 1000):
                    out.append(dict(val=val, decay=decay, alpha=alpha,
                                    tau=naive_tau(data, val, decay, alpha),
                                    tau_last=naive_tau(data, val, decay, alpha, "hist_pitcher_last_score")))
        df = pd.DataFrame(out)
        piv = df.pivot_table(index=["decay", "alpha"], columns="val", values="tau").round(4)
        print("\n[이력 점수(hist_pitcher_score)만으로 투수 순위 → tau-b]")
        print(piv.to_string())
        print("\n[직전 시즌 단독 점수(last_score) → tau-b]  (decay 무관)")
        print(df[df.decay == 1.0].pivot_table(index="alpha", columns="val", values="tau_last").round(4).to_string())
        return

    results = []
    for val in args.val:
        for name in args.exps:
            r = run_exp(data, name, val, args.decay, args.alpha, tuple(args.drop), args.tag, args.dump,
                        args.train_from, args.era, args.seeds, args.tm_std, args.eval_gt, args.hist_gt, args.train_gt)
            results.append(r)
            print(f"{name + args.tag:16s} val={val} decay={args.decay} alpha={args.alpha:g} | "
                  f"총점 {r['total']:.4f} | pitch {r['pitch']:.4f} | player {r['player']:.4f} "
                  f"(tau {r['tau']:.4f}, 투수 {r['n_pitchers']}) | feat {r['n_feat']} | {r['sec']}s", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS, sep="\t", mode="a", header=not RESULTS.exists(), index=False)


if __name__ == "__main__":
    main()
