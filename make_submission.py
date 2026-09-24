"""제출 파일 생성: submit/script.py, submit/requirements.txt, submit/model/model.pkl → submit.zip

script.py 는 src/features.py 전체 소스를 그대로 포함해 생성하므로 학습과 추론의 피처 코드가 항상 같다.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

INFERENCE_MAIN = '''

# ============================================================================
# 추론 (평가 서버에서 실행): ./data/test.csv + ./model/model.pkl → ./output/submission.csv
# ============================================================================
import joblib  # noqa: E402

ID_COL = "row_id"
PROB_COLS = ("prob_shadow", "prob_heart", "prob_failure")


def apply_post(probs, batch, bundle):
    """학습 시점에 고정된 상수만 쓰는 행별 후처리. 점수(100·Shadow+30·Heart)를 delta 점 이동시키도록
    Shadow<->Failure 확률 질량을 delta/100 만큼 옮긴다. 행 자신의 game_type 과 투수의 고정 이력 표본수만 참조한다."""
    post = bundle.get("post") or {}
    if not post:
        return probs
    probs = probs.copy()
    delta = np.zeros(len(batch))
    if post.get("f_shift"):
        delta += np.where(batch["game_type"].astype(str).to_numpy() == "F", float(post["f_shift"]), 0.0)
    if post.get("group_shift"):
        # F 행을 투수 그룹별로 다르게 이동: 이력에 1군(R) 투구가 없으면 pure, 있으면 mixed (학습 시점 고정 표)
        rn = bundle["history"]["pitcher_table"]["hist_pitcher_r_n"].reindex(batch["pitcher_id"].to_numpy()).fillna(0.0).to_numpy(dtype=float)
        is_f = batch["game_type"].astype(str).to_numpy() == "F"
        gs = post["group_shift"]
        delta += np.where(is_f, np.where(rn > 0, float(gs.get("mixed", 0.0)), float(gs.get("pure", 0.0))), 0.0)
    if post.get("rookie_r_shift"):
        # 이력 없는 투수(hist_pitcher_n == 0)의 R 행 점수 이동
        hn = bundle["history"]["pitcher_table"]["hist_pitcher_n"].reindex(batch["pitcher_id"].to_numpy()).fillna(0.0).to_numpy(dtype=float)
        is_r = batch["game_type"].astype(str).to_numpy() != "F"
        delta += np.where(is_r & (hn == 0), float(post["rookie_r_shift"]), 0.0)
    if post.get("fshare_r_shift"):
        # 이력 F(퓨처스) 비중이 높은 투수의 R 행 이동: {"thr": [0.5, 0.9], "values": [-1, -2]} → f_share > thr 중 가장 큰 thr 의 값
        fs = bundle["history"]["pitcher_table"]["hist_pitcher_f_share"].reindex(batch["pitcher_id"].to_numpy()).fillna(-1.0).to_numpy(dtype=float)
        is_r = batch["game_type"].astype(str).to_numpy() != "F"
        thr, vals = np.asarray(post["fshare_r_shift"]["thr"], float), np.asarray(post["fshare_r_shift"]["values"], float)
        idx = np.searchsorted(thr, fs, side="right") - 1
        delta += np.where(is_r & (idx >= 0), vals[np.clip(idx, 0, len(vals) - 1)], 0.0)
    if post.get("vdrop_shift"):
        # 투수의 고정 Trackman 이력 (릴리스 구속 − 존 도달 구속) z-score × k 점 이동. 표에 없는 투수(z 결측)는 0.
        z = bundle["history"]["pitcher_table"]["hist_pitcher_vdrop_z"].reindex(batch["pitcher_id"].to_numpy()).fillna(0.0).to_numpy(dtype=float)
        delta += float(post["vdrop_shift"]) * z
    if post.get("n_offset"):
        table = bundle["history"]["pitcher_table"]["hist_pitcher_n"]
        hn = table.reindex(batch["pitcher_id"].to_numpy()).fillna(0.0).to_numpy(dtype=float)
        edges, values = np.asarray(post["n_offset"]["edges"], float), np.asarray(post["n_offset"]["values"], float)
        delta += values[np.clip(np.searchsorted(edges, hn, side="right") - 1, 0, len(values) - 1)]
    # Failure 질량 m 을 Shadow(비율 s)와 Heart(1-s)로 옮기면 점수는 m·(100s + 30(1-s)) 만큼 변한다.
    s = float(post.get("shadow_frac", 1.0))
    m = delta / (30.0 + 70.0 * s)
    lower = -np.minimum(probs[:, 0] / s if s > 0 else np.inf, probs[:, 1] / (1 - s) if s < 1 else np.inf)
    m = np.clip(m, lower, probs[:, 2])
    probs[:, 0] += m * s
    probs[:, 1] += m * (1 - s)
    probs[:, 2] -= m
    if post.get("field_shift"):
        # 행 자신의 사전 정보 열(test.csv 에 있는 열)만 쓰는 선형 이동. 계수·중심은 학습 시점 고정 상수.
        # Failure 확률은 그대로 두고 Shadow<->Heart 질량만 옮긴다(점수 70 점당 질량 1) → Brier 비용이 S<->F 경로의 1/3~1/5.
        d2 = np.zeros(len(batch))
        for t in post["field_shift"]:
            name = t["col"]
            if "=" in name:  # 지시변수: "열=값" → (행의 그 열 == 값)
                cn, val = name.split("=", 1)
                col = batch[cn]
                try:
                    v = (col.to_numpy(dtype=float) == float(val)).astype(float)
                except (TypeError, ValueError):
                    v = (col.astype(str).to_numpy() == val).astype(float)
            else:
                v = batch[name].to_numpy(dtype=float)
            d2 += float(t["coef"]) * (v - float(t["center"]))
        if post.get("field_scale_hist"):
            # 투수의 고정 이력 표본수 구간별로 field-shift 세기를 달리한다(이력이 적을수록 모델 추정이 약함).
            hn = bundle["history"]["pitcher_table"]["hist_pitcher_n"].reindex(batch["pitcher_id"].to_numpy()).fillna(0.0).to_numpy(dtype=float)
            fe = np.asarray(post["field_scale_hist"]["edges"], float)
            fv = np.asarray(post["field_scale_hist"]["values"], float)
            d2 = d2 * fv[np.clip(np.searchsorted(fe, hn, side="right") - 1, 0, len(fv) - 1)]
        if post.get("field_r_only"):
            d2 = np.where(batch["game_type"].astype(str).to_numpy() == "F", 0.0, d2)
        if post.get("field_path") == "opt":
            # 점수 v=(100,30,0). sum(dp)=0, v·dp=d2 를 만족하는 최소노름 방향 dp = d2·(v−mean v)/((v−mean v)·v).
            # 같은 점수 이동에 Brier 비용이 Shadow<->Heart 경로의 절반 수준.
            vec = np.array([0.01075945, -0.00253163, -0.00822782])
            dp = np.outer(d2, vec)
            lim = np.ones(len(probs))
            for k in range(3):
                neg = dp[:, k] < 0
                if neg.any():
                    lim[neg] = np.minimum(lim[neg], (probs[neg, k] - 1e-9) / -dp[neg, k])
            probs += dp * np.clip(lim, 0.0, 1.0)[:, None]
        else:
            mv = np.clip(d2 / 70.0, -probs[:, 0] + 1e-9, probs[:, 1] - 1e-9)
            probs[:, 0] += mv
            probs[:, 1] -= mv
    return probs


def make_submission(bundle, test, sample, chunk_size=50000):
    """test 순서대로 행별 독립 예측. 이력은 학습 시점에 고정된 표를 조회만 한다."""
    if not test[ID_COL].reset_index(drop=True).equals(sample[ID_COL].reset_index(drop=True)):
        raise ValueError("test.csv와 sample_submission.csv의 row_id와 순서가 같아야 합니다.")
    mode = bundle.get("mode", "model")
    hist_only, offset = mode == "hist_only", mode == "offset"
    pipelines = [] if hist_only else (bundle.get("pipelines") or [bundle["pipeline"]])
    class_indices = None if (hist_only or offset) else [list(pipelines[0].classes_).index(label) for label in CLASSES]
    drop_prefixes = tuple(bundle.get("drop_prefixes", ()))
    rate_cols = [f"hist_pitcher_{g.lower()}_rate" for g in CLASSES]
    predictions = []
    for start in range(0, len(test), chunk_size):
        batch = test.iloc[start:start + chunk_size]
        features = build_features(batch, bundle["history"], bundle["drop_ids"])
        if hist_only:
            # 진단용: 학습 시점에 고정된 투수 이력 비율을 그대로 확률로 사용
            predictions.append(features.loc[:, rate_cols].to_numpy(dtype=float))
            continue
        if offset:
            # 이력 비율의 로그를 기준점으로, 상황 피처 모델의 raw score 를 더해 softmax
            init = np.log(np.clip(features.loc[:, rate_cols].to_numpy(dtype=float), 1e-6, 1))
            sit = features.loc[:, bundle["feature_columns"]]
            raw = np.mean([np.asarray(p.predict_proba(sit, raw_score=True), dtype=float) for p in pipelines], axis=0)
            z = raw + init
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            predictions.append(e / e.sum(axis=1, keepdims=True))
            continue
        if drop_prefixes:
            features = features.drop(columns=[c for c in features.columns if c.startswith(drop_prefixes)])
        full = features
        features = features.loc[:, bundle["feature_columns"]]
        probabilities = np.mean([np.asarray(p.predict_proba(features), dtype=float) for p in pipelines], axis=0)
        probabilities = probabilities[:, class_indices]
        r_model = bundle.get("r_model")
        if r_model:
            # 1군(R) 행: R 행만으로 학습한 LightGBM 확률을 w 로 혼합 (행 자신의 피처와 R 전용 고정 이력만 사용)
            rmask = batch["game_type"].astype(str).to_numpy() == "R"
            if rmask.any():
                fr = build_features(batch.loc[rmask], r_model["history"], r_model["drop_ids"]).loc[:, r_model["feature_columns"]]
                pr_ = np.asarray(r_model["pipeline"].predict_proba(fr), dtype=float)[:, r_model["class_indices"]]
                probabilities[rmask] = r_model["w"] * pr_ + (1 - r_model["w"]) * probabilities[rmask]
        rookie = bundle.get("rookie")
        if rookie:
            # 학습 이력이 없는 투수의 행: 신인 전용 모델 확률을 w 로 혼합 (행 자신의 피처와 고정 이력만 사용)
            mask = full["hist_pitcher_n"].to_numpy(dtype=float) == 0
            if mask.any():
                pr = np.asarray(rookie["pipeline"].predict_proba(full.loc[mask, rookie["feature_columns"]]), dtype=float)[:, rookie["class_indices"]]
                probabilities[mask] = rookie["w"] * pr + (1 - rookie["w"]) * probabilities[mask]
        f_sub = bundle.get("f_sub")
        if f_sub:
            # F(퓨처스) 행: F 행만으로 학습한 서브모델 확률을 w 로 혼합 (행 자신의 피처와 고정 이력만 사용)
            fmask = batch["game_type"].astype(str).to_numpy() == "F"
            if fmask.any():
                pf = np.asarray(f_sub["pipeline"].predict_proba(full.loc[fmask, f_sub["feature_columns"]]), dtype=float)[:, f_sub["class_indices"]]
                probabilities[fmask] = f_sub["w"] * pf + (1 - f_sub["w"]) * probabilities[fmask]
        hb = (bundle.get("post") or {}).get("hist_blend")
        if hb:
            # 이력이 충분한 투수(hist_pitcher_n >= edge)의 행: 모델 확률을 그 투수의 고정 이력 등급 비율과 혼합.
            # RF 의 투수 수준 성분(잡음 포함)과 직접 이력 추정의 앙상블. 행 자신의 투수 고정 표만 사용.
            hn = full["hist_pitcher_n"].to_numpy(dtype=float)
            edges, weights = np.asarray(hb["edges"], float), np.asarray(hb["weights"], float)
            w = np.where(hn >= edges[0], weights[np.clip(np.searchsorted(edges, hn, side="right") - 1, 0, len(weights) - 1)], 0.0)
            ph = full.loc[:, rate_cols].to_numpy(dtype=float).clip(0.0, 1.0)
            ph = ph / ph.sum(axis=1, keepdims=True)
            probabilities = (1.0 - w)[:, None] * probabilities + w[:, None] * ph
        predictions.append(apply_post(probabilities, batch, bundle))
    probabilities = np.concatenate(predictions, axis=0)
    if not np.isfinite(probabilities).all():
        raise ValueError("예측 확률에 결측값 또는 무한대가 있습니다.")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise ValueError("행별 예측 확률의 합이 1이 아닙니다.")
    submission = sample.copy()
    submission[list(PROB_COLS)] = probabilities
    return submission


def main():
    root = Path("./")
    test = pd.read_csv(root / "data/test.csv", dtype={ID_COL: str})
    sample = pd.read_csv(root / "data/sample_submission.csv", dtype={ID_COL: str})
    bundle = joblib.load(root / "model/model.pkl")
    submission = make_submission(bundle, test, sample)
    (root / "output").mkdir(exist_ok=True)
    submission.to_csv(root / "output/submission.csv", index=False)
    print(f"제출 파일 저장 완료: ./output/submission.csv ({len(submission):,}행)")


if __name__ == "__main__":
    main()
'''


def main():
    model_path = ROOT / "model" / "model.pkl"
    if not model_path.exists():
        sys.exit("model/model.pkl 이 없습니다. 먼저 bash run_train.sh 를 실행하세요.")
    submit = ROOT / "submit"
    if submit.exists():
        shutil.rmtree(submit)
    (submit / "model").mkdir(parents=True)
    shutil.copy(model_path, submit / "model" / "model.pkl")

    features_src = (ROOT / "src" / "features.py").read_text(encoding="utf-8")
    (submit / "script.py").write_text(features_src + INFERENCE_MAIN, encoding="utf-8")
    (submit / "requirements.txt").write_text("lightgbm==4.7.0\n", encoding="utf-8")

    zip_path = ROOT / "submit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submit / "script.py", "script.py")
        zf.write(submit / "requirements.txt", "requirements.txt")
        zf.write(submit / "model" / "model.pkl", "model/model.pkl")
    print(f"생성: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    for n in zipfile.ZipFile(zip_path).namelist():
        print("  ", n)


if __name__ == "__main__":
    main()
