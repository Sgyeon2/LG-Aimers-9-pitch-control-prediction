"""학습된 model/model.pkl 에 추론 후처리 상수(bundle["post"])를 넣는다. 재학습 없이 F 이동·이력 표본수 오프셋을 실험.

  python3 patch_post.py --src model/model_s22.pkl --out model/model.pkl --f-shift 3
  python3 patch_post.py --src model/model_s22.pkl --out model/model.pkl --n-offset "0:-1.5,1:-2.5,301:-2,1001:-0.7,3000:0"
n-offset 형식: "하한:오프셋,..." (hist_pitcher_n 이 하한 이상 다음 하한 미만이면 그 오프셋을 점수에 더함)
"""
import argparse, json
import joblib
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="model/model.pkl")
ap.add_argument("--out", default="model/model.pkl")
ap.add_argument("--f-shift", type=float, default=0.0)
ap.add_argument("--n-offset", default="")
ap.add_argument("--hist-blend", default="", help='이력 표본수 하한별 이력 혼합 비중, 예 "500:0.15,1000:0.3"')
ap.add_argument("--fshare-r-shift", default="", help='이력 F 비중 임계값별 R 행 이동, 예 "0.5:-1,0.9:-2" (f_share > 임계값)')
ap.add_argument("--rookie-r-shift", type=float, default=0.0, help="이력 없는 투수의 R 행 점수 이동")
ap.add_argument("--group-shift", default="", help='F 행 그룹별 이동, 예 "pure:-2,mixed:0" (pure = 이력에 1군 투구 없음)')
ap.add_argument("--shadow-frac", type=float, default=1.0, help="이동 질량 중 Shadow 로 가는 비율 (나머지는 Heart)")
ap.add_argument("--vdrop-shift", type=float, default=0.0, help="투수 Trackman 이력 (rel_speed − zone_speed) z-score 1 당 점수 이동 (예 -0.5)")
ap.add_argument("--field-shift", default="", help='행 필드 기반 점수 이동 "열:중심:계수,...", 예 "balls_before:0.9:-1,strikes_before:0.87:0.5" (Shadow<->Heart 경로, pFailure 고정)')
ap.add_argument("--field-path", default="sh", choices=("sh", "opt"), help="field-shift 질량 경로: sh=Shadow<->Heart(기존), opt=최소노름(같은 점수 이동에 Brier 비용 최소)")
ap.add_argument("--field-scale-hist", default="", help='이력 표본수 하한별 field-shift 배율, 예 "0:1.5,1:1.0"')
ap.add_argument("--field-r-only", action="store_true", help="field-shift 를 R 행에만 적용")
ap.add_argument("--rookie-w", type=float, default=None, help="번들의 신인 전용 모델 혼합 가중치를 덮어씀 (재학습 없이)")
ap.add_argument("--r-w", type=float, default=None, help="번들의 R 전용 모델 혼합 가중치를 덮어씀 (재학습 없이)")
a = ap.parse_args()
b = joblib.load(a.src)
post = {}
if a.shadow_frac != 1.0:
    post["shadow_frac"] = a.shadow_frac
if a.f_shift:
    post["f_shift"] = a.f_shift
if a.n_offset:
    pairs = sorted((float(k), float(v)) for k, v in (kv.split(":") for kv in a.n_offset.split(",")))
    post["n_offset"] = {"edges": [k for k, _ in pairs], "values": [v for _, v in pairs]}
if a.hist_blend:
    pairs = sorted((float(k), float(v)) for k, v in (kv.split(":") for kv in a.hist_blend.split(",")))
    post["hist_blend"] = {"edges": [k for k, _ in pairs], "weights": [v for _, v in pairs]}
if a.rookie_r_shift:
    post["rookie_r_shift"] = a.rookie_r_shift
if a.fshare_r_shift:
    pairs = sorted((float(k), float(v)) for k, v in (kv.split(":") for kv in a.fshare_r_shift.split(",")))
    post["fshare_r_shift"] = {"thr": [k for k, _ in pairs], "values": [v for _, v in pairs]}
    tbl = b["history"]["pitcher_table"]
    if "hist_pitcher_f_share" not in tbl.columns:
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent))
        import pandas as pd
        from src.features import CLASSES, load_train
        cache = _P(__file__).resolve().parent / "experiments" / "train_merged.pkl"
        try:
            d = pd.read_pickle(cache) if cache.exists() else load_train(_P(__file__).resolve().parent / "data")
        except Exception as e:   # 캐시가 다른 numpy 버전으로 저장된 경우 등
            print(f"캐시 로드 실패({e}) → 원본 데이터에서 재계산")
            d = load_train(_P(__file__).resolve().parent / "data")
        lab = d[d.control_grade.isin(CLASSES) & d.season.lt(b["history"]["ref_season"])]
        fs = lab.groupby("pitcher_id").game_type.apply(lambda x: float((x == "F").mean()))
        tbl["hist_pitcher_f_share"] = fs.reindex(tbl.index).fillna(-1.0)
        b["history"]["pitcher_fallback"]["hist_pitcher_f_share"] = -1.0
        print(f"hist_pitcher_f_share 추가: >0.9 {int((tbl.hist_pitcher_f_share > 0.9).sum())}명, 0.5~0.9 {int(((tbl.hist_pitcher_f_share > 0.5) & (tbl.hist_pitcher_f_share <= 0.9)).sum())}명")
if a.group_shift:
    post["group_shift"] = {k: float(v) for k, v in (kv.split(":") for kv in a.group_shift.split(","))}
    tbl = b["history"]["pitcher_table"]
    if "hist_pitcher_r_n" not in tbl.columns:
        # 학습 데이터(2019~2024 라벨 행)에서 투수별 1군 투구 수를 계산해 고정 표에 추가. RF 입력(feature_columns)에는 영향 없음.
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent))
        import pandas as pd
        from src.features import CLASSES, load_train
        cache = _P(__file__).resolve().parent / "experiments" / "train_merged.pkl"
        try:
            d = pd.read_pickle(cache) if cache.exists() else load_train(_P(__file__).resolve().parent / "data")
        except Exception as e:   # 캐시가 다른 numpy 버전으로 저장된 경우 등
            print(f"캐시 로드 실패({e}) → 원본 데이터에서 재계산")
            d = load_train(_P(__file__).resolve().parent / "data")
        lab = d[d.control_grade.isin(CLASSES) & d.season.lt(b["history"]["ref_season"])]
        rn = lab[lab.game_type.eq("R")].groupby("pitcher_id").size().astype(float)
        tbl["hist_pitcher_r_n"] = rn.reindex(tbl.index).fillna(0.0)
        b["history"]["pitcher_fallback"]["hist_pitcher_r_n"] = 0.0
        print(f"hist_pitcher_r_n 추가: 투수 {len(tbl)}명 중 1군 이력 없음 {int((tbl.hist_pitcher_r_n == 0).sum())}명")
if a.vdrop_shift:
    post["vdrop_shift"] = a.vdrop_shift
    tbl = b["history"]["pitcher_table"]
    if "hist_pitcher_vdrop_z" not in tbl.columns:
        # 학습 시점 고정 표: train_trackman 의 P_ 투수별 (rel_speed − zone_speed) 평균(측정 50구 이상), 투수 간 z-score(±3 클립).
        import pandas as pd
        from pathlib import Path as _P
        tmf = pd.read_csv(_P(__file__).resolve().parent / "data" / "train_trackman.csv", usecols=["season", "pitcher_id", "rel_speed", "zone_speed"], dtype={"pitcher_id": str})
        tmf = tmf[tmf.pitcher_id.str.startswith("P_", na=False) & tmf.season.lt(b["history"]["ref_season"]) & tmf.rel_speed.notna() & tmf.zone_speed.notna()]
        vd = (tmf.rel_speed - tmf.zone_speed).groupby(tmf.pitcher_id).agg(["mean", "size"])
        vd = vd.loc[vd["size"] >= 50, "mean"]
        z = ((vd - vd.mean()) / vd.std()).clip(-3, 3)
        tbl["hist_pitcher_vdrop_z"] = z.reindex(tbl.index).fillna(0.0)
        b["history"]["pitcher_fallback"]["hist_pitcher_vdrop_z"] = 0.0
        print(f"hist_pitcher_vdrop_z 추가: z 보유 투수 {int((tbl.hist_pitcher_vdrop_z != 0).sum())}/{len(tbl)}, 평균 vdrop {vd.mean():.2f} km/h sd {vd.std():.2f}")
if a.field_shift:
    # 행 자신의 사전 정보 열만 사용하는 선형 이동. 계수·중심은 학습 시점 상수(test 집계 아님).
    terms = []
    for kv in a.field_shift.split(","):
        col, center, coef = kv.split(":")
        terms.append({"col": col.strip(), "center": float(center), "coef": float(coef)})
    post["field_shift"] = terms
    post["field_path"] = a.field_path
    if a.field_scale_hist:
        pr = sorted((float(k), float(v)) for k, v in (kv.split(":") for kv in a.field_scale_hist.split(",")))
        post["field_scale_hist"] = {"edges": [k for k, _ in pr], "values": [v for _, v in pr]}
    if a.field_r_only:
        post["field_r_only"] = True
    print("field_shift =", terms, "path =", a.field_path)
if a.rookie_w is not None and b.get("rookie"):
    b["rookie"]["w"] = float(a.rookie_w)
    post["rookie_w"] = float(a.rookie_w)
    print(f"신인 모델 가중치 -> {a.rookie_w}")
if a.r_w is not None and b.get("r_model"):
    b["r_model"]["w"] = float(a.r_w)
    post["r_w"] = float(a.r_w)
    print(f"R 전용 모델 가중치 -> {a.r_w}")
b["post"] = post
b["params"]["post"] = post
joblib.dump(b, a.out, compress=3)
print("post =", json.dumps(post), "->", a.out)
