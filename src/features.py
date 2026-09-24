"""학습·추론 공용 피처 모듈.

원칙: 선수 이력(history)은 학습 시점에 만들어 고정하고, 추론 시에는 조회만 한다.
test 데이터로는 어떤 집계도 만들지 않는다 (대회 규칙).

baseline 대비 변경점
- make_history: 시즌 거리 기반 지수 감쇠 가중치(decay), 스무딩 강도(alpha) 파라미터화
- extra=True 이면 투수/타자 이력 점수(100·Shadow + 30·Heart), 유효 표본 수, 마지막 시즌까지의 거리,
  직전 시즌 단독 점수·표본 수를 추가한다.
- build_features(drop_ids=True) 이면 pitcher_id/batter_id 원본 ID를 입력에서 제외한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("Shadow", "Heart", "Failure")
GRADE_SCORE = np.array([100.0, 30.0, 0.0])
BASE_COLUMNS29 = (
    "row_id", "season", "game_month", "game_dayofweek", "inning",
    "top_bottom", "game_type", "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
    "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "num_runners_on", "base_state", "home_win_expectancy", "away_win_expectancy",
    "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id",
)
CAT_COLS = (
    "top_bottom", "game_type", "base_state", "pitcher_id", "batter_id",
    "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
)
ID_COLS = ("pitcher_id", "batter_id")
TRACKMAN_COLS = [
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side",
]
PLATE_COLS = ["plate_loc_height", "plate_loc_side"]      # train_outcomes 의 도달 위치 (사후 정보, 이력 집계 전용)
SPREAD_COLS = ["rel_height", "rel_side", "rel_speed", *PLATE_COLS]   # 투수별 표준편차(일관성) 이력
FLAG_COLS = ["BALL_YN", "REVERSE_YN"]   # 원천 판정 플래그: 등급이 없는(Unlabeled) 행에도 있음 → 이력 전용
HEART_BOX = (0.263, 0.537, 0.963)    # 라벨 정의(운영진 공지): |side| ≤ 0.263, 0.537 ≤ height ≤ 0.963
SHADOW_BOX = (0.373, 0.427, 1.073)   # |side| ≤ 0.373, 0.427 ≤ height ≤ 1.073. 그 밖 또는 REVERSE_YN=T → Failure


def soft_grade_probs(df: pd.DataFrame, sigma: float) -> np.ndarray:
    """도달 좌표에 sd=sigma(m) 등방 가우시안을 씌운 등급 확률 [Shadow, Heart, Failure] (이력 집계 전용).
    경계 1cm 안팎이 100/0 으로 갈리는 하드 라벨의 표본 잡음을 줄인 투수 위치 성향 추정. REVERSE_YN=T 는 Failure 확정."""
    from scipy.stats import norm
    s = df["plate_loc_side"].to_numpy(dtype=float)
    h = df["plate_loc_height"].to_numpy(dtype=float)

    def pbox(hs, lo, hi):
        return (norm.cdf((hs - s) / sigma) - norm.cdf((-hs - s) / sigma)) * (norm.cdf((hi - h) / sigma) - norm.cdf((lo - h) / sigma))

    p_heart, p_shadow_box = pbox(*HEART_BOX), pbox(*SHADOW_BOX)
    p = np.column_stack([p_shadow_box - p_heart, p_heart, 1.0 - p_shadow_box])
    p[df["REVERSE_YN"].to_numpy() == "T"] = [0.0, 0.0, 1.0]
    return p


def load_train(data_dir: Path) -> pd.DataFrame:
    """baseline과 동일하게 train + control_grade + Trackman 물리량을 row_id로 연결."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"row_id": str})
    outcomes = pd.read_csv(
        data_dir / "train_outcomes.csv", usecols=["row_id", "control_grade", *PLATE_COLS, *FLAG_COLS],
        dtype={"row_id": str},
    )
    trackman = pd.read_csv(
        data_dir / "train_trackman.csv", usecols=["row_id", *TRACKMAN_COLS],
        dtype={"row_id": str},
    ).dropna(subset=["row_id"])
    data = train.merge(outcomes, on="row_id", how="left", validate="one_to_one")
    data = data.merge(trackman, on="row_id", how="left", validate="one_to_one")
    data[TRACKMAN_COLS + PLATE_COLS] = data[TRACKMAN_COLS + PLATE_COLS].replace([np.inf, -np.inf], np.nan)
    return data


def _season_weights(seasons: pd.Series, ref_season: int, decay: float) -> np.ndarray:
    """ref_season 직전 시즌은 1, 그 이전은 decay, decay^2, ... (decay=1이면 baseline과 동일)."""
    gap = ref_season - seasons.to_numpy(dtype=float)
    return np.power(float(decay), gap - 1.0)


def _role_rates(onehot, w, ids, alpha, prior, excess=None):
    """선수별 가중 등급 카운트 → 스무딩된 비율.

    excess 가 주어지면(era 보정) 시즌 리그 평균 대비 편차를 0으로 수축시킨 뒤 prior 에 더한다:
        rate = prior + Σ(w·excess) / (Σw + alpha)
    """
    wonehot = pd.DataFrame(onehot * w[:, None], columns=CLASSES)
    wc = wonehot.groupby(ids).sum()
    n_eff = wc.sum(axis=1)
    if excess is None:
        rates = (wc + alpha * prior).div(n_eff + alpha, axis=0)
    else:
        wex = pd.DataFrame(excess * w[:, None], columns=CLASSES).groupby(ids).sum()
        rates = wex.div(n_eff + alpha, axis=0) + prior
    return n_eff, rates


def make_history(past: pd.DataFrame, ref_season: int, decay: float = 1.0,
                 alpha: float = 100.0, extra: bool = True, decay_mid: float = 0.5,
                 era_adjust: bool = False, tm_standardize: bool = False,
                 era_by_gt: bool = False, spread: bool = False, f_share: bool = False,
                 ball_hist: bool = False, rev_split: bool = False, alpha_rev: float = 50.0,
                 alpha_zone: float = 400.0, soft_sigma: float = 0.0) -> dict:
    """ref_season 이전 시즌(past)만으로 선수별 이력 표를 만든다.

    기본 비율은 decay 가중(기본 1.0 = baseline과 동일). extra=True 이면
    장기 점수, 중기(decay_mid) 점수, 직전 시즌 단독 점수를 함께 만들어
    모델이 horizon 별 가중치를 학습하게 한다.
    """
    labeled = past.loc[past.control_grade.isin(CLASSES)]
    w = _season_weights(labeled.season, ref_season, decay)
    onehot = (labeled.control_grade.to_numpy()[:, None] == np.asarray(CLASSES)).astype(float)

    def _prior_excess(mat):
        """등급 행렬(원핫 또는 소프트 확률)의 prior 와 era 보정 편차."""
        total = (mat * w[:, None]).sum(axis=0)
        pr = total / total.sum() if total.sum() > 0 else np.full(3, 1 / 3)
        ex = None
        if era_adjust and len(labeled):
            # 시즌별 리그 평균 대비 편차. prior 는 가장 최근 시즌의 리그 분포로 둔다.
            if era_by_gt:
                # 시즌×경기유형(R/F)별 평균 대비 편차: 퓨처스 라벨 수준 변화(2023~)까지 제거. prior 는 최근 시즌 R 분포.
                keys = pd.MultiIndex.from_arrays([labeled.season.to_numpy(), labeled.game_type.to_numpy()])
                league = pd.DataFrame(mat, columns=CLASSES, index=keys).groupby(level=[0, 1]).mean()
                ex = mat - league.reindex(keys).to_numpy()
                last = league.index.get_level_values(0).max()
                pr = (league.loc[(last, "R")] if (last, "R") in league.index else league.loc[last].mean()).to_numpy()
            else:
                league = pd.DataFrame(mat, columns=CLASSES).groupby(labeled.season.to_numpy()).mean()
                ex = mat - league.reindex(labeled.season.to_numpy()).to_numpy()
                pr = league.loc[league.index.max()].to_numpy()
        return pr, ex

    prior, excess = _prior_excess(onehot)
    prior_score = float(prior @ GRADE_SCORE)
    soft = None
    if soft_sigma > 0 and all(c in labeled.columns for c in PLATE_COLS + ["REVERSE_YN"]):
        # 소프트 라벨 이력(투수만): 하드 등급 대신 좌표 커널 확률로 비율·점수를 집계 (같은 era 보정, 같은 alpha).
        soft = soft_grade_probs(labeled, soft_sigma)
        miss = ~np.isfinite(soft).all(axis=1)
        soft[miss] = onehot[miss]
        prior_s, excess_s = _prior_excess(soft)

    history = {"ref_season": ref_season, "decay": decay, "alpha": alpha, "prior": prior,
               "era_adjust": era_adjust, "tm_standardize": tm_standardize, "soft_sigma": soft_sigma}
    for role in ("pitcher", "batter"):
        ids = labeled[f"{role}_id"].to_numpy()
        if soft is not None and role == "pitcher":
            mat, pr_, ex_ = soft, prior_s, excess_s
        else:
            mat, pr_, ex_ = onehot, prior, excess
        ps_ = float(pr_ @ GRADE_SCORE)
        n_eff, rates = _role_rates(mat, w, ids, alpha, pr_, ex_)
        n_raw = pd.Series(1.0).repeat(len(ids)).groupby(ids).sum().reindex(rates.index)

        table = pd.DataFrame(index=rates.index)
        table[f"hist_{role}_n"] = n_raw
        fallback = {f"hist_{role}_n": 0.0}
        for i, grade in enumerate(CLASSES):
            col = f"hist_{role}_{grade.lower()}_rate"
            table[col] = rates[grade]
            fallback[col] = float(pr_[i])

        if extra:
            table[f"hist_{role}_neff"] = n_eff
            fallback[f"hist_{role}_neff"] = 0.0
            table[f"hist_{role}_score"] = rates.to_numpy() @ GRADE_SCORE
            fallback[f"hist_{role}_score"] = ps_
            # 중기 horizon
            w_mid = _season_weights(labeled.season, ref_season, decay_mid)
            n_mid, r_mid = _role_rates(mat, w_mid, ids, alpha, pr_, ex_)
            table[f"hist_{role}_neff_mid"] = n_mid.reindex(rates.index)
            table[f"hist_{role}_score_mid"] = pd.Series(
                r_mid.to_numpy() @ GRADE_SCORE, index=r_mid.index).reindex(rates.index)
            fallback[f"hist_{role}_neff_mid"] = 0.0
            fallback[f"hist_{role}_score_mid"] = ps_
            # 직전 기록 시즌까지의 거리와 그 시즌 단독 점수
            last = labeled.season.groupby(ids).max()
            table[f"hist_{role}_gap"] = (ref_season - last.reindex(rates.index)).astype(float)
            fallback[f"hist_{role}_gap"] = 10.0
            is_last = labeled.season.to_numpy() == last.reindex(ids).to_numpy()
            n_last, r_last = _role_rates(mat[is_last], np.ones(int(is_last.sum())),
                                         ids[is_last], alpha, pr_,
                                         None if ex_ is None else ex_[is_last])
            table[f"hist_{role}_last_n"] = n_last.reindex(rates.index)
            table[f"hist_{role}_last_score"] = pd.Series(
                r_last.to_numpy() @ GRADE_SCORE, index=r_last.index).reindex(rates.index)
            fallback[f"hist_{role}_last_n"] = 0.0
            fallback[f"hist_{role}_last_score"] = ps_

        if f_share and role == "pitcher":
            # 이력 중 퓨처스(F) 투구 비중과 1군(R) 투구 수: F 위주 투수는 정규화된 이력보다 1군에서 약 3점 나쁨(구성 효과)
            is_f = (labeled.game_type.to_numpy() == "F").astype(float)
            n_f = pd.Series(is_f * w).groupby(ids).sum().reindex(rates.index)
            n_all = pd.Series(w).groupby(ids).sum().reindex(rates.index)
            table["hist_pitcher_f_share"] = n_f / n_all
            table["hist_pitcher_r_n"] = n_all - n_f
            fallback["hist_pitcher_f_share"] = -1.0   # 이력 없음 표시
            fallback["hist_pitcher_r_n"] = 0.0
        history[f"{role}_table"] = table.astype(float)
        history[f"{role}_fallback"] = fallback

    # 투수 Trackman 가중 평균
    measured = past.loc[past[TRACKMAN_COLS].notna().any(axis=1)]
    wm = _season_weights(measured.season, ref_season, decay)
    pid = measured.pitcher_id.to_numpy()
    x = measured[TRACKMAN_COLS]
    if tm_standardize:
        # 시즌별 측정 드리프트(장비 보정 변화) 제거: 시즌 내 z-score 후 평균
        grp = x.groupby(measured.season.to_numpy())
        x = (x - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    num = x.mul(wm, axis=0).groupby(pid).sum(min_count=1)
    den = x.notna().astype(float).mul(wm, axis=0).groupby(pid).sum()
    physics = (num / den).add_prefix("hist_tm_").add_suffix("_mean")
    physics.insert(0, "hist_tm_n", pd.Series(1.0, index=measured.index).groupby(pid).sum())
    history["pitcher_fallback"]["hist_tm_n"] = 0.0
    history["pitcher_table"] = history["pitcher_table"].join(physics, how="outer")
    if spread:
        # 투수별 릴리스 포인트·구속·도달 위치의 가중 표준편차 (제구 일관성). 시즌 내 표준화된 값 기준.
        cols = [c for c in SPREAD_COLS if c in past.columns]
        meas = past.loc[past[cols].notna().any(axis=1)]
        ws = _season_weights(meas.season, ref_season, decay)
        xs = meas[cols]
        grp = xs.groupby(meas.season.to_numpy())
        xs = (xs - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        pid_s = meas.pitcher_id.to_numpy()
        w_ok = xs.notna().astype(float).mul(ws, axis=0)
        den_s = w_ok.groupby(pid_s).sum()
        m1 = xs.mul(ws, axis=0).groupby(pid_s).sum(min_count=1) / den_s
        m2 = (xs ** 2).mul(ws, axis=0).groupby(pid_s).sum(min_count=1) / den_s
        sd = np.sqrt((m2 - m1 ** 2).clip(lower=0)).where(den_s >= 20)
        sd = sd.add_prefix("hist_sd_")
        history["pitcher_table"] = history["pitcher_table"].join(sd, how="outer")
    if ball_hist and all(c in past.columns for c in FLAG_COLS):
        # 원천 볼 판정·반대투구 비율: 등급 없는 행까지 써서 표본을 늘린 제구 지표 (시즌×경기유형 평균 대비 편차로 스무딩)
        flagged = past.loc[past[FLAG_COLS].notna().all(axis=1)]
        wb = _season_weights(flagged.season, ref_season, decay)
        yb = np.column_stack([(flagged[c].to_numpy() == "T").astype(float) for c in FLAG_COLS])
        pid_b = flagged.pitcher_id.to_numpy()
        keys = pd.MultiIndex.from_arrays([flagged.season.to_numpy(), flagged.game_type.to_numpy()])
        league_b = pd.DataFrame(yb, index=keys).groupby(level=[0, 1]).mean()
        last_b = league_b.index.get_level_values(0).max()
        prior_b = (league_b.loc[(last_b, "R")] if (last_b, "R") in league_b.index else league_b.loc[last_b].mean()).to_numpy()
        excess_b = yb - league_b.reindex(keys).to_numpy()
        n_b = pd.Series(wb).groupby(pid_b).sum()
        num_b = pd.DataFrame(excess_b * wb[:, None]).groupby(pid_b).sum()
        rates_b = num_b.div(n_b + alpha, axis=0) + prior_b
        tb = pd.DataFrame({"hist_pitcher_ball_rate": rates_b[0], "hist_pitcher_rev_rate": rates_b[1], "hist_pitcher_n_all": n_b})
        history["pitcher_fallback"].update({"hist_pitcher_ball_rate": float(prior_b[0]), "hist_pitcher_rev_rate": float(prior_b[1]),
                                            "hist_pitcher_n_all": 0.0})
        history["pitcher_table"] = history["pitcher_table"].join(tb, how="outer")
    if rev_split and "REVERSE_YN" in past.columns:
        # 라벨 정의(운영진 공지): Failure = 반대투구(REVERSE_YN=T) ∪ 존 밖. 두 성분은 투수별 시즌 간 지속성이 크게 달라
        # (반대투구 비율 tau≈0.45, 존 점수|비반대 ≈0.2) 각각 다른 강도(alpha_rev, alpha_zone)로 수축한 뒤 곱한 이력 점수를 둔다.
        # 기존 이력과 같은 방식으로 시즌(×경기유형) 평균 대비 편차를 쓰고 prior 도 같은 기준(최근 시즌 R)으로 맞춘다.
        rev = (labeled.REVERSE_YN.to_numpy() == "T").astype(float)
        nr = 1.0 - rev
        row_score = onehot @ GRADE_SCORE
        if era_adjust and era_by_gt:
            keys = pd.MultiIndex.from_arrays([labeled.season.to_numpy(), labeled.game_type.to_numpy()])
        elif era_adjust:
            keys = pd.Index(labeled.season.to_numpy())
        else:
            keys = pd.Index(np.zeros(len(labeled), dtype=int))
        lg_src = pd.DataFrame({"rev": rev, "zs_w": row_score * nr, "nr": nr, "cnt": 1.0}, index=keys)
        lg = lg_src.groupby(level=list(range(keys.nlevels))).sum()
        lg["rev"] = lg["rev"] / lg["cnt"]
        lg["zs"] = lg["zs_w"] / lg["nr"].replace(0, np.nan)
        if era_adjust and era_by_gt:
            last_s = lg.index.get_level_values(0).max()
            pr = lg.loc[(last_s, "R")] if (last_s, "R") in lg.index else lg.loc[last_s].mean()
        else:
            pr = lg.loc[lg.index.max()]
        prior_rev, prior_zs = float(pr["rev"]), float(pr["zs"])
        lk = lg.reindex(keys)
        ex_rev = rev - lk["rev"].to_numpy()
        ex_zs = (row_score - lk["zs"].to_numpy()) * nr
        pid_r = labeled.pitcher_id.to_numpy()
        n_w = pd.Series(w).groupby(pid_r).sum()
        nr_w = pd.Series(w * nr).groupby(pid_r).sum()
        revr = prior_rev + pd.Series(w * ex_rev).groupby(pid_r).sum() / (n_w + alpha_rev)
        zone = prior_zs + pd.Series(w * ex_zs).groupby(pid_r).sum() / (nr_w + alpha_zone)
        tb = pd.DataFrame({"hist_pitcher_revr": revr, "hist_pitcher_zone_nr": zone,
                           "hist_pitcher_comp_score": zone * (1.0 - revr)})
        history["pitcher_fallback"].update({"hist_pitcher_revr": prior_rev, "hist_pitcher_zone_nr": prior_zs,
                                            "hist_pitcher_comp_score": prior_zs * (1.0 - prior_rev)})
        history["pitcher_table"] = history["pitcher_table"].join(tb, how="outer")
    history["pitcher_table"] = history["pitcher_table"].fillna(history["pitcher_fallback"])
    return history


def build_features(df: pd.DataFrame, history: dict, drop_ids: bool = False) -> pd.DataFrame:
    """학습·추론 공용. 현재 행 + 고정 이력만 사용하며 다른 행을 참조하지 않는다."""
    x = df.loc[:, BASE_COLUMNS29[1:]].copy()
    x["count_balance"] = df["balls_before"] - df["strikes_before"]
    x["is_two_strikes"] = (df["strikes_before"] == 2).astype(int)
    x["runners_in_scoring_position"] = df["runner_on_2b"] + df["runner_on_3b"]
    x["abs_score_diff"] = df["score_diff_pitcher_team"].abs()
    pitcher_hand = pd.to_numeric(df["pitcher_hand"], errors="coerce")
    batter_hand = pd.to_numeric(df["batter_hand"], errors="coerce")
    known = pitcher_hand.notna() & batter_hand.notna()
    known &= pitcher_hand.ne(0) & batter_hand.ne(0)
    x["same_hand"] = np.where(known, pitcher_hand.eq(batter_hand).astype(float), np.nan)

    for role in ("pitcher", "batter"):
        table = history[f"{role}_table"]
        joined = table.reindex(df[f"{role}_id"].to_numpy()).copy()
        joined.index = df.index
        joined = joined.fillna(history[f"{role}_fallback"]).astype(float)
        x = pd.concat([x, joined], axis=1)

    if drop_ids:
        x = x.drop(columns=list(ID_COLS))
    for col in ("pitcher_hand", "batter_hand"):
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("Int64")
    for col in cat_cols(drop_ids):
        x[col] = x[col].astype("object").where(x[col].notna(), "__MISSING__")
        x[col] = x[col].astype(str).astype("object")
    return x


def cat_cols(drop_ids: bool = False) -> list[str]:
    return [c for c in CAT_COLS if not (drop_ids and c in ID_COLS)]


def evaluate(probs: np.ndarray, y, pitcher_ids, min_pitches: int = 50) -> dict:
    """대회 산식. probs 열 순서는 CLASSES(Shadow, Heart, Failure)."""
    from scipy.stats import kendalltau

    one_hot = (np.asarray(y)[:, None] == np.asarray(CLASSES)).astype(float)
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    pitch = 100 * (1 - brier / 2)
    players = pd.DataFrame({
        "pid": np.asarray(pitcher_ids),
        "actual": one_hot @ GRADE_SCORE,
        "pred": probs @ GRADE_SCORE,
    }).groupby("pid").agg(n=("actual", "size"), actual=("actual", "mean"), pred=("pred", "mean"))
    elig = players.loc[players.n.ge(min_pitches)]
    actual, pred = elig.actual.round(10), elig.pred.round(10)
    tau = 0.0 if pred.nunique() < 2 else float(
        kendalltau(actual, pred, variant="b").statistic)
    player = 50 * (1 + tau)
    return {"total": 0.7 * pitch + 0.3 * player, "pitch": pitch, "player": player,
            "tau": tau, "brier": brier, "n_pitchers": int(len(elig))}
