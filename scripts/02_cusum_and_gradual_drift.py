"""
Track A - 실험 2: CUSUM을 오프라인 비교에 포함 + 점진적(gradual) Drift 합성 공격
================================================================================
(원본 docstring 동일 — 생략 없이 유지)
"""

import gzip
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

POSITION_RESIDUAL_THRESH = 4.0  # 표 1과 동일

# ★ CUSUM/Recall 계산의 시퀀스 단위. receiver+sender+attack_type 셋 다 있어야
#   서로 다른 receiver, 서로 다른 시나리오(variant)가 하나의 스트림으로 섞이지 않음.
GROUP_KEYS = ["receiver_id", "sender_id", "attack_type"]


# ----------------------------------------------------------------------
# 데이터 로드 (receiver_id 추가됨)
# ----------------------------------------------------------------------
def load_test_features(split: str = "test") -> pd.DataFrame:
    """
    필요 컬럼: receiver_id, sender_id, t, position_residual, rule_score,
              ml_prob, attack_type, y_true(0/1)
    """
    if split in ["val", "validation"]:
        data_path = Path("data/processed/veremi_nextgen/validation_features.pkl.gz")
    else:
        data_path = Path("data/processed/veremi_nextgen/test_features.pkl.gz")

    model_path = Path("models/veremi_nextgen_rf.joblib")
    meta_path = Path("models/veremi_nextgen_rf.metadata.json")

    with gzip.open(data_path, "rb") as f:
        df = pd.read_pickle(f)
    df = df.sort_values(["receiver_id", "sender_id", "timestamp"]).reset_index(drop=True)

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    _ = metadata["feature_names"]  # 필요시 사용

    r_pos = (df["position_residual"].abs() / 4.0).to_numpy()
    r_spd = (df["speed"] / 55.0).to_numpy()
    r_acc = (df["abs_acceleration"] / 12.0).to_numpy()
    r_dacc = (df["derived_acceleration"].abs() / 12.0).to_numpy()
    r_ares = (df["acceleration_residual"].abs() / 6.0).to_numpy()
    rule_score = np.maximum.reduce([r_pos, r_spd, r_acc, r_dacc, r_ares])

    bundle = joblib.load(model_path)
    ml_prob = bundle.predict_anomaly_probability(df)

    df_out = pd.DataFrame({
        "receiver_id": df["receiver_id"].to_numpy(),
        "sender_id": df["sender_id"].to_numpy(),
        "t": df["timestamp"].to_numpy(),
        "position_residual": df["position_residual"].to_numpy(),
        "rule_score": rule_score,
        "ml_prob": ml_prob,
        "attack_type": df["attack_class"].to_numpy(dtype=str),
        "y_true": df["label"].to_numpy(dtype=int),
    })
    return df_out


def load_raw_benign_stream_for_synthesis() -> pd.DataFrame:
    """필요 컬럼: sender_id, receiver_id, t, pos_x, pos_y, speed (원본, feature 추출 이전)"""
    raise NotImplementedError("load_raw_benign_stream_for_synthesis()를 채워주세요.")


def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """기존 feature engineering 파이프라인 연결. 18개 feature 반환."""
    raise NotImplementedError("extract_features()를 기존 파이프라인과 연결하세요.")


def rule_score_5signals(feature_row) -> float:
    r_pos = abs(feature_row["position_residual"]) / 4.0
    r_spd = feature_row["speed"] / 55.0
    r_acc = feature_row["abs_acceleration"] / 12.0
    r_dacc = abs(feature_row["derived_acceleration"]) / 12.0
    r_ares = abs(feature_row["acceleration_residual"]) / 6.0
    return float(max(r_pos, r_spd, r_acc, r_dacc, r_ares))


def ml_predict_proba(features_df: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(Path("models/veremi_nextgen_rf.joblib"))
    return bundle.predict_anomaly_probability(features_df)


# ----------------------------------------------------------------------
# CUSUM 구현 (원본과 동일)
# ----------------------------------------------------------------------
def cusum_scores(residuals: np.ndarray, decay: float, drift_ref: float) -> np.ndarray:
    s = 0.0
    out = np.zeros(len(residuals))
    for i, r in enumerate(residuals):
        s = max(0.0, s * decay + (abs(r) - drift_ref))
        out[i] = s
    return out


def compute_cusum_column(df: pd.DataFrame, decay: float, drift_ref: float) -> pd.DataFrame:
    """GROUP_KEYS(receiver+sender+attack_type) 단위로 CUSUM을 독립적으로 누적."""
    df_sorted = df.sort_values(GROUP_KEYS + ["t"]).copy()

    def _apply(s):
        return pd.Series(cusum_scores(s.values, decay, drift_ref), index=s.index)

    cusum_series = (
        df_sorted.groupby(GROUP_KEYS, group_keys=False)["position_residual"]
        .apply(_apply)
    )
    df_sorted["cusum"] = cusum_series.reindex(df_sorted.index)
    return df_sorted


def grid_search_cusum_params(df_val, decay_grid, drift_ref_grid, cusum_thresh_grid):
    best = {"f1": -1}
    for decay in decay_grid:
        for drift_ref in drift_ref_grid:
            scored = compute_cusum_column(df_val, decay, drift_ref)
            for thr in cusum_thresh_grid:
                combined = np.maximum(
                    scored["rule_score"].values,
                    (scored["cusum"].values >= thr).astype(float),
                )
                pred = (combined >= 1.0).astype(int)
                tp = np.sum((scored["y_true"] == 1) & (pred == 1))
                fp = np.sum((scored["y_true"] == 0) & (pred == 1))
                fn = np.sum((scored["y_true"] == 1) & (pred == 0))
                precision = tp / (tp + fp) if (tp + fp) else 0
                recall = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                if f1 > best["f1"]:
                    best = {"f1": f1, "decay": decay, "drift_ref": drift_ref, "cusum_thresh": thr}
    return best


def _prf(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    return precision, recall, f1, fpr


def evaluate_rule_with_cusum(df_test, decay, drift_ref, cusum_thresh):
    """[A] Rule-only vs Rule+CUSUM: 전체 P/R/F1/FPR + 공격유형별 Recall (배경수준 자동판정 포함)."""
    scored = compute_cusum_column(df_test, decay, drift_ref)
    scored["rule_plus_cusum_score"] = np.maximum(
        scored["rule_score"], (scored["cusum"] >= cusum_thresh).astype(float)
    )
    scored["pred_rule_only"] = (scored["rule_score"] >= 1.0).astype(int)
    scored["pred_rule_cusum"] = (scored["rule_plus_cusum_score"] >= 1.0).astype(int)

    y_true = scored["y_true"].values
    p1, r1, f1_1, fpr1 = _prf(y_true, scored["pred_rule_only"].values)
    p2, r2, f2, fpr2 = _prf(y_true, scored["pred_rule_cusum"].values)

    print("\n=== 전체 성능: Rule-only vs Rule+CUSUM ===")
    print(f"  Rule-only  : P={p1:.4f} R={r1:.4f} F1={f1_1:.4f} FPR={fpr1:.4f}")
    print(f"  Rule+CUSUM : P={p2:.4f} R={r2:.4f} F1={f2:.4f} FPR={fpr2:.4f}")

    print("\n=== 공격 유형별 Recall: Rule-only vs Rule+CUSUM ===")
    for atk in [a for a in scored["attack_type"].unique() if a != "benign"]:
        mask = scored["attack_type"].isin([atk, "benign"])
        sub = scored[mask]
        rec_rule = sub.loc[sub["y_true"] == 1, "pred_rule_only"].mean()
        rec_cusum = sub.loc[sub["y_true"] == 1, "pred_rule_cusum"].mean()
        diff = rec_cusum - fpr2
        verdict = "⚠️ 배경수준" if diff < 0.05 else "✅ 실질탐지"
        print(f"  {atk:24s} Rule-only={rec_rule:.3f}  Rule+CUSUM={rec_cusum:.3f}  "
              f"(개선폭={rec_cusum - rec_rule:+.3f})  전체FPR대비차이={diff:+.3f} {verdict}")

    return scored


# ----------------------------------------------------------------------
# [B] 점진적 drift 합성 공격 생성 및 평가 (원본과 동일, 아직 미구현)
# ----------------------------------------------------------------------
def inject_gradual_drift(df_raw: pd.DataFrame, rate_m_per_s: float, onset_t: float = 5.0) -> pd.DataFrame:
    df = df_raw.copy()
    elapsed = (df["t"] - onset_t).clip(lower=0)
    offset = rate_m_per_s * elapsed
    df["pos_x"] = df["pos_x"] + offset
    df["is_attack"] = (df["t"] >= onset_t).astype(int)
    df["attack_onset_t"] = onset_t
    return df


def run_gradual_drift_experiment(rate_grid_m_per_s, decay, drift_ref, cusum_thresh, ml_thresh):
    print("\n=== [B] 점진적 Drift 속도별 Recall: Rule-only vs Rule+CUSUM vs ML-only ===")
    raw_benign = load_raw_benign_stream_for_synthesis()

    results = []
    for rate in rate_grid_m_per_s:
        df_attacked_raw = inject_gradual_drift(raw_benign, rate_m_per_s=rate)
        feat = extract_features(df_attacked_raw)
        feat = feat.sort_values(["sender_id", "t"])

        feat["rule_score"] = feat.apply(rule_score_5signals, axis=1)
        per_sender_cusum = feat.groupby("sender_id")["position_residual"].apply(
            lambda s: cusum_scores(s.values, decay, drift_ref)
        )
        feat["cusum"] = np.concatenate(per_sender_cusum.values)
        feat["rule_cusum_score"] = np.maximum(feat["rule_score"], (feat["cusum"] >= cusum_thresh).astype(float))
        feat["ml_prob"] = ml_predict_proba(feat)

        y_true = feat["is_attack"].values
        pred_rule = (feat["rule_score"].values >= 1.0).astype(int)
        pred_rule_cusum = (feat["rule_cusum_score"].values >= 1.0).astype(int)
        pred_ml = (feat["ml_prob"].values >= ml_thresh).astype(int)

        recall_rule = pred_rule[y_true == 1].mean() if (y_true == 1).any() else np.nan
        recall_rule_cusum = pred_rule_cusum[y_true == 1].mean() if (y_true == 1).any() else np.nan
        recall_ml = pred_ml[y_true == 1].mean() if (y_true == 1).any() else np.nan

        results.append({"rate_m_per_s": rate, "recall_rule": recall_rule,
                         "recall_rule_cusum": recall_rule_cusum, "recall_ml": recall_ml})
        print(f"  rate={rate:5.2f} m/s  Rule={recall_rule:.3f}  Rule+CUSUM={recall_rule_cusum:.3f}  "
              f"ML={recall_ml:.3f}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print(">>> [Step 1] Validation 데이터 로드 중...")
    df_val = load_test_features(split="validation")

    print(">>> [Step 2] Test 데이터 로드 중...")
    df_test = load_test_features(split="test")

    print(">>> [Step 3] Validation 셋에서 CUSUM 최적 파라미터 탐색 중...")
    best_params = grid_search_cusum_params(
        df_val,
        decay_grid=[0.9, 0.95, 0.98, 1.0],
        drift_ref_grid=[0.5, 1.0, 1.5, 2.0],
        cusum_thresh_grid=[2.0, 4.0, 6.0, 8.0, 10.0],
    )
    print("Validation에서 선정된 최적 CUSUM 파라미터:", best_params)

    print("\n>>> [Step 4] Test 셋에 CUSUM 결합 성능 평가:")
    evaluate_rule_with_cusum(
        df_test,
        decay=best_params["decay"],
        drift_ref=best_params["drift_ref"],
        cusum_thresh=best_params["cusum_thresh"],
    )

    # [B]는 load_raw_benign_stream_for_synthesis / extract_features 구현 후 사용
    # ML_THRESH = 0.57
    # run_gradual_drift_experiment(
    #     rate_grid_m_per_s=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    #     decay=best_params["decay"], drift_ref=best_params["drift_ref"],
    #     cusum_thresh=best_params["cusum_thresh"], ml_thresh=ML_THRESH,
    # )