"""
Track A - 실험 3: 표 2/3 성능 차이에 대한 Bootstrap 신뢰구간
================================================================
논문 4.3의 표 2(Rule/ML/Hybrid 비교), 표 3(공격 유형별 Recall)에 나온
성능 차이가 통계적으로 유의한지(표본 변동으로 설명되지 않는지) 확인합니다.

세 가지를 계산합니다.
  1) Message-level Precision/Recall/F1/FPR의 95% CI (방법별)
  2) 방법 간 쌍대 차이(예: ML - Hybrid)의 95% CI → 0을 포함하지 않으면 유의미
  3) Node-level(95th percentile 집계, cluster bootstrap by sender)의 95% CI

=====================  여기를 실제 데이터에 맞게 채우세요  =====================
아래 load_predictions() 함수 안에서, 이미 갖고 계신 test set 예측 결과를 불러와
아래 5개 배열/시리즈로 반환하도록 수정하세요. (재학습 불필요 — 저장해 둔 예측값만 있으면 됩니다)

  y_true       : (N,) 0=benign, 1=attack (message-level 이진 라벨)
  attack_type  : (N,) 문자열. 'benign' 또는 'constantPositionOffset' 등 (표 3 재현용)
  sender_id    : (N,) 송신자 ID (node-level cluster bootstrap용)
  scores       : dict[str, np.ndarray]  예) {
                     'rule':   a_rule 배열 (0~1, threshold 1.0 이상이면 공격 판정),
                     'ml':     calibration 이후의 P(attack) 배열,
                     'hybrid': 0.45*rule + 0.55*ml (3.5식 그대로)
                 }
  thresholds   : dict[str, float]  각 방법의 판정 임계값 (표 2 만들 때 쓴 값 그대로)
===========================================================================
"""

import numpy as np
import pandas as pd
# from numpy import split  ← 이 줄은 삭제 (불필요하고 혼란만 유발)

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
ALPHA = 0.05  # 95% CI


def load_predictions(split: str = "test"):
    """test_features.pkl.gz(또는 validation_features.pkl.gz)를 로드하고
    Rule / ML / Hybrid 점수를 계산하여 반환합니다."""
    import gzip
    import json
    from pathlib import Path
    import joblib

    data_path = Path(
        f"data/processed/veremi_nextgen/"
        f"{'validation_features' if split in ('val', 'validation') else 'test_features'}.pkl.gz"
    )
    model_path = Path("models/veremi_nextgen_rf.joblib")
    meta_path = Path("models/veremi_nextgen_rf.metadata.json")

    with gzip.open(data_path, "rb") as f:
        df = pd.read_pickle(f)

    y_true = df["label"].to_numpy(dtype=int)
    attack_type = df["attack_class"].to_numpy(dtype=str)
    sender_id = df["sender_id"].to_numpy()
    receiver_id = df["receiver_id"].to_numpy()
    scenario_id = df["dataset_variant"].to_numpy(dtype=str)   # ← 추가

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    thresholds = {k: float(v) for k, v in metadata["thresholds"].items()}
    hw = metadata["hybrid_weights"]

    r_pos = (df["position_residual"].abs() / 4.0).to_numpy()
    r_spd = (df["speed"] / 55.0).to_numpy()
    r_acc = (df["abs_acceleration"] / 12.0).to_numpy()
    r_dacc = (df["derived_acceleration"].abs() / 12.0).to_numpy()
    r_ares = (df["acceleration_residual"].abs() / 6.0).to_numpy()
    rule_score = np.maximum.reduce([r_pos, r_spd, r_acc, r_dacc, r_ares])

    bundle = joblib.load(model_path)
    ml_score = bundle.predict_anomaly_probability(df)

    hybrid_score = hw["rule"] * rule_score + hw["ml"] * ml_score

    scores = {"rule": rule_score, "ml": ml_score, "hybrid": hybrid_score}

    return y_true, attack_type, sender_id, receiver_id, scenario_id, scores, thresholds
# ----------------------------------------------------------------------
# 지표 계산 (수정 불필요)
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def node_level_scores(scores, sender_id, receiver_id, scenario_id, y_true, method):
    df = pd.DataFrame({
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "scenario_id": scenario_id,
        "score": scores[method],
        "y_true": y_true,
    })
    agg = df.groupby(["scenario_id", "sender_id", "receiver_id"]).agg(
        node_score=("score", lambda s: np.percentile(s, 95)),
        node_label=("y_true", "max"),
    )
    return agg["node_score"].values, agg["node_label"].values, agg.index.values


def percentile_ci(samples, alpha=ALPHA):
    lo = np.percentile(samples, 100 * alpha / 2)
    hi = np.percentile(samples, 100 * (1 - alpha / 2))
    return lo, hi


# ----------------------------------------------------------------------
# 1) Message-level bootstrap
# ----------------------------------------------------------------------
def message_level_bootstrap(y_true, scores, thresholds, methods):
    n = len(y_true)
    boot_metrics = {m: {"precision": [], "recall": [], "f1": [], "fpr": []} for m in methods}
    boot_f1_diff = []  # ML - Hybrid (필요시 다른 쌍으로 변경)

    for _ in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n, size=n)  # message-level resample with replacement
        yt = y_true[idx]

        f1_this_iter = {}
        for m in methods:
            pred = (scores[m][idx] >= thresholds[m]).astype(int)
            met = compute_metrics(yt, pred)
            for k, v in met.items():
                boot_metrics[m][k].append(v)
            f1_this_iter[m] = met["f1"]

        if "ml" in methods and "hybrid" in methods:
            boot_f1_diff.append(f1_this_iter["ml"] - f1_this_iter["hybrid"])

    print(f"\n=== [1] Message-level 95% CI (bootstrap, n_iter={N_BOOTSTRAP}) ===")
    for m in methods:
        for metric in ["precision", "recall", "f1", "fpr"]:
            lo, hi = percentile_ci(boot_metrics[m][metric])
            point = np.mean(boot_metrics[m][metric])
            print(f"  {m:8s} {metric:10s}: {point:.3f}  [{lo:.3f}, {hi:.3f}]")

    if boot_f1_diff:
        lo, hi = percentile_ci(boot_f1_diff)
        sig = "유의함 (CI가 0 미포함)" if (lo > 0 or hi < 0) else "유의하지 않음 (CI가 0 포함)"
        print(f"\n  ML F1 - Hybrid F1 차이: mean={np.mean(boot_f1_diff):.4f}, "
              f"95% CI=[{lo:.4f}, {hi:.4f}] -> {sig}")


# ----------------------------------------------------------------------
# 2) 공격 유형별 (표 3) bootstrap
# ----------------------------------------------------------------------
def per_attack_bootstrap(y_true, attack_type, scores, thresholds, methods, attack_names):
    print(f"\n=== [2] Per-attack Recall 95% CI (bootstrap, n_iter={N_BOOTSTRAP}) ===")
    for atk in attack_names:
        mask = (attack_type == atk) | (attack_type == "benign")
        yt_full = y_true[mask]
        n = mask.sum()
        for m in methods:
            sc = scores[m][mask]
            recalls = []
            for _ in range(N_BOOTSTRAP):
                idx = RNG.integers(0, n, size=n)
                pred = (sc[idx] >= thresholds[m]).astype(int)
                met = compute_metrics(yt_full[idx], pred)
                recalls.append(met["recall"])
            lo, hi = percentile_ci(recalls)
            print(f"  {atk:24s} {m:8s} recall: {np.mean(recalls):.3f}  [{lo:.3f}, {hi:.3f}]")


# ----------------------------------------------------------------------
# 3) Node-level cluster bootstrap (송신자 단위로 리샘플)
# ----------------------------------------------------------------------
def node_level_cluster_bootstrap(y_true, sender_id, receiver_id, scenario_id, scores, thresholds, methods):
    print(f"\n=== [3] Node-level 95% CI (bootstrap, n_iter={N_BOOTSTRAP}) ===")

    for m in methods:
        node_score, node_label, node_ids = node_level_scores(
            scores, sender_id, receiver_id, scenario_id, y_true, m
        )
        n_nodes = len(node_ids)

        # --- 진단용: bootstrap 없이 전체 데이터 1회 point estimate ---
        pred_full = (node_score >= thresholds[m]).astype(int)
        met_full = compute_metrics(node_label, pred_full)
        print(f"  [진단] {m:8s} n_nodes={n_nodes}  point F1={met_full['f1']:.5f}  "
              f"(phase2 값과 직접 비교)")
        # -----------------------------------------------

        f1s = []
        for _ in range(N_BOOTSTRAP):
            idxs = RNG.integers(0, n_nodes, size=n_nodes)
            pred = (node_score[idxs] >= thresholds[m]).astype(int)
            met = compute_metrics(node_label[idxs], pred)
            f1s.append(met["f1"])
        lo, hi = percentile_ci(f1s)
        print(f"  {m:8s} node F1: {np.mean(f1s):.3f}  [{lo:.3f}, {hi:.3f}]")

def select_node_threshold_on_validation(y_val, sender_id_val, scores_val, method, grid=None):
    """Validation set에서 node-level F1을 최대화하는 threshold 선정 (4.2 원칙과 동일)."""
    if grid is None:
        grid = np.round(np.arange(0.05, 1.00, 0.01), 2)
    node_score, node_label, _ = node_level_scores(scores_val, sender_id_val, y_val, method)
    best_thr, best_f1 = None, -1
    for thr in grid:
        pred = (node_score >= thr).astype(int)
        met = compute_metrics(node_label, pred)
        if met["f1"] > best_f1:
            best_f1, best_thr = met["f1"], thr
    return best_thr, best_f1




if __name__ == "__main__":
    y_true, attack_type, sender_id, receiver_id, scenario_id, scores, thresholds = load_predictions()
    methods = list(scores.keys())
    attack_names = [a for a in np.unique(attack_type) if a != "benign"]

    message_level_bootstrap(y_true, scores, thresholds, methods)
    per_attack_bootstrap(y_true, attack_type, scores, thresholds, methods, attack_names)
    node_level_cluster_bootstrap(y_true, sender_id, receiver_id, scenario_id, scores, thresholds, methods)