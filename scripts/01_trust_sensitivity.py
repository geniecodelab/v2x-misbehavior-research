"""
Track A - 실험 1: Message/Node Trust 가중치 · Hysteresis 파라미터 Sensitivity 분석
========================================================================
논문 3.6의 Message Trust 가중치(w_rule, w_ml, w_cam), Node Trust EMA(alpha),
그리고 상태 전이 hysteresis(연속 위반/회복 횟수)가 지금은 근거 없이 고정되어
있습니다. 이 스크립트는 각 파라미터를 그리드로 스윕하면서

  - 탐지 지연(공격 시작~ATTACK 상태 도달까지 시간)
  - 오탐률(정상 시퀀스가 ATTACK/SUSPECT로 잘못 전이하는 비율)
  - miss_rate(공격 시퀀스가 끝까지 탐지되지 않는 비율)

를 함께 측정해 Pareto curve(트레이드오프 곡선)를 만들고, 그 위에 현재
논문에서 실제 사용 중인 값(CURRENT_DEFAULTS)을 표시해 "왜 이 값을 골랐는지"를
정량적으로 뒷받침합니다.

=====================  2단계 구조 (4.2 Validation/Test 분리 원칙 준수)  =====================
[수정] 이전 버전은 그리드서치 자체를 test_features.pkl.gz 위에서 돌리고 있었음.
이는 4.2절 "파라미터 탐색은 Validation에서만, Test는 최종 1회 평가에만" 원칙과
정면으로 모순됨 (test set을 보고 파라미터를 고르는 셈이라 데이터 누수와 동일한
문제). 아래처럼 두 단계로 분리함:

  [1단계] Validation set 그리드서치
      - VALIDATION_SPLIT_FILE(기본: validation_features.pkl.gz)을 로드해
        그리드서치(3528개 조합)를 수행.
      - CURRENT_DEFAULTS가 Pareto-optimal인지 확인하고, 이를 지배(dominate)하는
        조합이 있으면 RECOMMENDED_DEFAULTS 후보로 자동 제안.
      - 이 결과(trust_sensitivity_results_validation.csv)는 파라미터를
        "고르는" 데만 쓰고, 논문에 최종 성능 수치로 인용하지 않는다.

  [2단계] Test set 최종 평가
      - TEST_SPLIT_FILE(기본: test_features.pkl.gz)을 로드.
      - 그리드서치를 다시 하지 않고, 1단계에서 이미 확정된 두 조합
        (CURRENT_DEFAULTS, RECOMMENDED_DEFAULTS)만 각각 정확히 1회 평가.
      - 이 결과(trust_sensitivity_final_test_eval.csv)가 논문 3.6에 인용할
        최종 수치.

=====================  파일 경로 확인 필요  =====================
같은 디렉토리에 test_features.pkl.gz / train_features.pkl.gz /
validation_features.pkl.gz 세 개가 있다고 하셨는데, validation 파일의
정확한 이름이 다를 수 있습니다(예: val_features.pkl.gz). 아래
VALIDATION_SPLIT_FILE 상수만 실제 파일명에 맞게 확인/수정하세요.
===========================================================================
"""

from pathlib import Path
import gzip
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 데이터 경로 설정 -- 실제 파일명에 맞게 확인하세요.
# ----------------------------------------------------------------------
DATA_DIR = Path("data/processed/veremi_nextgen")
VALIDATION_SPLIT_FILE = DATA_DIR / "validation_features.pkl.gz"  # 그리드서치용
TEST_SPLIT_FILE = DATA_DIR / "test_features.pkl.gz"              # 최종 1회 평가용
MODEL_PATH = Path("models/veremi_nextgen_rf.joblib")
META_PATH = Path("models/veremi_nextgen_rf.metadata.json")

# 그리드 탐색 범위 설정 (validation sweep에서만 사용)
W_RULE_GRID = np.round(np.arange(0.0, 1.001, 0.05), 2)  # w_ml = 1 - w_rule
ALPHA_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]  # Node Trust EMA 계수
K_ATTACK_GRID = [1, 2, 3, 5, 7, 10]  # ATTACK 전이 필요 연속 위반 횟수
K_RECOVER_GRID = [1, 3, 5, 10]  # NORMAL 복귀 필요 연속 정상 횟수

TRUST_LOW_THRESH = 0.5  # NORMAL -> SUSPECT 임계값
TRUST_ATTACK_THRESH = 0.3  # SUSPECT -> ATTACK 임계값

# 논문에 지금까지 쓰던 값 (지난 validation 탐색 이전 기준)
CURRENT_DEFAULTS = {
    "w_rule": 0.45,
    "alpha": 0.2,
    "k_attack": 3,
    "k_recover": 5,
}


def load_message_stream(data_path: Path) -> pd.DataFrame:
  """지정한 split 파일을 로드하고 Trust 모듈 민감도 분석용 데이터프레임을 구성합니다."""
  with gzip.open(data_path, "rb") as f:
    df = pd.read_pickle(f)

  with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
  _ = metadata["feature_names"]  # 참고용 (현재 로직에서 직접 사용하지 않음)

  # 1. Rule Score (5개 물리 신호 정규화 최댓값)
  r_pos = (df["position_residual"].abs() / 4.0).to_numpy()
  r_spd = (df["speed"] / 55.0).to_numpy()
  r_acc = (df["abs_acceleration"] / 12.0).to_numpy()
  r_dacc = (df["derived_acceleration"].abs() / 12.0).to_numpy()
  r_ares = (df["acceleration_residual"].abs() / 6.0).to_numpy()
  a_rule = np.maximum.reduce([r_pos, r_spd, r_acc, r_dacc, r_ares])

  # 2. ML Probability (calibration 포함, bundle 내장 calibrator 사용)
  bundle = joblib.load(MODEL_PATH)
  p_ml = bundle.predict_anomaly_probability(df)

  # 3. 데이터프레임 구성 및 정렬
  df_stream = pd.DataFrame({
      "scenario_id": df["dataset_variant"].to_numpy(dtype=str),
      "sender_id": df["sender_id"].to_numpy(),
      "t": df["timestamp"].to_numpy(),
      "a_rule": a_rule,
      "p_ml": p_ml,
      "label": df["label"].to_numpy(dtype=int),
  })
  df_stream = df_stream.sort_values(
      ["scenario_id", "sender_id", "t"]
  ).reset_index(drop=True)

  # 4. 시퀀스별 공격 여부 및 최초 공격 시점(onset_t) 계산
  df_stream["is_attack_seq"] = df_stream.groupby(
      ["scenario_id", "sender_id"]
  )["label"].transform(lambda s: bool((s == 1).any()))
  df_stream["_atk_t"] = np.where(
      df_stream["label"] == 1, df_stream["t"], np.nan
  )
  df_stream["attack_onset_t"] = df_stream.groupby(
      ["scenario_id", "sender_id"]
  )["_atk_t"].transform("min")
  df_stream = df_stream.drop(columns=["_atk_t", "label"])

  return df_stream


def build_cached_groups(df: pd.DataFrame):
  """송신자별 시퀀스를 사전 캐싱 (그리드서치와 단일 평가에서 공용으로 사용)."""
  cached_groups = []
  for _, g in df.groupby(["scenario_id", "sender_id"]):
    g = g.sort_values("t")
    is_attack = bool(g["is_attack_seq"].iloc[0])
    onset_t = g["attack_onset_t"].iloc[0] if is_attack else np.nan
    cached_groups.append({
        "a_rule": g["a_rule"].to_numpy(),
        "p_ml": g["p_ml"].to_numpy(),
        "t": g["t"].to_numpy(),
        "is_attack": is_attack,
        "onset_t": onset_t,
    })
  return cached_groups


def simulate_node_trust_and_state(trust_msg_seq, alpha, k_attack, k_recover):
  """EMA 누적 및 hysteresis 상태머신 시뮬레이션"""
  n = len(trust_msg_seq)
  state = np.zeros(n, dtype=int)
  consec_violation = 0
  consec_ok = 0
  cur_state = 0
  prev_trust = 1.0
  first_attack_idx = None

  for i, tm in enumerate(trust_msg_seq):
    prev_trust = alpha * tm + (1.0 - alpha) * prev_trust
    violated = prev_trust < TRUST_LOW_THRESH
    hard_violated = prev_trust < TRUST_ATTACK_THRESH

    if violated:
      consec_violation += 1
      consec_ok = 0
    else:
      consec_ok += 1
      consec_violation = 0

    if cur_state == 0 and consec_violation >= k_attack:
      cur_state = 1 if not hard_violated else 2
    elif cur_state == 1 and hard_violated and consec_violation >= k_attack:
      cur_state = 2
    elif cur_state in (1, 2) and consec_ok >= k_recover:
      cur_state = 0

    state[i] = cur_state
    if cur_state == 2 and first_attack_idx is None:
      first_attack_idx = i

  return state, first_attack_idx


def evaluate_single_config(cached_groups, w_rule, alpha, k_attack, k_recover):
  """캐싱된 시퀀스에 대해 단일 파라미터 조합의 fpr/latency_median/miss_rate 계산."""
  w_ml = 1.0 - w_rule
  latencies = []
  fp_flags = []
  for item in cached_groups:
    trust_msg = np.clip(
        1.0 - (w_rule * item["a_rule"] + w_ml * item["p_ml"]), 0.0, 1.0
    )
    state, first_idx = simulate_node_trust_and_state(
        trust_msg, alpha, k_attack, k_recover
    )
    if item["is_attack"]:
      if first_idx is not None:
        latencies.append(max(0.0, item["t"][first_idx] - item["onset_t"]))
      else:
        latencies.append(np.nan)
    else:
      fp_flags.append(bool(np.any(state == 2)))

  lat_med = np.nanmedian(latencies) if latencies else np.nan
  miss = np.mean([np.isnan(x) for x in latencies]) if latencies else np.nan
  fpr = np.mean(fp_flags) if fp_flags else np.nan
  return {
      "w_rule": w_rule, "alpha": alpha, "k_attack": k_attack, "k_recover": k_recover,
      "latency_median": lat_med, "miss_rate": miss, "fpr": fpr,
  }


def run_sensitivity_sweep_fast(cached_groups):
  """전체 그리드에 대해 evaluate_single_config를 반복 (validation sweep 전용)."""
  total_combos = (
      len(W_RULE_GRID) * len(ALPHA_GRID) * len(K_ATTACK_GRID) * len(K_RECOVER_GRID)
  )
  print(f">>> 총 {total_combos}개 파라미터 조합 시뮬레이션 시작 (validation set)...")

  results = []
  count = 0
  for w_rule in W_RULE_GRID:
    for alpha in ALPHA_GRID:
      for k_attack in K_ATTACK_GRID:
        for k_recover in K_RECOVER_GRID:
          results.append(
              evaluate_single_config(cached_groups, w_rule, alpha, k_attack, k_recover)
          )
          count += 1
          if count % 300 == 0 or count == total_combos:
            print(f"    진행률: {count}/{total_combos} ({count/total_combos*100:.1f}%)")

  return pd.DataFrame(results)


def find_dominating_recommendation(results_df, defaults):
  """defaults를 fpr/latency/miss_rate 세 지표 모두에서 지배하는 조합 중
  (fpr, miss_rate, latency_median) 순으로 가장 우수한 하나를 추천한다.
  지배하는 조합이 없으면 None을 반환 (= 현재값이 이미 Pareto-optimal)."""
  df = results_df.dropna(subset=["fpr", "latency_median", "miss_rate"])
  match = df[
      np.isclose(df["w_rule"], defaults["w_rule"])
      & np.isclose(df["alpha"], defaults["alpha"])
      & (df["k_attack"] == defaults["k_attack"])
      & (df["k_recover"] == defaults["k_recover"])
  ]
  if match.empty:
    print(f"!! 경고: CURRENT_DEFAULTS={defaults} 조합이 그리드에 없습니다.")
    return None, None

  base = match.iloc[0]
  dominating = df[
      (df["fpr"] <= base["fpr"])
      & (df["latency_median"] <= base["latency_median"])
      & (df["miss_rate"] <= base["miss_rate"])
      & (
          (df["fpr"] < base["fpr"])
          | (df["latency_median"] < base["latency_median"])
          | (df["miss_rate"] < base["miss_rate"])
      )
  ]
  if dominating.empty:
    print("[검증 결과] CURRENT_DEFAULTS는 validation 기준으로 이미 Pareto-optimal입니다. "
          "교체 근거 없음 -> 그대로 유지 권장.")
    return base, None

  best = dominating.sort_values(["fpr", "miss_rate", "latency_median"]).iloc[0]
  print(f"[검증 결과] CURRENT_DEFAULTS를 모든 지표에서 지배하는 조합 {len(dominating)}개 발견.")
  print("  현재값 (validation):", dict(base[["w_rule", "alpha", "k_attack", "k_recover",
                                             "fpr", "latency_median", "miss_rate"]]))
  print("  추천값 (validation):", dict(best[["w_rule", "alpha", "k_attack", "k_recover",
                                             "fpr", "latency_median", "miss_rate"]]))
  return base, best


def plot_pareto(results_df, current_defaults, recommended_defaults=None, out_path=None):
  """파레토 산점도 (validation sweep 결과 기준)"""
  if out_path is None:
    out_path = Path(__file__).resolve().parent / "trust_sensitivity_pareto.png"
  fig, ax = plt.subplots(figsize=(7, 5.5))
  ax.scatter(
      results_df["fpr"], results_df["latency_median"],
      alpha=0.35, s=20, color="#1f77b4", label="Grid Combinations (Validation)",
  )

  d = current_defaults
  match = results_df[
      np.isclose(results_df["w_rule"], d["w_rule"])
      & np.isclose(results_df["alpha"], d["alpha"])
      & (results_df["k_attack"] == d["k_attack"])
      & (results_df["k_recover"] == d["k_recover"])
  ]
  if not match.empty:
    ax.scatter(
        match["fpr"], match["latency_median"], color="red", marker="*", s=250,
        label=f"Current Default (w_r={d['w_rule']}, α={d['alpha']})", zorder=5,
    )

  if recommended_defaults is not None:
    r = recommended_defaults
    rmatch = results_df[
        np.isclose(results_df["w_rule"], r["w_rule"])
        & np.isclose(results_df["alpha"], r["alpha"])
        & (results_df["k_attack"] == r["k_attack"])
        & (results_df["k_recover"] == r["k_recover"])
    ]
    if not rmatch.empty:
      ax.scatter(
          rmatch["fpr"], rmatch["latency_median"], color="green", marker="*", s=250,
          label=f"Recommended (w_r={r['w_rule']}, α={r['alpha']})", zorder=5,
      )

  ax.set_xlabel("Node-level FPR (False Positive Rate)", fontsize=11)
  ax.set_ylabel("Detection Latency Median (s)", fontsize=11)
  ax.set_title("Node Trust Sensitivity (Validation Set): Latency vs FPR", fontsize=12)
  ax.grid(True, linestyle="--", alpha=0.5)
  ax.legend(loc="upper right")
  fig.tight_layout()
  fig.savefig(str(out_path), dpi=200)
  print(f">>> 파레토 산점도 그래프 저장 완료: {out_path}")


if __name__ == "__main__":
  # ==================== [1단계] Validation에서 그리드서치 ====================
  print("=" * 70)
  print("[1단계] Validation set 로드 및 그리드서치")
  print("=" * 70)
  df_val = load_message_stream(VALIDATION_SPLIT_FILE)
  cached_val = build_cached_groups(df_val)
  results_val = run_sensitivity_sweep_fast(cached_val)
  results_val.to_csv("trust_sensitivity_results_validation.csv", index=False)
  print("\n=== Validation 그리드서치 결과 저장: trust_sensitivity_results_validation.csv ===")

  print(f"\n(참고: validation에서 관측된 miss_rate 범위 = "
        f"[{results_val['miss_rate'].min():.4f}, {results_val['miss_rate'].max():.4f}])")

  base_perf, recommended = find_dominating_recommendation(results_val, CURRENT_DEFAULTS)

  plot_pareto(results_val, CURRENT_DEFAULTS,
              recommended_defaults=recommended.to_dict() if recommended is not None else None)

  # ==================== [2단계] Test set에서 최종 1회 평가 ====================
  print("\n" + "=" * 70)
  print("[2단계] Test set 최종 평가 (그리드서치 아님 - 확정된 후보 2개만 평가)")
  print("=" * 70)
  df_test = load_message_stream(TEST_SPLIT_FILE)
  cached_test = build_cached_groups(df_test)

  final_rows = []
  cur = evaluate_single_config(cached_test, **CURRENT_DEFAULTS)
  cur["label"] = "CURRENT_DEFAULTS"
  final_rows.append(cur)

  if recommended is not None:
    rec_params = {
        "w_rule": float(recommended["w_rule"]),
        "alpha": float(recommended["alpha"]),
        "k_attack": int(recommended["k_attack"]),
        "k_recover": int(recommended["k_recover"]),
    }
    rec = evaluate_single_config(cached_test, **rec_params)
    rec["label"] = "RECOMMENDED_DEFAULTS"
    final_rows.append(rec)

  final_df = pd.DataFrame(final_rows)
  final_df.to_csv("trust_sensitivity_final_test_eval.csv", index=False)
  print("\n[Test set 최종 평가 결과 -- 이 표를 논문 3.6에 인용]")
  print(final_df.to_string(index=False))
  print("\n=== 저장 완료: trust_sensitivity_final_test_eval.csv ===")