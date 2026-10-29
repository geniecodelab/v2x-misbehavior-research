import _paths
import pandas as pd
import numpy as np
from scipy import stats

file_path = _paths.P("test_features.pkl.gz")
print(f"데이터 로딩 중... : {file_path}")
df = pd.read_pickle(file_path)

# ------------------------------------------------------------------
# 0. 기본 설정 (논문 3.3절, 표1 값 그대로 재구성 — 실제 코드와 다르면 여기만 수정)
# ------------------------------------------------------------------
TH_POS = 4.0        # 위치 잔차 허용 오차 (m)
TH_SPEED = 55.0      # 최대 속도 (m/s)
TH_ACC = 12.0        # 최대 가속도 (m/s^2)
TH_ACC_CONS = 6.0    # 가속도 일관성 허용 오차 (m/s^2)

CUSUM_DECAY = 0.98
CUSUM_DRIFT_REF = 1.0
CUSUM_THRESHOLD = 10.0

ATTACK_TYPES = [
    'constantPositionOffset',
    'randomPositionOffset',
    'constantSpeedOffset',
    'randomSpeedOffset',
]

# ------------------------------------------------------------------
# 1. Rule anomaly score (5개 신호 정규화 후 최댓값)
# ------------------------------------------------------------------
def rule_score(row):
    s1 = row['position_residual'] / TH_POS
    s2 = row['speed'] / TH_SPEED
    s3 = row['abs_acceleration'] / TH_ACC
    s4 = row['derived_acceleration'] / TH_ACC
    s5 = row['acceleration_residual'] / TH_ACC_CONS
    return max(s1, s2, s3, s4, s5)

# ------------------------------------------------------------------
# 2. (receiver, sender) 그룹별 CUSUM 계산 -> Rule score와 CUSUM 결합 score 반환
# ------------------------------------------------------------------
def compute_scores_for_group(group):
    group = group.sort_values('timestamp').copy()
    rule_scores = group.apply(rule_score, axis=1).values

    cusum_state = 0.0
    cusum_scores = np.zeros(len(group))
    for i, r in enumerate(group['position_residual'].values):
        cusum_state = max(0.0, CUSUM_DECAY * cusum_state + (r - CUSUM_DRIFT_REF))
        cusum_scores[i] = cusum_state / CUSUM_THRESHOLD

    rule_cusum_scores = np.maximum(rule_scores, cusum_scores)
    group['rule_score'] = rule_scores
    group['rule_cusum_score'] = rule_cusum_scores
    return group

def compute_scores(subset_df):
    results = []
    for (recv, send), g in subset_df.groupby(['receiver_id', 'sender_id']):
        results.append(compute_scores_for_group(g))
    return pd.concat(results, ignore_index=True)

# ------------------------------------------------------------------
# 3. Benign 기준 FPR 계산 (전체 benign 데이터 기준, 논문 방식과 동일)
# ------------------------------------------------------------------
print("\nBenign 데이터 처리 중 (전체 FPR 기준선 계산)...")
benign_df = df[df['dataset_variant'] == 'benign'].copy()
benign_scored = compute_scores(benign_df)

fpr_rule = (benign_scored['rule_score'] >= 1.0).mean()
fpr_rule_cusum = (benign_scored['rule_cusum_score'] >= 1.0).mean()

print(f"Rule-only 전체 FPR       : {fpr_rule:.4f}")
print(f"Rule+CUSUM 전체 FPR      : {fpr_rule_cusum:.4f}")

# ------------------------------------------------------------------
# 4. 공격 유형별 Recall + 분포 비교
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print(f"{'공격유형':25s} {'N':>6s} {'RuleRecall':>11s} {'CUSUMRecall':>12s} "
      f"{'전체FPR':>8s} {'Recall-FPR':>11s} {'판정':>10s}")
print("=" * 90)

for atk in ATTACK_TYPES:
    atk_df = df[(df['dataset_variant'] == atk) & (df['label'] == 1)].copy()
    if len(atk_df) == 0:
        print(f"{atk:25s} 데이터 없음")
        continue

    atk_scored = compute_scores(atk_df)

    recall_rule = (atk_scored['rule_score'] >= 1.0).mean()
    recall_rule_cusum = (atk_scored['rule_cusum_score'] >= 1.0).mean()

    diff = recall_rule_cusum - fpr_rule_cusum
    # 판정: recall이 FPR보다 눈에 띄게 높지 않으면(절대차 5%p 미만) "배경오탐 수준"으로 의심
    verdict = "⚠️ 배경수준" if diff < 0.05 else "✅ 실질탐지"

    print(f"{atk:25s} {len(atk_scored):6d} {recall_rule:11.4f} {recall_rule_cusum:12.4f} "
          f"{fpr_rule_cusum:8.4f} {diff:11.4f} {verdict:>10s}")

# ------------------------------------------------------------------
# 5. Constant Position Offset: residual 분포 자체가 정상과 다른지 직접 검정
# ------------------------------------------------------------------
print("\n" + "=" * 90)
print("Constant Position Offset 공격 residual vs Benign residual 분포 비교")
print("=" * 90)

cpo_df = df[(df['dataset_variant'] == 'constantPositionOffset') & (df['label'] == 1)]
cpo_res = cpo_df['position_residual'].dropna()
benign_res = benign_df['position_residual'].dropna()

print(f"공격 메시지 수: {len(cpo_res)}, 정상 메시지 수: {len(benign_res)}")
print(f"공격 residual  : mean={cpo_res.mean():.4f}, std={cpo_res.std():.4f}, median={cpo_res.median():.4f}")
print(f"정상 residual  : mean={benign_res.mean():.4f}, std={benign_res.std():.4f}, median={benign_res.median():.4f}")

ks_stat, ks_p = stats.ks_2samp(cpo_res, benign_res)
print(f"\nKolmogorov-Smirnov 검정: statistic={ks_stat:.4f}, p-value={ks_p:.6f}")
if ks_p > 0.05:
    print("⚠️  두 분포가 통계적으로 구분되지 않음 (p > 0.05) "
          "-> position_residual만으로는 이 공격을 원리적으로 못 잡는다는 증거")
else:
    print(f"두 분포는 통계적으로 다름 (p <= 0.05). "
          f"다만 KS statistic={ks_stat:.4f}이 실질적으로 얼마나 큰 차이인지는 별도 판단 필요.")

print("\n완료.")