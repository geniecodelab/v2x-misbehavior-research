import _paths
import pandas as pd

file_path = _paths.P("test_features.pkl.gz")

print(f"데이터 로딩 중... : {file_path}")
df = pd.read_pickle(file_path)
print("사용 가능한 컬럼들:", df.columns.tolist())

id_col = 'sender_id'
recv_col = 'receiver_id'
time_col = 'timestamp'
label_col = 'label'

# 진단에 쓸 컬럼들 (존재하는 것만 사용)
diag_cols_candidates = [
    time_col, 'receive_timestamp', label_col, 'dataset_variant', 'attack_class',
    'attack_family', 'message_id', 'position_residual', 'delta_position',
    'pos_x', 'pos_y'
]
diag_cols = [c for c in diag_cols_candidates if c in df.columns]

# 1. sender 단독이 아니라 (receiver, sender) 쌍으로 그룹화
pair_counts = (
    df.groupby([recv_col, id_col])[label_col]
    .apply(lambda s: (s > 0).sum())
    .sort_values(ascending=False)
)
attacker_pairs = pair_counts[pair_counts > 0].index.tolist()

print(f"\n공격 라벨이 존재하는 (receiver, sender) 쌍: 총 {len(attacker_pairs)}개")

def diagnose_pair(recv_id, send_id, max_rows_print=40):
    sub = df[(df[recv_col] == recv_id) & (df[id_col] == send_id)].sort_values(time_col).reset_index(drop=True)
    sub['label_bin'] = (sub[label_col] > 0).astype(int)

    print(f"\n{'='*70}")
    print(f"--- Receiver: {recv_id} <- Sender: {send_id} ---")
    print(f"총 메시지 수: {len(sub)}개")

    # timestamp 역행(비단조) 여부 확인 -> 서로 다른 시뮬레이션 런이 이어붙었을 가능성
    ts_diff = sub[time_col].diff()
    n_backwards = (ts_diff < 0).sum()
    if n_backwards > 0:
        print(f"⚠️  timestamp 역행 지점 {n_backwards}개 발견 -> 서로 다른 런이 이어붙었을 가능성 있음")

    # 라벨 토글 지점 찾기
    toggle_idx = sub.index[sub['label_bin'].diff().abs() == 1].tolist()

    if not toggle_idx:
        print("라벨 토글 없음 (계속 0이거나 계속 1).")
        return

    print(f"라벨 토글 지점: {len(toggle_idx)}개")

    variant_col = 'dataset_variant' if 'dataset_variant' in sub.columns else None
    class_col = 'attack_class' if 'attack_class' in sub.columns else None

    suspicious = 0
    for idx in toggle_idx:
        lo = max(0, idx - 1)
        hi = min(len(sub) - 1, idx)
        row_before = sub.iloc[lo]
        row_after = sub.iloc[hi]

        variant_changed = (
            variant_col is not None and
            row_before[variant_col] != row_after[variant_col]
        )
        if variant_changed:
            suspicious += 1

    print(f"토글 지점 중 dataset_variant도 함께 바뀐 경우: {suspicious} / {len(toggle_idx)}")
    if variant_col and suspicious == len(toggle_idx) and len(toggle_idx) > 0:
        print("❌ 결론: 모든 라벨 토글이 시나리오(variant) 경계와 정확히 일치 -> 서로 다른 시뮬레이션 런이 병합되어 발생한 아티팩트로 강하게 의심됨")
    elif variant_col and suspicious > 0:
        print(f"⚠️  결론: 일부만 variant 경계와 일치 ({suspicious}/{len(toggle_idx)}) -> 원인이 섞여 있을 수 있음, 아래 상세 표 확인 필요")
    elif variant_col:
        print("✅ 결론: 토글이 variant 경계와 무관하게 발생 -> 데이터 병합 문제는 아니며, 동일 런 내에서 공격이 간헐적으로 켜짐/꺼짐을 시사")

    # 상세 테이블 출력 (앞부분만)
    print(f"\n--- 상세 테이블 (최대 {max_rows_print}행) ---")
    show_cols = ['label_bin'] + diag_cols
    show_cols = [c for c in dict.fromkeys(show_cols) if c in sub.columns]
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(sub[show_cols].head(max_rows_print).to_string())


# 가장 메시지 수가 많은 쌍부터 상위 3개 진단
for recv_id, send_id in attacker_pairs[:3]:
    diagnose_pair(recv_id, send_id)

# 추가: 특정 sender_id가 여러 dataset_variant에 걸쳐 등장하는지 전체적으로 확인
print(f"\n{'='*70}")
print("전체 데이터에서 sender_id별 등장 variant 개수 (상위 10개, 2개 이상만)")
if 'dataset_variant' in df.columns:
    variant_spread = df.groupby(id_col)['dataset_variant'].nunique().sort_values(ascending=False)
    print(variant_spread[variant_spread > 1].head(10))
else:
    print("dataset_variant 컬럼이 없어 확인 불가")