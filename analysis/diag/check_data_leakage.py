import _paths
import pandas as pd

df = pd.read_pickle(_paths.P("test_features.pkl.gz"))

col = 'distance_to_road_edge'

# 1. benign에도 정말 값이 있는지, 아니면 전부 NaN인지
benign = df[df['dataset_variant'] == 'benign']
cpo = df[df['dataset_variant'] == 'constantPositionOffset']

print("benign 결측 비율:", benign[col].isna().mean())
print("CPO 결측 비율:", cpo[col].isna().mean())

print("\nbenign 통계:\n", benign[col].describe())
print("\nCPO 통계:\n", cpo[col].describe())

# 2. 값이 둘 다 있다면, 분포가 실제로 다른지 (KS 검정)
from scipy import stats
b = benign[col].dropna()
c = cpo[col].dropna()
if len(b) > 0 and len(c) > 0:
    ks_stat, ks_p = stats.ks_2samp(b, c)
    print(f"\nKS statistic={ks_stat:.4f}, p={ks_p:.6f}")