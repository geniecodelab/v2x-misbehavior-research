import _paths
import pandas as pd

df = pd.read_pickle(_paths.P("test_features.pkl.gz"))

TARGET = 'constantPositionOffset'
vdf = df[df['dataset_variant'] == TARGET]

pairs = (
    vdf[vdf['label'] > 0]
    .groupby(['receiver_id', 'sender_id'])
    .size()
    .sort_values(ascending=False)
    .index.tolist()
)

onset, no_onset = 0, 0
for recv, send in pairs[:50]:
    sub = vdf[(vdf.receiver_id == recv) & (vdf.sender_id == send)].sort_values('timestamp')
    labels = (sub['label'] > 0).astype(int).tolist()
    if labels[0] == 0 and 1 in labels:
        onset += 1
    else:
        no_onset += 1

print(f"[{TARGET}] 상위 50쌍 중 — Onset 있음: {onset} / Onset 없음(등장부터 1): {no_onset}")