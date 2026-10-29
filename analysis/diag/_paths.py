"""
_paths.py: diag/ 안의 일회성 점검 스크립트 공용 경로 설정.

이 스크립트들은 원래 V2X_IDS 최상위에서 "scored.parquet" 처럼 파일 이름만 적어
현재 작업 폴더 기준으로 읽었다. 정리 후에는 모듈이 analysis/ 로, 산출물이
artifacts/ 로 옮겨졌으므로, 어디서 실행하든 같은 파일을 찾도록 여기서 맞춰 준다.

    import _paths                      # analysis/ 를 import 경로에 추가
    import f5_map_check as f
    df = pd.read_parquet(_paths.A("scored.parquet"))          # artifacts/
    df = pd.read_pickle(_paths.P("test_features.pkl.gz"))     # data/processed/
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # ...\V2X_IDS
ANALYSIS = ROOT / "analysis"
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed" / "veremi_nextgen"

if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))


def A(name: str) -> str:
    """artifacts/ 안의 산출물 경로를 문자열로 돌려준다."""
    return str(ARTIFACTS / name)


def P(name: str) -> str:
    """data/processed/veremi_nextgen/ 안의 feature 파일 경로를 돌려준다.

    train_veremi_baseline.py 를 한 번 이상 돌려야 생긴다. 없으면 무엇을 먼저
    실행해야 하는지 알려 주고 멈춘다.
    """
    path = PROCESSED / name
    if not path.exists():
        raise SystemExit(
            f"{path} 가 없습니다.\n"
            "먼저 `python scripts/train_veremi_baseline.py` 로 feature 캐시를 만드세요."
        )
    return str(path)
