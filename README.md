# V2X 오정보 탐지 연구 실험 코드 (VeReMi NextGen)

V2X 메시지의 위치·속도 위조를 탐지하는 모델을 만들다가, **위치를 일정하게 옮기는 공격(Constant Position Offset, CPO)이 어떤 모델로도 잘 탐지되지 않는 문제**를 만났고, 그 원인을 분석해 여러 편의 연구로 발전시킨 과정의 실험 코드입니다. KISIA ICT 융합보안 크루 3기 활동의 일부이며, 논문의 공동 제1저자로서 작성한 실험·분석 코드를 담고 있습니다.

## 출발점: 모델을 바꿔도 CPO는 잡히지 않았다

같은 데이터와 같은 분할에서 여러 모델을 비교했습니다. 모든 모델이 속도 위조는 90% 이상 탐지했지만, CPO는 10% 아래였습니다.

| 모델 | constantPositionOffset 재현율 | constantSpeedOffset 재현율 |
|---|---:|---:|
| Decision Tree | 0.092 | 0.923 |
| Logistic Regression | 0.081 | 0.905 |
| XGBoost | 0.082 | 0.919 |

(`reports/phase2_baselines_evaluation.md`, `reports/phase2_baselines_metrics.json`, 테스트 셋 기준)

위치 변화량·속도 같은 운동학적 특징은 위치를 평행이동해도 값이 변하지 않아서, 이 특징만 쓰는 모델은 모델 종류와 무관하게 구조적으로 CPO를 탐지하기 어렵다는 가설을 세우고 수식과 실험으로 확인했습니다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `analysis/` | 도로 지도(SUMO 도로망) 정합성 검사로 위치 위조를 탐지하는 F5 검사기, 지도를 아는 적응형 공격자(A1/A1h/A2) 모델 |
| `timewarp/` | 자신의 과거 궤적을 시간만 옮겨 보내는 공격의 자기모순을 이론과 실험으로 분석 |
| `evidence_timing/` | 미래 상태 예측형 위치 위조의 증거가 언제 드러나는지(최소 탐지 시간) |
| `phy_cpo/` | 수신 신호 세기(RSSI)를 독립 증거로 쓰는 CPO 검사 |
| `scripts/` | 신뢰도 파라미터 민감도, CUSUM, 부트스트랩 신뢰구간, 베이스라인 모델 비교, 특징 중요도 |
| `reports/` | 베이스라인 비교 결과, 데이터 출처 기록 |
| `docs/RESEARCH.md` | 폴더별 상세 설명 |

## 관련 논문

- 도로 지도 정합성 검사를 이용한 V2X Constant Position Offset 공격의 탐지 조건 분석 (정보보호학회 CISC-W'26)
- 미래 상태 예측 기반 위치 위조 공격의 최소 탐지 시간 분석 (정보처리학회 ACK 2026)
- V2X 환경에서 CPO 공격의 실제 위치와 주장 위치에 대한 외부 관측 조건 분석 (정보처리학회 ACK 2026)

(폴더와 논문의 대응은 `docs/RESEARCH.md`를 참고하세요.)

## 실행

데이터는 저장소에 포함하지 않습니다. 아래 순서로 준비하세요.

```bash
pip install -r requirements.txt
python scripts/download_veremi_pilot.py        # VeReMi NextGen 일부 내려받기 (출처·MD5는 reports/dataset-provenance.md)
export VEREMI_EXTRACTED_DIR=data/raw/veremi_nextgen/extracted   # Windows: set VEREMI_EXTRACTED_DIR=...
```

폴더별 실행 명령은 [`docs/RESEARCH.md`](docs/RESEARCH.md)와 [`evidence_timing/README.md`](evidence_timing/README.md)에 있습니다. `analysis/`와 `timewarp/`는 `shapely`, `pyarrow`가 필요합니다.

## 팀 코드와의 관계

`scripts/train_baselines.py`는 팀의 IDS 엔진(`v2x_ids` 패키지: 특징 추출·평가 모듈)을 사용합니다. 엔진은 이 저장소에 포함하지 않았습니다. 나머지 폴더(`analysis`, `timewarp`, `evidence_timing`, `phy_cpo`)는 엔진 없이 독립적으로 실행됩니다.

## 공개 범위

실험 결과 파일(CSV·그림·로그)은 논문 공개 후 추가할 예정입니다. 현재는 코드와 베이스라인 비교 결과만 포함합니다.

## 데이터와 라이선스

VeReMi NextGen 데이터셋(CC BY 4.0)을 사용했습니다. 출처와 인용 정보는 `reports/dataset-provenance.md`에 있습니다.
