# 논문 실험 코드 안내

이 문서는 각 폴더가 어떤 논문 실험에 해당하는지, 무엇을 하는 코드인지 정리한 것입니다.
VeReMi NextGen 데이터는 저장소에 포함하지 않으며, 출처와 내려받는 법은 [README](../README.md)에 있습니다.

## 한눈에 보기

| 위치 | 성격 | 한 줄 요약 |
| --- | --- | --- |
| `analysis/` | 논문 실험 (지도 정합성) | SUMO 도로망을 기준으로 위치 위조를 잡는 F5 탐지기와 그 공격자 모델 |
| `timewarp/` | 논문 실험 (시간 왜곡) | "과거 위치를 그대로 보내는" 적응형 공격이 왜 자기모순을 남기는지 이론+실험 |
| `evidence_timing/` | 논문 실험 (증거 발생 시점) | 미래 상태 예측형 위조의 증거가 **언제** 드러나는지 (ACK 2026 논문) |
| `phy_cpo/` | 논문 실험 (물리계층) | 수신 신호 세기(RSSI)로 Constant Position Offset 을 잡는 독립 검사기 |
| `scripts/0*.py` 외 | 기존 Phase 2 보강 | 민감도 분석·CUSUM·부트스트랩 CI·베이스라인 모델 비교·피처 중요도 |
| `artifacts/`, `runs/` | 산출물 | 중간 산출물과 과거 실행 결과 (저장소에 포함하지 않음, 재생성 가능) |


## 실행 환경 (중요)

추가분은 두 가상환경으로 나뉩니다.

| 대상 | 환경 | 이유 |
| --- | --- | --- |
| `analysis/`, `timewarp/` | 분석용 가상환경 (`venv`) | `shapely`(도로 폴리곤 연산), `pyarrow` 필요 |
| `evidence_timing/`, `phy_cpo/` | 기본 가상환경으로 충분 | numpy/pandas/scipy/matplotlib 만 사용, `shapely` 불필요 |

산출물은 파일 이름만 주면 `artifacts/` 에서 찾습니다
(`timewarp/common.py` 의 `path()`, `analysis/diag/_paths.py` 의 `A()`).
그래서 폴더를 정리한 뒤에도 예전 명령을 그대로 쓸 수 있습니다.

```bash
python timewarp/pilot4.py --scored scored_tol5.parquet --max-senders 150
```

---

## 1. `analysis/` — 지도 정합성 기반 위치 위조 탐지 (F5)

메시지가 말하는 위치가 **실제 도로 위에 있을 수 있는 위치인지**를 SUMO 도로망
(`ingolstadt.net.xml`) 과 대조하는 계열입니다. Ground Truth 없이 지도만으로
동작한다는 점이 기존 rule/ML 탐지기와 다릅니다.

| 파일 | 역할 |
| --- | --- |
| `f5_map_check.py` | **핵심 라이브러리 겸 CLI.** `build-map`(net.xml → 차로 중심선·주행가능영역 pickle) → `align`(데이터 좌표계와 도로망 좌표계 정합 검증, 필요 시 평행이동 보정) → `score`(메시지별 도로 이탈 거리 `d_off`, 주행 방향 오차 `heading_err` 계산) → `evaluate`(validation benign 으로 임계값 선정 후 test 평가, 오프셋의 종/횡 분해 분석) |
| `road_geom.py` | 도로 형상 유틸. 각진 SUMO 폴리라인을 1 m 재표본화 + 가우시안 평활해서 접선각 `theta(s)` 와 곡률 `kappa(s)` 를 계산 (numpy/shapely 만 사용, scipy 불필요) |
| `semi_synth.py` | **준합성 공격 생성.** test benign 궤적에 크기·방향을 지정한 오프셋 `b` 를 주입해, F5 가 어느 크기부터 탐지하는지 측정. 방향은 송신자 첫 이동 헤딩 기준(0=전방, 90=좌측 횡방향 …)이고 메시지별 종/횡 성분도 함께 기록 |
| `adaptive_attack.py` | **지도를 아는 적응형 공격자(A1/A2).** 위치만 τ초 뒤 값으로 보내는 순진형(A1), 헤딩까지 맞춘 A1h, 위치·속도·헤딩 전부 τ초 뒤로 보내는 일관형(A2)을 만들고 `R_scalar`/`R_vector`/`F5_off`/`F5_head` 및 이들의 OR 결합(COMB)으로 탐지 |
| `spline_compare.py` | 각진 SUMO 도로 vs 평활 곡선 중 어느 쪽이 benign 을 더 잘 설명하는지 비교하는 분석 전용 스크립트 |
| `diag/` | **일회성 점검 스크립트 모음** (아래) |

### `analysis/diag/` — 점검·진단 스크립트

논문 수치를 만들기 전에 "이 값이 정말 맞나"를 확인한 짧은 스크립트들입니다.
재현 파이프라인의 일부가 아니라 기록용입니다.

| 파일 | 확인한 것 |
| --- | --- |
| `_paths.py` | 공용 경로 helper. `import _paths` 하면 `analysis/` 가 import 경로에 붙고 `_paths.A("scored.parquet")` 로 `artifacts/` 산출물을 찾습니다 |
| `diag_offroad.py` | benign 메시지의 도로 이탈 비율과 `d_off` 분위수 — 임계값 설정 근거 |
| `diag_heading.py` | benign 헤딩 오차 분포, 90° 초과 이상치가 어느 송신자·어느 위치에 몰려 있는지 |
| `diag_lateral.py` | 근처 차로 중 진행 방향이 비슷한 차로만 골라 횡방향 오차를 다시 계산 |
| `diag_noise.py` / `diag_noise2.py` | 측위 노이즈 부호 규약 확인(보고 위치 ± noise 중 어느 쪽이 참값인지), 같은 메시지를 여러 수신자가 받았을 때 노이즈가 공유되는지 |
| `diag_split.py` | Train/Validation/Test 분할의 좌표 범위와 `d_off` 분포가 서로 비슷한지 |
| `check_data_leakage.py` | `distance_to_road_edge` 피처가 benign 에는 결측이고 공격에만 값이 있는 식의 누수가 아닌지 |
| `check_onset.py` / `find_onset.py` | 공격이 실제로 시작되는 시점(onset) 찾기 — 탐지 지연 계산의 기준점 |
| `find_residual.py` | 논문 3.3절 임계값(위치 잔차 4 m, 최대 속도 55 m/s, 최대 가속도 12 m/s²)을 test feature 에 직접 적용해 rule 탐지 실패 원인을 분해 |
| `f5_synthetic_test.py` | 실데이터가 아닌 합성 고속도로 궤적으로 F5 자체가 정상 동작하는지 sanity check |
| `merge_splits.py` | train/validation/test feature parquet 3개를 `all_features.parquet` 하나로 합치는 유틸 |

## 2. `timewarp/` — 시간 왜곡(time warping) 공격의 자기모순

"과거 궤적을 그대로 재생하면 물리적으로 완벽한 위조가 되는가?"에 답하는
논문2 pilot 실험입니다. 결론은 **위치만 늦추면 속도와 어긋나고(자기모순),
속도까지 맞추면 가속도 한계를 넘는다**는 쪽입니다.

| 파일 | 역할 |
| --- | --- |
| `common.py` | timewarp 공용 설정. `analysis/` 의 `f5_map_check`·`road_geom` 을 복사하지 않고 그대로 불러 쓰고, 산출물을 `artifacts/` 에서 찾는 `path()` 와 결측 구간을 거르는 `load_benign()` 제공 |
| `tau_model.py` | **이론 모델.** `y(t) = p(s(t)), s(t) = t + τ(t)` 로 정식화. mode "a"(위치만 왜곡 → 자기모순 잔차 `r ≈ v·|τ̇|·dt`) 와 mode "b"(속도·가속도까지 일관되게 변환 → 대신 가속도 한계 위반) |
| `pilot_theory.py` | 위 이론값을 수치로 확인. 명제 1(자기모순 잔차), 명제 2(τ 를 Δτ 만큼 회복하는 최소 시간 `T_min = 2√(Δτ·v/a_lim)`) |
| `pilot2.py` | E1 탐지 가능성 1차 확인 + "불가능(infeasible) 영역" 분리. mode a/b 를 같이 돌려 mode b 의 탐지율이 0(=자기모순이 안 생김)임을 확인 |
| `pilot3.py` | E1'. 잔차를 미래 메시지 기준으로 다시 정의하고, 진짜 대조군 `none`(τ=0) 을 추가. `A2_fixed`/`A2_rate`/`A1_fixed` 비교, 집계를 송신자 중앙값이 아닌 전체 메시지 기준으로 변경 |
| `pilot4.py` | E3+/E4+. 탐지 한계 확장과 **공격자 예측 차수 비교** — 1차 CV(등속), 2차 CA(등가속+요레이트), 3차 CJ(저크+요 가속도) |
| `pilot5.py` | τ 스윕. 차수별로 (a) 탐지율 vs τ, (b) 탐지율 vs 공격 이득 `v·τ` 두 그림 — 공격자가 얻을 수 있는 최대 이득(=유효 상한)을 보여주는 핵심 그림 |
| `plot_e5.py` | `results/e5_tau_sweep_summary.csv` 만 읽어 그림을 다시 그림 (matplotlib 설치 후 재실행용, 실험은 다시 돌리지 않음) |

## 3. `evidence_timing/` — 증거는 언제 드러나는가 (ACK 2026 논문)

미래 상태를 예측해서 보내는 위조가 **언제** 들키는지를 다룹니다.
자체 README([`evidence_timing/README.md`](../evidence_timing/README.md))에
재현 방법·결과·원본 대비 수정 사항·남은 문제가 상세히 적혀 있습니다.

| 파일 | 역할 |
| --- | --- |
| `prepare_data.py` | VeReMi NextGen 배포본이 **이중 zip**(`InTAS_highway_2.zip` → `{Train,Test,Validation}/InTAS_highway_2.zip` → `veh_*.json`)이라, 본 실험이 기대하는 `veh_*.json` 폴더 구조로 풀어 주는 전처리 |
| `v2x_evidence_timing.py` | **본 실험.** 정상 잔차 통계 추정 → 예측 차수(1/2/3)별 위조 메시지 생성 → 가설 H1(불일치 D 가 식 (2) 예측과 맞는가) / H2(최적 탐지 지연 `k*` 가 식 (4)를 따르는가) / H3(실제 탐지기가 최적보다 느린가) 검증 |
| `sweep_alpha_and_ma.py` | 메모 X. α 를 바꿔 가며 `k*` 의 로그-로그 기울기가 이론값 +1/3 에 맞는지, 이동 평균 창 길이의 영향은 어떤지 |
| `arl_matched_ma.py` | 메모 X-2. 이동 평균은 경보가 몰려 나오므로 "메시지당 오탐 1%"로 맞추면 탐지기별 평균 오탐 간격(ARL)이 달라집니다. ARL 을 맞춰 공정하게 비교하도록 보정 |
| `results_run.log`, `results_sweep_run.log` | 위 두 실행의 로그 |

> 원본 스크립트에서 고친 것 중 결과를 가르는 두 가지: ① `sendTime` 이 나노초 정수라
> 초로 그냥 읽으면 속도·가속도가 전부 무의미해지는 문제, ② 통신 범위를 벗어나 비콘이
> 수십 초 끊기는 구간을 거르지 않으면 정상 잔차 σ 가 0.21 m → 161 m 로 767배 틀어지는
> 문제. 자세한 내용은 해당 폴더 README 에 있습니다.

## 4. `phy_cpo/` — RSSI 기반 Constant Position Offset 탐지

메시지 내용만 보는 기존 탐지기가 가장 못 잡는 공격이 Constant Position Offset
(ML Recall 0.09) 입니다. 이 폴더는 **물리계층 수신 신호 세기**라는 독립된 근거로
그 공격을 잡을 수 있는지 봅니다. 보고된 거리로 계산한 예상 수신 세기와 실제 수신
세기의 차이 `rho` 를 (송신, 수신) 쌍별로 이동 평균해서 판정합니다.

| 파일 | 단계 | 역할 |
| --- | --- | --- |
| `s0_schema.py` | 0 | 데이터 구조 확인 — 수신자 위치 `q`, 송신자 실제 위치 `p`(무공격 실행과 대조), 공격 라벨, 공격자별 오프셋 `b` 가 정말 일정한지 |
| `s1_messages.py` | 1 | (수신자, 메시지) 테이블 구성. 조건 A = 데이터셋 기본 CPO(\|b\| 31~92 m), 조건 B = 같은 궤적에 선행연구 조건 `b=(250, −150)` m 를 주입한 것 |
| `s2_rssi.py` | 2 | 수신 세기 합성. `P_rx = P_tx − L0 − 10n·log10(d) + X`, 섀도잉 `X` 는 이동 거리 기반 1차 자기회귀. 조건 A/B 가 **같은 X 실현값**을 공유하도록 해 공정 비교 |
| `s3_detect.py` | 3 | 검사기와 임계값. 임계값 τ 는 **검증 분할의 정상 메시지에서만** \|s\| 의 99 백분위수로 선정(오탐률 1%). 유효 표본 수 `K_eff` 도 함께 추정 |
| `s4_analysis.py` | 4 | E1~E5 분석, 그림, `out/results.json` 생성. 평가는 전부 Test 분할 |
| `diag_e1_e2.py` | — | E1·E2 불합격 원인 진단 (섀도잉 상관 → `K_eff` → 양측/단측 → `b` 추정 순서, 근사식 유효 범위 `\|b\| ≪ d` 포함) |
| `out/` | — | 결과물: `00_schema_report.md`, `01_dataset_stats.md`, `02_params.md`, `03_threshold.md`, `results.json`, `fig2_detection_rate.png`, `fig3_complementarity.png`, `figA_residual_check.png` |
| `work/` | — | 중간 parquet(`messages`, 파라미터 조합별 `rssi_*`/`detections_*`), `params.json`, `thresholds.json` |

`work/` 의 태그는 파라미터 조합을 뜻합니다 — `n2_s4_dc25` = 경로손실 지수 n=2,
섀도잉 σ=4 dB, 상관거리 d_corr=25 m. 기본 조합 외에 n=2.7, σ=2/6, d_corr=0 을
함께 돌려 민감도를 봅니다.

## 5. `scripts/` 추가분 — 기존 Phase 2 결과 보강

기존 `scripts/` 의 `train_veremi_baseline.py`, `download_veremi_pilot.py`,
`export_schemas.py` 는 그대로 두고 다음이 추가됐습니다.
실행 순서와 주의사항은 `scripts/README.md` 에 있습니다.

| 파일 | 역할 |
| --- | --- |
| `01_trust_sensitivity.py` | Message/Node Trust 가중치(`w_rule`, `w_ml`, `w_cam`), EMA `alpha`, 상태 전이 hysteresis 를 그리드 스윕 → 탐지 지연 / 오탐률 / miss rate 트레이드오프. 지금까지 근거 없이 고정돼 있던 값에 근거를 만드는 실험. 결과: `trust_sensitivity_pareto.png` |
| `02_cusum_and_gradual_drift.py` | CUSUM 을 오프라인 비교에 포함하고, 점진적(gradual) drift 합성 공격을 만들어 CPO Recall 이 개선되는지 확인. **개선이 안 나와도 실패가 아님** — constant offset 이 메시지 간 차분에서 상쇄된다는 것 자체가 "카메라/GT 교차검증 필요" 주장의 근거 |
| `03_bootstrap_ci.py` | 표 2/3 의 성능 차이(특히 ML F1 0.829 vs Hybrid 0.820)가 표본 변동 범위인지 95% 부트스트랩 CI 로 확인. Node-level 은 sender 단위 cluster bootstrap |
| `train_baselines.py` | Decision Tree / Logistic Regression / XGBoost 를 같은 feature 캐시로 학습해 RandomForest 와 비교. 결과는 `reports/phase2_baselines_evaluation.md`, `reports/phase2_baselines_metrics.json` |
| `extract_feature_importance.py` | RandomForest 의 `feature_importances_` 를 뽑아 `reports/feature_importance.png` 생성 |

베이스라인 비교 결과 (같은 Test set. 비교 대상 RandomForest 는 P 0.9470 / R 0.7373 / F1 0.8291):

| Method | Precision | Recall | F1 | FPR | PR-AUC | Node F1 | Fit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decision_tree | 0.9495 | 0.7341 | 0.8280 | 0.0303 | 0.8896 | 0.8423 | 1.37s |
| logistic_regression | 0.8748 | 0.7333 | 0.7978 | 0.0813 | 0.8831 | 0.6801 | 35.21s |
| xgboost | 0.9584 | 0.7279 | 0.8274 | 0.0245 | 0.8976 | 0.8582 | 8.56s |

## 6. 산출물 폴더 (`artifacts/`, `runs/`)

둘 다 `.gitignore` 에 등록돼 있고 재생성 가능합니다.

```text
artifacts/    지도·스코어링 중간 산출물 (수백 MB 급)
  ingolstadt.net.xml            SUMO 도로망 원본
  map.pkl, transform.json       build-map / align 결과
  scored*.parquet (+.meta.json) score 결과 (tol5 / tol25 는 정합 허용오차 차이)

runs/         과거 실행 결과 모음
  results*/                              timewarp pilot 실행 결과 (v2~v5, w1)
  adaptive_quick/, adaptive_v1/, adaptive_v2/   적응형 공격 실험
  semi_v1/                               준합성 오프셋 주입 실험
  spline_cmp/                            스플라인 차수 비교
  trust_sensitivity_*.csv / .png         01_trust_sensitivity.py 결과
  0{1,2,3}_*_result.txt                  Track A 실험 실행 로그
```

`artifacts/` 는 `f5_map_check.py` 의 `build-map` → `align` → `score` 로 다시 만들 수 있습니다.
