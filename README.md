# SMT 사전 공정(납도포) 불량 분석 대시보드

SMT(표면실장기술) 공정 중 전체 불량의 60~70%를 차지하는 납도포(솔더 프린팅) 단계의 불량을
사전에 탐지하기 위한 분석·모델링 대시보드입니다. 공정 센서 데이터와 SPI(Solder Paste
Inspection) 이미지를 결합(Early Fusion)해 불량 여부를 예측하며, 이미지 인코더(Custom
CNN / ResNet50) × 분류기(RandomForest / XGBoost) 2x2 조합 실험 결과를 비교할 수 있습니다.

**배포된 앱:** https://app-smt-axbootcamp-jyf4uagjwi9e7b9r5pyke3.streamlit.app/

---

## 주요 기능

Streamlit 사이드바 메뉴 기준 6개 페이지로 구성되어 있습니다.

| 페이지 | 설명 |
|---|---|
| 개요 | 파이프라인 현황 KPI, 6대 불량 유형 요약, 2x2 실험 설계, 데이터 구성 |
| EDA | 라벨별 센서 분포, 상관관계 히트맵, 이미지 임베딩 PCA 산점도, 요약 통계 |
| 전처리 확인 | 센서-이미지 매칭 현황, 결측치/이상치 점검, 원본 시계열 vs 변환 후 피처 비교 |
| 모델 학습·비교 (2x2) | (A) 2x2 고정 실험 결과 비교 (B) 앱 내 커스텀 모델 학습 (C) 통합 비교 |
| 예측 데모 | 최종 선정 모델로 저장된 샘플 조회 예측 / 새 이미지+센서 데이터 업로드 예측 |
| 필터/조회 | 불량 유형·이상치 여부·파일명 기준 샘플 검색, 썸네일 그리드 |

혼동행렬과 SHAP 기반 피처 중요도(센서 vs 이미지 기여도)는 별도 파일 없이
`fusion_model.pkl` + `fusion_features.csv`만 있으면 앱이 그 자리에서 직접 계산합니다.

---

## 기술 스택

- **Frontend/App**: Streamlit
- **데이터 처리**: Pandas, NumPy
- **시각화**: Plotly
- **모델링**: scikit-learn (RandomForest, LogisticRegression), XGBoost
- **이미지 인코더**: TensorFlow/Keras (Custom CNN, ResNet50)
- **모델 해석**: SHAP
- **배포**: Streamlit Community Cloud

---

## 프로젝트 구조

```
.
├── app.py                          # 메인 Streamlit 앱
├── requirements.txt                # 파이썬 패키지 의존성
├── thumbnails/                     # 샘플 이미지 썸네일 (file_base.jpg)
│
├── fusion_features.csv             # 병합 피처 데이터 (센서 + 이미지 임베딩 + 라벨)
├── data_quality_report.json        # 매칭/결측치/이상치 리포트
├── raw_sensor_sequences.json       # 원본 센서 시계열 (file_base -> 시퀀스)
├── sample_metadata.csv             # 샘플별 메타데이터 (defect_type, is_outlier 등)
│
├── model_comparison_results.csv    # 2x2 실험(인코더 x 분류기) 성능 비교표
├── model_meta.pkl                  # 최종 선정 모델 정보 (encoder_name, classifier_name)
├── fusion_model.pkl                # 최종 선정 Fusion 분류기
├── feature_cols.pkl                # fusion_model이 기대하는 피처 컬럼 순서
├── best_image_model.keras          # 최종 선정 이미지 인코더
├── val_roc_data.json               # (선택) 4개 조합 ROC curve 데이터
│
├── all_results.json                # 앱 내 커스텀 학습 결과 (자동 생성/누적)
└── feature_importance_all.csv      # 앱 내 커스텀 학습 피처 중요도 (자동 생성/누적)
```

`app.py` 상단 docstring에 각 파일이 노트북의 어느 단계에서 만들어지는지, 어떤 컬럼 구조를
가져야 하는지 자세히 적혀 있습니다.

---

## 로컬에서 실행하기

```bash
git clone https://github.com/juunghaa/streamlit-smt-axbootcamp.git
cd streamlit-smt-axbootcamp

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속하면 됩니다.

### 필요 데이터 파일 체크리스트

앱은 필요한 파일이 없어도 죽지 않고 해당 섹션만 비활성화되지만, 전체 기능을 보려면
위 [프로젝트 구조](#프로젝트-구조)에 있는 파일들이 `app.py`와 같은 폴더에 있어야 합니다.
사이드바의 **데이터 파일 연결 상태** 메뉴에서 현재 어떤 파일이 연결/누락됐는지 확인할 수
있습니다.

---

## 배포 (Streamlit Community Cloud)

1. 이 저장소를 GitHub에 push (대용량 모델/이미지 파일이 있다면 Git LFS 권장)
2. [share.streamlit.io](https://share.streamlit.io) 에서 GitHub 계정 연동 후 New app
3. Repository / Branch / Main file path(`app.py`) 지정 후 Deploy
4. `requirements.txt`의 패키지가 자동으로 설치됩니다

> 무료 플랜은 메모리가 약 1GB로 제한되어 있어, TensorFlow + XGBoost + SHAP을 동시에
> 로드하면 한도에 걸릴 수 있습니다. 리소스가 부족하면 "예측 데모 > 새 이미지 업로드"
> 기능(TensorFlow 필요)을 선택적으로 비활성화하는 것도 방법입니다.

---

## 알려진 이슈

- `fusion_model.pkl`을 불러올 때 `ModuleNotFoundError: No module named 'dill'` 오류가
  나면 `pip install dill` 후 다시 실행하세요. (`requirements.txt`에 이미 포함되어 있습니다.)
- `all_results.json`, `feature_importance_all.csv`는 앱 실행 중 자동 생성/갱신되는
  파일이라, Streamlit Community Cloud처럼 파일시스템이 초기화되는 환경에서는 앱을
  재시작/재배포할 때마다 초기화됩니다.

---

## 데이터 안내

이 저장소의 데이터는 부트캠프 프로젝트용 예시 데이터입니다. 실제 생산 라인 데이터를
연동할 경우, 회사 정책에 따라 저장소를 Private으로 전환하고 접근 권한을 관리하는 것을
권장합니다.
