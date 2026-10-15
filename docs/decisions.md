# SMT 모델 선정과 평가 데이터 분리 결정

> 프로젝트 기간: 2026.07–09 · 프로젝트 종료 후 Colab 노트북 출력과 당시 팀의 결정 내용을 정리한 기록

## 범위와 근거

모델 학습과 평가는 6인 팀이 Google Colab에서 조합을 나눠 진행했다. 이 저장소에는 별도로 구현한 Streamlit 대시보드의 배포 코드와 자료가 있다. 아래의 **최종 발표 수치**는 팀이 발표에 채택한 실행본 기준이다. 저장소의 `model_comparison_results.csv`와 8종 비교 노트북은 각각 다른 실행 결과이므로 수치를 섞지 않았다.

## 결정 1 — 센서 단독과 이미지·센서 Fusion을 모두 평가

- **상황:** 센서만으로 충분하다는 의견과 이미지 결합이 필요하다는 의견이 갈렸다. 초기 가설은 이미지가 센서의 한계를 보완해 Fusion 성능을 높인다는 것이었다.
- **선택:** 이미지 단독, 센서 단독, 이미지 인코더 2종(Custom CNN·ResNet50) × 센서 분류기 2종(RF·XGBoost)의 Early Fusion 4조합을 평가했다. 팀원들이 모델 조합을 나눠 실행하고 결과를 취합했다.
- **근거:** 최종 발표 기준 실행에서 ResNet50+XGBoost Fusion은 Accuracy 0.9995, F1 0.9996, ROC-AUC 1.0000이었다. 센서 단독 RF·XGBoost도 같은 수치였고, 이미지 단독 ResNet50은 Accuracy 0.9623, F1 0.9688, ROC-AUC 0.9940이었다.
- **결론:** 팀은 발표·데모용으로 ResNet50+XGBoost Fusion을 채택했다. 최종 실행에서는 센서 단독과 지표상 차이가 없어 이미지 추가의 성능 이득을 확인하지 못했다. 이미지의 현장 설명·검수 가치는 별도 평가가 필요하다.
- **해석의 한계:** 8종 비교 실행에서는 Fusion 최고 F1이 1.0000, 센서 단독 최고 F1이 0.9996이었다. 이는 최종 발표 기준과 다른 실행이다. 매우 높은 수치만으로 현장 일반화나 멀티모달의 일관된 우위를 주장할 수 없다. 센서 피처와 라벨 생성 방식의 관계, 촬영 세션을 분리한 평가, 반복 실행이 추가로 필요하다.

## 결정 2 — Train·Validation 촬영 시간 겹침을 Train에서 제외

- **상황:** Train과 Validation의 촬영 시간대가 겹쳐 같은 촬영 환경의 단서가 평가에 영향을 줄 가능성이 있었다.
- **선택:** 겹치는 시간 구간의 Train 샘플 1,154건을 제외했다. Train은 16,347건에서 15,193건으로 줄었고 Validation 2,043건은 유지했다. 제거 후 이미지·센서 매칭률은 양쪽 모두 100%였다.
- **이유:** Validation을 유지하면서 시간상 겹치는 Train 샘플을 학습에서 제외하기 위해서다.
- **한계:** 동일 조건에서 제거 전·후 성능을 비교한 출력은 확인되지 않는다. 시간 겹침 제거만으로 라벨 생성 방식이나 촬영 세션에 따른 모든 편향이 해결됐다고 볼 수 없다.

## 근거 자료

- [최종 발표 기준 실행 노트북 (Colab, 읽기 전용)](https://colab.research.google.com/drive/1tIppNNjtLAXvdL8wcsAKu8SUCOtGAFat)
- [단독 4종·Fusion 4조합 비교 노트북 (Colab, 읽기 전용)](https://colab.research.google.com/drive/1kqYAj5BSE-90oiR6nG7jQeCLptl0XC8R)
- [`model_comparison_results.csv`](../model_comparison_results.csv): 대시보드에 포함한 별도 실행 결과이며 위 발표 수치의 출처가 아님
- [`data_quality_report.json`](../data_quality_report.json): 앱에 포함한 2,657건의 품질 점검 결과이며 전체 실험의 Train 15,193건·Validation 2,043건과 범위가 다름
