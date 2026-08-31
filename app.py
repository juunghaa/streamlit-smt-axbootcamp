"""
SMT 사전 공정(납도포) 불량 분석 - 통합 대시보드 (v2)
사용법: streamlit run app.py

사이드바 메뉴 (총 6개)
  - 개요           : 파이프라인 상태 + 2x2 실험 설계 + 6대 불량유형 요약
  - EDA            : 라벨 분포 / 센서 분포 / 상관관계 / 임베딩 산점도
  - 전처리 확인    : 매칭 현황·결측치·이상치·원본 vs 변환 후 비교
  - 모델 학습·비교 : (A) 2x2 고정 실험 결과 (B) 커스텀 학습 (C) 통합비교
  - 예측 데모      : 최종 선정 모델로 샘플조회 예측 / 신규 이미지+센서예측
  - 필터/조회      : 샘플 단위 검색 + 썸네일 그리드

[필요 파일] (앱과 같은 폴더에 두면 자동 인식 — 없어도 앱은 죽지 않고 해당 섹션만 비활성화됨)

  ■ 커스텀 학습/EDA/필터 용 (기존 파이프라인, 코랩 Cell 7-1 / Cell 9 계열)
    - fusion_features.csv                              (file_base, label, 센서피처, emb_*)
    - data_quality_report.json                         (매칭/결측치/이상치 리포트)
    - raw_sensor_sequences.json                         (file_base -> 원본 시계열)
    - sample_metadata.csv                                (file_base, defect_type, is_outlier, *_mean ...)
    - thumbnails/*.jpg                                   (file_base.jpg 썸네일)
    - all_results.json, feature_importance_all.csv        (앱 내 커스텀 학습 결과, 자동 생성/누적)

  ■ 2x2 고정 실험(이 노트북: Custom CNN/ResNet50 x RandomForest/XGBoost) 결과물
    - model_comparison_results.csv   (노트북 Cell 16 summary_df 저장본. 컬럼: 인코더,분류기,Accuracy,
                                       Precision,Recall,F1,AUC,파라미터(M),추론속도(ms/장),조합)
    - model_meta.pkl                 (노트북 Cell 19. {"encoder_name":..., "classifier_name":...})
    - fusion_model.pkl               (노트북 Cell 19. 최종 선정된 Fusion 분류기)
    - feature_cols.pkl               (노트북 Cell 19. fusion_model이 기대하는 피처 컬럼 순서)
    - best_image_model.keras         (노트북 Cell 19. 최종 선정된 이미지 인코더, 예측 데모에서 사용)
    - val_roc_data.json  (선택)      (없으면 ROC curve 탭은 자동으로 생략됩니다. 아래 스니펫으로 노트북
                                       Cell 16 마지막에 추가해서 내려받으면 4개 조합 ROC를 한 화면에서 비교 가능)
        >>> import json
        >>> roc_export = {
        ...     f"{enc}+{clf}": {
        ...         "fpr": list(map(float, roc_curve(a["val_y"], a["val_proba"])[0])),
        ...         "tpr": list(map(float, roc_curve(a["val_y"], a["val_proba"])[1])),
        ...         "auc": float(roc_auc_score(a["val_y"], a["val_proba"])),
        ...     } for (enc, clf), a in artifacts.items()
        ... }
        >>> json.dump(roc_export, open(f"{SAVE_DIR}/val_roc_data.json", "w"))

  ※ confusion matrix / SHAP / 최종 모델 예측은 fusion_model.pkl + feature_cols.pkl + fusion_features.csv
    (label 포함) 조합만 있으면 앱이 그 자리에서 직접 계산합니다. 별도 export 없이도 동작합니다.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.set_page_config(page_title="SMT 납도포 불량 분석 대시보드", layout="wide")

DATA_DIR = Path(__file__).parent
THUMB_DIR = DATA_DIR / "thumbnails"

RESULTS_PATH = DATA_DIR / "all_results.json"
IMPORTANCE_PATH = DATA_DIR / "feature_importance_all.csv"
FUSION_PATH = DATA_DIR / "fusion_features.csv"
QUALITY_PATH = DATA_DIR / "data_quality_report.json"
RAW_SEQ_PATH = DATA_DIR / "raw_sensor_sequences.json"
METADATA_PATH = DATA_DIR / "sample_metadata.csv"

MODEL_COMPARISON_PATH = DATA_DIR / "model_comparison_results.csv"
MODEL_META_PATH = DATA_DIR / "model_meta.pkl"
FUSION_MODEL_PATH = DATA_DIR / "fusion_model.pkl"
FEATURE_COLS_PATH = DATA_DIR / "feature_cols.pkl"
IMAGE_MODEL_PATH = DATA_DIR / "best_image_model.keras"
ROC_DATA_PATH = DATA_DIR / "val_roc_data.json"

SENSOR_COLS = ["temperature", "humidity", "vibration", "acceleration", "noise"]
DEFECT_LABEL_MAP = {0: "정상", 1: "불량(미납)"}

DEFECT_INFO_DF = pd.DataFrame([
    ("1. 미납 (Missing)", "패드에 솔더 전혀 미도포", "전기적 단선(Open)"),
    ("2. 납부족 (Insufficient)", "도포량/면적 부족", "냉납(Cold Joint)"),
    ("3. 납쇼트 (Short/Bridge)", "인접 패드 간 브릿지", "회로 합선"),
    ("4. 납볼 (Solder Ball)", "패드 외곽 미세 납 입자", "2차 쇼트 위험"),
    ("5. 납좌표 밀림 (Shifted)", "패드 중심 이탈", "톰스톤 현상"),
    ("6. 납형성 불량 (Deform)", "도포 형태 불규칙", "기공(Void)"),
], columns=["불량 유형", "현상", "주요 영향"])


# ══════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════════════════
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


PICKLE_LOAD_ERRORS = {}


def load_pickle(path: Path):
    """joblib.load 래퍼. 파일이 없으면 None, 로드 중 오류가 나면 오류 메시지를 PICKLE_LOAD_ERRORS에
    기록하고 None을 반환합니다 (다른 페이지가 함께 죽지 않도록 예외를 여기서 막습니다)."""
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        PICKLE_LOAD_ERRORS[path.name] = f"{type(e).__name__}: {e}"
        return None


def save_all_results(all_results: dict):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def save_importance(model_name: str, imp_df: pd.DataFrame):
    imp_df = imp_df.copy()
    imp_df["model_name"] = model_name
    if IMPORTANCE_PATH.exists():
        existing = pd.read_csv(IMPORTANCE_PATH)
        existing = existing[existing["model_name"] != model_name]
        imp_df = pd.concat([existing, imp_df], ignore_index=True)
    imp_df.to_csv(IMPORTANCE_PATH, index=False)


@st.cache_data
def load_fusion_features():
    if not FUSION_PATH.exists():
        return None
    return pd.read_csv(FUSION_PATH)


@st.cache_resource
def load_final_model_bundle():
    """fusion_model.pkl + feature_cols.pkl + model_meta.pkl 를 함께 로드. 하나라도 없으면 None."""
    meta = load_pickle(MODEL_META_PATH)
    fusion_model = load_pickle(FUSION_MODEL_PATH)
    feature_cols = load_pickle(FEATURE_COLS_PATH)
    if meta is None or fusion_model is None or feature_cols is None:
        return None
    return {"meta": meta, "fusion_model": fusion_model, "feature_cols": feature_cols}


@st.cache_resource
def load_image_model_bundle():
    """best_image_model.keras (+ ResNet50 등 전처리) 로드. tensorflow 미설치/파일 없음/로드 오류 시 None."""
    if not IMAGE_MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        return None
    try:
        image_model = keras.models.load_model(str(IMAGE_MODEL_PATH))
        feature_extractor = keras.Model(inputs=image_model.input, outputs=image_model.get_layer("feature_layer").output)
    except Exception as e:
        PICKLE_LOAD_ERRORS["best_image_model.keras"] = f"{type(e).__name__}: {e}"
        return None
    meta = load_pickle(MODEL_META_PATH) or {}
    encoder_name = meta.get("encoder_name", "resnet50")
    preprocess_fns = {
        "custom_cnn": lambda arr: arr / 255.0,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
    }
    return {
        "image_model": image_model,
        "feature_extractor": feature_extractor,
        "encoder_name": encoder_name,
        "preprocess_fn": preprocess_fns.get(encoder_name, lambda arr: arr / 255.0),
    }


def build_status_table(rows: list) -> pd.DataFrame:
    """[(파일명, 존재여부, 관련파일명 or None), ...] -> 사이드바 상태 점검용 표.
    관련파일명이 PICKLE_LOAD_ERRORS에 있으면 '로드 실패'로 표시합니다."""
    out = []
    for item in rows:
        name, ok = item[0], item[1]
        err_key = item[2] if len(item) > 2 else None
        if err_key and err_key in PICKLE_LOAD_ERRORS:
            status = "로드 실패"
        else:
            status = "연결됨" if ok else "없음"
        out.append({"파일": name, "상태": status})
    return pd.DataFrame(out)


def final_bundle_missing_message() -> str:
    """final_bundle_global이 None일 때, 파일이 아예 없는 것인지 로드 중 오류가 난 것인지 구분해서 안내."""
    if PICKLE_LOAD_ERRORS:
        detail = "; ".join(f"{k} ({v})" for k, v in PICKLE_LOAD_ERRORS.items())
        return (f"모델 파일을 찾았지만 불러오는 중 오류가 발생했습니다: {detail}. "
                "`ModuleNotFoundError: No module named 'dill'` 오류라면 터미널에서 `pip install dill` 실행 후 "
                "앱을 다시 시작해보세요.")
    return "`fusion_model.pkl`, `feature_cols.pkl`, `model_meta.pkl` 파일이 이 앱과 같은 폴더에 필요합니다."


def build_sensor_stats_from_sequence(seq: dict) -> dict:
    """원본 sensor_sequence(dict of list) -> fusion_features와 동일한 요약통계 피처로 변환."""
    row = {}
    for col in SENSOR_COLS:
        if col not in seq:
            continue
        vals = np.array(seq[col], dtype=float)
        if len(vals) == 0:
            continue
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_std"] = vals.std()
        row[f"{col}_min"] = vals.min()
        row[f"{col}_max"] = vals.max()
        row[f"{col}_first"] = vals[0]
        row[f"{col}_last"] = vals[-1]
        row[f"{col}_range"] = vals.max() - vals.min()
        x = np.arange(len(vals))
        row[f"{col}_slope"] = np.polyfit(x, vals, 1)[0] if len(vals) > 1 else 0.0
    return row


# ══════════════════════════════════════════════════════════════════════════
# 데이터 로드 (여러 페이지 공용)
# ══════════════════════════════════════════════════════════════════════════
fusion_df_global = load_fusion_features()
all_results_global = load_json(RESULTS_PATH)
importance_df_global = load_csv(IMPORTANCE_PATH)
quality_report_global = load_json(QUALITY_PATH)
metadata_global = load_csv(METADATA_PATH)
model_comparison_global = load_csv(MODEL_COMPARISON_PATH)
roc_data_global = load_json(ROC_DATA_PATH)
final_bundle_global = load_final_model_bundle()

# ══════════════════════════════════════════════════════════════════════════
# 사이드바 : 네비게이션 + 데이터 상태
# ══════════════════════════════════════════════════════════════════════════
st.sidebar.title("SMT 불량 분석")
st.sidebar.caption("사전 공정(납도포) · SPI 검사 대시보드")

PAGES = ["개요", "EDA", "전처리 확인", "모델 학습·비교 (2x2)", "예측 데모", "필터/조회"]
page = st.sidebar.radio("메뉴", PAGES, label_visibility="collapsed")

with st.sidebar.expander("데이터 파일 연결 상태", expanded=False):
    st.caption("각 페이지가 정상적으로 표시되려면 아래 파일들이 앱과 같은 폴더에 있어야 합니다. "
               "파일이 없는 섹션은 자동으로 안내 문구로 대체됩니다.")
    st.markdown("기본 데이터")
    st.dataframe(build_status_table([
        ("fusion_features.csv", fusion_df_global is not None),
        ("data_quality_report.json", quality_report_global is not None),
        ("sample_metadata.csv", metadata_global is not None),
        ("thumbnails/", THUMB_DIR.exists()),
    ]), use_container_width=True, hide_index=True)

    st.markdown("2x2 실험 결과물")
    st.dataframe(build_status_table([
        ("model_comparison_results.csv", model_comparison_global is not None),
        ("model_meta.pkl", MODEL_META_PATH.exists(), "model_meta.pkl"),
        ("fusion_model.pkl", final_bundle_global is not None, "fusion_model.pkl"),
        ("feature_cols.pkl", FEATURE_COLS_PATH.exists(), "feature_cols.pkl"),
        ("best_image_model.keras", IMAGE_MODEL_PATH.exists()),
        ("val_roc_data.json (선택)", roc_data_global is not None),
    ]), use_container_width=True, hide_index=True)

    st.markdown("앱 내 커스텀 학습")
    st.dataframe(build_status_table([
        ("all_results.json", all_results_global is not None),
    ]), use_container_width=True, hide_index=True)

    if PICKLE_LOAD_ERRORS:
        st.markdown("파일 로드 오류 상세")
        for fname, err in PICKLE_LOAD_ERRORS.items():
            st.caption(f"{fname}: {err}")
        st.caption("`ModuleNotFoundError: No module named 'dill'` 오류라면, 터미널에서 "
                   "`pip install dill` 실행 후 앱을 다시 시작하면 대부분 해결됩니다.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 0. 개요
# ════════════════════════════════════════════════════════════════════════════════
if page == "개요":
    st.title("SMT 사전 공정(납도포) 불량 분석 — 개요")
    st.caption("전체 파이프라인의 현재 상태와 데이터 구성을 한눈에 요약해서 보여주는 페이지입니다.")
    st.caption("SMT 전체 불량의 60~70%가 납도포 단계에서 발생합니다. 이 대시보드는 센서(공정) + 이미지(SPI) 데이터를 "
               "결합해 불량을 사전에 탐지하는 파이프라인을 실험/모니터링합니다.")

    st.subheader("파이프라인 현황")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("병합 데이터셋", f"{len(fusion_df_global)} 건" if fusion_df_global is not None else "없음")
    k2.metric("커스텀 학습 모델 수", len(all_results_global) if all_results_global else 0)
    if model_comparison_global is not None:
        best_idx = model_comparison_global["F1"].idxmax() if "F1" in model_comparison_global.columns else 0
        best_combo = model_comparison_global.loc[best_idx]
        k3.metric("2x2 최고 조합", best_combo.get("조합", "-"), f"F1 {best_combo.get('F1', 0):.3f}")
    else:
        k3.metric("2x2 최고 조합", "-")
    if final_bundle_global is not None:
        m = final_bundle_global["meta"]
        k4.metric("최종 선정 모델", f"{m.get('encoder_name','?')}+{m.get('classifier_name','?')}")
    else:
        k4.metric("최종 선정 모델", "-")
    if quality_report_global:
        k5.metric("이상치 샘플", quality_report_global.get("total_outlier_samples", "-"))
    else:
        k5.metric("이상치 샘플", "-")

    st.divider()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("6대 불량 유형")
        st.dataframe(DEFECT_INFO_DF, use_container_width=True, hide_index=True)
        st.caption("※ 현재 학습 데이터는 '미납' 이진 분류 기준이며, 구조상 다중분류(6종)로 확장 가능합니다.")

        st.subheader("2x2 실험 설계")
        design_df = pd.DataFrame(
            {"": ["Custom CNN", "ResNet50"], "Random Forest": ["조합 1", "조합 3"], "XGBoost": ["조합 2", "조합 4"]}
        ).set_index("")
        st.dataframe(design_df, use_container_width=True)
        st.caption("이미지 인코더 2종 × Fusion 분류기 2종 = 4가지 조합을 전부 학습·평가해 최적 조합을 실측으로 선정 "
                   "(Early Fusion: 이미지 임베딩 + 센서 요약피처를 하나로 합쳐 하나의 분류기로 학습).")
    with c2:
        st.subheader("현재 데이터 구성")
        if fusion_df_global is not None:
            id_cols = [c for c in ["file_base", "label", "img_label"] if c in fusion_df_global.columns]
            sensor_n = len([c for c in fusion_df_global.columns if c not in id_cols and not c.startswith("emb_")])
            image_n = len([c for c in fusion_df_global.columns if c.startswith("emb_")])
            fig = px.pie(
                pd.DataFrame({"구분": ["센서 피처", "이미지 임베딩"], "개수": [sensor_n, image_n]}),
                names="구분", values="개수", hole=0.55,
                color="구분", color_discrete_map={"센서 피처": "#4C78A8", "이미지 임베딩": "#F58518"},
            )
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            if "label" in fusion_df_global.columns:
                lab_counts = fusion_df_global["label"].map(DEFECT_LABEL_MAP).value_counts().reset_index()
                lab_counts.columns = ["라벨", "개수"]
                fig2 = px.bar(lab_counts, x="라벨", y="개수", color="라벨", text_auto=True,
                              color_discrete_map={"정상": "#4C78A8", "불량(미납)": "#E45756"})
                fig2.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("fusion_features.csv 를 넣으면 여기에 피처/라벨 구성이 표시됩니다.")

    st.divider()
    st.info("왼쪽 사이드바에서 페이지를 이동하세요. **EDA**에서 데이터 특성을 살펴보고, **전처리 확인**에서 원본 대비 "
            "변환 품질을 점검하고, **모델 학습·비교(2x2)**에서 4가지 조합의 성능을 비교한 뒤, **예측 데모**에서 최종 "
            "모델로 직접 예측해볼 수 있습니다.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1. EDA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("탐색적 데이터 분석 (EDA)")
    st.caption("병합 데이터(fusion_features.csv)를 기준으로 센서 피처의 분포, 상관관계, 이미지 임베딩 구조를 살펴보는 페이지입니다.")
    fusion_df = fusion_df_global

    if fusion_df is None:
        st.error(f"`{FUSION_PATH.name}` 파일이 필요합니다.")
    else:
        id_cols = [c for c in ["file_base", "label", "img_label"] if c in fusion_df.columns]
        sensor_mean_cols = [c for c in fusion_df.columns if c.endswith("_mean") and not c.startswith("emb_")]
        all_sensor_cols = [c for c in fusion_df.columns if c not in id_cols and not c.startswith("emb_")]
        emb_cols = [c for c in fusion_df.columns if c.startswith("emb_")]
        has_label = "label" in fusion_df.columns

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("전체 샘플", len(fusion_df))
        k2.metric("센서 피처 수", len(all_sensor_cols))
        k3.metric("이미지 임베딩 차원", len(emb_cols))
        k4.metric("결측치 셀 수", int(fusion_df[all_sensor_cols].isna().sum().sum()) if all_sensor_cols else 0)

        st.divider()
        tab1, tab2, tab3, tab4 = st.tabs(["센서 분포", "상관관계", "이미지 임베딩(PCA)", "요약 통계"])

        with tab1:
            st.subheader("라벨별 센서 피처 분포")
            if sensor_mean_cols:
                c1, c2 = st.columns([1, 3])
                with c1:
                    picked_feat = st.selectbox("피처 선택", sensor_mean_cols)
                    chart_kind = st.radio("그래프 종류", ["히스토그램", "박스플롯"], horizontal=False)
                with c2:
                    plot_df = fusion_df.copy()
                    if has_label:
                        plot_df["라벨"] = plot_df["label"].map(DEFECT_LABEL_MAP)
                        color_arg = "라벨"
                    else:
                        color_arg = None
                    if chart_kind == "히스토그램":
                        fig = px.histogram(plot_df, x=picked_feat, color=color_arg, barmode="overlay", opacity=0.6,
                                            color_discrete_map={"정상": "#4C78A8", "불량(미납)": "#E45756"})
                    else:
                        fig = px.box(plot_df, x=color_arg, y=picked_feat, color=color_arg,
                                     color_discrete_map={"정상": "#4C78A8", "불량(미납)": "#E45756"})
                    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("`_mean`으로 끝나는 센서 요약 피처가 없습니다.")

        with tab2:
            st.subheader("센서 피처 상관관계 히트맵")
            corr_cols = sensor_mean_cols if sensor_mean_cols else all_sensor_cols
            if len(corr_cols) >= 2:
                corr = fusion_df[corr_cols].corr()
                fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
                fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

                if has_label:
                    st.subheader("라벨과의 상관관계 (절댓값 상위)")
                    label_corr = fusion_df[all_sensor_cols + ["label"]].corr()["label"].drop("label")
                    label_corr = label_corr.reindex(label_corr.abs().sort_values(ascending=False).index).head(15)
                    fig2 = px.bar(label_corr.reset_index().rename(columns={"index": "피처", "label": "상관계수"}),
                                  x="상관계수", y="피처", orientation="h", color="상관계수", color_continuous_scale="RdBu_r")
                    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("상관관계를 계산할 센서 피처가 부족합니다.")

        with tab3:
            st.subheader("이미지 임베딩 2D 시각화 (PCA)")
            if len(emb_cols) >= 2:
                n_show = st.slider("표시할 샘플 수(속도용)", 50, min(2000, len(fusion_df)), min(500, len(fusion_df)))
                sample_df = fusion_df.sample(n=min(n_show, len(fusion_df)), random_state=42)
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(sample_df[emb_cols])
                plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1]})
                if has_label:
                    plot_df["라벨"] = sample_df["label"].map(DEFECT_LABEL_MAP).values
                    color_arg = "라벨"
                else:
                    color_arg = None
                fig = px.scatter(plot_df, x="PC1", y="PC2", color=color_arg, opacity=0.7,
                                  color_discrete_map={"정상": "#4C78A8", "불량(미납)": "#E45756"})
                fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"PCA 2개 성분이 설명하는 분산 비율: {pca.explained_variance_ratio_.sum()*100:.1f}%")
            else:
                st.info("이미지 임베딩(emb_ 로 시작하는 컬럼)이 없습니다.")

        with tab4:
            st.subheader("피처 요약 통계")
            describe_cols = all_sensor_cols if all_sensor_cols else fusion_df.columns.tolist()
            st.dataframe(fusion_df[describe_cols].describe().T, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2. 전처리 확인
# ════════════════════════════════════════════════════════════════════════════════
elif page == "전처리 확인":
    st.title("전처리 확인")
    st.caption("이미지-센서 매칭 현황, 결측치·이상치 점검 결과, 그리고 원본 시계열이 요약 통계 피처로 어떻게 "
               "변환됐는지 샘플 단위로 확인하는 페이지입니다.")
    quality_report = quality_report_global
    raw_sequences = load_json(RAW_SEQ_PATH)
    sample_metadata = metadata_global

    if quality_report is None or sample_metadata is None:
        st.error("`data_quality_report.json`, `sample_metadata.csv` 파일이 필요합니다.")
    else:
        st.subheader("데이터 매칭 및 품질 개요")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("센서 JSON 수", quality_report["total_sensor_json"])
        c2.metric("이미지 수", quality_report["total_images"])
        c3.metric("매칭된 샘플", quality_report["matched_samples"])
        c4.metric("센서만 있음", quality_report["sensor_only_unmatched"])
        c5.metric("이미지만 있음", quality_report["image_only_unmatched"])

        if quality_report["sensor_only_unmatched"] > 0 or quality_report["image_only_unmatched"] > 0:
            st.warning("센서 JSON과 이미지가 file_base 기준으로 짝지어지지 않은 샘플이 있습니다. 폴더/파일명 매칭을 다시 확인해보세요.")

        st.divider()
        col_missing, col_outlier = st.columns(2)

        with col_missing:
            st.subheader("결측치")
            if quality_report["samples_with_any_missing"] == 0:
                st.success("결측치가 있는 샘플이 없습니다.")
            else:
                st.error(f"결측치가 있는 샘플: {quality_report['samples_with_any_missing']}개")
                miss_df = pd.DataFrame(
                    quality_report["features_with_missing"].items(), columns=["피처", "결측 개수"]
                ).sort_values("결측 개수", ascending=False)
                st.dataframe(miss_df, use_container_width=True, hide_index=True)

        with col_outlier:
            st.subheader("이상치 (IQR 기준)")
            st.metric("이상치로 판정된 샘플 수", quality_report["total_outlier_samples"])
            outlier_top = pd.DataFrame(
                quality_report["outlier_per_feature_top10"].items(), columns=["피처", "이상치 개수"]
            )
            fig_out = px.bar(outlier_top, x="이상치 개수", y="피처", orientation="h")
            fig_out.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_out, use_container_width=True)

        st.divider()
        st.subheader("전처리 전/후 비교 (샘플 단위)")
        st.caption("왼쪽: 원본 센서 시계열 (변환 전) / 오른쪽: 추출된 요약 통계 피처 (변환 후)")

        sample_options = sample_metadata["file_base"].tolist()
        picked = st.selectbox("샘플 선택 (file_base)", sample_options)
        row = sample_metadata[sample_metadata["file_base"] == picked].iloc[0]

        outlier_status = "이상치" if row["is_outlier"] else "정상 범위"
        c1, c2, c3 = st.columns([2, 2, 1])
        c1.markdown(f"**판정:** {row['defect_type']}")
        c2.markdown(f"**이상치 스코어:** {row['outlier_score']} ({outlier_status})")
        thumb_path = THUMB_DIR / f"{picked}.jpg"
        if thumb_path.exists():
            c3.image(str(thumb_path), width=110)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**변환 전: 원본 센서 시계열**")
            if raw_sequences and picked in raw_sequences:
                seq = raw_sequences[picked]
                fig_raw = go.Figure()
                for col in SENSOR_COLS:
                    if col in seq:
                        fig_raw.add_trace(go.Scatter(y=seq[col], mode="lines", name=col))
                fig_raw.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="시점", yaxis_title="측정값")
                st.plotly_chart(fig_raw, use_container_width=True)
            else:
                st.info("이 샘플의 원본 시계열 데이터를 찾을 수 없습니다.")

        with c2:
            st.markdown("**변환 후: 요약 통계 피처 (mean 기준)**")
            mean_cols = [c for c in sample_metadata.columns if c.endswith("_mean")]
            if mean_cols:
                after_df = pd.DataFrame({"피처": mean_cols, "값": [row[c] for c in mean_cols]})
                fig_after = px.bar(after_df, x="값", y="피처", orientation="h")
                fig_after.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_after, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3. 모델 학습·비교 (2x2)
# ════════════════════════════════════════════════════════════════════════════════
elif page == "모델 학습·비교 (2x2)":
    st.title("모델 학습·비교")
    st.caption("이미지 인코더(Custom CNN/ResNet50) × Fusion 분류기(RandomForest/XGBoost) 2x2 실험 결과를 비교하고, "
               "필요하면 다른 설정으로 직접 새 모델을 학습해서 함께 비교하는 페이지입니다.")
    tab_grid, tab_train, tab_all = st.tabs(["2x2 고정 실험 결과", "커스텀 학습", "통합 비교"])

    # -------------------- (A) 2x2 고정 실험 결과 --------------------
    with tab_grid:
        if model_comparison_global is None:
            st.info(f"`{MODEL_COMPARISON_PATH.name}` 파일이 없습니다. 노트북 Cell 16의 `summary_df`를 "
                    "`summary_df.to_csv('model_comparison_results.csv', index=False)` 로 저장해 이 폴더에 넣어주세요.")
        else:
            cdf = model_comparison_global.copy()
            best_row = cdf.sort_values("F1", ascending=False).iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("최고 조합", best_row.get("조합", "-"))
            k2.metric("F1", f"{best_row['F1']:.3f}")
            k3.metric("AUC", f"{best_row['AUC']:.3f}")
            k4.metric("Accuracy", f"{best_row['Accuracy']:.3f}")

            st.divider()
            left, right = st.columns([1.3, 1])
            with left:
                st.subheader("조합별 성능표 (Validation 기준)")
                fmt_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1", "AUC"] if c in cdf.columns]
                st.dataframe(
                    cdf.style.format({c: "{:.3f}" for c in fmt_cols}).background_gradient(cmap="Blues", subset=["F1"]),
                    use_container_width=True, hide_index=True,
                )
                metric_pick = st.radio("히트맵 지표", fmt_cols, horizontal=True, key="grid_metric")
                if "인코더" in cdf.columns and "분류기" in cdf.columns:
                    pivot = cdf.pivot(index="인코더", columns="분류기", values=metric_pick)
                    fig = px.imshow(pivot, text_auto=".3f", color_continuous_scale="Blues", aspect="auto")
                    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            with right:
                st.subheader("지표별 막대비교")
                metric_bar = st.radio("지표 선택", fmt_cols, horizontal=True, key="bar_metric")
                fig_bar = px.bar(cdf.sort_values(metric_bar), x=metric_bar, y="조합", orientation="h",
                                  text_auto=".3f", color=metric_bar, color_continuous_scale="Blues")
                fig_bar.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

                if "파라미터(M)" in cdf.columns and "추론속도(ms/장)" in cdf.columns:
                    st.subheader("효율성 (파라미터 수 / 추론속도)")
                    fig_eff = px.scatter(cdf, x="파라미터(M)", y="추론속도(ms/장)", color="인코더", size="AUC",
                                          hover_name="조합", text="조합")
                    fig_eff.update_traces(textposition="top center")
                    fig_eff.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_eff, use_container_width=True)

            if "인코더" in cdf.columns and "분류기" in cdf.columns:
                st.divider()
                st.subheader("주효과 분석 (2x2 요인설계) — 어느 축이 성능에 더 큰 영향을 줬는가")
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown("**인코더별 평균 성능**")
                    st.dataframe(cdf.groupby("인코더")[fmt_cols].mean().round(4), use_container_width=True)
                with m2:
                    st.markdown("**분류기별 평균 성능**")
                    st.dataframe(cdf.groupby("분류기")[fmt_cols].mean().round(4), use_container_width=True)

            if roc_data_global is not None:
                st.divider()
                st.subheader("ROC Curve 비교 (4개 조합)")
                fig_roc = go.Figure()
                for combo_name, d in roc_data_global.items():
                    fig_roc.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                                                  name=f"{combo_name} (AUC={d.get('auc', 0):.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="랜덤 기준선",
                                              line=dict(dash="dash", color="gray")))
                fig_roc.update_layout(height=460, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                                       margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.caption("`val_roc_data.json` 을 추가하면 4개 조합의 ROC Curve를 한 화면에서 비교할 수 있습니다. "
                           "(상단 파일 안내 참고)")

            st.divider()
            st.subheader("최종 선정 모델 상세 분석 (혼동행렬 · SHAP)")
            if final_bundle_global is None:
                st.info(f"이 자리에서 혼동행렬과 SHAP 피처중요도를 계산하려면 최종 모델이 필요합니다. {final_bundle_missing_message()}")
            elif fusion_df_global is None or "label" not in fusion_df_global.columns:
                st.info("`fusion_features.csv`(label 포함)가 있어야 혼동행렬/SHAP을 계산할 수 있습니다.")
            else:
                meta = final_bundle_global["meta"]
                fusion_model = final_bundle_global["fusion_model"]
                feature_cols = final_bundle_global["feature_cols"]
                missing_cols = [c for c in feature_cols if c not in fusion_df_global.columns]
                if missing_cols:
                    st.warning(f"fusion_features.csv에 모델이 기대하는 피처가 {len(missing_cols)}개 없습니다. "
                               f"(예: {missing_cols[:5]}) — 계산을 건너뜁니다.")
                else:
                    st.caption(f"현재 로드된 모델: **{meta.get('encoder_name','?')} + {meta.get('classifier_name','?')}** "
                               "· 아래 결과는 `fusion_features.csv`에 있는 데이터 전체를 기준으로 계산됩니다.")
                    X_eval = fusion_df_global[feature_cols]
                    y_eval = fusion_df_global["label"]
                    pred = fusion_model.predict(X_eval)
                    proba = fusion_model.predict_proba(X_eval)[:, 1]

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{accuracy_score(y_eval, pred):.3f}")
                    m2.metric("Precision", f"{precision_score(y_eval, pred, zero_division=0):.3f}")
                    m3.metric("Recall", f"{recall_score(y_eval, pred, zero_division=0):.3f}")
                    m4.metric("F1", f"{f1_score(y_eval, pred):.3f}")
                    m5.metric("AUC", f"{roc_auc_score(y_eval, proba):.3f}")

                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**혼동행렬**")
                        cm = confusion_matrix(y_eval, pred)
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm, x=["예측: 정상", "예측: 불량"], y=["실제: 정상", "실제: 불량"],
                            text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
                        ))
                        fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with d2:
                        st.markdown("**SHAP 기반 센서 vs 이미지 기여도**")
                        try:
                            import shap
                            explainer = shap.TreeExplainer(fusion_model)
                            shap_values = explainer.shap_values(X_eval)
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]
                            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                                shap_values = shap_values[:, :, 1]
                            mean_abs_shap = np.abs(shap_values).mean(axis=0)
                            shap_importance = pd.Series(mean_abs_shap, index=feature_cols)
                            sensor_only = [c for c in feature_cols if not c.startswith("emb_")]
                            image_only = [c for c in feature_cols if c.startswith("emb_")]
                            contrib = pd.DataFrame({
                                "구분": ["센서", "이미지(임베딩)"],
                                "SHAP 중요도 합": [shap_importance[sensor_only].sum() if sensor_only else 0,
                                                shap_importance[image_only].sum() if image_only else 0],
                            })
                            fig_contrib = px.pie(contrib, names="구분", values="SHAP 중요도 합", hole=0.5,
                                                  color="구분", color_discrete_map={"센서": "#4C78A8", "이미지(임베딩)": "#F58518"})
                            fig_contrib.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_contrib, use_container_width=True)

                            st.markdown("**상위 SHAP 피처 (전역 중요도)**")
                            top_shap = shap_importance.sort_values(ascending=True).tail(15).reset_index()
                            top_shap.columns = ["feature", "importance"]
                            fig_shap_bar = px.bar(top_shap, x="importance", y="feature", orientation="h")
                            fig_shap_bar.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                            st.plotly_chart(fig_shap_bar, use_container_width=True)
                        except ImportError:
                            st.warning("`shap` 패키지가 설치되어 있지 않습니다. `pip install shap` 후 다시 시도하세요.")

    # -------------------- (B) 커스텀 학습 --------------------
    with tab_train:
        fusion_df = fusion_df_global
        if fusion_df is None:
            st.error(f"`{FUSION_PATH.name}` 파일이 필요합니다. 코랩에서 저장한 fusion_features.csv를 이 앱과 같은 폴더에 넣어주세요.")
        else:
            id_cols = [c for c in ["file_base", "label", "img_label"] if c in fusion_df.columns]
            all_feature_cols = [c for c in fusion_df.columns if c not in id_cols]
            sensor_cols_all = [c for c in all_feature_cols if not c.startswith("emb_")]
            image_cols_all = [c for c in all_feature_cols if c.startswith("emb_")]

            s1, s2, s3 = st.columns(3)
            s1.metric("전체 샘플", len(fusion_df))
            s2.metric("센서 피처", len(sensor_cols_all))
            s3.metric("이미지 피처", len(image_cols_all))

            left, right = st.columns([1, 1.4])
            with left:
                st.subheader("① 기본 설정")
                algo = st.selectbox("알고리즘", ["XGBoost", "RandomForest", "LogisticRegression"])
                feature_set = st.selectbox("사용할 피처", ["전체(센서+이미지)", "센서만", "이미지만"])
                model_name = st.text_input("모델 이름", value=f"{algo}_{feature_set.split('(')[0]}")

                st.subheader("② 데이터 분할 / 검증")
                test_size = st.slider("Test set 비율", 0.1, 0.4, 0.2, step=0.05)
                random_state = st.number_input("random_state", value=42, step=1)
                cv_folds = st.slider("교차검증 fold 수", 3, 10, 5)
                shuffle_cv = st.checkbox("CV 시 셔플", value=True)

                st.subheader("③ 클래스 불균형 / 임계값")
                balance_classes = st.checkbox("클래스 불균형 자동 보정", value=False)
                decision_threshold = st.slider("분류 임계값", 0.05, 0.95, 0.5, step=0.05)

                st.subheader("④ 이미지 임베딩 PCA (선택)")
                use_pca = st.checkbox("PCA로 이미지 임베딩 축소", value=False)
                pca_components = None
                if use_pca and len(image_cols_all) > 0:
                    pca_components = st.slider("PCA 성분 수", 2, min(100, len(image_cols_all)), 20)

                with st.expander("하이퍼파라미터 세부 조정", expanded=False):
                    if algo == "XGBoost":
                        n_estimators = st.slider("n_estimators", 50, 800, 300, step=50)
                        max_depth = st.slider("max_depth", 2, 12, 4)
                        learning_rate = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], value=0.05)
                        subsample = st.slider("subsample", 0.5, 1.0, 0.8, step=0.05)
                        colsample_bytree = st.slider("colsample_bytree", 0.5, 1.0, 0.8, step=0.05)
                        min_child_weight = st.slider("min_child_weight", 1, 10, 1)
                        reg_lambda = st.select_slider("reg_lambda (L2)", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
                    elif algo == "RandomForest":
                        n_estimators = st.slider("n_estimators", 50, 800, 300, step=50)
                        max_depth = st.slider("max_depth", 2, 30, 8)
                        min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 1)
                        max_features = st.select_slider("max_features", ["sqrt", "log2", None], value="sqrt")
                    else:
                        c_value = st.select_slider("C (규제 강도)", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
                        penalty = st.selectbox("penalty", ["l2", "l1"])
                        solver = "liblinear" if penalty == "l1" else "lbfgs"

            if feature_set == "전체(센서+이미지)":
                feature_cols = all_feature_cols
            elif feature_set == "센서만":
                feature_cols = sensor_cols_all
            else:
                feature_cols = image_cols_all

            with right:
                st.subheader("⑤ 학습 실행")
                st.caption(f"선택된 피처 수: **{len(feature_cols)}개**  |  피처셋: {feature_set}")
                run = st.button("학습 시작", type="primary", use_container_width=True)

                if run:
                    with st.spinner(f"{model_name} 학습 중..."):
                        X = fusion_df[feature_cols].copy()
                        y = fusion_df["label"]

                        used_image_cols = [c for c in feature_cols if c in image_cols_all]
                        used_sensor_cols = [c for c in feature_cols if c in sensor_cols_all]
                        if use_pca and pca_components and len(used_image_cols) > 0:
                            pca = PCA(n_components=pca_components, random_state=random_state)
                            emb_reduced = pca.fit_transform(X[used_image_cols])
                            emb_cols_new = [f"emb_pca_{i}" for i in range(pca_components)]
                            X = pd.concat(
                                [X[used_sensor_cols].reset_index(drop=True),
                                 pd.DataFrame(emb_reduced, columns=emb_cols_new)],
                                axis=1,
                            )
                            feature_cols_used = used_sensor_cols + emb_cols_new
                            explained = pca.explained_variance_ratio_.sum()
                        else:
                            feature_cols_used = feature_cols
                            explained = None

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y
                        )
                        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

                        if algo == "XGBoost":
                            model = xgb.XGBClassifier(
                                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                subsample=subsample, colsample_bytree=colsample_bytree,
                                min_child_weight=min_child_weight, reg_lambda=reg_lambda,
                                scale_pos_weight=pos_weight if balance_classes else 1.0,
                                eval_metric="logloss", random_state=random_state,
                            )
                        elif algo == "RandomForest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                max_features=max_features, class_weight="balanced" if balance_classes else None,
                                random_state=random_state,
                            )
                        else:
                            model = make_pipeline(
                                StandardScaler(),
                                LogisticRegression(C=c_value, penalty=penalty, solver=solver, max_iter=3000,
                                                    class_weight="balanced" if balance_classes else None),
                            )

                        model.fit(X_train, y_train)
                        proba = model.predict_proba(X_test)[:, 1]
                        pred = (proba >= decision_threshold).astype(int)

                        skf = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle_cv,
                                               random_state=random_state if shuffle_cv else None)
                        cv_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

                        if algo == "LogisticRegression":
                            coefs = np.abs(model.named_steps["logisticregression"].coef_[0])
                            imp = pd.Series(coefs, index=feature_cols_used).sort_values(ascending=False)
                        else:
                            imp = pd.Series(model.feature_importances_, index=feature_cols_used).sort_values(ascending=False)

                        sensor_in_set = [c for c in feature_cols_used if not c.startswith("emb_")]
                        image_in_set = [c for c in feature_cols_used if c.startswith("emb_")]

                        entry = {
                            "model_name": model_name,
                            "accuracy": float(accuracy_score(y_test, pred)),
                            "f1": float(f1_score(y_test, pred)),
                            "precision": float(precision_score(y_test, pred, zero_division=0)),
                            "recall": float(recall_score(y_test, pred, zero_division=0)),
                            "auc": float(roc_auc_score(y_test, proba)),
                            "cv_auc_scores": [float(s) for s in cv_scores],
                            "cv_auc_mean": float(cv_scores.mean()),
                            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
                            "confusion_matrix_labels": ["정상", "불량"],
                            "sensor_importance_sum": float(imp[sensor_in_set].sum()) if sensor_in_set else 0.0,
                            "image_importance_sum": float(imp[image_in_set].sum()) if image_in_set else 0.0,
                            "n_train": int(len(X_train)), "n_test": int(len(X_test)), "n_total": int(len(X)),
                            "threshold": decision_threshold,
                            "used_pca": bool(use_pca and pca_components),
                            "pca_explained_variance": float(explained) if explained is not None else None,
                        }

                        all_results = load_json(RESULTS_PATH) or {}
                        all_results[model_name] = entry
                        save_all_results(all_results)

                        imp_df = imp.reset_index()
                        imp_df.columns = ["feature", "importance"]
                        imp_df["feature_type"] = imp_df["feature"].apply(lambda x: "이미지(ResNet)" if x.startswith("emb_") else "센서")
                        save_importance(model_name, imp_df.head(30))

                    st.success(f"'{model_name}' 학습 완료!")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Accuracy", f"{entry['accuracy']*100:.1f}%")
                    m2.metric("Precision", f"{entry['precision']:.3f}")
                    m3.metric("Recall", f"{entry['recall']:.3f}")
                    m4.metric("F1", f"{entry['f1']:.3f}")
                    m5.metric("AUC (5-fold CV)", f"{entry['cv_auc_mean']:.3f}")
                    if entry["used_pca"]:
                        st.caption(f"PCA 적용됨 · 설명된 분산 비율: {entry['pca_explained_variance']*100:.1f}%")
                else:
                    st.info("왼쪽에서 설정을 조정한 뒤 '학습 시작'을 누르세요. 결과는 자동 저장되어 [통합 비교] 탭에 누적됩니다.")

    # -------------------- (C) 통합 비교 --------------------
    with tab_all:
        all_results = all_results_global
        importance_df = importance_df_global
        if all_results is None or importance_df is None:
            st.info("아직 커스텀 학습된 모델이 없습니다. '커스텀 학습' 탭에서 먼저 모델을 학습해주세요.")
        else:
            model_names = list(all_results.keys())
            compare_df = pd.DataFrame(
                [{"모델": name, "Accuracy": r["accuracy"], "F1": r["f1"], "Precision": r.get("precision"),
                  "Recall": r.get("recall"), "AUC (Holdout)": r["auc"], "AUC (5-fold CV)": r["cv_auc_mean"]}
                 for name, r in all_results.items()]
            ).sort_values("AUC (5-fold CV)", ascending=False)

            best_row = compare_df.iloc[0]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("최고 커스텀 모델", best_row["모델"])
            k2.metric("AUC (5-fold CV)", f"{best_row['AUC (5-fold CV)']:.3f}")
            k3.metric("F1", f"{best_row['F1']:.3f}")
            k4.metric("학습된 모델 수", len(model_names))

            def safe_fmt(x):
                return "-" if pd.isna(x) else f"{x:.3f}"

            st.dataframe(
                compare_df.style.format({c: safe_fmt for c in ["Accuracy", "F1", "Precision", "Recall", "AUC (Holdout)", "AUC (5-fold CV)"]})
                .background_gradient(cmap="Blues", subset=["AUC (5-fold CV)"]),
                use_container_width=True, hide_index=True,
            )

            metric_for_chart = st.radio("비교 지표", ["Accuracy", "F1", "Precision", "Recall", "AUC (Holdout)", "AUC (5-fold CV)"], horizontal=True)
            fig_compare = px.bar(compare_df.sort_values(metric_for_chart), x=metric_for_chart, y="모델", orientation="h",
                                  text_auto=".3f", color=metric_for_chart, color_continuous_scale="Blues")
            fig_compare.update_layout(height=max(280, len(model_names) * 50), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_compare, use_container_width=True)

            st.divider()
            selected_model = st.selectbox("모델 상세 보기", model_names)
            r = all_results[selected_model]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
            c2.metric("Precision", f"{r.get('precision', 0):.3f}")
            c3.metric("Recall", f"{r.get('recall', 0):.3f}")
            c4.metric("F1", f"{r['f1']:.3f}")
            c5.metric("AUC (5-fold CV)", f"{r['cv_auc_mean']:.3f}")

            d1, d2 = st.columns(2)
            with d1:
                cm = r["confusion_matrix"]
                labels = r.get("confusion_matrix_labels", ["정상", "불량"])
                fig_cm = go.Figure(data=go.Heatmap(z=cm, x=[f"예측: {l}" for l in labels], y=[f"실제: {l}" for l in labels],
                                                     text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
                fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_cm, use_container_width=True)
            with d2:
                contrib = pd.DataFrame({"구분": ["센서", "이미지(ResNet)"],
                                         "중요도 합": [r["sensor_importance_sum"], r["image_importance_sum"]]})
                fig_contrib = px.pie(contrib, names="구분", values="중요도 합", hole=0.5, color="구분",
                                      color_discrete_map={"센서": "#4C78A8", "이미지(ResNet)": "#F58518"})
                fig_contrib.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_contrib, use_container_width=True)

            model_imp = importance_df[importance_df["model_name"] == selected_model].sort_values("importance", ascending=True)
            if len(model_imp) > 0:
                top_n = st.slider("표시할 피처 개수", 5, len(model_imp), min(15, len(model_imp)))
                fig_bar = px.bar(model_imp.tail(top_n), x="importance", y="feature", color="feature_type", orientation="h",
                                  color_discrete_map={"센서": "#4C78A8", "이미지(ResNet)": "#F58518"})
                fig_bar.update_layout(height=max(320, top_n * 22), margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4. 예측 데모
# ════════════════════════════════════════════════════════════════════════════════
elif page == "예측 데모":
    st.title("최종 선정 모델로 예측 데모")
    st.caption("2x2 실험에서 최종 선정된 모델을 사용해, 저장된 샘플을 조회하거나 새 이미지·센서 데이터를 입력해 "
               "불량 확률을 직접 예측해보는 페이지입니다.")

    if final_bundle_global is None:
        st.error(final_bundle_missing_message())
    else:
        meta = final_bundle_global["meta"]
        fusion_model = final_bundle_global["fusion_model"]
        feature_cols = final_bundle_global["feature_cols"]
        st.success(f"로딩된 최종 모델: **{meta.get('encoder_name','?')} + {meta.get('classifier_name','?')}** "
                   f"(피처 {len(feature_cols)}개)")

        mode = st.radio("예측 방식", ["저장된 데이터에서 샘플 선택", "새 이미지 + 센서 데이터 업로드"], horizontal=True)
        st.divider()

        # ---- 모드 A: 기존 데이터에서 조회 ----
        if mode == "저장된 데이터에서 샘플 선택":
            if fusion_df_global is None:
                st.info("`fusion_features.csv`가 있어야 저장된 샘플을 조회할 수 있습니다.")
            else:
                missing_cols = [c for c in feature_cols if c not in fusion_df_global.columns]
                if missing_cols:
                    st.warning(f"fusion_features.csv에 모델이 기대하는 피처가 없습니다. (예: {missing_cols[:5]})")
                else:
                    id_col = "file_base" if "file_base" in fusion_df_global.columns else None
                    options = fusion_df_global[id_col].tolist() if id_col else list(range(len(fusion_df_global)))
                    picked = st.selectbox("샘플 선택", options)
                    row_df = fusion_df_global[fusion_df_global[id_col] == picked] if id_col else fusion_df_global.iloc[[picked]]
                    X_row = row_df[feature_cols]

                    proba = fusion_model.predict_proba(X_row)[0, 1]
                    pred_label = DEFECT_LABEL_MAP[int(proba >= 0.5)]

                    c1, c2 = st.columns([1, 2])
                    with c1:
                        thumb_path = THUMB_DIR / f"{picked}.jpg"
                        if thumb_path.exists():
                            st.image(str(thumb_path), width=200)
                        st.metric("불량 확률", f"{proba*100:.1f}%")
                        st.markdown(f"### 예측 결과: {pred_label}")
                        if "label" in row_df.columns:
                            actual = DEFECT_LABEL_MAP[int(row_df['label'].iloc[0])]
                            st.caption(f"실제 라벨: {actual}")
                    with c2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=proba * 100,
                            gauge={"axis": {"range": [0, 100]},
                                   "bar": {"color": "#E45756" if proba >= 0.5 else "#4C78A8"},
                                   "steps": [{"range": [0, 50], "color": "#EAF2FB"}, {"range": [50, 100], "color": "#FBEAEA"}]},
                            title={"text": "불량(미납) 확률(%)"},
                        ))
                        fig_gauge.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
                        st.plotly_chart(fig_gauge, use_container_width=True)

        # ---- 모드 B: 새 이미지 + 센서 업로드 ----
        else:
            image_bundle = load_image_model_bundle()
            st.caption("이미지 파일 + 원본 센서 시퀀스 JSON(`sensor_data[0].sensor_sequence` 구조)을 올리면, "
                       "요약통계 변환 → 이미지 임베딩 추출 → Fusion 모델 예측까지 한 번에 수행합니다.")
            c1, c2 = st.columns(2)
            with c1:
                img_file = st.file_uploader("SPI 이미지 업로드", type=["jpg", "jpeg", "png"])
                if img_file is not None:
                    st.image(img_file, width=200)
            with c2:
                json_file = st.file_uploader("센서 시퀀스 JSON 업로드", type=["json"])

            if image_bundle is None:
                st.warning("`best_image_model.keras` 를 불러올 수 없습니다 (파일 없음 또는 tensorflow 미설치). "
                           "이미지 임베딩 없이는 예측할 수 없습니다.")
            elif st.button("예측 실행", type="primary"):
                if img_file is None or json_file is None:
                    st.error("이미지와 센서 JSON을 모두 업로드해주세요.")
                else:
                    try:
                        import tensorflow as tf
                        from PIL import Image

                        raw = json.load(json_file)
                        seq = raw.get("sensor_data", [{}])[0].get("sensor_sequence", raw) if "sensor_data" in raw else raw
                        sensor_row = build_sensor_stats_from_sequence(seq)

                        img = Image.open(img_file).convert("RGB").resize((128, 128))
                        arr = np.array(img, dtype=np.float32)[None, ...]
                        arr = image_bundle["preprocess_fn"](arr)
                        emb = image_bundle["feature_extractor"].predict(arr, verbose=0)[0]
                        emb_row = {f"emb_{i}": float(v) for i, v in enumerate(emb)}

                        full_row = {**sensor_row, **emb_row}
                        missing = [c for c in feature_cols if c not in full_row]
                        if missing:
                            st.error(f"모델이 기대하는 피처를 만들지 못했습니다. (누락 {len(missing)}개, 예: {missing[:5]})")
                        else:
                            X_new = pd.DataFrame([full_row])[feature_cols]
                            proba = fusion_model.predict_proba(X_new)[0, 1]
                            pred_label = DEFECT_LABEL_MAP[int(proba >= 0.5)]
                            st.success(f"예측 완료: **{pred_label}** (불량 확률 {proba*100:.1f}%)")
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number", value=proba * 100,
                                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#E45756" if proba >= 0.5 else "#4C78A8"}},
                                title={"text": "불량(미납) 확률(%)"},
                            ))
                            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                            st.plotly_chart(fig_gauge, use_container_width=True)
                    except Exception as e:
                        st.error(f"예측 중 오류가 발생했습니다: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5. 필터 / 조회
# ════════════════════════════════════════════════════════════════════════════════
elif page == "필터/조회":
    st.title("필터 / 조회")
    st.caption("불량 유형, 이상치 여부, 파일명으로 원하는 샘플을 검색하고 썸네일로 훑어보는 페이지입니다.")
    sample_metadata = metadata_global

    if sample_metadata is None:
        st.error("`sample_metadata.csv` 파일이 필요합니다.")
    else:
        f1, f2, f3 = st.columns(3)
        with f1:
            defect_types = sorted(sample_metadata["defect_type"].unique())
            picked_types = st.multiselect("불량 유형", defect_types, default=defect_types)
        with f2:
            outlier_filter = st.selectbox("이상치 여부", ["전체", "이상치만", "정상 범위만"])
        with f3:
            search = st.text_input("file_base 검색", "")

        if "date" in sample_metadata.columns:
            dates = sorted(sample_metadata["date"].dropna().unique())
            picked_dates = st.multiselect("날짜", dates, default=dates)
        else:
            picked_dates = None

        filtered = sample_metadata[sample_metadata["defect_type"].isin(picked_types)]
        if outlier_filter == "이상치만":
            filtered = filtered[filtered["is_outlier"]]
        elif outlier_filter == "정상 범위만":
            filtered = filtered[~filtered["is_outlier"]]
        if search:
            filtered = filtered[filtered["file_base"].str.contains(search, case=False, na=False)]
        if picked_dates is not None:
            filtered = filtered[filtered["date"].isin(picked_dates)]

        st.caption(f"조회 결과: {len(filtered)}개 / 전체 {len(sample_metadata)}개")

        page_size = 24
        total_pages = max(1, (len(filtered) - 1) // page_size + 1)
        page_num = st.number_input("페이지", min_value=1, max_value=total_pages, value=1)
        page_df = filtered.iloc[(page_num - 1) * page_size: page_num * page_size]

        cols = st.columns(6)
        for i, (_, row) in enumerate(page_df.iterrows()):
            with cols[i % 6]:
                thumb_path = THUMB_DIR / f"{row['file_base']}.jpg"
                if thumb_path.exists():
                    st.image(str(thumb_path), use_container_width=True)
                else:
                    st.caption("(썸네일 없음)")
                st.caption(f"{row['file_base']}\n{row['defect_type']}")
                if row["is_outlier"]:
                    st.caption("이상치")

        with st.expander("표로 보기"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)