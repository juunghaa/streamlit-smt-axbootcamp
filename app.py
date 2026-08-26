"""
SMT 사전 공정(납도포) 불량 분석 - 통합 대시보드
사용법: streamlit run app.py

[개요]          - 전체 파이프라인 상태를 한눈에 요약
[모델 학습]     - fusion_features.csv 하나로 코랩 없이 새 모델 학습/실험
[모델 비교]     - 학습된 모델들 성능/피처중요도 비교
[전처리 확인]   - 매칭 현황, 결측치, 이상치, 전/후 비교
[필터/조회]     - 샘플 단위 조회 + 썸네일

필요 파일 (기존과 동일한 이름, 같은 폴더에 두면 자동 인식):
  - fusion_features.csv                              (코랩 Cell 9 결과)
  - all_results.json, feature_importance_all.csv     (모델 학습 탭에서 자동 생성/갱신)
  - data_quality_report.json, raw_sensor_sequences.json,
    sample_metadata.csv, thumbnails/                 (코랩 Cell 7-1 결과)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

st.set_page_config(page_title="SMT 납도포 불량 분석 대시보드", page_icon="🔧", layout="wide")

DATA_DIR = Path(__file__).parent
THUMB_DIR = DATA_DIR / "thumbnails"
RESULTS_PATH = DATA_DIR / "all_results.json"
IMPORTANCE_PATH = DATA_DIR / "feature_importance_all.csv"
FUSION_PATH = DATA_DIR / "fusion_features.csv"
QUALITY_PATH = DATA_DIR / "data_quality_report.json"
RAW_SEQ_PATH = DATA_DIR / "raw_sensor_sequences.json"
METADATA_PATH = DATA_DIR / "sample_metadata.csv"

SENSOR_COLS = ["temperature", "humidity", "vibration", "acceleration", "noise"]
DEFECT_LABEL_MAP = {0: "정상", 1: "불량(미납)"}


# ------------------------------------------------------------------
# 공통 유틸
# ------------------------------------------------------------------
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


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


def status_badge(ok: bool, label: str):
    return f"🟢 {label}" if ok else f"⚪ {label}"


# ------------------------------------------------------------------
# 데이터 로드 (여러 페이지에서 공용으로 사용)
# ------------------------------------------------------------------
fusion_df_global = load_fusion_features()
all_results_global = load_json(RESULTS_PATH)
importance_df_global = load_csv(IMPORTANCE_PATH)
quality_report_global = load_json(QUALITY_PATH)
metadata_global = load_csv(METADATA_PATH)

# ------------------------------------------------------------------
# 사이드바 : 네비게이션 + 데이터 상태
# ------------------------------------------------------------------
st.sidebar.title("🔧 SMT 불량 분석")
st.sidebar.caption("사전 공정(납도포) · SPI 검사 대시보드")

PAGES = ["🏠 개요", "🧪 모델 학습", "📊 모델 비교", "🔍 전처리 확인", "🗂️ 필터/조회"]
page = st.sidebar.radio("메뉴", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.markdown("**데이터 파일 상태**")
st.sidebar.markdown(status_badge(fusion_df_global is not None, "fusion_features.csv"))
st.sidebar.markdown(status_badge(all_results_global is not None, "all_results.json"))
st.sidebar.markdown(status_badge(quality_report_global is not None, "data_quality_report.json"))
st.sidebar.markdown(status_badge(metadata_global is not None, "sample_metadata.csv"))
st.sidebar.markdown(status_badge(THUMB_DIR.exists(), "thumbnails/"))

if fusion_df_global is not None:
    st.sidebar.divider()
    st.sidebar.metric("총 샘플 수", len(fusion_df_global))
    if "label" in fusion_df_global.columns:
        defect_rate = fusion_df_global["label"].mean() * 100
        st.sidebar.metric("불량 비율", f"{defect_rate:.1f}%")


# ====================================================================
# PAGE 0. 개요 (한눈에 보기)
# ====================================================================
if page == "🏠 개요":
    st.title("🏠 SMT 사전 공정(납도포) 불량 분석 — 개요")
    st.caption("SMT 전체 불량의 60~70%가 납도포 단계에서 발생합니다. 이 대시보드는 센서(공정) + 이미지(SPI) 데이터를 결합해 불량을 사전에 탐지하는 파이프라인을 실험/모니터링합니다.")

    st.subheader("파이프라인 현황")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("병합 데이터셋", f"{len(fusion_df_global)} 건" if fusion_df_global is not None else "없음")
    k2.metric("학습된 모델 수", len(all_results_global) if all_results_global else 0)
    if all_results_global:
        best_name = max(all_results_global, key=lambda k: all_results_global[k]["cv_auc_mean"])
        best_auc = all_results_global[best_name]["cv_auc_mean"]
        k3.metric("최고 성능 모델", best_name, f"AUC {best_auc:.3f}")
    else:
        k3.metric("최고 성능 모델", "-")
    if quality_report_global:
        k4.metric("이상치 샘플", quality_report_global.get("total_outlier_samples", "-"))
    else:
        k4.metric("이상치 샘플", "-")

    st.divider()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("6대 불량 유형")
        defect_info = pd.DataFrame([
            ("1. 미납 (Missing)", "패드에 솔더 전혀 미도포", "전기적 단선(Open)"),
            ("2. 납부족 (Insufficient)", "도포량/면적 부족", "냉납(Cold Joint)"),
            ("3. 납쇼트 (Short/Bridge)", "인접 패드 간 브릿지", "회로 합선"),
            ("4. 납볼 (Solder Ball)", "패드 외곽 미세 납 입자", "2차 쇼트 위험"),
            ("5. 납좌표 밀림 (Shifted)", "패드 중심 이탈", "톰스톤 현상"),
            ("6. 납형성 불량 (Deform)", "도포 형태 불규칙", "기공(Void)"),
        ], columns=["불량 유형", "현상", "주요 영향"])
        st.dataframe(defect_info, use_container_width=True, hide_index=True)
        st.caption("※ 현재 학습 데이터는 '미납' 이진 분류 기준이며, 구조상 다중분류(6종)로 확장 가능합니다.")
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
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("fusion_features.csv 를 넣으면 여기에 피처 구성이 표시됩니다.")

    st.divider()
    st.info("👉 왼쪽 사이드바에서 페이지를 이동하세요. **모델 학습**에서 새 모델을 만들고, **모델 비교**에서 성능을 비교하고, **전처리 확인 / 필터·조회**에서 원본 데이터 품질을 점검할 수 있습니다.")


# ====================================================================
# PAGE 1. 모델 학습
# ====================================================================
elif page == "🧪 모델 학습":
    st.title("🧪 모델 학습")
    fusion_df = fusion_df_global

    if fusion_df is None:
        st.error(
            f"`{FUSION_PATH.name}` 파일이 필요합니다. "
            "코랩 Cell 9에서 저장한 fusion_features.csv를 이 앱과 같은 폴더에 넣어주세요."
        )
    else:
        id_cols = [c for c in ["file_base", "label", "img_label"] if c in fusion_df.columns]
        all_feature_cols = [c for c in fusion_df.columns if c not in id_cols]
        sensor_cols_all = [c for c in all_feature_cols if not c.startswith("emb_")]
        image_cols_all = [c for c in all_feature_cols if c.startswith("emb_")]

        # ---- 상단 요약 카드 ----
        s1, s2, s3 = st.columns(3)
        s1.metric("전체 샘플", len(fusion_df))
        s2.metric("센서 피처", len(sensor_cols_all))
        s3.metric("이미지 피처", len(image_cols_all))

        st.divider()
        left, right = st.columns([1, 1.4])

        # ---------------- 왼쪽: 설정 패널 ----------------
        with left:
            st.subheader("① 기본 설정")
            algo = st.selectbox("알고리즘", ["XGBoost", "RandomForest", "LogisticRegression"])
            feature_set = st.selectbox("사용할 피처", ["전체(센서+이미지)", "센서만", "이미지만"])
            model_name = st.text_input("모델 이름", value=f"{algo}_{feature_set.split('(')[0]}")

            st.subheader("② 데이터 분할 / 검증")
            test_size = st.slider("Test set 비율", 0.1, 0.4, 0.2, step=0.05)
            random_state = st.number_input("random_state (재현성)", value=42, step=1)
            cv_folds = st.slider("교차검증 fold 수", 3, 10, 5)
            shuffle_cv = st.checkbox("CV 시 셔플", value=True)

            st.subheader("③ 클래스 불균형 / 임계값")
            balance_classes = st.checkbox(
                "클래스 불균형 자동 보정",
                value=False,
                help="XGBoost: scale_pos_weight 자동 계산 / RandomForest·LogisticRegression: class_weight='balanced'",
            )
            decision_threshold = st.slider(
                "분류 임계값 (기본 0.5)", 0.05, 0.95, 0.5, step=0.05,
                help="불량(1)로 판정할 확률 컷오프. 미탐(재현율)이 중요하면 낮추고, 오탐(정밀도)이 중요하면 높이세요.",
            )

            st.subheader("④ 이미지 임베딩 차원 축소 (선택)")
            use_pca = st.checkbox(
                "PCA로 이미지 임베딩(512차원) 축소",
                value=False,
                help="샘플 수 대비 이미지 피처(512차원)가 과도하게 많을 때 과적합 방지에 도움이 됩니다.",
            )
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
                    min_child_weight = st.slider("min_child_weight (과적합 방지)", 1, 10, 1)
                    reg_lambda = st.select_slider("reg_lambda (L2)", [0.0, 0.1, 0.5, 1.0, 2.0, 5.0], value=1.0)
                elif algo == "RandomForest":
                    n_estimators = st.slider("n_estimators", 50, 800, 300, step=50)
                    max_depth = st.slider("max_depth", 2, 30, 8)
                    min_samples_leaf = st.slider("min_samples_leaf", 1, 20, 1)
                    max_features = st.select_slider("max_features", ["sqrt", "log2", None], value="sqrt")
                else:
                    c_value = st.select_slider("C (규제 강도, 작을수록 강함)", [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], value=1.0)
                    penalty = st.selectbox("penalty", ["l2", "l1"])
                    solver = "liblinear" if penalty == "l1" else "lbfgs"

        # 피처셋 결정
        if feature_set == "전체(센서+이미지)":
            feature_cols = all_feature_cols
        elif feature_set == "센서만":
            feature_cols = sensor_cols_all
        else:
            feature_cols = image_cols_all

        # ---------------- 오른쪽: 실행 & 결과 ----------------
        with right:
            st.subheader("⑤ 학습 실행")
            st.caption(f"선택된 피처 수: **{len(feature_cols)}개**  |  피처셋: {feature_set}")
            run = st.button("🚀 학습 시작", type="primary", use_container_width=True)

            if run:
                with st.spinner(f"{model_name} 학습 중..."):
                    X = fusion_df[feature_cols].copy()
                    y = fusion_df["label"]

                    # PCA (이미지 임베딩에만 적용, 센서 피처는 그대로 유지)
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
                            n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, subsample=subsample,
                            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
                            reg_lambda=reg_lambda,
                            scale_pos_weight=pos_weight if balance_classes else 1.0,
                            eval_metric="logloss", random_state=random_state,
                        )
                    elif algo == "RandomForest":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, max_features=max_features,
                            class_weight="balanced" if balance_classes else None,
                            random_state=random_state,
                        )
                    else:
                        model = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                C=c_value, penalty=penalty, solver=solver, max_iter=3000,
                                class_weight="balanced" if balance_classes else None,
                            ),
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

                    sensor_in_set = [c for c in feature_cols_used if c in sensor_cols_all or c.startswith("emb_pca_") is False and c in sensor_cols_all]
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
                        "n_train": int(len(X_train)),
                        "n_test": int(len(X_test)),
                        "n_total": int(len(X)),
                        "threshold": decision_threshold,
                        "used_pca": bool(use_pca and pca_components),
                        "pca_explained_variance": float(explained) if explained is not None else None,
                    }

                    all_results = load_json(RESULTS_PATH) or {}
                    all_results[model_name] = entry
                    save_all_results(all_results)

                    imp_df = imp.reset_index()
                    imp_df.columns = ["feature", "importance"]
                    imp_df["feature_type"] = imp_df["feature"].apply(
                        lambda x: "이미지(ResNet)" if x.startswith("emb_") else "센서"
                    )
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

                st.caption("자세한 비교는 왼쪽 사이드바 → **모델 비교** 페이지에서 확인하세요.")
            else:
                st.info("왼쪽에서 설정을 조정한 뒤 '학습 시작'을 누르세요. 학습 결과는 자동 저장되어 [모델 비교] 페이지에 누적됩니다.")

        st.divider()
        with st.expander("💡 추가로 시도해볼 수 있는 변수 조정 아이디어"):
            st.markdown(
                "- **PCA 성분 수**를 줄이면 이미지 피처의 과적합 위험이 줄지만 정보 손실이 생깁니다 (5~50 사이 비교 추천)\n"
                "- **클래스 불균형 보정**을 켜고 끄면서 Recall/Precision 트레이드오프 비교\n"
                "- **분류 임계값**을 0.3~0.4로 낮추면 불량 미탐지(Recall)를 줄일 수 있습니다 (공정 특성상 미탐이 더 치명적인 경우 유용)\n"
                "- **센서만 / 이미지만** 피처셋으로 각각 학습해 어느 모달리티가 더 기여하는지 비교\n"
                "- XGBoost의 `min_child_weight`, `reg_lambda`를 올리면 작은 데이터셋에서 과적합을 줄이는 데 도움\n"
                "- (향후) 6대 불량 전체 데이터가 준비되면 `label`을 다중클래스로 바꾸고 `objective='multi:softprob'`로 확장 가능"
            )

        st.divider()
        st.caption(
            f"현재 `{FUSION_PATH.name}` 에는 샘플 {len(fusion_df)}개, "
            f"센서 피처 {len(sensor_cols_all)}개, 이미지 피처 {len(image_cols_all)}개가 있습니다."
        )


# ====================================================================
# PAGE 2. 모델 비교
# ====================================================================
elif page == "📊 모델 비교":
    st.title("📊 모델 비교")
    all_results = all_results_global
    importance_df = importance_df_global

    if all_results is None or importance_df is None:
        st.info("아직 학습된 모델이 없습니다. [모델 학습] 페이지에서 먼저 모델을 학습해주세요.")
    else:
        model_names = list(all_results.keys())
        compare_df = pd.DataFrame(
            [
                {
                    "모델": name,
                    "Accuracy": r["accuracy"],
                    "F1": r["f1"],
                    "Precision": r.get("precision"),
                    "Recall": r.get("recall"),
                    "AUC (Holdout)": r["auc"],
                    "AUC (5-fold CV 평균)": r["cv_auc_mean"],
                }
                for name, r in all_results.items()
            ]
        ).sort_values("AUC (5-fold CV 평균)", ascending=False)

        # ---- 상단 요약 카드: 최고 성능 모델 ----
        best_row = compare_df.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏆 최고 모델", best_row["모델"])
        k2.metric("AUC (5-fold CV)", f"{best_row['AUC (5-fold CV 평균)']:.3f}")
        k3.metric("F1", f"{best_row['F1']:.3f}")
        k4.metric("학습된 모델 수", len(model_names))

        st.divider()
        st.subheader("전체 모델 성능 비교")
        st.dataframe(
            compare_df.style.format(
                {c: "{:.3f}" for c in ["Accuracy", "F1", "Precision", "Recall", "AUC (Holdout)", "AUC (5-fold CV 평균)"]}
            ).background_gradient(cmap="Blues", subset=["AUC (5-fold CV 평균)"]),
            use_container_width=True,
            hide_index=True,
        )

        metric_for_chart = st.radio(
            "비교 지표", ["Accuracy", "F1", "Precision", "Recall", "AUC (Holdout)", "AUC (5-fold CV 평균)"], horizontal=True
        )
        fig_compare = px.bar(
            compare_df.sort_values(metric_for_chart),
            x=metric_for_chart, y="모델", orientation="h", text_auto=".3f",
            color=metric_for_chart, color_continuous_scale="Blues",
        )
        fig_compare.update_layout(height=max(300, len(model_names) * 60), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_compare, use_container_width=True)

        st.divider()
        st.subheader("모델 상세 보기")
        selected_model = st.selectbox("모델 선택", model_names)
        r = all_results[selected_model]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{r.get('precision', 0):.3f}")
        c3.metric("Recall", f"{r.get('recall', 0):.3f}")
        c4.metric("F1", f"{r['f1']:.3f}")
        c5.metric("AUC (5-fold CV)", f"{r['cv_auc_mean']:.3f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion Matrix**")
            cm = r["confusion_matrix"]
            labels = r.get("confusion_matrix_labels", ["정상", "불량"])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=[f"예측: {l}" for l in labels], y=[f"실제: {l}" for l in labels],
                text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)
            if r.get("threshold") is not None:
                st.caption(f"분류 임계값: {r['threshold']}" + (" · PCA 적용됨" if r.get("used_pca") else ""))

        with right:
            st.markdown("**센서 vs 이미지 기여도**")
            contrib = pd.DataFrame({
                "구분": ["센서", "이미지(ResNet)"],
                "중요도 합": [r["sensor_importance_sum"], r["image_importance_sum"]],
            })
            fig_contrib = px.pie(
                contrib, names="구분", values="중요도 합", hole=0.5, color="구분",
                color_discrete_map={"센서": "#4C78A8", "이미지(ResNet)": "#F58518"},
            )
            fig_contrib.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_contrib, use_container_width=True)

        st.markdown("**상위 피처 중요도**")
        model_imp = importance_df[importance_df["model_name"] == selected_model].sort_values("importance", ascending=True)
        if len(model_imp) > 0:
            top_n = st.slider("표시할 피처 개수", 5, len(model_imp), min(15, len(model_imp)))
            fig_bar = px.bar(
                model_imp.tail(top_n), x="importance", y="feature", color="feature_type", orientation="h",
                color_discrete_map={"센서": "#4C78A8", "이미지(ResNet)": "#F58518"},
                labels={"importance": "중요도", "feature": "피처", "feature_type": "구분"},
            )
            fig_bar.update_layout(height=max(350, top_n * 24), margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)


# ====================================================================
# PAGE 3. 전처리 확인
# ====================================================================
elif page == "🔍 전처리 확인":
    st.title("🔍 전처리 확인")
    quality_report = quality_report_global
    raw_sequences = load_json(RAW_SEQ_PATH)
    sample_metadata = metadata_global

    if quality_report is None or sample_metadata is None:
        st.error("`data_quality_report.json`, `sample_metadata.csv` 파일이 필요합니다. (코랩 Cell 7-1 실행 후 다운로드)")
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
            fig_out.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_out, use_container_width=True)

        st.divider()
        st.subheader("전처리 전/후 비교 (샘플 단위)")
        st.caption("왼쪽: 원본 센서 시계열 (변환 전) / 오른쪽: 추출된 요약 통계 피처 (변환 후)")

        sample_options = sample_metadata["file_base"].tolist()
        picked = st.selectbox("샘플 선택 (file_base)", sample_options)
        row = sample_metadata[sample_metadata["file_base"] == picked].iloc[0]

        badge = "🔴 이상치" if row["is_outlier"] else "🟢 정상 범위"
        st.markdown(f"**판정:** {row['defect_type']}  |  **이상치 스코어:** {row['outlier_score']}  |  {badge}")

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

        thumb_path = THUMB_DIR / f"{picked}.jpg"
        if thumb_path.exists():
            st.image(str(thumb_path), caption=picked, width=200)


# ====================================================================
# PAGE 4. 필터 / 조회
# ====================================================================
elif page == "🗂️ 필터/조회":
    st.title("🗂️ 필터 / 조회")
    sample_metadata = metadata_global

    if sample_metadata is None:
        st.error("`sample_metadata.csv` 파일이 필요합니다. (코랩 Cell 7-1 실행 후 다운로드)")
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
                    st.caption("🔴 이상치")

        with st.expander("표로 보기"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)
            