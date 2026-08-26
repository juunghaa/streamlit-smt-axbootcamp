"""
SMT 사전 공정(납도포) 불량 분석 - 통합 대시보드
사용법: streamlit run app.py

[모델 학습] 탭 - 이 파일이 있으면 코랩 없이 바로 새 모델을 학습/비교할 수 있습니다.
  - fusion_features.csv   (코랩 Cell 9에서 이미 저장하던 파일 그대로)

[모델 비교] [전처리 확인] [필터/조회] 탭 - 기존과 동일하게 아래 파일을 사용합니다.
  - all_results.json, feature_importance_all.csv   (모델 학습 탭에서 자동 생성/갱신됨.
    코랩에서 만든 것도 같은 이름으로 두면 함께 합쳐서 보입니다)
  - data_quality_report.json, raw_sensor_sequences.json, sample_metadata.csv, thumbnails/
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
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

SENSOR_COLS = ["temperature", "humidity", "vibration", "acceleration", "noise"]


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


st.title("🔧 SMT 사전 공정(납도포) 불량 분석 대시보드")

tab_train, tab_model, tab_quality, tab_filter = st.tabs(
    ["🧪 모델 학습", "📊 모델 비교", "🔍 전처리 확인", "🗂️ 필터/조회"]
)

# =================================================================
# TAB 0. 모델 학습 (코랩 없이 여기서 바로)
# =================================================================
with tab_train:
    fusion_df = load_fusion_features()

    if fusion_df is None:
        st.error(
            f"`{FUSION_PATH.name}` 파일이 필요합니다. "
            "코랩 Cell 9에서 저장한 fusion_features.csv를 이 앱과 같은 폴더에 넣어주세요. "
            "(이미지 임베딩 추출까지 끝난 병합 피처 테이블이라, 이 파일 하나만 있으면 "
            "이후 모델 실험은 코랩 없이 여기서 계속할 수 있습니다)"
        )
    else:
        id_cols = [c for c in ["file_base", "label", "img_label"] if c in fusion_df.columns]
        all_feature_cols = [c for c in fusion_df.columns if c not in id_cols]
        sensor_cols_all = [c for c in all_feature_cols if not c.startswith("emb_")]
        image_cols_all = [c for c in all_feature_cols if c.startswith("emb_")]

        st.subheader("새 모델 학습")
        c1, c2, c3 = st.columns(3)
        with c1:
            algo = st.selectbox("알고리즘", ["XGBoost", "RandomForest", "LogisticRegression"])
        with c2:
            feature_set = st.selectbox("사용할 피처", ["전체(센서+이미지)", "센서만", "이미지만"])
        with c3:
            model_name = st.text_input("모델 이름", value=f"{algo}_{feature_set.split('(')[0]}")

        with st.expander("하이퍼파라미터 (기본값으로도 충분히 동작합니다)"):
            if algo == "XGBoost":
                n_estimators = st.slider("n_estimators", 50, 500, 300, step=50)
                max_depth = st.slider("max_depth", 2, 10, 4)
                learning_rate = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.2], value=0.05)
            elif algo == "RandomForest":
                n_estimators = st.slider("n_estimators", 50, 500, 300, step=50)
                max_depth = st.slider("max_depth", 2, 20, 8)
            else:
                c_value = st.select_slider("C (규제 강도, 작을수록 강함)", [0.01, 0.1, 1.0, 10.0], value=1.0)

        if feature_set == "전체(센서+이미지)":
            feature_cols = all_feature_cols
        elif feature_set == "센서만":
            feature_cols = sensor_cols_all
        else:
            feature_cols = image_cols_all

        if st.button("🚀 학습 시작", type="primary"):
            with st.spinner(f"{model_name} 학습 중..."):
                X = fusion_df[feature_cols]
                y = fusion_df["label"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                if algo == "XGBoost":
                    model = xgb.XGBClassifier(
                        n_estimators=n_estimators, max_depth=max_depth,
                        learning_rate=learning_rate, subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", random_state=42,
                    )
                elif algo == "RandomForest":
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, max_depth=max_depth, random_state=42
                    )
                else:
                    model = make_pipeline(
                        StandardScaler(), LogisticRegression(C=c_value, max_iter=2000)
                    )

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]

                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

                # 피처 중요도 (모델 종류에 따라 다르게 추출)
                if algo == "LogisticRegression":
                    coefs = np.abs(model.named_steps["logisticregression"].coef_[0])
                    imp = pd.Series(coefs, index=feature_cols).sort_values(ascending=False)
                else:
                    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

                sensor_in_set = [c for c in feature_cols if c in sensor_cols_all]
                image_in_set = [c for c in feature_cols if c in image_cols_all]

                entry = {
                    "model_name": model_name,
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred)),
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

            st.success(f"'{model_name}' 학습 완료! 아래에서 바로 결과를 확인하거나, [모델 비교] 탭에서 다른 모델과 비교하세요.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{entry['accuracy']*100:.1f}%")
            m2.metric("F1", f"{entry['f1']:.3f}")
            m3.metric("AUC (Holdout)", f"{entry['auc']:.3f}")
            m4.metric("AUC (5-fold CV)", f"{entry['cv_auc_mean']:.3f}")

        st.divider()
        st.caption(
            f"현재 `{FUSION_PATH.name}` 에는 샘플 {len(fusion_df)}개, "
            f"센서 피처 {len(sensor_cols_all)}개, 이미지 피처 {len(image_cols_all)}개가 있습니다."
        )

# =================================================================
# TAB 1. 모델 비교
# =================================================================
with tab_model:
    all_results = load_json(RESULTS_PATH)
    importance_df = load_csv(IMPORTANCE_PATH)

    if all_results is None or importance_df is None:
        st.info("아직 학습된 모델이 없습니다. [모델 학습] 탭에서 먼저 모델을 학습해주세요.")
    else:
        model_names = list(all_results.keys())

        st.subheader("전체 모델 성능 비교")
        compare_df = pd.DataFrame(
            [
                {
                    "모델": name,
                    "Accuracy": r["accuracy"],
                    "F1": r["f1"],
                    "AUC (Holdout)": r["auc"],
                    "AUC (5-fold CV 평균)": r["cv_auc_mean"],
                }
                for name, r in all_results.items()
            ]
        ).sort_values("AUC (5-fold CV 평균)", ascending=False)

        st.dataframe(
            compare_df.style.format(
                {c: "{:.3f}" for c in ["Accuracy", "F1", "AUC (Holdout)", "AUC (5-fold CV 평균)"]}
            ).background_gradient(cmap="Blues", subset=["AUC (5-fold CV 평균)"]),
            use_container_width=True,
            hide_index=True,
        )

        metric_for_chart = st.radio(
            "비교 지표", ["Accuracy", "F1", "AUC (Holdout)", "AUC (5-fold CV 평균)"], horizontal=True
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

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{r['accuracy']*100:.1f}%")
        c2.metric("F1 Score", f"{r['f1']:.3f}")
        c3.metric("AUC (Holdout)", f"{r['auc']:.3f}")
        c4.metric("AUC (5-fold CV 평균)", f"{r['cv_auc_mean']:.3f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion Matrix**")
            cm = r["confusion_matrix"]
            labels = r.get("confusion_matrix_labels", ["정상", "불량"])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=[f"예측: {l}" for l in labels], y=[f"실제: {l}" for l in labels],
                text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False,
            ))
            fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

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
            fig_contrib.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
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

# =================================================================
# TAB 2. 전처리 확인
# =================================================================
with tab_quality:
    quality_report = load_json(DATA_DIR / "data_quality_report.json")
    raw_sequences = load_json(DATA_DIR / "raw_sensor_sequences.json")
    sample_metadata = load_csv(DATA_DIR / "sample_metadata.csv")

    if quality_report is None or sample_metadata is None:
        st.error("`data_quality_report.json`, `sample_metadata.csv` 파일이 필요합니다. (colab_export_quality.py 실행 후 다운로드)")
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
            fig_out.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_out, use_container_width=True)

        st.divider()
        st.subheader("전처리 전/후 비교 (샘플 단위)")
        st.caption("왼쪽: 원본 센서 시계열 (변환 전) / 오른쪽: 추출된 요약 통계 피처 (변환 후)")

        sample_options = sample_metadata["file_base"].tolist()
        picked = st.selectbox("샘플 선택 (file_base)", sample_options)
        row = sample_metadata[sample_metadata["file_base"] == picked].iloc[0]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**변환 전: 원본 센서 시계열**")
            if raw_sequences and picked in raw_sequences:
                seq = raw_sequences[picked]
                fig_raw = go.Figure()
                for col in SENSOR_COLS:
                    if col in seq:
                        fig_raw.add_trace(go.Scatter(y=seq[col], mode="lines", name=col))
                fig_raw.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="시점", yaxis_title="측정값")
                st.plotly_chart(fig_raw, use_container_width=True)
            else:
                st.info("이 샘플의 원본 시계열 데이터를 찾을 수 없습니다.")

        with c2:
            st.markdown("**변환 후: 요약 통계 피처 (mean 기준)**")
            mean_cols = [c for c in sample_metadata.columns if c.endswith("_mean")]
            if mean_cols:
                after_df = pd.DataFrame({"피처": mean_cols, "값": [row[c] for c in mean_cols]})
                fig_after = px.bar(after_df, x="값", y="피처", orientation="h")
                fig_after.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_after, use_container_width=True)

        badge = "🔴 이상치" if row["is_outlier"] else "🟢 정상 범위"
        st.markdown(f"**판정:** {row['defect_type']}  |  **이상치 스코어:** {row['outlier_score']}  |  {badge}")

        thumb_path = THUMB_DIR / f"{picked}.jpg"
        if thumb_path.exists():
            st.image(str(thumb_path), caption=picked, width=200)

# =================================================================
# TAB 3. 필터 / 조회
# =================================================================
with tab_filter:
    sample_metadata = load_csv(DATA_DIR / "sample_metadata.csv")

    if sample_metadata is None:
        st.error("`sample_metadata.csv` 파일이 필요합니다. (colab_export_quality.py 실행 후 다운로드)")
    else:
        st.subheader("조건별 샘플 조회")

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
        page = st.number_input("페이지", min_value=1, max_value=total_pages, value=1)
        page_df = filtered.iloc[(page - 1) * page_size: page * page_size]

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
            