# app.py — Streamlit 리뷰 예측(회귀 전용)
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

VEC_PKL = MODELS / "tfidf_vectorizer.pkl"
REG_JOBLIB = MODELS / "rf_reg.joblib"

def _assert_files_exist(paths):
    miss = [p for p in paths if not p.exists()]
    if miss:
        st.error(f"모델 파일 누락: {[str(p) for p in miss]}")
        st.stop()

# =========================
# 전처리/토크나이즈
# =========================
POS_EMO = "😀😃😄😁😆🙂😊😍🤩😋😉👍🙌🎉❤💖💗💓💞💕✨😻🥰🤗😺😸"
NEG_EMO = "😞😟😠😡😢😭🤮😒😕🙁☹👎💢😣😖🤬😤💔😿😹"
URL_RE = re.compile(r"(https?:\/\/[^\s]+)")
HTML_RE = re.compile(r"<[^>]+>")
MULTI_SPACE = re.compile(r"\s+")

def _replace_emojis(text: str) -> str:
    text = re.sub(f"[{re.escape(POS_EMO)}]+", " [EMO_POS] ", text)
    text = re.sub(f"[{re.escape(NEG_EMO)}]+", " [EMO_NEG] ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " [EMO] ", text)
    return text

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" [URL] ", s)
    s = _replace_emojis(s)
    s = re.sub(r"[^0-9A-Za-z가-힣\.\,\!\?\[\]_ ]+", " ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    return s

def tokenize_and_join(s: str) -> str:
    return " ".join(re.findall(r"[가-힣A-Za-z0-9]{2,}", _clean_text(s)))

# =========================
# 호환 패치 (RF)
# =========================
def _patch_rf_monotonic(reg_model):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        rf = None
        if isinstance(reg_model, Pipeline):
            last = reg_model.steps[-1][1]
            if isinstance(last, RandomForestRegressor):
                rf = last
        elif isinstance(reg_model, RandomForestRegressor):
            rf = reg_model
        if rf is not None and getattr(rf, "estimators_", None) is not None:
            for est in rf.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
    except Exception:
        pass
    return reg_model

# =========================
# 모델 로드 (캐시)
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    _assert_files_exist([VEC_PKL, REG_JOBLIB])
    vec = joblib.load(VEC_PKL)
    reg = joblib.load(REG_JOBLIB)
    reg = _patch_rf_monotonic(reg)
    return vec, reg

# =========================
# 위험도 판정
# =========================
def risk_level(avg_score: float) -> str:
    # Safe ≥ 4.10, Low ≥ 4.00, Medium ≥ 3.90, High < 3.90
    if avg_score >= 4.10:
        return "Safe"
    if avg_score >= 4.00:
        return "Low"
    if avg_score >= 3.90:
        return "Medium"
    return "High"

def risk_color(level: str) -> str:
    return {
        "Safe":   "#2e7d32",
        "Low":    "#558b2f",
        "Medium": "#f9a825",
        "High":   "#c62828",
    }.get(level, "#333333")

# ==========================================================
#                          UI
# ==========================================================
st.title("⭐ 리뷰 예측 데모")

vec, reg = load_models()

# ── 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp)
    X = vec.transform([toks])
    score = float(np.clip(reg.predict(X)[0], 1, 5))  # 1~5로 클립
    st.metric("예측 점수", f"{score:.2f} ★")

st.divider()

# ── 배치 예측
st.subheader("배치 예측 (CSV 업로드)")
csv = st.file_uploader("CSV 업로드 (필수 컬럼: review_text)", type=["csv"])

if csv is not None:
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        st.error(f"CSV 로딩 실패: {e}")
    else:
        if "review_text" not in df.columns:
            st.error("CSV에 'review_text' 컬럼이 없습니다.")
        else:
            # 예측
            toks = df["review_text"].fillna("").astype(str).map(tokenize_and_join)
            X = vec.transform(toks)
            df["pred_score"] = np.clip(reg.predict(X), 1, 5).round(2)

            # 화면 표시용 컬럼 구성 (query는 숨김)
            view_cols = []
            if "review_text" in df.columns:
                view_cols.append("review_text")
            if "review_date" in df.columns:
                view_cols.append("review_date")
            view_cols.append("pred_score")

            df_view = df.loc[:, view_cols].rename(
                columns={
                    "review_text": "리뷰",
                    "review_date": "날짜",
                    "pred_score":  "예측 별점",
                }
            )

            st.dataframe(df_view, use_container_width=True)

            # ===== 평균 & 위험도 =====
            avg = float(df["pred_score"].mean())
            level = risk_level(avg)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("나의 평균 평점", f"{avg:.2f} ★")
            with col2:
                st.markdown(
                    f"""
                    <div style="padding:10px 12px;border-radius:10px;
                                background:{risk_color(level)};color:#fff;
                                display:inline-block;font-weight:600;">
                        위험도: {level}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ===== 다운로드 =====
            st.download_button(
                "결과 CSV 다운로드",
                df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
