# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---- (선택) skops: 있으면 우선 사용 ----
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

def safe_skops_load(path: Path):
    if not HAS_SKOPS:
        raise RuntimeError("skops 미설치")
    p = str(path)
    try:
        types = get_untrusted_types(file=p)
    except TypeError:
        try:
            types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()
    # 보안: 우리가 쓸 가능성이 있는 모듈만 허용
    _ = [t for t in types if t.startswith(ALLOWED_PREFIXES)]
    return skops_load(p, trusted=types)

# -------------------- 기본 설정 --------------------
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")

# -------------------- 모델 경로 --------------------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

VEC_SKOPS = MODELS / "tfidf_vectorizer.skops"
VEC_PKL   = MODELS / "tfidf_vectorizer.pkl"
REG_SKOPS = MODELS / "rf_reg.skops"
REG_JBL   = MODELS / "rf_reg.joblib"

def _assert_files_exist():
    have_vec = VEC_SKOPS.exists() or VEC_PKL.exists()
    have_reg = REG_SKOPS.exists() or REG_JBL.exists()
    if not (have_vec and have_reg):
        st.error(
            "모델 파일 누락입니다. models/ 폴더에 다음 중 하나씩은 있어야 합니다.\n\n"
            "• tfidf_vectorizer.(skops | pkl)\n"
            "• rf_reg.(skops | joblib)\n"
        )
        st.stop()

# -------------------- 전처리/토크나이즈 --------------------
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

# -------------------- 호환 패치: RF monotonic_cst --------------------
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

# -------------------- 모델 로더 (캐시) --------------------
@st.cache_resource(show_spinner=True)
def load_models():
    _assert_files_exist()

    # 벡터라이저
    if VEC_SKOPS.exists() and HAS_SKOPS:
        vectorizer = safe_skops_load(VEC_SKOPS)
    else:
        vectorizer = joblib.load(VEC_PKL)

    # 회귀 모델
    if REG_SKOPS.exists() and HAS_SKOPS:
        reg = safe_skops_load(REG_SKOPS)
    else:
        reg = joblib.load(REG_JBL)

    reg = _patch_rf_monotonic(reg)
    return vectorizer, reg

# ==========================================================
#                           UI
# ==========================================================
st.title("⭐ 리뷰 예측 데모")
vec, reg = load_models()

# ---------------- 단일 예측 ----------------
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")

if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp)
    X = vec.transform([toks])

    pred_score = float(np.clip(reg.predict(X)[0], 1, 5))  # 1~5 범위로 클립
    st.metric("예측 점수", f"{pred_score:.2f} ★")

st.divider()

# ---------------- 배치 예측 ----------------
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
            toks = df["review_text"].fillna("").astype(str).map(tokenize_and_join)
            X = vec.transform(toks)
            df["pred_score"] = np.clip(reg.predict(X), 1, 5).round(2)

            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "결과 CSV 다운로드",
                df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
