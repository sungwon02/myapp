import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------- skops 안전 로더 --------
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

def safe_skops_load(path: Path):
    """skops 0.10+ : trusted 타입을 지정해 안전 로드"""
    if not HAS_SKOPS:
        raise RuntimeError("skops가 설치되어 있지 않습니다. pip install skops")

    p = str(path)
    try:
        types = get_untrusted_types(file=p)   # 0.10 계열
    except TypeError:
        try:
            types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()

    bad = [t for t in types if not t.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"허용되지 않은 타입 감지: {bad[:5]}")
    return skops_load(p, trusted=types)

# -------- 모델/벡터 경로 --------
BASE = Path(r"C:\Users\kimsw\TF-IDF_벡터")

VEC_SKOPS = BASE / "tfidf_vectorizer.skops"
VEC_PKL   = BASE / "tfidf_vectorizer.pkl"
CLS_SKOPS = BASE / "sgd_logistic_cls.skops"
CLS_PKL   = BASE / "sgd_logistic_cls.joblib"
REG_SKOPS = BASE / "rf_reg.skops"
REG_PKL   = BASE / "rf_reg.joblib"

# -------- 토크나이저/전처리 --------
POS_EMO = "😀😃😄😁😆🙂😊😍🤩😋😉👍🙌🎉❤💖💗💓💞💕✨😻🥰🤗😺😸"
NEG_EMO = "😞😟😠😡😢😭🤮😒😕🙁☹👎💢😣😖🤬😤💔😿😹"
URL_RE = re.compile(r"(https?:\/\/[^\s]+)")
HTML_RE = re.compile(r"<[^>]+>")
MULTI_SPACE = re.compile(r"\s+")

def replace_emojis(text: str) -> str:
    text = re.sub(f"[{re.escape(POS_EMO)}]+", " [EMO_POS] ", text)
    text = re.sub(f"[{re.escape(NEG_EMO)}]+", " [EMO_NEG] ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", " [EMO] ", text)
    return text

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" [URL] ", s)
    s = replace_emojis(s)
    s = re.sub(r"[^0-9A-Za-z가-힣\.\,\!\?\[\]_ ]+", " ", s)
    s = MULTI_SPACE.sub(" ", s).strip()
    return s

def tokenize_and_join(s: str) -> str:
    return " ".join(re.findall(r"[가-힣A-Za-z0-9]{2,}", clean_text(s)))

# -------- 모델 로더 (캐시) --------
@st.cache_resource(show_spinner=True)
def load_models():
    # Vectorizer
    if VEC_SKOPS.exists() and HAS_SKOPS:
        vectorizer = safe_skops_load(VEC_SKOPS)
    else:
        vectorizer = joblib.load(VEC_PKL)

    # Classifier
    if CLS_SKOPS.exists() and HAS_SKOPS:
        cls = safe_skops_load(CLS_SKOPS)
    else:
        cls = joblib.load(CLS_PKL)

    # Regressor (+ 1.3→1.6 호환 패치)
    if REG_SKOPS.exists() and HAS_SKOPS:
        reg = safe_skops_load(REG_SKOPS)
    else:
        reg = joblib.load(REG_PKL)

    # RF 회귀기 호환 패치 (monotonic_cst)
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        rf = None
        if isinstance(reg, Pipeline):
            last = reg.steps[-1][1]
            if isinstance(last, RandomForestRegressor):
                rf = last
        elif isinstance(reg, RandomForestRegressor):
            rf = reg
        if rf is not None and getattr(rf, "estimators_", None) is not None:
            for est in rf.estimators_:
                if not hasattr(est, "monotonic_cst"):
                    setattr(est, "monotonic_cst", None)
    except Exception:
        pass

    return vectorizer, cls, reg

# ---------------- UI ----------------
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")
st.title("⭐ 리뷰 별점 예측 데모")

vec, cls, reg = load_models()

# 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp)
    X = vec.transform([toks])
    cls_star = int(cls.predict(X)[0])
    reg_star = float(np.clip(reg.predict(X)[0], 1, 5))
    c1, c2 = st.columns(2)
    with c1: st.metric("분류(정수)", f"{cls_star} ★")
    with c2: st.metric("회귀(연속)", f"{reg_star:.2f} ★")

    # 간단 설명(로지스틱 가중치 × TF-IDF)
    if hasattr(cls, "coef_"):
        feats = vec.get_feature_names_out()
        try:
            proba = cls.predict_proba(X)[0]
            st.caption(f"최대 클래스 확률: {np.max(proba):.3f}")
        except Exception:
            pass
        y_label = cls.predict(X)[0]
        classes = getattr(cls, "classes_", None)
        y_idx = int(np.where(classes == y_label)[0][0]) if classes is not None else int(y_label) - 1
        coef = cls.coef_[y_idx]
        xv = X.toarray().ravel()
        contrib = coef * xv
        idx = np.argsort(contrib)[::-1]
        rows = []
        for j in idx:
            if xv[j] == 0: continue
            rows.append({"term": str(feats[j]), "tfidf": float(xv[j]), "coef": float(coef[j]), "contrib": float(contrib[j])})
            if len(rows) >= 8: break
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()

# 배치 예측
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
            df["pred_star_cls"] = cls.predict(X)
            df["pred_star_reg"] = np.clip(reg.predict(X), 1, 5).round(2)
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "결과 CSV 다운로드",
                df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
