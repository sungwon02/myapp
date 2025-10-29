# app.py — Streamlit 리뷰 예측 데모 (skops 먼저, 없으면 joblib)

import os, re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---- skops 옵션 로드(있으면 사용) ----
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

def safe_skops_load(path: Path):
    """
    skops 0.10+ 보안모드: 파일 내 타입을 검사하고 trusted 목록으로 로드
    """
    if not HAS_SKOPS:
        raise RuntimeError("skops 미설치")
    p = str(path)
    # 서로 다른 시그니처 대응
    try:
        types = get_untrusted_types(file=p)
    except TypeError:
        try:
            types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()
    bad = [t for t in types if not t.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"허용되지 않은 타입 감지: {bad[:5]}")
    return skops_load(p, trusted=types)

# =========================
#  전처리 & 토크나이즈
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

def clean_text(s: str) -> str:
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
    # 두 글자 이상 영/숫/한글 토큰만
    return " ".join(re.findall(r"[가-힣A-Za-z0-9]{2,}", clean_text(s)))

# =========================
#  모델 경로 선택 & 로드
# =========================
def _pick_models_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "models",                # 권장
        Path("./models").resolve(),
        Path(os.getcwd()) / "models",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

def _patch_rf_monotonic(reg):
    """sklearn 1.3→1.6 호환 패치 (DecisionTreeRegressor.monotonic_cst)"""
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
    return reg

def _load_one(base: Path, stem: str):
    """
    같은 이름의 .skops가 있으면 skops로, 아니면 joblib로 로드
    stem: 파일 앞부분(ex. 'tfidf_vectorizer', 'sgd_logistic_cls', 'rf_reg')
    """
    sk = base / f"{stem}.skops"
    jl = base / f"{stem}.joblib"
    pkl = base / f"{stem}.pkl"

    if sk.exists() and HAS_SKOPS:
        return safe_skops_load(sk)
    if jl.exists():
        return joblib.load(jl)
    if pkl.exists():
        return joblib.load(pkl)
    raise FileNotFoundError(f"{stem}(.skops|.joblib|.pkl) 파일이 없습니다.")

@st.cache_resource(show_spinner=True)
def load_models():
    BASE = _pick_models_dir()

    # 배포 경로 디버그(문제시 바로 확인)
    st.write("🔎 CWD:", os.getcwd())
    st.write("📁 BASE(models):", str(BASE))
    try:
        st.write("📄 models contents:", sorted(p.name for p in BASE.glob("*")))
    except Exception:
        pass

    try:
        vectorizer = _load_one(BASE, "tfidf_vectorizer")
        cls        = _load_one(BASE, "sgd_logistic_cls")
        reg        = _load_one(BASE, "rf_reg")
    except FileNotFoundError as e:
        st.error(
            "모델 파일을 찾을 수 없습니다.\n\n"
            f"- 찾은 BASE 폴더: {BASE}\n"
            f"- BASE 목록: {[p.name for p in BASE.glob('*')]}\n"
            f"- 에러: {e}\n\n"
            "👉 리포지토리 루트에 `models/` 폴더를 두고 다음 파일들 중 하나 형식으로 준비하세요.\n"
            "   • tfidf_vectorizer.(skops|pkl)\n"
            "   • sgd_logistic_cls.(skops|joblib)\n"
            "   • rf_reg.(skops|joblib)\n"
        )
        st.stop()

    reg = _patch_rf_monotonic(reg)
    return vectorizer, cls, reg

# =========================
#  Streamlit UI
# =========================
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")
st.title("⭐ 리뷰 별점 예측 데모")

vec, cls, reg = load_models()

# ── 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp)
    X = vec.transform([toks])

    cls_star = int(cls.predict(X)[0])                   # 분류(정수)
    reg_star = float(np.clip(reg.predict(X)[0], 1, 5))  # 회귀(연속, 1~5 클립)

    c1, c2 = st.columns(2)
    with c1: st.metric("분류(정수)", f"{cls_star} ★")
    with c2: st.metric("회귀(연속)", f"{reg_star:.2f} ★")

    # 간단 설명(로지스틱: 가중치 × TF-IDF)
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
            if xv[j] == 0:
                continue
            rows.append({
                "term": str(feats[j]),
                "tfidf": float(xv[j]),
                "coef": float(coef[j]),
                "contrib": float(contrib[j]),
            })
            if len(rows) >= 8:
                break
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

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
