# app.py — Streamlit 리뷰 예측( predict_safe.py 동치화 버전 )
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import streamlit as st

# ---------- 공통: 스레드/백엔드 제한 ----------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# skops 안전 로더
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

def safe_skops_load(path: Path):
    p = str(path)
    try:
        types = get_untrusted_types(file=p)
    except TypeError:
        try:
            types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()
    bad = [t for t in types if not t.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"skops 파일 비허용 타입 포함: {bad[:5]} ...")
    return skops_load(p, trusted=types)

# ---------- RF 하위 추정기 패치 ----------
def patch_rf_monotonic(reg):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        rf = None
        if "Pipeline" in str(type(reg)):
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

# ---------- 텍스트 정제/토크나이즈 ----------
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

def get_tokenizer():
    # 1) mecab
    try:
        from mecab import MeCab
        m = MeCab()
        def tok(text):
            return [w for (w, p) in m.pos(text)
                    if p.startswith(("NN","VV","VA","MAG","IC","XR"))]
        return "mecab_python", tok
    except Exception:
        pass
    # 2) kiwi
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        def tok(text):
            return [t.form for t in kiwi.tokenize(text)
                    if t.tag.startswith(("N","V","MAG","IC","XR","MM"))]
        return "kiwi", tok
    except Exception:
        pass
    # 3) fallback: regex
    def tok(text):
        return re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    return "simple", tok

TOK_NAME, TOKENIZE = get_tokenizer()

def tokenize_and_join(text: str, stopwords:set) -> str:
    toks = TOKENIZE(clean_text(text))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return " ".join(toks)

# ---------- neg/pos 열가중 ----------
def apply_column_scaling_csc(X_csc, vocab: dict, terms: list, scale: float):
    hits = [vocab[t] for t in terms if t in vocab]
    for j in hits:
        start, end = X_csc.indptr[j], X_csc.indptr[j+1]
        if end > start:
            X_csc.data[start:end] *= scale
    return X_csc

def maybe_apply_negpos_bonus(X_csr, vec, base_dir: Path):
    summ = base_dir / "tfidf_summary.json"
    if not summ.exists():
        return X_csr
    try:
        js = json.loads(summ.read_text(encoding="utf-8"))
        info = js.get("tfidf", {})
        neg_terms = info.get("neg_terms", []) or []
        pos_terms = info.get("pos_terms", []) or []
        neg_bonus = float(info.get("neg_bonus", 1.0))
        pos_bonus = float(info.get("pos_bonus", 1.0))
        if (neg_terms or pos_terms) and hasattr(vec, "vocabulary_"):
            vocab = vec.vocabulary_
            X_csc = X_csr.tocsc(copy=True)
            if neg_terms and neg_bonus != 1.0:
                X_csc = apply_column_scaling_csc(X_csc, vocab, neg_terms, neg_bonus)
            if pos_terms and pos_bonus != 1.0:
                X_csc = apply_column_scaling_csc(X_csc, vocab, pos_terms, pos_bonus)
            return X_csc.tocsr(copy=False)
    except Exception:
        return X_csr
    return X_csr

# ---------- 자산 로드 ----------
def load_vectorizer_with_compat(base: Path, model_expected_n: int | None):
    cands = []
    sk = base / "tfidf_vectorizer.skops"
    pk = base / "tfidf_vectorizer.pkl"
    if sk.exists() and HAS_SKOPS:
        cands.append(("skops", sk))
    if pk.exists():
        cands.append(("pkl", pk))
    if not cands:
        raise FileNotFoundError("tfidf_vectorizer.(skops|pkl) 없음")

    last_err, chosen = None, None
    for kind, path in cands:
        try:
            vec = safe_skops_load(path) if kind == "skops" else joblib.load(path)
            n_vec = len(vec.get_feature_names_out())
            if (model_expected_n is None) or (n_vec == model_expected_n):
                chosen = vec
                break
            else:
                last_err = f"{path.name} features {n_vec} != expected {model_expected_n}"
        except Exception as e:
            last_err = f"load {path.name} failed: {e!r}"
    if chosen is None:
        raise ValueError(f"적합한 TF-IDF 벡터라이저를 찾지 못함: {last_err}")
    return chosen

def load_assets(base_dir: Path):
    base = Path(base_dir)
    # SGD (설명용)
    if HAS_SKOPS and (base / "sgd_logistic_cls.skops").exists():
        cls = safe_skops_load(base / "sgd_logistic_cls.skops")
    elif (base / "sgd_logistic_cls.joblib").exists():
        cls = joblib.load(base / "sgd_logistic_cls.joblib")
    else:
        cls = None
    model_expected_n_cls = getattr(cls, "n_features_in_", None) if cls is not None else None

    # RF
    if HAS_SKOPS and (base / "rf_reg.skops").exists():
        reg = safe_skops_load(base / "rf_reg.skops")
    elif (base / "rf_reg.joblib").exists():
        reg = joblib.load(base / "rf_reg.joblib")
    else:
        raise FileNotFoundError("rf_reg.(skops|joblib) 없음")
    reg = patch_rf_monotonic(reg)
    model_expected_n_reg = getattr(reg, "n_features_in_", None)

    # Vectorizer: (가능하면) SGD 기준으로 맞추되, RF와도 일치 확인
    vec = load_vectorizer_with_compat(base, model_expected_n_cls or model_expected_n_reg)
    n_vec = len(vec.get_feature_names_out())
    if (model_expected_n_reg is not None) and (n_vec != model_expected_n_reg):
        raise ValueError(f"벡터라이저 피처수 {n_vec} != RF 기대 {model_expected_n_reg}")

    # 불용어
    stop = set()
    for sw in [base.parent / "stopwords_ko.txt", base / "stopwords_ko.txt"]:
        if sw.exists():
            stop = {x.strip() for x in sw.read_text(encoding="utf-8").splitlines() if x.strip()}
            break

    return vec, reg, cls, stop, base

# ---------- 위험도 ----------
def risk_level(avg_score: float) -> str:
    if avg_score >= 4.10: return "Safe"
    if avg_score >= 4.00: return "Low"
    if avg_score >= 3.90: return "Medium"
    return "High"

def risk_color(level: str) -> str:
    return {"Safe":"#2e7d32","Low":"#558b2f","Medium":"#f9a825","High":"#c62828"}.get(level, "#333")

# ---------- 부정적 토큰(상위3) : (coef5 - coef1) * tfidf 가장 음수 ----------
def negative_top_terms(vec, cls, Xrow: csr_matrix, topk=3):
    if cls is None or not hasattr(cls, "coef_"):  # SGD 없으면 불가
        return []
    feats = vec.get_feature_names_out()
    coef = cls.coef_       # shape: [n_classes, n_features]
    classes = getattr(cls, "classes_", np.arange(coef.shape[0]))
    # class 1과 class 5 인덱스 찾기
    def find_idx(c):
        idx = np.where(classes == c)[0]
        return int(idx[0]) if len(idx) else None
    i1, i5 = find_idx(1), find_idx(5)
    if i1 is None or i5 is None:
        return []
    delta = coef[i5] - coef[i1]   # 양수면 별5 쪽, 음수면 별1 쪽
    x = Xrow.toarray().ravel()
    contrib = delta * x
    # 음수(하락) 기여가 큰 순서
    idx = np.argsort(contrib)[:topk]
    rows = []
    for j in idx:
        if x[j] == 0: 
            continue
        rows.append({"term": str(feats[j]), "tfidf": float(x[j]), "coef_delta": float(delta[j]), "score_pull": float(contrib[j])})
    return rows

# ---------- Streamlit UI ----------
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.title("⭐ 리뷰 예측 데모 (safe-동일화)")

# 경로 입력(고정 경로 쓰면 수정)
base_dir = MODELS
vec, reg, cls, stop, base = load_assets(base_dir)

with st.expander("디버그(모델/벡터라이저 일치 확인)"):
    st.write({
        "tokenizer": TOK_NAME,
        "stopwords": len(stop),
        "vectorizer_features": len(vec.get_feature_names_out()),
        "rf_n_features_in_": getattr(reg, "n_features_in_", None),
        "sgd_n_features_in_": getattr(cls, "n_features_in_", None) if cls is not None else None,
        "negpos_summary": (base / "tfidf_summary.json").exists()
    })

sgd_explain = st.toggle("SGD 기반 설명(부정 토큰) 표시", value=True, help="SGD가 있을 때만 표시")

# 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp, stop)
    X = vec.transform([toks])
    X = maybe_apply_negpos_bonus(X, vec, base)
    score = float(np.clip(reg.predict(X)[0], 1, 5))
    st.metric("예측 점수", f"{score:.2f} ★")

st.divider()

# 배치 예측
st.subheader("배치 예측 (CSV 업로드)")
csv = st.file_uploader("CSV 업로드 (필수 컬럼: review_text)", type=["csv"])

if csv is not None:
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        st.error(f"CSV 로딩 실패: {e}")
        st.stop()

    if "review_text" not in df.columns:
        st.error("CSV에 'review_text' 컬럼이 없습니다.")
        st.stop()

    texts = df["review_text"].fillna("").astype(str).tolist()
    toks = [tokenize_and_join(t, stop) for t in texts]
    X = vec.transform(toks)
    X = maybe_apply_negpos_bonus(X, vec, base)

    pred = np.clip(reg.predict(X), 1, 5)
    df_out = df.copy()
    df_out["pred_score"] = np.round(pred, 2)

    # 화면용 표
    view_cols = [c for c in ("review_text","review_date") if c in df_out.columns] + ["pred_score"]
    st.dataframe(df_out.loc[:, view_cols].rename(columns={"review_text":"리뷰","review_date":"날짜","pred_score":"예측 별점"}), use_container_width=True)

    # 평균/위험도
    avg = float(df_out["pred_score"].mean())
    level = risk_level(avg)
    c1,c2 = st.columns([1,1])
    with c1:
        st.metric("평균 평점", f"{avg:.2f} ★")
    with c2:
        st.markdown(f"""
        <div style="padding:10px 12px;border-radius:10px;background:{risk_color(level)};color:#fff;display:inline-block;font-weight:600;">
            위험도: {level}
        </div>""", unsafe_allow_html=True)

    # 부정 토큰 TOP3 (High/Medium && SGD on)
    if sgd_explain and level in {"High","Medium"} and cls is not None:
        # 전체 문서의 부정 기여 합을 토큰별로 누적하여 TOP3
        feats = vec.get_feature_names_out()
        agg = {}
        # (coef5 - coef1) * tfidf 음수 방향 합
        coef = cls.coef_
        classes = getattr(cls, "classes_", np.arange(coef.shape[0]))
        i1 = np.where(classes == 1)[0]
        i5 = np.where(classes == 5)[0]
        if len(i1) and len(i5):
            delta = coef[int(i5[0])] - coef[int(i1[0])]
            for i in range(X.shape[0]):
                row = X[i].toarray().ravel()
                contrib = delta * row
                # 음수(하락)만 누적
                neg_idx = np.where(contrib < 0)[0]
                for j in neg_idx:
                    if row[j] == 0: 
                        continue
                    agg[j] = agg.get(j, 0.0) + contrib[j]
            if agg:
                worst = sorted(agg.items(), key=lambda kv: kv[1])[:3]
                labels = [f"{feats[j]} (Σ {aggv:.2f})" for j, aggv in worst]
                st.markdown(f"**리뷰 속 부정적 단어 TOP3:** " + ", ".join(labels))

    # 다운로드
    st.download_button(
        "결과 CSV 다운로드",
        df_out.to_csv(index=False, encoding="utf-8-sig"),
        file_name="predictions.csv",
        mime="text/csv",
    )
