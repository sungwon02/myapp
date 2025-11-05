# app.py — Streamlit 리뷰 예측(스크립트와 정합, RF예측 + SGD설명)
# -*- coding: utf-8 -*-
from __future__ import annotations
import re, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import csr_matrix

# ============= 기본 UI =============
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")
st.title("⭐ 리뷰 예측")

ROOT   = Path(__file__).resolve().parent
MODELS = ROOT / "models"

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

# ============= skops 안전 로더 =============
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

def safe_skops_load(path: Path):
    p = str(path)
    types = None
    try:
        types = get_untrusted_types(file=p)
    except TypeError:
        try:
            types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()
    bad = [t for t in types if not t.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(f"비허용 타입 포함: {bad[:5]} in {p}")
    return skops_load(p, trusted=types)

def _patch_rf_monotonic(reg_pipeline):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        rf = None
        if isinstance(reg_pipeline, Pipeline):
            last = reg_pipeline.steps[-1][1]
            if isinstance(last, RandomForestRegressor):
                rf = last
        elif isinstance(reg_pipeline, RandomForestRegressor):
            rf = reg_pipeline
        if rf is None or getattr(rf, "estimators_", None) is None:
            return reg_pipeline
        for est in rf.estimators_:
            if not hasattr(est, "monotonic_cst"):
                setattr(est, "monotonic_cst", None)
    except Exception:
        pass
    return reg_pipeline

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
                last_err = f"{path.name}: {n_vec} vs expected {model_expected_n}"
        except Exception as e:
            last_err = f"load fail {path.name}: {e!r}"
    if chosen is None:
        raise ValueError(f"호환 벡터라이저 없음. 마지막 오류: {last_err}")
    return chosen

# ============= 토크나이즈/정규화 (스크립트 동일) =============
POS_EMO = "😀😃😄😁😆🙂😊😍🤩😋😉👍🙌🎉❤💖💗💓💞💕✨😻🥰🤗😺😸"
NEG_EMO = "😞😟😠😡😢😭🤮😒😕🙁☹👎💢😣😖🤬😤💔😿😹"
URL_RE   = re.compile(r"(https?:\/\/[^\s]+)")
HTML_RE  = re.compile(r"<[^>]+>")
MULTI_S  = re.compile(r"\s+")

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
    s = MULTI_S.sub(" ", s).strip()
    return s

def get_tokenizer():
    # 1) mecab
    try:
        from mecab import MeCab
        m = MeCab()
        def tok(text):
            return [w for (w, p) in m.pos(text) if p.startswith(("NN","VV","VA","MAG","IC","XR"))]
        return "mecab_python", tok
    except Exception:
        pass
    # 2) kiwi
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        def tok(text):
            return [t.form for t in kiwi.tokenize(text) if t.tag.startswith(("N","V","MAG","IC","XR","MM"))]
        return "kiwi", tok
    except Exception:
        pass
    # 3) fallback
    def tok(text):
        return re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    return "simple", tok

TOK_NAME, TOKENIZE = get_tokenizer()

def tokenize_and_join(text: str, stopwords:set) -> str:
    toks = TOKENIZE(clean_text(text))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return " ".join(toks)

# ============= neg/pos 보너스 (스크립트 동일) =============
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

# ============= 모델 로드 (스크립트와 동일 정책) =============
@st.cache_resource(show_spinner=True)
def load_assets():
    # SGD (설명용) 있으면 로드
    cls = None
    if HAS_SKOPS and (MODELS / "sgd_logistic_cls.skops").exists():
        cls = safe_skops_load(MODELS / "sgd_logistic_cls.skops")
    elif (MODELS / "sgd_logistic_cls.joblib").exists():
        cls = joblib.load(MODELS / "sgd_logistic_cls.joblib")
    else:
        cls = None
    n_exp_cls = getattr(cls, "n_features_in_", None) if cls is not None else None

    # RF
    if HAS_SKOPS and (MODELS / "rf_reg.skops").exists():
        reg = safe_skops_load(MODELS / "rf_reg.skops")
    elif (MODELS / "rf_reg.joblib").exists():
        reg = joblib.load(MODELS / "rf_reg.joblib")
    else:
        raise FileNotFoundError("rf_reg.(skops|joblib) 없음")
    reg = _patch_rf_monotonic(reg)
    n_exp_reg = getattr(reg, "n_features_in_", None)

    # Vectorizer (cls 기준으로 우선 맞추고, 불일치 시 에러)
    vec = load_vectorizer_with_compat(MODELS, n_exp_cls)
    n_vec = len(vec.get_feature_names_out())
    if (n_exp_reg is not None) and (n_vec != n_exp_reg):
        raise ValueError(f"벡터라이저({n_vec}) != RF 기대({n_exp_reg})")

    # stopwords
    stop = set()
    for sw in [MODELS.parent / "stopwords_ko.txt", MODELS / "stopwords_ko.txt"]:
        if Path(sw).exists():
            stop = {x.strip() for x in Path(sw).read_text(encoding="utf-8").splitlines() if x.strip()}
            break

    return vec, reg, cls, stop

vec, reg, sgd, stopwords = load_assets()

# ============= 위험도 =============
def risk_level(avg_score: float) -> str:
    if avg_score >= 4.10: return "Safe"
    if avg_score >= 4.00: return "Low"
    if avg_score >= 3.90: return "Medium"
    return "High"

def risk_color(level: str) -> str:
    return {"Safe": "#2e7d32","Low": "#558b2f","Medium": "#f9a825","High": "#c62828"}.get(level, "#333")

# ============= SGD 기여도(부정 TOP3 집계) =============
def negative_top3_across_rows(X: csr_matrix, vec, sgd, tfidf_min=0.10, k=3):
    """
    5점(가장 긍정) 클래스의 계수 기준으로 contrib=coef*tfidf가 음수인 토큰들만 합산,
    전체 행에 대해 누적 후 가장 마이너스가 큰 3개 반환.
    """
    if (sgd is None) or (not hasattr(sgd, "coef_")):
        return []
    feats = np.asarray(vec.get_feature_names_out())
    classes = getattr(sgd, "classes_", None)
    if classes is None:
        idx5 = sgd.coef_.shape[0]-1
    else:
        idx_arr = np.where(classes == 5)[0]
        idx5 = int(idx_arr[0]) if len(idx_arr) else sgd.coef_.shape[0]-1
    coef = sgd.coef_[idx5]
    contrib_sum = defaultdict(float)

    # 행 단위 누적
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        mask = (row >= tfidf_min)
        if not np.any(mask): 
            continue
        contrib = coef[mask] * row[mask]
        # 음수만
        neg_idx = np.where(contrib < 0)[0]
        for local_j in neg_idx:
            j = np.where(mask)[0][local_j]
            contrib_sum[int(j)] += float(contrib[local_j])

    if not contrib_sum:
        return []
    # 가장 마이너스(작은 값) 3개
    items = sorted(contrib_sum.items(), key=lambda x: x[1])[:k]
    return [(feats[j], v) for j, v in items]

# ============= UI: 단일 예측 =============
st.subheader("단일 텍스트 예측")
colA, colB = st.columns([3,1])
with colA:
    inp = st.text_area("리뷰 텍스트 입력", height=140, placeholder="리뷰를 붙여넣으세요…")
with colB:
    show_sgd = st.toggle("설명 사용(부정 단어)", value=True, help="부정 단어 TOP3를 계산")

if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp, stopwords)
    X = vec.transform([toks])
    X = maybe_apply_negpos_bonus(X, vec, MODELS)
    score = float(np.clip(reg.predict(X)[0], 1, 5))
    st.metric("예측 점수", f"{score:.2f} ★")

    if show_sgd:
        neg_top3 = negative_top3_across_rows(X, vec, sgd, tfidf_min=0.10, k=3)
        if neg_top3:
            parts = [f"{w} (Σ {v:.2f})" for w, v in neg_top3]
            st.markdown(f"**리뷰 속 부정적 단어 TOP3:** " + ", ".join(parts))

st.divider()

# ============= UI: 배치 예측 =============
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
            texts = df["review_text"].fillna("").astype(str).tolist()
            toks  = [tokenize_and_join(t, stopwords) for t in texts]
            X     = vec.transform(toks)
            X     = maybe_apply_negpos_bonus(X, vec, MODELS)

            pred  = np.clip(reg.predict(X), 1, 5)
            df_out = df.copy()
            df_out["예측 별점"] = np.round(pred, 2)

            # 표
            view_cols = []
            if "review_text" in df_out.columns: view_cols.append("review_text")
            if "review_date" in df_out.columns: view_cols.append("review_date")
            view_cols.append("예측 별점")
            st.dataframe(df_out.loc[:, view_cols].rename(columns={
                "review_text":"리뷰","review_date":"날짜"
            }), use_container_width=True)

            # 평균/위험도
            avg = float(np.round(pred.mean(), 2))
            level = risk_level(avg)
            c1, c2 = st.columns([1,1])
            with c1:
                st.metric("평균 평점", f"{avg:.2f} ★")
            with c2:
                st.markdown(
                    f"""<div style="padding:10px 12px;border-radius:10px;background:{risk_color(level)};color:#fff;display:inline-block;font-weight:600;">위험도: {level}</div>""",
                    unsafe_allow_html=True,
                )

            # 부정 TOP3 (High/Medium일 때만)
            if level in ("High","Medium") and show_sgd:
                neg_top3 = negative_top3_across_rows(X, vec, sgd, tfidf_min=0.10, k=3)
                if neg_top3:
                    parts = [f"{w} (Σ {v:.2f})" for w, v in neg_top3]
                    st.markdown(f"**리뷰 속 부정적 단어 TOP3:** " + ", ".join(parts))

            # 다운로드
            st.download_button(
                "결과 CSV 다운로드",
                df_out.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
