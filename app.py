# app.py — Streamlit 리뷰 예측( predict_safe.py 와 동치 파이프라인 )
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────────────────────
# (0) 커널/스레드 안전 (predict_safe.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ─────────────────────────────────────────────────────────────────────────────
# 기본 경로
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")
ROOT   = Path(__file__).resolve().parent
MODELS = ROOT / "models"

# ─────────────────────────────────────────────────────────────────────────────
# skops 로더 (있으면 사용) + 안전 로드
# ─────────────────────────────────────────────────────────────────────────────
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
        raise RuntimeError(
            "skops 파일에 비허용 타입이 포함되어 있습니다.\n"
            f"- 파일: {p}\n- 비허용 예: {bad[:5]} ..."
        )
    return skops_load(p, trusted=types)

def _patch_rf_monotonic(reg):
    # RandomForestRegressor 하위 트리 monotonic_cst 누락 보정
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        rf = None
        if hasattr(reg, "steps"):  # Pipeline
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

# ─────────────────────────────────────────────────────────────────────────────
# 전처리/토크나이즈 (predict_safe.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────
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

@st.cache_data(show_spinner=False)
def load_stopwords() -> set[str]:
    # stopwords_ko.txt가 models/ 또는 상위에 있으면 불러오기
    for sw in [ROOT / "stopwords_ko.txt", MODELS / "stopwords_ko.txt", ROOT.parent / "stopwords_ko.txt"]:
        if sw.exists():
            with open(sw, encoding="utf-8") as f:
                return {x.strip() for x in f if x.strip()}
    return set()

def tokenize_and_join(s: str, stop:set[str]) -> str:
    toks = TOKENIZE(clean_text(s))
    if stop:
        toks = [t for t in toks if t not in stop]
    return " ".join(toks)

# ─────────────────────────────────────────────────────────────────────────────
# neg/pos 열가중 (predict_safe.py 동일)
# ─────────────────────────────────────────────────────────────────────────────
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
        with open(summ, "r", encoding="utf-8") as f:
            js = json.load(f)
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

# ─────────────────────────────────────────────────────────────────────────────
# 벡터라이저/모델 로드 (predict_safe.py와 동일한 우선순위)
# ─────────────────────────────────────────────────────────────────────────────
def load_vectorizer_with_compat(base: Path, model_expected_n: int|None):
    cands = []
    sk = base / "tfidf_vectorizer.skops"
    pk = base / "tfidf_vectorizer.pkl"
    if sk.exists() and HAS_SKOPS: cands.append(("skops", sk))
    if pk.exists():               cands.append(("pkl",   pk))
    if not cands:
        raise FileNotFoundError("tfidf_vectorizer.(skops|pkl) 가 없습니다.")
    last_err, chosen = None, None
    for kind, path in cands:
        try:
            vec = safe_skops_load(path) if kind=="skops" else joblib.load(path)
            n_vec = len(vec.get_feature_names_out())
            if (model_expected_n is None) or (n_vec == model_expected_n):
                chosen = vec; break
            else:
                last_err = f"'{path.name}' has {n_vec} features, model expects {model_expected_n}"
        except Exception as e:
            last_err = f"load fail {path.name}: {e!r}"
    if chosen is None:
        raise ValueError(f"적합한 TF-IDF 벡터라이저를 찾지 못했습니다. 마지막 오류: {last_err}")
    return chosen

@st.cache_resource(show_spinner=True)
def load_assets():
    base = MODELS
    # 1) sgd classifier (설명/부정 토큰 추출용)
    if HAS_SKOPS and (base / "sgd_logistic_cls.skops").exists():
        cls = safe_skops_load(base / "sgd_logistic_cls.skops")
    elif (base / "sgd_logistic_cls.joblib").exists():
        cls = joblib.load(base / "sgd_logistic_cls.joblib")
    else:
        st.error("sgd_logistic_cls.(skops|joblib) 파일이 없습니다.")
        st.stop()
    model_expected_n_cls = getattr(cls, "n_features_in_", None)

    # 2) rf regressor
    if HAS_SKOPS and (base / "rf_reg.skops").exists():
        reg = safe_skops_load(base / "rf_reg.skops")
    elif (base / "rf_reg.joblib").exists():
        reg = joblib.load(base / "rf_reg.joblib")
    else:
        st.error("rf_reg.(skops|joblib) 파일이 없습니다.")
        st.stop()
    reg = _patch_rf_monotonic(reg)
    model_expected_n_reg = getattr(reg, "n_features_in_", None)

    # 3) vectorizer (분류 기준으로 맞추고, 회귀와도 일치 확인)
    vec = load_vectorizer_with_compat(base, model_expected_n_cls)
    n_vec = len(vec.get_feature_names_out())
    if (model_expected_n_reg is not None) and (n_vec != model_expected_n_reg):
        st.error(
            "[특징 불일치] 분류 모델과 맞는 벡터라이저를 찾았지만, 회귀 모델과는 피처 수가 다릅니다.\n"
            f"- vectorizer: {n_vec}, rf_reg expects: {model_expected_n_reg}"
        )
        st.stop()

    # stopwords(optional)
    stop = load_stopwords()
    return vec, cls, reg, stop, base

# ─────────────────────────────────────────────────────────────────────────────
# 위험도/색상
# ─────────────────────────────────────────────────────────────────────────────
def risk_level(avg_score: float) -> str:
    if avg_score >= 4.10: return "Safe"
    if avg_score >= 4.00: return "Low"
    if avg_score >= 3.90: return "Medium"
    return "High"

def risk_color(level: str) -> str:
    return {"Safe":"#2e7d32","Low":"#558b2f","Medium":"#f9a825","High":"#c62828"}.get(level,"#333")

# ─────────────────────────────────────────────────────────────────────────────
# 부정 기여 토큰 TOP-K (멀티라인 전체 집계)
#  - 각 리뷰별로 SGD가 예측한 class의 coef를 사용해 X.multiply(coef) 후 열 방향 합
#  - 합계가 가장 음수(부정적)인 피처 상위 K 반환
# ─────────────────────────────────────────────────────────────────────────────
def negative_topk_tokens(vec, cls, X:csr_matrix, k=3):
    feats = vec.get_feature_names_out()
    classes = getattr(cls, "classes_", None)
    y = cls.predict(X)
    totals = np.zeros(X.shape[1], dtype=np.float64)

    # 클래스별로 묶어 벡터화된 곱 수행 (빠름)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if idx.size == 0:
            continue
        Xsub = X[idx]
        if classes is None:
            y_idx = int(c) - 1
        else:
            pos = np.where(classes == c)[0]
            y_idx = int(pos[0]) if len(pos) else int(c) - 1
        coef = np.asarray(cls.coef_[y_idx]).ravel()  # (n_features,)
        # 요소곱 후 열합 (sparse-friendly)
        contrib_sum = Xsub.multiply(coef).sum(axis=0).A1  # shape (n_features,)
        totals += contrib_sum

    # 가장 음수인 항목 k개
    neg_idx = np.argsort(totals)[:k]
    out = []
    for j in neg_idx:
        if totals[j] >= 0:
            break
        out.append((feats[j], float(totals[j])))
    return out  # [(token, total_contrib),...]

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("⭐ 리뷰 별점 예측)")

vec, cls, reg, stop, base = load_assets()

st.caption(f"tokenizer = **{TOK_NAME}**, stopwords = {len(stop)}개")

# ── 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp, stop)
    X = vec.transform([toks])
    X = maybe_apply_negpos_bonus(X, vec, base)
    score = float(np.clip(reg.predict(X)[0], 1, 5))
    st.metric("예측 별점", f"{score:.2f} ★")

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
            texts = df["review_text"].fillna("").astype(str).tolist()
            toks  = [tokenize_and_join(t, stop) for t in texts]
            X = vec.transform(toks)
            X = maybe_apply_negpos_bonus(X, vec, base)

            # 예측 (predict_safe.py 와 동일: RF 회귀 점수 → 1~5 클립)
            pred = np.clip(reg.predict(X), 1, 5).round(2)

            # 화면 표시용
            view_cols = []
            if "review_text" in df.columns: view_cols.append("review_text")
            if "review_date" in df.columns: view_cols.append("review_date")
            view_cols.append("pred_score")

            out = df.copy()
            out["pred_score"] = pred
            out_view = out.loc[:, view_cols].rename(columns={
                "review_text":"리뷰", "review_date":"날짜", "pred_score":"예측 별점"
            })
            st.dataframe(out_view, use_container_width=True)

            # 평균 & 위험도
            avg = float(out["pred_score"].mean())
            level = risk_level(avg)

            c1, c2 = st.columns([1,1])
            with c1:
                st.metric("평균 평점", f"{avg:.2f} ★")
            with c2:
                st.markdown(
                    f"""<div style="padding:10px 12px;border-radius:10px;
                                 background:{risk_color(level)};color:#fff;
                                 display:inline-block;font-weight:600;">
                         위험도: {level}
                        </div>""",
                    unsafe_allow_html=True
                )

            # 위험도가 Medium/High 이면 부정적 기여 토큰 TOP3
            if level in ("Medium", "High"):
                neg_top3 = negative_topk_tokens(vec, cls, X, k=3)
                if neg_top3:
                    pretty = " · ".join([f"{tok} (∑ {val:.2f})" for tok, val in neg_top3])
                    st.markdown(
                        f"**리뷰 속 부정적 단어 TOP3**: {pretty}"
                    )

            # 다운로드
            st.download_button(
                "결과 CSV 다운로드",
                out.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
