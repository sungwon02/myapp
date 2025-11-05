# app.py — Streamlit 리뷰 예측 (predict_safe.py 방식 반영)
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, re, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="리뷰 예측/분석", page_icon="⭐", layout="wide")

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

# 파일 경로(우선 skops, 없으면 pkl/joblib)
VEC_SKOPS = MODELS / "tfidf_vectorizer.skops"
VEC_PKL   = MODELS / "tfidf_vectorizer.pkl"
SGD_SKOPS = MODELS / "sgd_logistic_cls.skops"
SGD_JOB   = MODELS / "sgd_logistic_cls.joblib"
RF_SKOPS  = MODELS / "rf_reg.skops"
RF_JOB    = MODELS / "rf_reg.joblib"
SUMMARY   = MODELS / "tfidf_summary.json"

# ============== skops 안전 로더 ==============
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False

ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")

def safe_skops_load(path: Path):
    p = str(path)
    types = None
    if HAS_SKOPS:
        try:
            # skops 0.10+ 시그니처 호환
            try:
                types = get_untrusted_types(file=p)
            except TypeError:
                types = get_untrusted_types(path=p)
        except TypeError:
            types = get_untrusted_types()
    bad = [t for t in (types or []) if not t.startswith(ALLOWED_PREFIXES)]
    if bad:
        raise RuntimeError(
            "skops 파일에 비허용 타입이 포함되어 있습니다.\n"
            f"- 파일: {p}\n- 비허용 예: {bad[:5]} ..."
        )
    return skops_load(p, trusted=types) if HAS_SKOPS else None

# ============== RF 호환 패치 ==============
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
# 전처리/토크나이즈 (predict_safe와 동일)
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

def get_tokenizer():
    # 간결화: regex 기본, mecab/kiwi 있으면 자동 사용
    try:
        from mecab import MeCab
        m = MeCab()
        def tok(text):
            return [w for (w, p) in m.pos(text)
                    if p.startswith(("NN","VV","VA","MAG","IC","XR"))]
        return "mecab_python", tok
    except Exception:
        pass
    try:
        from kiwipiepy import Kiwi
        kiwi = Kiwi()
        def tok(text):
            return [t.form for t in kiwi.tokenize(text)
                    if t.tag.startswith(("N","V","MAG","IC","XR","MM"))]
        return "kiwi", tok
    except Exception:
        pass
    def tok(text):
        return re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    return "simple", tok

TOK_NAME, TOKENIZE = get_tokenizer()

def tokenize_and_join(text: str, stopwords:set) -> str:
    toks = TOKENIZE(_clean_text(text))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return " ".join(toks)

# =========================
# neg/pos 열가중 (predict_safe와 동일)
# =========================
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

# =========================
# 모델 로드 (predict_safe 규칙 반영)
# =========================
def _load_vectorizer(model_expected_n: int | None):
    # skops 우선, 불일치 시 pkl 시도
    candidates = []
    if VEC_SKOPS.exists() and HAS_SKOPS:
        candidates.append(("skops", VEC_SKOPS))
    if VEC_PKL.exists():
        candidates.append(("pkl", VEC_PKL))
    if not candidates:
        st.error("tfidf_vectorizer.(skops|pkl) 가 없습니다."); st.stop()

    last_err = None
    for kind, path in candidates:
        try:
            vec = safe_skops_load(path) if kind == "skops" else joblib.load(path)
            n_vec = len(vec.get_feature_names_out())
            if (model_expected_n is None) or (n_vec == model_expected_n):
                return vec
            else:
                last_err = (f"{path.name} features={n_vec}, "
                            f"but model expects {model_expected_n}")
        except Exception as e:
            last_err = f"{path.name} load failed: {e!r}"
    st.error(f"적합한 TF-IDF 벡터라이저를 찾지 못했습니다. 마지막 오류: {last_err}")
    st.stop()

def _load_sgd():
    if HAS_SKOPS and SGD_SKOPS.exists():
        return safe_skops_load(SGD_SKOPS)
    if SGD_JOB.exists():
        return joblib.load(SGD_JOB)
    return None  # SGD 선택적

def _load_rf():
    if HAS_SKOPS and RF_SKOPS.exists():
        return _patch_rf_monotonic(safe_skops_load(RF_SKOPS))
    if RF_JOB.exists():
        return _patch_rf_monotonic(joblib.load(RF_JOB))
    st.error("rf_reg.(skops|joblib) 이 없습니다."); st.stop()

@st.cache_resource(show_spinner=True)
def load_assets():
    # stopwords: 상위/동일 폴더 탐색
    stop = set()
    for sw in [ROOT / "stopwords_ko.txt", MODELS / "stopwords_ko.txt"]:
        if sw.exists():
            stop = {x.strip() for x in sw.read_text(encoding="utf-8").splitlines() if x.strip()}
            break

    sgd = _load_sgd()  # 없으면 None
    rf  = _load_rf()

    # 벡터라이저는 "SGD 기준"으로 맞추고, RF와도 일치 확인
    model_expected_n_cls = getattr(sgd, "n_features_in_", None) if sgd is not None else None
    vec = _load_vectorizer(model_expected_n_cls)

    n_vec = len(vec.get_feature_names_out())
    model_expected_n_reg = getattr(rf, "n_features_in_", None)
    if (model_expected_n_reg is not None) and (n_vec != model_expected_n_reg):
        st.error(
            f"[특징 불일치] vectorizer={n_vec}, rf_reg expects={model_expected_n_reg}\n"
            "→ 학습 시 동일 벡터라이저로 훈련했는지 확인하세요."
        )
        st.stop()

    return vec, sgd, rf, stop

# =========================
# SGD 설명(탑텀) 도우미
# =========================
def explain_top_terms(vec, cls, Xrow: csr_matrix, topk=8):
    feats = vec.get_feature_names_out()
    try:
        proba = cls.predict_proba(Xrow)[0]
    except Exception:
        proba = None
    y_label = cls.predict(Xrow)[0]
    classes = getattr(cls, "classes_", None)
    if classes is None:
        y_idx = int(y_label) - 1
    else:
        idx_arr = np.where(classes == y_label)[0]
        y_idx = int(idx_arr[0]) if len(idx_arr) else int(y_label) - 1
    if not hasattr(cls, "coef_"):
        return int(y_label), (proba.tolist() if proba is not None else None), []
    coef = cls.coef_[y_idx]
    x = Xrow.toarray().ravel()
    contrib = coef * x
    idx = np.argsort(contrib)[::-1]
    rows = []
    feats_arr = np.asarray(feats)
    for j in idx:
        if x[j] == 0:
            continue
        rows.append({
            "term": str(feats_arr[j]),
            "tfidf": float(x[j]),
            "coef": float(coef[j]),
            "contrib": float(contrib[j])
        })
        if len(rows) >= topk:
            break
    return int(y_label), (proba.tolist() if proba is not None else None), rows

# =========================
# 위험도(이전 UI 유지)
# =========================
def risk_level(avg_score: float) -> str:
    if avg_score >= 4.10: return "Safe"
    if avg_score >= 4.00: return "Low"
    if avg_score >= 3.90: return "Medium"
    return "High"

def risk_color(level: str) -> str:
    return {"Safe":"#2e7d32","Low":"#558b2f","Medium":"#f9a825","High":"#c62828"}.get(level,"#333")

# ==========================================================
#                          UI
# ==========================================================
st.title("⭐ 리뷰 예측 데모 (predict_safe 동기화)")

vec, sgd, reg, stop = load_assets()
st.caption(f"Tokenizer = **{TOK_NAME}**, Stopwords = **{len(stop)}**개, "
           f"벡터 특성수 = **{len(vec.get_feature_names_out())}**")

show_sgd = st.toggle("SGD 분류 결과/설명도 함께 보기", value=False)

# ── 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp, stop)
    X = vec.transform([toks])

    # 열가중(neg/pos) 동기화
    X2 = maybe_apply_negpos_bonus(X, vec, MODELS)

    # RF 회귀
    score = float(np.clip(reg.predict(X2)[0], 1, 5))
    st.metric("RF 예측 점수", f"{score:.2f} ★")

    # SGD (옵션)
    if show_sgd and sgd is not None:
        y, proba, terms = explain_top_terms(vec, sgd, X2, topk=8)
        st.write(f"**SGD 예측 클래스(별점)**: {y}")
        if proba is not None:
            st.write("**확률**:", np.round(proba, 3))
        st.write("**Top terms (coef·contrib)**:")
        st.json(terms)

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
            toks = df["review_text"].fillna("").astype(str).map(lambda s: tokenize_and_join(s, stop))
            X = vec.transform(toks)
            X2 = maybe_apply_negpos_bonus(X, vec, MODELS)

            # RF 회귀
            df["pred_star_reg"] = np.clip(reg.predict(X2), 1, 5).round(2)

            if show_sgd and sgd is not None:
                # SGD 분류 + 확률 + 설명
                cls_pred = sgd.predict(X2)
                df["pred_star_cls"] = cls_pred
                try:
                    probas = sgd.predict_proba(X2)
                    df["pred_confidence"] = np.max(probas, axis=1).round(3)
                except Exception:
                    df["pred_confidence"] = np.nan
                # top_terms JSON (상위 5개로 축약)
                rows = []
                for i in range(X2.shape[0]):
                    _, _, contribs = explain_top_terms(vec, sgd, X2[i], topk=5)
                    rows.append(json.dumps(contribs, ensure_ascii=False))
                df["top_terms"] = rows

            # 화면 표시
            view_cols = []
            if "review_text" in df.columns: view_cols.append("review_text")
            if "review_date" in df.columns: view_cols.append("review_date")
            view_cols.append("pred_star_reg")
            if show_sgd and sgd is not None:
                view_cols += ["pred_star_cls","pred_confidence","top_terms"]

            df_view = df.loc[:, view_cols].rename(columns={
                "review_text":"리뷰","review_date":"날짜","pred_star_reg":"RF 예측 별점",
                "pred_star_cls":"SGD 예측 별점","pred_confidence":"SGD 확신도"
            })

            st.dataframe(df_view, use_container_width=True)

            # 평균 & 위험도
            avg = float(df["pred_star_reg"].mean())
            level = risk_level(avg)
            c1, c2 = st.columns([1,1])
            with c1: st.metric("RF 평균 평점", f"{avg:.2f} ★")
            with c2:
                st.markdown(
                    f"""<div style="padding:10px 12px;border-radius:10px;
                                   background:{risk_color(level)};color:#fff;
                                   display:inline-block;font-weight:600;">
                            위험도: {level}
                        </div>""",
                    unsafe_allow_html=True,
                )

            # 다운로드
            st.download_button(
                "결과 CSV 다운로드",
                df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
