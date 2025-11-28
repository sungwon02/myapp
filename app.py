# app.py — Streamlit 리뷰 예측(회귀 + Kiwi + neg/pos + SGD기반 부정 토큰 TOP3)
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, re
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

# 벡터/모델 경로 (pkl/skops 모두 지원)
VEC_PKL   = MODELS / "tfidf_vectorizer.pkl"
VEC_SKOPS = MODELS / "tfidf_vectorizer.skops"
REG_JOB   = MODELS / "rf_reg.joblib"
REG_SKOPS = MODELS / "rf_reg.skops"
SGD_JOB   = MODELS / "sgd_logistic_cls.joblib"
SGD_SKOPS = MODELS / "sgd_logistic_cls.skops"
SUMMARY   = MODELS / "tfidf_summary.json"

# (선택) skops 로더
ALLOWED_PREFIXES = ("sklearn.", "numpy.", "scipy.", "xgboost.", "lightgbm.")
try:
    from skops.io import load as skops_load, get_untrusted_types
    HAS_SKOPS = True
except Exception:
    HAS_SKOPS = False


def _assert_files_exist(paths):
    miss = [p for p in paths if not p.exists()]
    if miss:
        st.error(f"모델 파일 누락: {[str(p) for p in miss]}")
        st.stop()


def _safe_skops_load(path: Path):
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
# 전처리/토크나이즈 (Kiwi 고정)
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


# ---- Kiwi 토크나이저 고정 (predict_safe.py와 동일 품사군) ----
try:
    from kiwipiepy import Kiwi
    _KIWI = Kiwi()
    _TOKENIZER_NAME = "kiwi"
except Exception as e:
    raise RuntimeError(
        "Kiwi 토크나이저를 로드하지 못했습니다. "
        "requirements.txt에 'kiwipiepy'를 추가/설치하고 다시 시도하세요."
    ) from e


def _tokenize(text: str) -> list[str]:
    return [t.form for t in _KIWI.tokenize(text)
            if t.tag.startswith(("N", "V", "MAG", "IC", "XR", "MM"))]


def tokenize_and_join(s: str, stopwords: set[str] | None = None) -> str:
    toks = _tokenize(_clean_text(s))
    if stopwords:
        toks = [t for t in toks if t not in stopwords]
    return " ".join(toks)


# =========================
# neg/pos 열가중
# =========================
def _apply_column_scaling_csc(X_csc, vocab: dict, terms: list, scale: float):
    hits = [vocab[t] for t in terms if t in vocab]
    for j in hits:
        start, end = X_csc.indptr[j], X_csc.indptr[j + 1]
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
                X_csc = _apply_column_scaling_csc(X_csc, vocab, neg_terms, neg_bonus)
            if pos_terms and pos_bonus != 1.0:
                X_csc = _apply_column_scaling_csc(X_csc, vocab, pos_terms, pos_bonus)
            return X_csc.tocsr(copy=False)
    except Exception:
        return X_csr
    return X_csr


# =========================
# 모델 로드 (캐시)
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    # 벡터라이저
    if VEC_SKOPS.exists() and HAS_SKOPS:
        vec = _safe_skops_load(VEC_SKOPS)
    else:
        _assert_files_exist([VEC_PKL])
        vec = joblib.load(VEC_PKL)

    # RF 회귀
    if REG_SKOPS.exists() and HAS_SKOPS:
        reg = _safe_skops_load(REG_SKOPS)
    else:
        _assert_files_exist([REG_JOB])
        reg = joblib.load(REG_JOB)
    reg = _patch_rf_monotonic(reg)

    # SGD 분류 (설명용)
    cls = None
    if SGD_SKOPS.exists() and HAS_SKOPS:
        cls = _safe_skops_load(SGD_SKOPS)
    elif SGD_JOB.exists():
        cls = joblib.load(SGD_JOB)

    # stopwords
    stop = set()
    for sw in [ROOT.parent / "stopwords_ko.txt",
               ROOT / "stopwords_ko.txt",
               MODELS / "stopwords_ko.txt",
               MODELS / "stopwords_ko1.txt"]:
        if Path(sw).exists():
            with open(sw, encoding="utf-8") as f:
                stop = {x.strip() for x in f if x.strip()}
            break

    dbg = {
        "tokenizer": _TOKENIZER_NAME,
        "stopwords": len(stop),
        "vectorizer_features": len(vec.get_feature_names_out()) if hasattr(vec, "get_feature_names_out") else None,
        "rf_n_features_in_": getattr(reg, "n_features_in_", None),
        "negpos_summary": SUMMARY.exists(),
        "has_sgd": cls is not None,
    }
    return vec, reg, cls, stop, dbg


# =========================
# 위험도 판정
# =========================
def risk_level(avg_score: float) -> str:
    if avg_score > 4.06:
        return "Safe"
    if avg_score > 3.84:
        return "Low"
    if avg_score > 3.60:
        return "Medium"
    return "High"


def risk_color(level: str) -> str:
    return {
        "Safe": "#2e7d32",
        "Low": "#558b2f",
        "Medium": "#f9a825",
        "High": "#c62828",
    }.get(level, "#333333")


# =========================
# 부정 토큰 TOP3 (네가 말한 방식 + 화이트리스트)
# =========================
def worst_tokens_from_batch_by_margin(X, vec, cls, stopwords: set[str], topk=3):
    """
    X: (n_samples, n_features) sparse
    vec: vectorizer
    cls: SGDClassifier
    stopwords: 우리가 읽어온 불용어 세트
    """
    if cls is None or not hasattr(cls, "coef_"):
        return []

    feats = vec.get_feature_names_out()
    coef = cls.coef_
    classes = getattr(cls, "classes_", np.arange(coef.shape[0]))

    # 1점 / 5점 계수 찾기
    i1 = np.where(classes == 1)[0]
    i5 = np.where(classes == 5)[0]
    if not len(i1) or not len(i5):
        return []

    w1 = coef[int(i1[0])]
    w5 = coef[int(i5[0])]
    delta = w5 - w1  # (coef5 - coef1)

    # ── 여기 추가된 화이트리스트 ──
    DISPLAY_WHITELIST = {
        # 서비스/응대
        "서비스", "불친절", "친절", "응대", "직원", "사장", "사장님",
        # 가격/가성비
        "가격", "비싸", "저렴", "가성비", "할인",
        # 위생/청결/이물질
        "위생", "청결", "더럽", "더러", "깨끗", "깔끔", "냄새", "벌레", "머리카락", "이물질",
        # 대기/접근성
        "대기", "웨이팅", "줄", "주차", "자리",
        # 맛/품질
        "맛", "맛없", "신선", "퀄리티", "식감",
        # 양/구성
        "양", "구성", "창렬",
        # 분위기
        "분위기", "시끄럽","공간", "인테리어",
        # 기타 부정
        "오래", "느림"
    }
    
    BLOCKLIST = {
        "다시","진짜","정말","아쉽",
    }

    agg = {}
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        contrib = delta * row
        neg_idx = np.where(contrib < 0)[0]
        for j in neg_idx:
            if row[j] == 0:
                continue
            agg[j] = agg.get(j, 0.0) + contrib[j]

    if not agg:
        return []

    sorted_items = sorted(agg.items(), key=lambda kv: kv[1])  # 가장 음수부터

    cleaned: list[tuple[str, float]] = []
    for j, val in sorted_items:
        tok = feats[j]

        # 우선 불용어 / 블랙리스트 / 한글자 / 숫자·기호 제거
        if tok in stopwords or tok in BLOCKLIST:
            continue
        if len(tok) < 2:
            continue
        if re.fullmatch(r"[0-9\W_]+", tok):
            continue

        # 화이트리스트 안에 있으면 바로 후보
        if tok in DISPLAY_WHITELIST:
            cleaned.append((tok, val))

        if len(cleaned) >= topk:
            break

    # 화이트리스트로 3개를 못 채웠으면 남은 것들에서 그냥 채움
    if len(cleaned) < topk:
        for j, val in sorted_items:
            tok = feats[j]
            # 여기서도 블랙리스트 제거
            if tok in stopwords or tok in BLOCKLIST or len(tok) < 2:
                continue
            if re.fullmatch(r"[0-9\W_]+", tok):
                continue
            if (tok, val) in cleaned:
                continue
            cleaned.append((tok, val))
            if len(cleaned) >= topk:
                break

    labels = [f"{tok}" for tok, val in cleaned]
    return labels


# ==========================================================
#                          UI
# ==========================================================
st.title("⭐ 가게 리뷰 예측 ⭐")

vec, reg, cls, stopwords, dbg = load_models()

# ── 단일 예측
st.subheader("단일 텍스트 예측")
inp = st.text_area("리뷰 텍스트 입력", height=160, placeholder="리뷰를 붙여넣으세요…")
if st.button("예측하기") and inp.strip():
    toks = tokenize_and_join(inp, stopwords)
    X = vec.transform([toks])
    X = maybe_apply_negpos_bonus(X, vec, MODELS)
    score = float(np.clip(reg.predict(X)[0], 1, 5))
    st.metric("예측 점수", f"{score:.2f} ★")

st.divider()

# ── 배치 예측
st.subheader("대량 리뷰 예측 (CSV 업로드)")
csv = st.file_uploader("CSV 업로드 (필수 Column: review_text)", type=["csv"])

if csv is not None:
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        st.error(f"CSV 로딩 실패: {e}")
    else:
        if "review_text" not in df.columns:
            st.error("CSV에 'review_text' 컬럼이 없습니다.")
        else:
            toks = df["review_text"].fillna("").astype(str).map(lambda s: tokenize_and_join(s, stopwords))
            X = vec.transform(toks)
            Xb = maybe_apply_negpos_bonus(X, vec, MODELS)
            df["pred_score"] = np.clip(reg.predict(Xb), 1, 5).round(2)

            # 표시용
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
                    "pred_score": "예측 별점",
                }
            )
            st.dataframe(df_view, use_container_width=True)

            # 평균 & 위험도
            avg = float(df["pred_score"].mean())
            level = risk_level(avg)
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("평균 평점", f"{avg:.2f} ★")
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

            # 부정 토큰 TOP3
            if level in {"High", "Medium"}:
                if cls is not None:
                    worst3 = worst_tokens_from_batch_by_margin(Xb, vec, cls, stopwords, topk=3)
                    st.markdown("**리뷰 속 부정적 단어 TOP3:** " + ", ".join(worst3))
                else:
                    st.markdown("**리뷰 속 부정적 단어 TOP3:** 분류 모델이 없어 표시할 수 없습니다.")

            # 다운로드
            st.download_button(
                "결과 CSV 다운로드",
                df.to_csv(index=False, encoding="utf-8-sig"),
                file_name="predictions.csv",
                mime="text/csv",
            )
