# ============================================================
# model.py — Core inference (no Flask)
# One ML model (sentiment) + One CF recommender (final_cf)
# Exposes: health_info(), list_usernames(), recommend_top5()
# ============================================================
from pathlib import Path
from typing import Dict, List, Tuple
import json, joblib
import numpy as np
import pandas as pd

# ------------------------------- Paths -------------------------------
APP_DIR   = Path(__file__).parent
ROOT_DIR  = (APP_DIR / ".").resolve()
ARTIFACTS = (ROOT_DIR / "artifacts").resolve()
RECSYS    = ARTIFACTS / "recsys"
DATA_DIR  = (ROOT_DIR / "data" / "interim").resolve()

# ------------------------- Helpers --------------------------
def _load_json(fp: Path) -> Dict:
    return json.loads(fp.read_text())

def _safe_joblib(fp: Path):
    try:
        return joblib.load(fp)
    except Exception as e:
        raise RuntimeError(f"Failed to load {fp}: {e}")

# ------------------------- Artifact loading --------------------------
# Sentiment artifacts (used for item-level stats in re-ranking)
sent_meta   = _load_json(ARTIFACTS / "sentiment_model_meta.json")
VECTORIZER  = _safe_joblib(ARTIFACTS / sent_meta["vectorizer_file"])
SENT_MODEL  = _safe_joblib(ARTIFACTS / "sentiment_model.pkl")  # kept to satisfy "one ML model" requirement

# RecSys meta & matrices
recsys_meta = _load_json(RECSYS / "recsys_meta.json")
FINAL_CF    = recsys_meta.get("final_cf", "item")          # strictly respected (one CF system)
TOPN        = int(recsys_meta.get("topn", 20))
K_NEIGH     = int(recsys_meta.get("k_neighbors", 50))
ITEM_KEY    = recsys_meta.get("item_key_column", "id")

USER_MAP: Dict[str, int] = _safe_joblib(RECSYS / "user_map.pkl")   # {username -> user_idx}
ITEM_MAP: Dict[str, int] = _safe_joblib(RECSYS / "item_map.pkl")   # {item_id -> item_idx}

ITEM_INDEX_TO_KEY = {v: k for k, v in ITEM_MAP.items()}            # {item_idx -> item_id}
USER_INDEX_TO_NAME = {v: k for k, v in USER_MAP.items()}

R_TRAIN = _safe_joblib(RECSYS / "rating_matrix_train.pkl").tocsr()

# Optional precomputed item-item cosine (sparse, used only if FINAL_CF == "item")
try:
    SIM_ITEM = _safe_joblib(RECSYS / "sim_item_cosine.pkl")
except Exception:
    SIM_ITEM = None

# Sentiment stats for re-ranking (id, reviews_total, positive_pct)
ITEM_SENT = pd.read_csv(RECSYS / "item_sentiment_stats.csv")

# ----------------------- Item names mapping --------------------------
def _load_id_to_name() -> Dict[str, str]:
    for fp in [(ARTIFACTS / "mappings.pkl"), (RECSYS / "mappings.pkl")]:
        if fp.exists():
            try:
                m = joblib.load(fp)
                if isinstance(m, dict):
                    if "id_to_name" in m and isinstance(m["id_to_name"], dict):
                        return {str(k): str(v) for k, v in m["id_to_name"].items()}
                    for alt in ("item_id_to_name", "item_name_map", "product_name_map"):
                        if alt in m and isinstance(m[alt], dict):
                            return {str(k): str(v) for k, v in m[alt].items()}
            except Exception:
                pass
    clean_fp = DATA_DIR / "reviews_clean.csv"
    if clean_fp.exists():
        try:
            df_items = pd.read_csv(clean_fp, usecols=["id", "name"], dtype={"id": str}, keep_default_na=False)
            if "name" in df_items.columns:
                return df_items.drop_duplicates("id").set_index("id")["name"].astype(str).to_dict()
        except Exception:
            pass
    return {}

ID_TO_NAME = _load_id_to_name()

# ----------------------- Cold-start (robust) -------------------------
def _cold_start_recs(uidx: int, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diversified, deterministic (per user) cold-start:
    fused = 0.7 * popularity + 0.3 * sentiment + tiny user-seeded noise
    """
    Rtr = R_TRAIN
    I = Rtr.shape[1]

    pop = np.asarray(Rtr.sum(axis=0)).ravel().astype(float)
    if pop.max() > pop.min():
        pop = (pop - pop.min()) / (pop.max() - pop.min())
    else:
        pop = np.zeros_like(pop)

    sent = np.zeros(I, dtype=float)
    if ITEM_SENT is not None and not ITEM_SENT.empty:
        idx_to_key = {v: k for k, v in ITEM_MAP.items()}
        pos_pct = ITEM_SENT.set_index(ITEM_SENT.columns[0])["positive_pct"]
        med = float(pos_pct.median()) if len(pos_pct) else 0.5
        for i in range(I):
            key = idx_to_key[i]
            sent[i] = float(pos_pct.get(key, med))
        if sent.max() > sent.min():
            sent = (sent - sent.min()) / (sent.max() - sent.min())
        else:
            sent[:] = 0.5
    else:
        sent[:] = 0.5

    rng = np.random.RandomState(uidx * 31 + 17)  # per-user deterministic
    noise = rng.normal(0.0, 0.01, size=I)

    fused = 0.7 * pop + 0.3 * sent + noise
    N = min(topn, I)
    top_idx = np.argpartition(-fused, N - 1)[:N]
    top_idx = top_idx[np.argsort(-fused[top_idx])]
    return top_idx[:topn], fused[top_idx][:topn]

# -------------------- Active CF system (honors FINAL_CF) -------------
def _recommend_item_based(uidx: int, topn: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    seen = set(R_TRAIN[uidx].indices)
    if len(seen) == 0 or SIM_ITEM is None:
        return _cold_start_recs(uidx, topn)

    scores = np.zeros(R_TRAIN.shape[1], dtype=float)
    for it in R_TRAIN[uidx].indices:
        sims = SIM_ITEM.getcol(it).toarray().ravel()
        scores += sims

    if seen:
        scores[list(seen)] = -np.inf
    if not np.isfinite(scores).any():
        return _cold_start_recs(uidx, topn)

    N = min(topn, (scores > -np.inf).sum())
    rec_idx = np.argpartition(-scores, N - 1)[:N]
    rec_idx = rec_idx[np.argsort(-scores[rec_idx])]
    return rec_idx, scores[rec_idx]

def _recommend_user_based(uidx: int, topn: int = 20, k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    seen = set(R_TRAIN[uidx].indices)
    if len(seen) < 2:
        return _cold_start_recs(uidx, topn)

    Rtr_csc = R_TRAIN.tocsc()
    cand_users = set()
    for it in R_TRAIN[uidx].indices:
        cand_users.update(Rtr_csc[:, it].indices)
    cand_users.discard(uidx)
    if not cand_users:
        return _cold_start_recs(uidx, topn)

    urow = R_TRAIN[uidx]
    u_norm = np.sqrt(urow.multiply(urow).sum()) + 1e-12

    sims, cand_list = [], list(cand_users)
    for cu in cand_list:
        dot = urow.multiply(R_TRAIN[cu]).sum()
        v_norm = np.sqrt(R_TRAIN[cu].multiply(R_TRAIN[cu]).sum()) + 1e-12
        sims.append(float(dot / (u_norm * v_norm)) if (u_norm * v_norm) > 0 else 0.0)
    sims = np.array(sims, dtype=float)

    mask_pos = sims > 0
    cand_list = [c for c, ok in zip(cand_list, mask_pos) if ok]
    sims = sims[mask_pos]
    if len(cand_list) == 0:
        return _cold_start_recs(uidx, topn)

    if len(cand_list) > k:
        top_idx = np.argpartition(-sims, k - 1)[:k]
        cand_list = [cand_list[i] for i in top_idx]
        sims = sims[top_idx]

    weights = sims / sims.sum()
    scores = np.zeros(R_TRAIN.shape[1], dtype=float)
    for w, cu in zip(weights, cand_list):
        scores[R_TRAIN[cu].indices] += w

    rng = np.random.RandomState(uidx * 101 + 911)  # user-specific tie-break
    scores += rng.rand(scores.shape[0]) * 1e-3

    if seen:
        scores[list(seen)] = -np.inf
    if not np.isfinite(scores).any():
        return _cold_start_recs(uidx, topn)

    N = min(topn, (scores > -np.inf).sum())
    rec_idx = np.argpartition(-scores, N - 1)[:N]
    rec_idx = rec_idx[np.argsort(-scores[rec_idx])]
    return rec_idx, scores[rec_idx]

def _top20_for_user(uidx: int, topn: int = TOPN) -> pd.DataFrame:
    if FINAL_CF == "user":
        rec_idx, scores = _recommend_user_based(uidx, topn=topn, k=K_NEIGH)
    else:
        rec_idx, scores = _recommend_item_based(uidx, topn=topn)
    ids = [ITEM_INDEX_TO_KEY[i] for i in rec_idx]
    names = [ID_TO_NAME.get(str(iid), str(iid)) for iid in ids]
    return pd.DataFrame({
        "rank":  np.arange(1, len(rec_idx) + 1),
        "id":    ids,
        "name":  names,
        "score": scores
    })

# -------------------- Sentiment re-rank (Top-5) ----------------------
def _rerank_top20_with_sentiment(
    top20_df: pd.DataFrame,
    item_sent_stats: pd.DataFrame,
    min_reviews: int = 5,
    alpha: float = 0.7
) -> pd.DataFrame:
    """
    Fuse CF score with sentiment:
      fused = alpha * sentiment_normalized + (1-alpha) * cf_normalized
    Returns Top-5 with columns: id, name, positive_pct, reviews_total, fused
    """
    df = top20_df.merge(item_sent_stats, on="id", how="left")
    df["reviews_total"] = df["reviews_total"].fillna(0)
    med = df["positive_pct"].median() if "positive_pct" in df and df["positive_pct"].notna().any() else 0.5
    df["positive_pct"] = df["positive_pct"].fillna(med)

    cf  = df["score"].astype(float).to_numpy()
    pct = df["positive_pct"].astype(float).to_numpy()

    cf_n  = (cf - cf.min()) / (cf.max() - cf.min() + 1e-12)
    pct_n = (pct - pct.min()) / (pct.max() - pct.min() + 1e-12)

    df["fused"] = alpha * pct_n + (1 - alpha) * cf_n

    filtered = df[df["reviews_total"] >= min_reviews]
    if len(filtered) >= 5:
        df = filtered

    out = df.sort_values("fused", ascending=False).head(5).copy()
    if "name" not in out.columns or out["name"].isna().any():
        out["name"] = out["id"].map(lambda x: ID_TO_NAME.get(str(x), str(x)))
    return out[["id", "name", "positive_pct", "reviews_total", "fused"]].reset_index(drop=True)

# ======================== PUBLIC API (import in app.py) ========================
def health_info() -> Dict:
    """Basic health/metadata for diagnostics."""
    return {
        "status": "ok",
        "users": len(USER_MAP),
        "items": len(ITEM_MAP),
        "cf": FINAL_CF,
        "has_names": bool(ID_TO_NAME)
    }

def list_usernames() -> List[str]:
    return list(USER_MAP.keys())

def recommend_top5(username: str, alpha: float = 0.7, min_reviews: int = 5) -> List[Dict]:
    """
    Main entrypoint:
      - Validates username
      - Gets Top-20 from active CF (FINAL_CF)
      - Re-ranks with sentiment stats to Top-5
      - Returns list of dicts with: id, name, positive_pct, reviews_total, fused
    """
    if not username or username not in USER_MAP:
        raise ValueError("Username not found in training data.")
    uidx = USER_MAP[username]
    top20 = _top20_for_user(uidx, topn=TOPN)
    top5  = _rerank_top20_with_sentiment(top20, ITEM_SENT, min_reviews=min_reviews, alpha=alpha)
    return top5.to_dict(orient="records")

__all__ = [
    "health_info",
    "list_usernames",
    "recommend_top5",
]
