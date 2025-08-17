# ============================================================
# Phase 8 — Deployment (Flask, local) - Version 4
# Why: Serve personalized recommendations via a web interface.
# app.py — Single-CF + Sentiment re-rank (Top-5) with Item Names
# ============================================================
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, json, random

# ------------------------------- Paths -------------------------------
APP_DIR   = Path(__file__).parent
ROOT_DIR  = (APP_DIR / "..").resolve()
ARTIFACTS = (ROOT_DIR / "artifacts").resolve()
RECSYS    = ARTIFACTS / "recsys"
DATA_DIR  = (ROOT_DIR / "data" / "interim").resolve()

# ------------------------- Helpers --------------------------
def _load_json(fp: Path):
    return json.loads(fp.read_text())

def _safe_joblib(fp: Path):
    try:
        return joblib.load(fp)
    except Exception as e:
        raise RuntimeError(f"Failed to load {fp}: {e}")

# ------------------------- Artifact loading --------------------------
# Sentiment artifacts (Phase 4)
sent_meta   = _load_json(ARTIFACTS / "sentiment_model_meta.json")
vectorizer  = _safe_joblib(ARTIFACTS / sent_meta["vectorizer_file"])
sent_model  = _safe_joblib(ARTIFACTS / "sentiment_model.pkl")

# RecSys meta & matrices (Phases 5–6)
recsys_meta = _load_json(RECSYS / "recsys_meta.json")
final_cf    = recsys_meta.get("final_cf", "item")          # "user" or "item"
TOPN        = int(recsys_meta.get("topn", 20))
K_NEIGH     = int(recsys_meta.get("k_neighbors", 50))
ITEM_KEY    = recsys_meta.get("item_key_column", "id")     # expected "id"

user_map = _safe_joblib(RECSYS / "user_map.pkl")           # {username -> user_idx}
item_map = _safe_joblib(RECSYS / "item_map.pkl")           # {item_id -> item_idx}
item_index_to_key = {v: k for k, v in item_map.items()}    # {item_idx -> item_id}
user_index_to_name = {v: k for k, v in user_map.items()}

R_train = _safe_joblib(RECSYS / "rating_matrix_train.pkl").tocsr()

# Optional: precomputed item-item cosine (sparse)
try:
    SIM_ITEM = _safe_joblib(RECSYS / "sim_item_cosine.pkl")
except Exception:
    SIM_ITEM = None

# Sentiment stats for re-ranking (id, reviews_total, positive_pct)
item_sent = pd.read_csv(RECSYS / "item_sentiment_stats.csv")

# ----------------------- Item names mapping --------------------------
# We try artifacts/mappings.pkl first; if missing, fallback to reviews_clean.csv
def _load_id_to_name():
    # 1) Try a consolidated mappings.pkl
    for fp in [(ARTIFACTS / "mappings.pkl"), (RECSYS / "mappings.pkl")]:
        if fp.exists():
            try:
                m = joblib.load(fp)
                if isinstance(m, dict):
                    # common keys we used in notebook
                    if "id_to_name" in m and isinstance(m["id_to_name"], dict):
                        return {str(k): str(v) for k, v in m["id_to_name"].items()}
                    # sometimes stored as item_id_to_name, product_name_map, etc.
                    for alt in ("item_id_to_name", "item_name_map", "product_name_map"):
                        if alt in m and isinstance(m[alt], dict):
                            return {str(k): str(v) for k, v in m[alt].items()}
            except Exception:
                pass
    # 2) Fallback: build from cleaned reviews
    clean_fp = DATA_DIR / "reviews_clean.csv"
    if clean_fp.exists():
        try:
            df_items = pd.read_csv(clean_fp, usecols=["id", "name"], dtype={"id": str}, keep_default_na=False)
            if "name" in df_items.columns:
                return df_items.drop_duplicates("id").set_index("id")["name"].astype(str).to_dict()
        except Exception:
            pass
    # 3) Final fallback: empty (UI will show id in place of name)
    return {}

id_to_name = _load_id_to_name()

# ----------------------- Cold-start (robust) -------------------------
def _cold_start_recs(uidx, Rtr, topn, item_sent_df=None):
    """
    Diversified, deterministic (per user) cold-start:
    fused = 0.7 * popularity + 0.3 * sentiment + tiny user-seeded noise
    """
    I = Rtr.shape[1]
    pop = np.asarray(Rtr.sum(axis=0)).ravel().astype(float)
    if pop.max() > pop.min():
        pop = (pop - pop.min()) / (pop.max() - pop.min())
    else:
        pop = np.zeros_like(pop)

    sent = np.zeros(I, dtype=float)
    if item_sent_df is not None and not item_sent_df.empty:
        idx_to_key = {v: k for k, v in item_map.items()}
        pos_pct = item_sent_df.set_index(item_sent_df.columns[0])["positive_pct"]
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

# -------------------- User- and Item-based CF ------------------------
def recommend_item_based(Rtr, uidx, topn=20):
    seen = set(Rtr[uidx].indices)
    if len(seen) == 0 or SIM_ITEM is None:
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    # score: SIM_ITEM @ user profile (sum over items the user interacted with)
    scores = np.zeros(Rtr.shape[1], dtype=float)
    prof_items = Rtr[uidx].indices
    for it in prof_items:
        sims = SIM_ITEM.getcol(it).toarray().ravel()  # sparse col -> dense vector
        scores += sims

    if seen:
        scores[list(seen)] = -np.inf

    if not np.isfinite(scores).any():
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    N = min(topn, (scores > -np.inf).sum())
    rec_idx = np.argpartition(-scores, N - 1)[:N]
    rec_idx = rec_idx[np.argsort(-scores[rec_idx])]
    return rec_idx, scores[rec_idx]

def recommend_user_based(Rtr, uidx, topn=20, k=50):
    seen = set(Rtr[uidx].indices)
    if len(seen) < 2:
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    Rtr_csc = Rtr.tocsc()
    cand_users = set()
    for it in Rtr[uidx].indices:
        cand_users.update(Rtr_csc[:, it].indices)
    cand_users.discard(uidx)
    if not cand_users:
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    # Cosine on implicit interactions (binary rows)
    urow = Rtr[uidx]
    u_norm = np.sqrt(urow.multiply(urow).sum()) + 1e-12

    sims, cand_list = [], list(cand_users)
    for cu in cand_list:
        dot = urow.multiply(Rtr[cu]).sum()
        v_norm = np.sqrt(Rtr[cu].multiply(Rtr[cu]).sum()) + 1e-12
        sims.append(float(dot / (u_norm * v_norm)) if (u_norm * v_norm) > 0 else 0.0)
    sims = np.array(sims, dtype=float)

    mask_pos = sims > 0
    cand_list = [c for c, ok in zip(cand_list, mask_pos) if ok]
    sims = sims[mask_pos]
    if len(cand_list) == 0:
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    if len(cand_list) > k:
        top_idx = np.argpartition(-sims, k - 1)[:k]
        cand_list = [cand_list[i] for i in top_idx]
        sims = sims[top_idx]

    weights = sims / sims.sum()
    scores = np.zeros(Rtr.shape[1], dtype=float)
    for w, cu in zip(weights, cand_list):
        scores[Rtr[cu].indices] += w

    # user-specific tiny tie-break noise
    rng = np.random.RandomState(uidx * 101 + 911)
    scores += rng.rand(scores.shape[0]) * 1e-3

    if seen:
        scores[list(seen)] = -np.inf

    if not np.isfinite(scores).any():
        return _cold_start_recs(uidx, Rtr, topn, item_sent_df=item_sent)

    N = min(topn, (scores > -np.inf).sum())
    rec_idx = np.argpartition(-scores, N - 1)[:N]
    rec_idx = rec_idx[np.argsort(-scores[rec_idx])]
    return rec_idx, scores[rec_idx]

# -------------------- Phase 6 wrapper (Top-20) -----------------------
def top20_for_user(uidx, topn=20):
    if final_cf == "user":
        rec_idx, scores = recommend_user_based(R_train, uidx, topn=topn, k=K_NEIGH)
    else:
        rec_idx, scores = recommend_item_based(R_train, uidx, topn=topn)
    # Map item_idx -> item_id + friendly name
    ids = [item_index_to_key[i] for i in rec_idx]
    names = [id_to_name.get(str(iid), str(iid)) for iid in ids]
    return pd.DataFrame({
        "rank":  np.arange(1, len(rec_idx) + 1),
        "id":    ids,
        "name":  names,
        "score": scores
    })

# -------------------- Phase 7 re-rank (Top-5) -----------------------
def rerank_top20_with_sentiment(top20_df, item_sent_stats, min_reviews=5, alpha=0.7):
    """
    Fuse CF score with sentiment:
      fused = alpha * sentiment_normalized + (1-alpha) * cf_normalized
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
    # Ensure item names are present even if sentiment join dropped them
    if "name" not in out.columns or out["name"].isna().any():
        out["name"] = out["id"].map(lambda x: id_to_name.get(str(x), str(x)))
    # Keep only nice columns for UI
    return out[["id", "name", "positive_pct", "reviews_total", "fused"]].reset_index(drop=True)

# ------------------------------ Flask --------------------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "users": len(user_map),
        "items": len(item_map),
        "cf": final_cf,
        "has_names": bool(id_to_name)
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    username = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        if username not in user_map:
            error = "Username not found. Use an existing username from the dataset."
        else:
            uidx = user_map[username]
            top20 = top20_for_user(uidx, topn=TOPN)                      # has id + name
            top5  = rerank_top20_with_sentiment(top20, item_sent, min_reviews=5, alpha=0.7)
            # Rename for template clarity
            top5 = top5.rename(columns={"name": "item_name"})
            result = top5.to_dict(orient="records")

    return render_template(
    "index.html",
    result=None if error else result,
    error=error,
    username=username,
    cf=final_cf,
    has_names=bool(id_to_name),
    users_count=len(user_map),
    items_count=len(item_map)
)

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    JSON API:
      { "username": "<existing username>", "alpha": 0.7, "min_reviews": 5 }
    Response includes item_name.
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
        username = str(payload.get("username", "")).strip()
        alpha = float(payload.get("alpha", 0.7))
        min_reviews = int(payload.get("min_reviews", 5))
        if username not in user_map:
            return jsonify({"error": "Username not found"}), 404
        uidx = user_map[username]
        top20 = top20_for_user(uidx, topn=TOPN)  # has id + name
        top5  = rerank_top20_with_sentiment(top20, item_sent, min_reviews=min_reviews, alpha=alpha)
        top5  = top5.rename(columns={"name": "item_name"})
        return jsonify({
            "username": username,
            "cf": final_cf,
            "top5": top5.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/random-user", methods=["GET"])
def random_user():
    """Return a random username from the dataset."""
    usernames = list(user_map.keys())
    if not usernames:
        return jsonify({"error": "No users found"}), 404
    random_username = random.choice(usernames)
    return jsonify({"username": random_username})

if __name__ == "__main__":
    # Flask looks for templates/ automatically; put index.html in app/templates/index.html
    # Run: python app_v4.py
    app.run(debug=True, port=5500)
