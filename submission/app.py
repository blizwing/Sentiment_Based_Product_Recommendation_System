# app.py (only the top imports and route bodies change)
from flask import Flask, render_template, request, jsonify
from model import health_info, list_usernames, recommend_top5

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return health_info()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    username = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        try:
            recommendations = recommend_top5(username, alpha=0.7, min_reviews=5)
            # Map the field names to match HTML template
            result = []
            for i, rec in enumerate(recommendations):
                result.append({
                    "id": rec["id"],
                    "item_name": rec["name"],  # HTML expects item_name, not name
                    "positive_pct": rec["positive_pct"],
                    "reviews_total": rec["reviews_total"],
                    "fused": rec["fused"]
                })
        except ValueError as e:
            error = str(e)

    meta = health_info()
    return render_template(
        "index.html",
        result=None if error else result,
        error=error,
        username=username,
        cf=meta["cf"],
        has_names=meta["has_names"],
        users_count=meta["users"],
        items_count=meta["items"],
    )

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        username = str(payload.get("username", "")).strip()
        alpha = float(payload.get("alpha", 0.7))
        min_reviews = int(payload.get("min_reviews", 5))
        top5 = recommend_top5(username, alpha=alpha, min_reviews=min_reviews)
        return jsonify({"username": username, "top5": top5})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/random-user", methods=["GET"])
def random_user():
    users = list_usernames()
    if not users:
        return jsonify({"error": "No users found"}), 404
    import random
    return jsonify({"username": random.choice(users)})
