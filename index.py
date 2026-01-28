import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, redirect

app = Flask(__name__)

# Load Pareto data
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "pareto_front.csv")
pf = pd.read_csv(CSV_PATH)
Y_MIN, Y_MAX = float(pf["Yield_pct"].min()), float(pf["Yield_pct"].max())
C_MIN, C_MAX = float(pf["Cost_index"].min()), float(pf["Cost_index"].max())
T_MIN, T_MAX = float(pf["Time_h"].min()), float(pf["Time_h"].max())

def norm01(x, xmin, xmax):
    if abs(xmax - xmin) < 1e-12:
        return 0.5
    return (x - xmin) / (xmax - xmin)

def minmax(arr: np.ndarray) -> np.ndarray:
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if abs(amax - amin) < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - amin) / (amax - amin)

@app.get("/")
def home():
    # Serve your UI from /public (Vercel serves public/ automatically)
    return redirect("/index.html", code=302)

@app.post("/api/recommend")
def recommend():
    data = request.get_json(force=True) or {}

    w_yield = float(data.get("w_yield", 0.5))
    w_cost  = float(data.get("w_cost", 0.3))
    w_time  = float(data.get("w_time", 0.2))

    min_yield = data.get("min_yield", None)
    max_cost  = data.get("max_cost", None)
    max_time  = data.get("max_time", None)

    df = pf.copy()

    # Apply constraints
    if min_yield is not None:
        df = df[df["Yield_pct"] >= float(min_yield)]
    if max_cost is not None:
        df = df[df["Cost_index"] <= float(max_cost)]
    if max_time is not None:
        df = df[df["Time_h"] <= float(max_time)]

    if len(df) == 0:
        return jsonify({"error": "No Pareto points match these constraints. Relax limits."}), 400

    # ✅ Global normalization (stable, weights actually matter)
    y_n = df["Yield_pct"].apply(lambda v: norm01(v, Y_MIN, Y_MAX)).to_numpy(float)
    c_n = df["Cost_index"].apply(lambda v: norm01(v, C_MIN, C_MAX)).to_numpy(float)
    t_n = df["Time_h"].apply(lambda v: norm01(v, T_MIN, T_MAX)).to_numpy(float)

    # Normalize weights
    s = w_yield + w_cost + w_time
    if s <= 0:
        return jsonify({"error": "Weights must sum to a positive value."}), 400
    w_yield, w_cost, w_time = w_yield/s, w_cost/s, w_time/s

    # Score (maximize)
    score = (w_yield * y_n) - (w_cost * c_n) - (w_time * t_n)

    df = df.assign(Score=score).sort_values("Score", ascending=False)

    best = df.iloc[0].to_dict()
    top = df.head(10).to_dict(orient="records")

    return jsonify({
        "best": best,
        "top": top,
        "debug": {
            "feasible_points": int(len(df)),
            "weights_used": {"yield": w_yield, "cost": w_cost, "time": w_time}
        }
    }), 200
