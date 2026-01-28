from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="static", static_url_path="")

PARETO_PATH = "pareto_front.csv"
pf = pd.read_csv(PARETO_PATH)

def minmax(series):
    smin, smax = float(series.min()), float(series.max())
    if abs(smax - smin) < 1e-12:
        return np.ones(len(series)) * 0.5
    return (series - smin) / (smax - smin)

@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/recommend")
def recommend():
    data = request.get_json(force=True)

    w_yield = float(data.get("w_yield", 0.5))
    w_cost  = float(data.get("w_cost", 0.3))
    w_time  = float(data.get("w_time", 0.2))

    min_yield = data.get("min_yield", None)
    max_cost  = data.get("max_cost", None)
    max_time  = data.get("max_time", None)

    df = pf.copy()

    # Optional constraints
    if min_yield is not None:
        df = df[df["Yield_pct"] >= float(min_yield)]
    if max_cost is not None:
        df = df[df["Cost_index"] <= float(max_cost)]
    if max_time is not None:
        df = df[df["Time_h"] <= float(max_time)]

    if len(df) == 0:
        return jsonify({"error": "No points match these constraints. Relax limits."}), 400

    # Normalize objectives within feasible set
    y_n = minmax(df["Yield_pct"].to_numpy())
    c_n = minmax(df["Cost_index"].to_numpy())
    t_n = minmax(df["Time_h"].to_numpy())

    # Normalize weights to sum to 1
    s = w_yield + w_cost + w_time
    if s <= 0:
        return jsonify({"error": "Weights must be positive."}), 400
    w_yield, w_cost, w_time = w_yield/s, w_cost/s, w_time/s

    # Score (maximize)
    score = w_yield*y_n - w_cost*c_n - w_time*t_n
    df = df.assign(Score=score)

    df_sorted = df.sort_values("Score", ascending=False)

    best = df_sorted.iloc[0].to_dict()
    top10 = df_sorted.head(10).to_dict(orient="records")

    return jsonify({"best": best, "top10": top10})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
