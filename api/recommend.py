import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import pandas as pd


# ---- Load Pareto data once per warm instance ----
# Vercel Python runtime uses project root as CWD, not /api.  [oai_citation:5‡Vercel](https://vercel.com/docs/functions/runtimes/python)
PARETO_PATH = os.path.join("data", "pareto_front.csv")
_pf_cache = None


def load_pareto():
    global _pf_cache
    if _pf_cache is None:
        _pf_cache = pd.read_csv(PARETO_PATH)
    return _pf_cache


def minmax(arr: np.ndarray) -> np.ndarray:
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if abs(amax - amin) < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - amin) / (amax - amin)


def recommend_from_pareto(
    pf: pd.DataFrame,
    w_yield: float,
    w_cost: float,
    w_time: float,
    min_yield=None,
    max_cost=None,
    max_time=None,
    top_k: int = 10,
):
    df = pf.copy()

    # Optional constraints
    if min_yield is not None:
        df = df[df["Yield_pct"] >= float(min_yield)]
    if max_cost is not None:
        df = df[df["Cost_index"] <= float(max_cost)]
    if max_time is not None:
        df = df[df["Time_h"] <= float(max_time)]

    if len(df) == 0:
        return None, None, "No Pareto points match these constraints. Relax limits."

    # Normalize objectives in remaining feasible set
    y = df["Yield_pct"].to_numpy(dtype=float)     # maximize
    c = df["Cost_index"].to_numpy(dtype=float)    # minimize
    t = df["Time_h"].to_numpy(dtype=float)        # minimize

    y_n = minmax(y)
    c_n = minmax(c)
    t_n = minmax(t)

    # Normalize weights
    s = w_yield + w_cost + w_time
    if s <= 0:
        return None, None, "Weights must sum to a positive value."

    w_yield, w_cost, w_time = w_yield / s, w_cost / s, w_time / s

    # Score to maximize
    score = (w_yield * y_n) - (w_cost * c_n) - (w_time * t_n)
    df = df.assign(Score=score)

    df_sorted = df.sort_values("Score", ascending=False).reset_index(drop=True)

    best = df_sorted.iloc[0].to_dict()
    top = df_sorted.head(top_k).to_dict(orient="records")
    return best, top, None


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/recommend":
            self._send_json(404, {"error": "Not found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            data = json.loads(raw) if raw else {}
        except Exception:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        # Read sliders
        try:
            w_yield = float(data.get("w_yield", 0.5))
            w_cost  = float(data.get("w_cost", 0.3))
            w_time  = float(data.get("w_time", 0.2))

            min_yield = data.get("min_yield", None)
            max_cost  = data.get("max_cost", None)
            max_time  = data.get("max_time", None)

            top_k = int(data.get("top_k", 10))
            top_k = max(1, min(top_k, 50))
        except Exception:
            self._send_json(400, {"error": "Bad parameter types"})
            return

        pf = load_pareto()
        best, top, err = recommend_from_pareto(
            pf,
            w_yield=w_yield, w_cost=w_cost, w_time=w_time,
            min_yield=min_yield, max_cost=max_cost, max_time=max_time,
            top_k=top_k
        )

        if err is not None:
            self._send_json(400, {"error": err})
            return

        self._send_json(200, {"best": best, "top": top})

    # Optional: allow browser preflight if you ever call from a different domain
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
