from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.decision_engine.costs import load_costs
from src.decision_engine.thresholds import optimize_threshold

logger = logging.getLogger(__name__)

def analyze_errors(cfg: Dict[str, Any], predictions_by_model: Dict[str, pd.DataFrame], features: pd.DataFrame, use_case: str) -> Dict[str, Any]:
    """Error analysis is about *understanding failures*, not generating more aggregates."""
    label_col = cfg["data"]["dataset"]["label_col"]
    id_col = cfg["data"]["dataset"]["id_col"]
    text_col = cfg["data"]["dataset"].get("text_col")

    costs = load_costs(cfg)
    use_case_cfg = costs["use_cases"][use_case]["binary"]

    out = {
        "top_false_positives": {},
        "top_false_negatives": {},
        "clusters": {},
    }

    for model_name, dfp in predictions_by_model.items():
        df = features[[c for c in features.columns if c in [id_col, text_col] or c not in []]].copy()
        df[label_col] = dfp[label_col].values
        df["y_score"] = dfp["y_score"].values

        # Pick model's best threshold for this use-case to identify FP/FN
        grid = _threshold_grid(cfg)
        best = optimize_threshold(df[label_col].to_numpy(), df["y_score"].to_numpy(), grid, use_case_cfg)
        t = best["threshold"]
        y_pred = (df["y_score"].to_numpy() >= t).astype(int)

        fp = df[(y_pred == 1) & (df[label_col] == 0)].copy()
        fn = df[(y_pred == 0) & (df[label_col] == 1)].copy()

        # Surface the most damaging errors: high-confidence wrong predictions
        fp["confidence"] = np.maximum(fp["y_score"], 1 - fp["y_score"])
        fn["confidence"] = np.maximum(fn["y_score"], 1 - fn["y_score"])

        fp_top = fp.sort_values("confidence", ascending=False).head(25)
        fn_top = fn.sort_values("confidence", ascending=False).head(25)

        out["top_false_positives"][model_name] = _to_records(fp_top, id_col, text_col)
        out["top_false_negatives"][model_name] = _to_records(fn_top, id_col, text_col)

        # Cluster errors to find recurring failure modes (cheap + explainable baseline)
        out["clusters"][model_name] = cluster_errors(cfg, fp, fn, text_col=text_col)

    return out

def _to_records(df: pd.DataFrame, id_col: str, text_col: str | None) -> List[Dict[str, Any]]:
    keep = [id_col, "y_score", "confidence"]
    if text_col and text_col in df.columns:
        keep.append(text_col)
    return df[keep].to_dict(orient="records")

def _threshold_grid(cfg: Dict[str, Any]) -> np.ndarray:
    g = cfg["evaluation"]["threshold_grid"]
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    return np.round(np.arange(start, stop + 1e-12, step), 6)

def cluster_errors(cfg: Dict[str, Any], fp: pd.DataFrame, fn: pd.DataFrame, text_col: str | None) -> Dict[str, Any]:
    # If no text, clustering on errors is weak; return counts only.
    if not text_col or text_col not in fp.columns:
        return {
            "available": False,
            "reason": "no_text_column",
            "fp_count": int(len(fp)),
            "fn_count": int(len(fn)),
        }

    combined = pd.concat([fp.assign(error_type="FP"), fn.assign(error_type="FN")], ignore_index=True)
    if len(combined) < 20:
        return {"available": False, "reason": "too_few_errors", "count": int(len(combined))}

    texts = combined[text_col].fillna("").astype(str).tolist()

    # TF-IDF baseline: it's cheap, interpretable, and sufficient to find repeated n-grams.
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = vec.fit_transform(texts)

    # Reduce dimensionality before clustering for stability
    svd = TruncatedSVD(n_components=min(50, X.shape[1]-1) if X.shape[1] > 1 else 1, random_state=42)
    Xr = svd.fit_transform(X)

    k = min(6, max(2, int(np.sqrt(len(combined) / 2))))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = km.fit_predict(Xr)
    combined["cluster"] = clusters

    # Summarize clusters by top terms and error mix
    terms = np.array(vec.get_feature_names_out())
    centroids = km.cluster_centers_

    summaries = []
    for c in range(k):
        sub = combined[combined["cluster"] == c]
        if len(sub) == 0:
            continue
        # approximate top terms using centroid projection back to tf-idf space is hard; use mean tfidf instead
        idxs = np.where(clusters == c)[0]
        tfidf_mean = np.asarray(X[idxs].mean(axis=0)).ravel()
        top = terms[np.argsort(tfidf_mean)[-10:]][::-1].tolist()
        summaries.append({
            "cluster": int(c),
            "count": int(len(sub)),
            "fp": int((sub["error_type"] == "FP").sum()),
            "fn": int((sub["error_type"] == "FN").sum()),
            "top_terms": top,
            "examples": sub[[text_col, "y_score", "error_type"]].head(5).to_dict(orient="records"),
        })

    return {
        "available": True,
        "n_clusters": int(k),
        "summaries": summaries,
    }
