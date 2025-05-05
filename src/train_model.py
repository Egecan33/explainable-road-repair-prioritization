"""
Train an XGBoost regressor to predict road-repair priority.
Label = previous heuristic score, so the model learns the same logic
but can capture nonlinear interactions.  Saves model + SHAP explainer.
"""

from pathlib import Path
import pickle
import shap
import xgboost as xgb
import pandas as pd
from data_ingest import get_graph
from prioritization_model import compute_features, load_poi
import yaml

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yml"))


def make_training_df():
    G = get_graph()
    poi = load_poi()

    # If no edges are damaged, temporarily flag 5 % of edges
    if all(
        data.get("status") != "damaged"
        for _, _, _, data in G.edges(keys=True, data=True)
    ):
        import random

        edges = list(G.edges(keys=True))
        sample = random.sample(edges, max(1, int(len(edges) * 0.05)))
        for u, v, k in sample:
            G[u][v][k]["status"] = "damaged"
            G[u][v][k]["delay"] = random.uniform(5, 20)

    feats = compute_features(G, poi)
    if feats.empty:
        raise ValueError("Even after auto-damage, no features available")

    # Use existing heuristic score as pseudo target
    w = CONFIG["weights"]
    feats["target"] = (
        (1 / (feats["dist_hospital"] + 1e-3)) * w["hospital_access"]
        + feats["delay"] * w["population_density"]
        + feats["centrality"] * w["network_centrality"]
    )
    return feats


def train():
    df = make_training_df()
    X = df[["length_km", "delay", "centrality", "dist_hospital"]]
    y = df["target"]
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8
    )
    model.fit(X, y)

    # Save model & SHAP explainer
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    pickle.dump(model, open(models_dir / "priority_xgb.pkl", "wb"))
    explainer = shap.Explainer(model)
    with open(models_dir / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    print("Model & explainer saved under /models")


if __name__ == "__main__":
    train()
