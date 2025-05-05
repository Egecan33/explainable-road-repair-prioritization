"""Score and rank damaged roads."""

import networkx as nx, pickle, json
from pathlib import Path
import pandas as pd
import yaml
from data_ingest import get_graph
from shapely.geometry import Point
import geopandas as gpd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import shap
from xai import shap_to_text  # new helper

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yml"))


def load_poi():
    poi_path = Path(__file__).resolve().parent.parent / "data" / "poi.geojson"
    return gpd.read_file(poi_path)


def compute_features(G, poi):
    """
    Build a DataFrame whose rows are *damaged* edges and whose columns are
    numeric features used by the prioritisation model.
    """
    features = []

    # GeoSeries of hospital centroids (needed for distance calc)
    hospitals = poi[poi["type"] == "hospital"].geometry.reset_index(drop=True)

    for u, v, k, data in G.edges(keys=True, data=True):
        # Skip anything that isn’t flagged as damaged
        if data.get("status") != "damaged":
            continue

        # --- basic edge metrics ------------------------------------------------
        length_km = data.get("length", 0) / 1_000
        delay_min = data.get("delay", 0)
        centrality = data.get("centrality", 0)

        # --- safe centroid for distance calculation ---------------------------
        if "geometry" in data and data["geometry"] is not None:
            # typical MultiDiGraph edge has a LineString geometry
            centroid_point = data["geometry"].centroid
        else:
            # some minor edges lack geometry; use midpoint of the two nodes
            centroid_x = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
            centroid_y = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
            from shapely.geometry import Point

            centroid_point = Point(centroid_x, centroid_y)

        # distance to nearest hospital (metres)
        if not hospitals.empty:
            dists = hospitals.distance(centroid_point)
            dist_hospital = dists.min()
        else:
            dist_hospital = 1e6  # large sentinel if no hospitals

        # ----------------------------------------------------------------------
        features.append(
            {
                "edge": (u, v, k),
                "length_km": length_km,
                "delay": delay_min,
                "centrality": centrality,
                "dist_hospital": dist_hospital,
            }
        )

    return pd.DataFrame(features)


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "priority_xgb.pkl"
EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"


def rank_roads():
    G = get_graph()
    poi = load_poi()

    # --------------------------------------------------
    # Apply externally‑saved damage events (with direction fallback)
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    applied = 0
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)

        for ev in events:
            if ev.get("event") != "damage":
                continue

            u, v, k = ev["edge"]
            try:
                k = int(k)
            except (TypeError, ValueError):
                pass

            # try u→v first, then v→u (graph is directed)
            if G.has_edge(u, v, key=k):
                G[u][v][k]["status"] = "damaged"
                G[u][v][k]["delay"] = ev.get("delay", 0)
                applied += 1
            elif G.has_edge(v, u, key=k):
                G[v][u][k]["status"] = "damaged"
                G[v][u][k]["delay"] = ev.get("delay", 0)
                applied += 1

    print(f"[rank_roads] applied {applied} damage events")
    # --------------------------------------------------

    feats = compute_features(G, poi)
    if feats.empty:
        return feats  # empty DF handled by app

    model = pickle.load(open(MODEL_PATH, "rb"))
    with open(EXPLAINER_PATH, "rb") as f:
        explainer = pickle.load(f)

    X = feats[["length_km", "delay", "centrality", "dist_hospital"]]
    feats["score"] = model.predict(X)

    # SHAP explanations → plain text
    shap_vals = explainer(X)
    feats["explanation"] = feats.apply(
        lambda row, sv=shap_vals: shap_to_text(row.name, sv, X.columns), axis=1
    )

    return feats.sort_values("score", ascending=False)


if __name__ == "__main__":
    print(rank_roads().head())
