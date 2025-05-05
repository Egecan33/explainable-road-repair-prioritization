"""Score and rank damaged roads."""

import networkx as nx, pickle, json
from pathlib import Path
import pandas as pd
import yaml
from data_ingest import get_graph
import networkx as nx
from shapely.geometry import Point
import geopandas as gpd
from sklearn.tree import DecisionTreeRegressor
import numpy as np

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


def rank_roads():
    G = get_graph()
    poi = load_poi()
    # compute betweenness centrality once and store
    # For MultiGraph, keys are (u, v, k)
    betweenness = nx.edge_betweenness_centrality(G, normalized=True, k=500)
    for (u, v, k), c in betweenness.items():
        G[u][v][k]["centrality"] = c
    # --------------------------------------------------
    # Apply any externally-saved damage events
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    n_applied = 0
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
        for ev in events:
            if ev.get("event") != "damage":
                continue
            u, v, k = ev["edge"]
            try:
                k = int(k)
            except (ValueError, TypeError):
                pass
            if G.has_edge(u, v, key=k):
                G[u][v][k]["status"] = "damaged"
                G[u][v][k]["delay"] = ev.get("delay", 0)
                n_applied += 1
    print(f"[rank_roads] applied {n_applied} damage events")
    # --------------------------------------------------
    feats = compute_features(G, poi)
    if feats.empty:
        return pd.DataFrame()
    # simple score
    w = CONFIG["weights"]
    feats["score"] = (
        (1 / (feats["dist_hospital"] + 0.001)) * w["hospital_access"]
        + feats["delay"] * w["population_density"]  # placeholder
        + feats["centrality"] * w["network_centrality"]
    )
    ranked = feats.sort_values("score", ascending=False)
    return ranked


if __name__ == "__main__":
    print(rank_roads().head())
