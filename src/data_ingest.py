import osmnx as ox
import networkx as nx
import pickle
from pathlib import Path
import yaml

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yml"))
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.pkl"
POLY_PATH = Path(__file__).resolve().parent.parent / "data" / "atasehir_poly.pkl"


def get_boundary_polygon():
    if POLY_PATH.exists():
        with open(POLY_PATH, "rb") as f:
            return pickle.load(f)
    gdf = ox.geocode_to_gdf(CONFIG["area_name"])
    poly = gdf.iloc[0].geometry
    with open(POLY_PATH, "wb") as f:
        pickle.dump(poly, f)
    return poly


def get_graph():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            G = pickle.load(f)
    else:
        poly = get_boundary_polygon()
        G = ox.graph_from_polygon(poly, network_type="drive", truncate_by_edge=True)
        # keep only the largest connected component
        try:
            G = ox.utils_graph.get_largest_component(G, strongly=False)
        except AttributeError:
            import networkx as nx

            largest_cc = max(nx.connected_components(G.to_undirected()), key=len)
            G = G.subgraph(largest_cc).copy()
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(G, f)
    return G


if __name__ == "__main__":
    G = get_graph()
    print("Graph loaded with", len(G.nodes), "nodes and", len(G.edges), "edges.")
