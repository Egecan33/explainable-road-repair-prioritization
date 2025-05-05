import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
import json
import random
import pandas as pd
import osmnx as ox
import networkx as nx

from data_ingest import get_graph
from prioritization_model import rank_roads

# ───────────────────────────────────────────────
st.set_page_config(page_title="Ataşehir Earthquake Response", layout="wide")

# ---------- Session-state ----------
if "damage_basket" not in st.session_state:
    st.session_state.damage_basket = []  # [(u,v,k), …]
if "focus_edges" not in st.session_state:
    st.session_state.focus_edges = []  # top-10 highlights
if "refresh_map" not in st.session_state:
    st.session_state.refresh_map = True
if "cached_map" not in st.session_state:
    st.session_state.cached_map = None

basket = st.session_state.damage_basket

# ---------- Sidebar ----------
st.sidebar.header("Damage tools")

# Clear everything
if st.sidebar.button("Clear all damage"):
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    if events_path.exists():
        events_path.unlink()
    basket.clear()
    st.session_state.refresh_map = True
    st.rerun()

# Mass-damage generator
st.sidebar.markdown("---")
st.sidebar.subheader("Scenario generator")
pct = st.sidebar.slider("Percentage of roads to damage", 5, 100, 50, step=5)
if st.sidebar.button("Generate mass damage scenario"):
    edges = list(get_graph().edges(keys=True))
    n_damage = max(1, int(len(edges) * pct / 100))
    chosen = random.sample(edges, n_damage)
    events = [
        {"event": "damage", "edge": [u, v, k], "delay": random.uniform(5, 20)}
        for u, v, k in chosen
    ]
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    with open(events_path, "w") as f:
        json.dump(events, f)
    st.session_state.refresh_map = True
    st.rerun()

# Manual basket
edges_all = list(get_graph().edges(keys=True))
sel_edge = st.sidebar.selectbox(
    "Select a road to add to basket",
    options=edges_all,
    format_func=lambda e: f"{e[0]}→{e[1]} (k={e[2]})",
)
if st.sidebar.button("Add selected to basket"):
    if sel_edge not in basket:
        basket.append(sel_edge)

st.sidebar.markdown("### Basket")
for i, e in enumerate(basket, 1):
    st.sidebar.write(f"{i}. {e[0]}→{e[1]} (k={e[2]})")

if basket and st.sidebar.button("Apply basket damage"):
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    events = json.load(open(events_path)) if events_path.exists() else []
    # ---- fixed line ----
    events.extend([{"event": "damage", "edge": list(e), "delay": 15} for e in basket])
    with open(events_path, "w") as f:
        json.dump(events, f)
    basket.clear()
    st.session_state.refresh_map = True
    st.rerun()

# ---------- Prioritisation ----------
ranked = rank_roads()  # adds "explanation" column
st.info(f"Damaged edges currently in model: **{len(ranked)}**")

if not ranked.empty:
    st.session_state.focus_edges = (
        ranked.sort_values("score", ascending=False).head(10)["edge"].tolist()
    )

# ---------- Build / reuse map ----------
if st.session_state.refresh_map or st.session_state.cached_map is None:
    G = get_graph()

    # apply damage events to this copy of the graph so the map shows them
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    if events_path.exists():
        with open(events_path) as f:
            for ev in json.load(f):
                if ev.get("event") == "damage":
                    u, v, k = ev["edge"]
                    G[u][v][k]["status"] = "damaged"
                    G[u][v][k]["delay"] = ev.get("delay", 0)

    lat0 = pd.Series(nx.get_node_attributes(G, "y")).iloc[:300].mean()
    lon0 = pd.Series(nx.get_node_attributes(G, "x")).iloc[:300].mean()
    m = folium.Map([lat0, lon0], zoom_start=13, tiles="cartodbpositron")

    # grey network
    for u, v, k, data in G.edges(keys=True, data=True):
        coords = (
            [(p[1], p[0]) for p in data["geometry"].coords]
            if "geometry" in data
            else [
                (G.nodes[u]["y"], G.nodes[u]["x"]),
                (G.nodes[v]["y"], G.nodes[v]["x"]),
            ]
        )
        folium.PolyLine(coords, color="#aaaaaa", weight=1, opacity=0.4).add_to(m)

    # POI layers
    poi_path = Path(__file__).resolve().parent.parent / "data" / "poi.geojson"
    if poi_path.exists():
        import geopandas as gpd

        poi = gpd.read_file(poi_path)

        for _, r in poi[poi["type"] == "hospital"].iterrows():
            folium.Marker(
                [r.geometry.y, r.geometry.x],
                icon=folium.Icon(color="red", icon="plus", prefix="fa"),
                tooltip=f'Hospital: {r["name"]} (cap={r["capacity"]})',
            ).add_to(m)

        for _, r in poi[poi["type"] == "shelter"].iterrows():
            folium.Marker(
                [r.geometry.y, r.geometry.x],
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                tooltip=f'Shelter: {r["name"]} (cap={r["capacity"]})',
            ).add_to(m)

        pops = poi[poi["type"] == "population"]
        if not pops.empty:
            HeatMap(
                [
                    [r.geometry.y, r.geometry.x, r["density"]]
                    for _, r in pops.iterrows()
                ],
                radius=15,
                blur=12,
                max_zoom=14,
            ).add_to(m)

    # damaged & focus
    bounds = []
    for _, row in ranked.iterrows():
        u, v, k = row["edge"]
        data = G[u][v][k]
        coords = (
            [(p[1], p[0]) for p in data["geometry"].coords]
            if "geometry" in data
            else [
                (G.nodes[u]["y"], G.nodes[u]["x"]),
                (G.nodes[v]["y"], G.nodes[v]["x"]),
            ]
        )
        is_focus = row["edge"] in st.session_state.focus_edges
        folium.PolyLine(
            coords,
            color="yellow" if is_focus else "red",
            weight=7 if is_focus else 4,
            tooltip=row["explanation"],
        ).add_to(m)
        bounds.extend(coords)

    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))

    st.session_state.cached_map = m
    st.session_state.refresh_map = False
else:
    m = st.session_state.cached_map

# ---------- Map to Streamlit ----------
map_state = st_folium(m, width=1100, height=600, returned_objects=["last_clicked"])

# click-to-basket
if map_state and map_state.get("last_clicked"):
    lat = map_state["last_clicked"]["lat"]
    lon = map_state["last_clicked"]["lng"]
    st.sidebar.info(f"Clicked: ({lat:.5f}, {lon:.5f})")
    if st.sidebar.button("Add nearest road to basket"):
        u_nn, v_nn, k_nn = ox.distance.nearest_edges(
            get_graph(), lon, lat, return_dist=False
        )
        if (u_nn, v_nn, k_nn) not in basket:
            basket.append((u_nn, v_nn, k_nn))
            st.sidebar.success("Added to basket.")

# ---------- Table & summary ----------
if not ranked.empty:
    st.subheader("Damaged roads (ranked)")
    st.dataframe(
        ranked[["edge", "score", "explanation"]].round(3), use_container_width=True
    )

    st.subheader("Top explanation reasons (count)")
    st.bar_chart(ranked["explanation"].value_counts().head(10))
else:
    st.info("No damaged roads yet – add via sidebar or map.")
