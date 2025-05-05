import streamlit as st

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
import json
import pandas as pd
import osmnx as ox
import networkx as nx
from data_ingest import get_graph
from prioritization_model import rank_roads
from xai import explain
import random

st.set_page_config(page_title="Ataşehir Earthquake Response", layout="wide")

# ───────────── Session state for “basket” ─────────────
if "damage_basket" not in st.session_state:
    st.session_state.damage_basket = []  # list of (u, v, k)
basket = st.session_state.damage_basket

# ───────────── Sidebar controls ─────────────
st.sidebar.header("Damage tools")


# Clear all damage
if st.sidebar.button("Clear all damage"):
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    if events_path.exists():
        events_path.unlink()  # remove the events file
    basket.clear()  # empty any queued edges
    st.rerun()

# --- Mass‑damage (percentage of all roads) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Scenario generator")
pct = st.sidebar.slider("Percentage of roads to damage", 5, 100, 50, step=5)
if st.sidebar.button("Generate mass damage scenario"):
    all_edges_full = list(get_graph().edges(keys=True))
    n_damage = max(1, int(len(all_edges_full) * pct / 100))
    chosen = random.sample(all_edges_full, n_damage)
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    events = []
    for u, v, k in chosen:
        events.append(
            {"event": "damage", "edge": [u, v, k], "delay": random.uniform(5, 20)}
        )
    with open(events_path, "w") as f:
        json.dump(events, f)
    st.rerun()

# Manual selectbox → add to basket
all_edges = list(get_graph().edges(keys=True))
sel_edge = st.sidebar.selectbox(
    "Select a road to add to basket",
    options=all_edges,
    format_func=lambda e: f"{e[0]}→{e[1]} (key={e[2]})",
)
if st.sidebar.button("Add selected to basket"):
    if sel_edge not in basket:
        basket.append(sel_edge)

# Basket summary + apply
st.sidebar.markdown("### Basket")
for idx, e in enumerate(basket):
    st.sidebar.write(f"{idx+1}. {e[0]}→{e[1]} (k={e[2]})")
if basket and st.sidebar.button("Apply basket damage"):
    events_path = Path(__file__).resolve().parent.parent / "data" / "events.json"
    if events_path.exists():
        with open(events_path) as f:
            events = json.load(f)
    else:
        events = []
    for u, v, k in basket:
        events.append({"event": "damage", "edge": [u, v, k], "delay": 15})
    with open(events_path, "w") as f:
        json.dump(events, f)
    basket.clear()
    st.rerun()

# ───────────── Prioritisation + explanations ─────────────
ranked = rank_roads()
num_damaged = len(ranked)
st.info(f"Damaged edges currently in model: **{num_damaged}**")
if not ranked.empty:
    ranked["explanation"] = ranked.apply(explain, axis=1)

    # ---- focus via data_editor check‑marks ----
    if "focus_edges" not in st.session_state:
        st.session_state.focus_edges = []

    # build a view with a checkbox column
    display_df = ranked[["edge", "score", "explanation"]].copy()
    display_df["focus"] = display_df["edge"].apply(
        lambda e: e in st.session_state.focus_edges
    )

    edited_df = st.data_editor(
        display_df,
        key="focus_table",
        num_rows="dynamic",
        use_container_width=True,
    )

    # update session_state with any rows whose checkbox is ticked
    st.session_state.focus_edges = [
        row["edge"] for _, row in edited_df.iterrows() if row["focus"]
    ]

# ───────────── Build Folium map ─────────────
G = get_graph()
cntr_lat = pd.Series(nx.get_node_attributes(G, "y")).iloc[:300].mean()
cntr_lon = pd.Series(nx.get_node_attributes(G, "x")).iloc[:300].mean()
m = folium.Map(location=[cntr_lat, cntr_lon], zoom_start=13, tiles="cartodbpositron")

# Base road network in grey
for u, v, k, data in G.edges(keys=True, data=True):
    if "geometry" in data:
        coords = [(pt[1], pt[0]) for pt in data["geometry"].coords]
    else:
        coords = [
            (G.nodes[u]["y"], G.nodes[u]["x"]),
            (G.nodes[v]["y"], G.nodes[v]["x"]),
        ]
    folium.PolyLine(coords, color="#aaaaaa", weight=1, opacity=0.4).add_to(m)

# POI layers (hospitals, shelters, population heatmap)
poi_path = Path(__file__).resolve().parent.parent / "data" / "poi.geojson"
if poi_path.exists():
    import geopandas as gpd

    poi = gpd.read_file(poi_path)

    # Hospitals
    for _, r in poi[poi["type"] == "hospital"].iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
            tooltip=f"Hospital: {r['name']} (cap={r['capacity']})",
        ).add_to(m)

    # Shelters
    for _, r in poi[poi["type"] == "shelter"].iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            icon=folium.Icon(color="blue", icon="home", prefix="fa"),
            tooltip=f"Shelter: {r['name']} (cap={r['capacity']})",
        ).add_to(m)

    # Population heatmap
    pops = poi[poi["type"] == "population"]
    if not pops.empty:
        HeatMap(
            [[r.geometry.y, r.geometry.x, r["density"]] for _, r in pops.iterrows()],
            radius=15,
            blur=12,
            max_zoom=14,
        ).add_to(m)

# Damaged roads in red
damaged_bounds = []

if not ranked.empty:
    for _, row in ranked.iterrows():
        u, v, k = row["edge"]
        data = G[u][v][k]
        if "geometry" in data:
            coords = [(pt[1], pt[0]) for pt in data["geometry"].coords]
        else:
            coords = [
                (G.nodes[u]["y"], G.nodes[u]["x"]),
                (G.nodes[v]["y"], G.nodes[v]["x"]),
            ]

        # Highlight if this is a focused edge
        is_focus = row["edge"] in st.session_state.focus_edges
        line_color = "yellow" if is_focus else "red"
        weight = 7 if is_focus else 4

        folium.PolyLine(
            coords,
            color=line_color,
            weight=weight,
            tooltip=row["explanation"],
        ).add_to(m)

        if is_focus:
            damaged_bounds.extend(coords)  # ensure zoom centers on focus
        else:
            damaged_bounds.extend(coords)

# Fit map to damaged area if any
if damaged_bounds:
    m.fit_bounds(damaged_bounds, padding=(20, 20))

# Map → Streamlit
map_state = st_folium(m, width=1100, height=600, returned_objects=["last_clicked"])

# Click-to-add (nearest edge → basket)
if map_state and map_state.get("last_clicked"):
    lat = map_state["last_clicked"]["lat"]
    lon = map_state["last_clicked"]["lng"]
    st.sidebar.info(f"Clicked: ({lat:.5f}, {lon:.5f})")
    if st.sidebar.button("Add nearest road to basket"):
        u_nn, v_nn, k_nn = ox.distance.nearest_edges(G, lon, lat, return_dist=False)
        edge_tuple = (u_nn, v_nn, k_nn)
        if edge_tuple not in basket:
            basket.append(edge_tuple)
            st.sidebar.success("Added to basket.")

# Table of priorities
if not ranked.empty:
    st.subheader("Priority List (current damaged roads)")
    st.dataframe(ranked[["edge", "score", "explanation"]])
    # Extra breakdown – top reasons
    reason_counts = ranked["explanation"].value_counts().head(10)
    st.subheader("Top explanation reasons (count)")
    st.bar_chart(reason_counts)
else:
    st.info("No damaged roads yet – add via sidebar or map.")
