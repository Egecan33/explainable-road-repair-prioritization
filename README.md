# Explainable Road-Repair Prioritization — Ataşehir Prototype

A research prototype combining an OpenStreetMap road network, simulated damage events, an XGBoost ranking model and SHAP explanations in a Streamlit map. It explores how to explain a repair-priority score; it is not a validated emergency-response system.

## Architecture and provenance

`data_ingest.py` loads or downloads the OSM driving network → `metadata_simulation.py` generates hospital, shelter and population points → damage events flag road segments → `compute_features()` builds model inputs → XGBoost scores roads → SHAP explains predictions → `app.py` displays the result.

Hospitals, shelters, population and damage are **simulated**, not verified municipal or earthquake observations. `train_model.py` learns a weighted heuristic as a pseudo-target, rather than observed repair outcomes. SHAP explains that model's behavior, not a causal benefit from repairing a road.

## Local exploration

Use a fresh Python 3.12 environment, not the committed `venv/` directory:

```bash
python -m venv .local-venv
source .local-venv/bin/activate
pip install -r requirements.txt
python src/data_ingest.py
python src/metadata_simulation.py
python src/train_model.py
streamlit run src/app.py
```

These commands are an exploratory source workflow, not an end-to-end verified installation. OSM ingestion may require network access. Inspect `config.yml` first. Scenario controls write `data/events.json`; metadata generation and training overwrite local artifacts. Do not run them against shared data.

`models/priority_xgb.pkl` **and** `models/shap_explainer.pkl` are required when damaged edges are ranked. They are loaded unconditionally in that path, despite previously being described as optional. Rebuild them together from trusted source and matching dependencies. Only load pickle files you trust; the committed pickle files were not deserialized in this review.

## Known correctness boundary: distance units

`compute_features()` uses planar geometry distance without projecting graph edges and POIs into a common metric CRS. On the geographic OSM coordinates, that distance is in degrees, although the feature is commented as metres. Generated POIs also lack an explicit CRS assignment. **Current distance-derived scores should not be interpreted as metric or operational priorities.**

A follow-up change must assign source CRSs, project both graph and POIs to a suitable local metric CRS, test known-distance fixtures, then retrain and regenerate the explainer. This changes feature values and learned behavior, so it is deliberately separate from documentation improvements.

## Validation and remaining work

First-party Python files received syntax checks only. Geospatial features, pickle compatibility, model training and the dashboard were not executed. No prediction accuracy, emergency effectiveness or deployment readiness is claimed. Dependencies need a reproducible lock and a supported OSMnx version check; the tracked environment needs cleanup. No project-wide license is declared.
