
# Earthquake Road Repair Prioritization – Ataşehir Prototype

This repository contains a **prototype framework** that simulates earthquake‑related road
damage in Ataşehir (Istanbul) and produces an **explainable ranking** of which roads to
repair first.  It then visualises the network and priorities in a Streamlit dashboard.

**Key folders**

```
.
├── README.md                  ← This file
├── requirements.txt           ← Python package requirements
├── config.yml                 ← Scenario + weighting settings
├── data/                      ← Simulated metadata & cached OSM graph
├── src/                       ← Core Python modules
│   ├── data_ingest.py         ← Download / load OpenStreetMap graph
│   ├── metadata_simulation.py ← Generate hospitals / shelters / population
│   ├── damage_simulation.py   ← Real‑time damage event generator
│   ├── prioritization_model.py← Score + rank damaged roads
│   ├── xai.py                 ← Human‑readable explanations
│   └── app.py                 ← Streamlit front‑end
└── models/                    ← Saved ML models (optional)
```

## Quick start (local)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Download OSM graph & generate simulated metadata
python src/data_ingest.py
python src/metadata_simulation.py
# Run dashboard
streamlit run src/app.py
```

*Created: 2025-04-06*


python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# fetch OSM graph
python src/data_ingest.py
# make some fake hospitals/shelters/population
python src/metadata_simulation.py
# (optional) start a damage-event loop in another terminal
python src/damage_simulation.py
# launch the dashboard
streamlit run src/app.py