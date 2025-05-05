"""Simple rule‑based explanation generator."""
import pandas as pd

def explain(row):
    reasons = []
    if row['dist_hospital'] < 1:
        reasons.append('close to hospital')
    if row['centrality'] > 0.05:
        reasons.append('network hub')
    if row['delay'] > 10:
        reasons.append('large detour time')
    if not reasons:
        reasons.append('general connectivity')
    return ', '.join(reasons)
