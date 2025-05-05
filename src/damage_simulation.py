
"""Generate random damage events for the road network."""
import networkx as nx, random, time, json, pickle
from pathlib import Path
from data_ingest import get_graph
import yaml

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / 'config.yml'))

def random_damage(G, n_segments=5):
    damaged_edges = random.sample(list(G.edges(keys=True)), n_segments)
    events = []
    for u, v, k in damaged_edges:
        G[u][v][k]['status'] = 'damaged'
        delay = random.uniform(5, 20)  # extra minutes travel time
        G[u][v][k]['delay'] = delay
        events.append({'event': 'damage', 'edge': (u, v, k), 'delay': delay})
    return events

def simulate_loop():
    G = get_graph()
    while True:
        events = random_damage(G, CONFIG['simulation']['max_segments_damaged'])
        print('New damage events:', events)
        with open(Path(__file__).resolve().parent.parent / 'data' / 'events.json', 'w') as f:
            json.dump(events, f)
        time.sleep(CONFIG['simulation']['damage_frequency_minutes'] * 60)

if __name__ == '__main__':
    simulate_loop()
