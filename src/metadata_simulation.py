import random
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from pathlib import Path
from data_ingest import get_graph, get_boundary_polygon

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def random_point_in_poly(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if poly.contains(p):
            return p


def simulate():
    G = get_graph()
    poly = get_boundary_polygon()

    # Hospitals
    hospitals = gpd.GeoDataFrame(
        {
            "name": [f"Hospital_{i}" for i in range(3)],
            "capacity": [random.randint(200, 600) for _ in range(3)],
            "geometry": [random_point_in_poly(poly) for _ in range(3)],
            "type": "hospital",
        }
    )

    # Shelters
    shelters = gpd.GeoDataFrame(
        {
            "name": [f"Shelter_{i}" for i in range(5)],
            "capacity": [random.randint(500, 2000) for _ in range(5)],
            "geometry": [random_point_in_poly(poly) for _ in range(5)],
            "type": "shelter",
        }
    )

    # Population points
    pop = gpd.GeoDataFrame(
        {
            "density": [random.randint(500, 2000) for _ in range(50)],
            "geometry": [random_point_in_poly(poly) for _ in range(50)],
            "type": "population",
        }
    )

    gdf = pd.concat([hospitals, shelters, pop]).reset_index(drop=True)
    gdf.to_file(DATA_DIR / "poi.geojson", driver="GeoJSON")
    print("Generated poi.geojson strictly within Ataşehir polygon.")


if __name__ == "__main__":
    simulate()
