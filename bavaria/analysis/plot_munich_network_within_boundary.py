#!/usr/bin/env python3
"""
Plot Munich road network within a provided city boundary.
"""

from pathlib import Path
import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BOUNDARY = (
    SCRIPT_DIR
    / "analysis_data"
    / "simulation_output_sq"
    / "munich_city_boundary.geojson"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "analysis_data" / "simulation_output_sq" / "munich_network_within_boundary.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Munich network within boundary.")
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY, help="Path to Munich boundary GeoJSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument(
        "--network-type",
        type=str,
        default="drive",
        choices=["drive", "walk", "bike", "all", "all_public"],
        help="OSMnx network type.",
    )
    args = parser.parse_args()

    boundary = gpd.read_file(args.boundary)
    if boundary.empty:
        raise ValueError(f"Boundary has no geometry: {args.boundary}")

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    else:
        boundary = boundary.to_crs("EPSG:4326")

    polygon = boundary.geometry.union_all()

    graph = ox.graph_from_polygon(polygon, network_type=args.network_type)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)

    fig, ax = plt.subplots(figsize=(12, 12))
    boundary.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=1.2, zorder=3)
    edges.plot(ax=ax, color="black", linewidth=0.35, alpha=0.85, zorder=2)

    ax.set_title(f"Munich Network within Boundary ({args.network_type})")
    ax.set_axis_off()
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Nodes: {len(nodes):,}")
    print(f"Edges: {len(edges):,}")
    print(f"Saved plot to: {args.output}")


if __name__ == "__main__":
    main()
