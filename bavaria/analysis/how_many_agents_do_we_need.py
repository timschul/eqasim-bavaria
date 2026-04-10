#!/usr/bin/env python3
"""
Count agents that never start or end a trip in Munich.

This script reads eqasim trips in chunks and performs a spatial point-in-polygon
check for both trip origins and destinations against a Munich boundary polygon.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRIPS_FILE = SCRIPT_DIR / "analysis_data" / "simulation_output_sq" / "eqasim_trips.csv"
DEFAULT_MUNICH_BOUNDARY = (
    SCRIPT_DIR
    / "analysis_data"
    / "simulation_output_sq"
    / "munich_city_boundary.geojson"
)
DEFAULT_OUTPUT_EXCLUDED = (
    SCRIPT_DIR / "analysis_data" / "simulation_output_sq" / "agents_never_munich.csv"
)


def load_munich_geometry(boundary_path: Path, target_crs: str) -> gpd.GeoSeries:
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {boundary_path}")

    if boundary.crs is None:
        raise ValueError(
            "Boundary CRS is missing. Please provide a boundary with valid CRS metadata."
        )

    munich_geom = boundary.geometry.union_all()
    munich_series = gpd.GeoSeries([munich_geom], crs=boundary.crs).to_crs(target_crs)
    return munich_series


def count_agents(
    trips_file: Path,
    munich_polygon: gpd.GeoSeries,
    csv_crs: str,
    chunk_size: int,
    delimiter: str,
) -> tuple[int, int, int, set]:
    all_agents: set = set()
    agents_with_munich_trip: set = set()

    required_cols = {
        "person_id",
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
    }

    for chunk in pd.read_csv(
        trips_file,
        sep=delimiter,
        chunksize=chunk_size,
        low_memory=False,
    ):
        missing = required_cols - set(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        chunk = chunk.dropna(
            subset=["person_id", "origin_x", "origin_y", "destination_x", "destination_y"]
        )
        if chunk.empty:
            continue

        chunk_agents = set(chunk["person_id"].astype("int64").tolist())
        all_agents.update(chunk_agents)

        origin_points = gpd.GeoSeries(
            gpd.points_from_xy(chunk["origin_x"], chunk["origin_y"]),
            crs=csv_crs,
        )
        dest_points = gpd.GeoSeries(
            gpd.points_from_xy(chunk["destination_x"], chunk["destination_y"]),
            crs=csv_crs,
        )

        # "start or end in Munich" => keep agent if either point is within Munich.
        # Include points on the city boundary as "in Munich" (within OR touches).
        in_munich = (
            (
                origin_points.within(munich_polygon.iloc[0])
                | origin_points.touches(munich_polygon.iloc[0])
            )
            | (
                dest_points.within(munich_polygon.iloc[0])
                | dest_points.touches(munich_polygon.iloc[0])
            )
        ).to_numpy()

        if in_munich.any():
            matching_ids = chunk.loc[in_munich, "person_id"].astype("int64").tolist()
            agents_with_munich_trip.update(matching_ids)

    excluded_agents = all_agents - agents_with_munich_trip
    return (
        len(all_agents),
        len(agents_with_munich_trip),
        len(excluded_agents),
        excluded_agents,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count agents that never have a trip origin or destination in Munich."
        )
    )
    parser.add_argument(
        "--trips-file",
        type=Path,
        default=DEFAULT_TRIPS_FILE,
        help="Path to eqasim_trips.csv",
    )
    parser.add_argument(
        "--munich-boundary",
        type=Path,
        default=DEFAULT_MUNICH_BOUNDARY,
        help="Path to Munich boundary geometry (GeoJSON/Shapefile/GPKG).",
    )
    parser.add_argument(
        "--csv-crs",
        type=str,
        default="EPSG:25832",
        help="CRS of trip coordinates in eqasim_trips.csv (default: EPSG:25832).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows per chunk to process.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=";",
        help="CSV delimiter for eqasim_trips.csv (default: ';').",
    )
    parser.add_argument(
        "--output-excluded",
        type=Path,
        default=DEFAULT_OUTPUT_EXCLUDED,
        help="Output CSV path for excluded agent ids.",
    )
    parser.add_argument(
        "--no-write-output",
        action="store_true",
        help="If set, do not write excluded agent ids CSV.",
    )
    args = parser.parse_args()

    if not args.trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {args.trips_file}")
    if not args.munich_boundary.exists():
        raise FileNotFoundError(f"Boundary file not found: {args.munich_boundary}")

    munich_polygon = load_munich_geometry(args.munich_boundary, args.csv_crs)

    total_agents, agents_to_keep, agents_to_exclude, excluded_ids = count_agents(
        trips_file=args.trips_file,
        munich_polygon=munich_polygon,
        csv_crs=args.csv_crs,
        chunk_size=args.chunk_size,
        delimiter=args.delimiter,
    )

    keep_share = (agents_to_keep / total_agents * 100.0) if total_agents else 0.0
    exclude_share = (agents_to_exclude / total_agents * 100.0) if total_agents else 0.0

    print(f"Total unique agents: {total_agents:,}")
    print(f"Agents with >=1 Munich start/end: {agents_to_keep:,} ({keep_share:.2f}%)")
    print(
        f"Agents with NEVER Munich start/end: {agents_to_exclude:,} ({exclude_share:.2f}%)"
    )

    if not args.no_write_output:
        args.output_excluded.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"person_id": sorted(excluded_ids)}).to_csv(
            args.output_excluded, index=False
        )
        print(f"Wrote excluded agent ids to: {args.output_excluded}")


if __name__ == "__main__":
    main()
