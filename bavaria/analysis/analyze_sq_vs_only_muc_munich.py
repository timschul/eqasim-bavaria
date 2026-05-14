"""Detailed analysis of Munich-area trips in two scenarios:

  A) Full Bavaria SQ run -- ``simulation_output_sq/eqasim_trips.csv``
  B) Munich-only SQ run  -- ``simulation_output_sq_ONLY_MUC/output_trips.csv.gz``

Focus areas:
  (1) Coordinate shift between A and B for the matched intersection.
  (2) Profile of the 1,515 agents that have Munich trips in A but none in B.
  (3) Mode share comparison and travel-time distribution comparison.

Spatial filter is the standard "origin OR destination within/touches the
Munich boundary" rule.

Trip identity:
  A: (person_id, person_trip_id)         -- person_trip_id is 0-based
  B: (person, trip_number - 1)           -- trip_number is 1-based
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "analysis_data"

DEFAULT_FILE_A = BASE / "simulation_output_sq" / "eqasim_trips.csv"
DEFAULT_FILE_B = BASE / "simulation_output_sq_ONLY_MUC" / "output_trips.csv.gz"
DEFAULT_BOUNDARY = BASE / "simulation_output_sq" / "munich_city_boundary.geojson"

PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def banner(title: str) -> None:
    print("\n" + "=" * 8 + " " + title + " " + "=" * 8)


def load_munich_polygon(boundary_path: Path, target_crs: str):
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {boundary_path}")
    geom = boundary.geometry.union_all()
    return gpd.GeoSeries([geom], crs=boundary.crs).to_crs(target_crs).iloc[0]


def hms_to_seconds(s: pd.Series) -> pd.Series:
    parts = s.astype(str).str.split(":", n=2, expand=True)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return h * 3600 + m * 60 + sec


def stream_munich(
    path: Path,
    munich_polygon,
    csv_crs: str,
    chunk_size: int,
    delimiter: str,
    *,
    label: str,
    cols: list[str],
    ox: str,
    oy: str,
    dx: str,
    dy: str,
) -> pd.DataFrame:
    print(f"[{label}] Streaming {path} ...")
    kept_chunks: list[pd.DataFrame] = []
    total_kept = 0
    for chunk in pd.read_csv(
        path,
        sep=delimiter,
        chunksize=chunk_size,
        usecols=cols,
        low_memory=False,
        compression="infer",
    ):
        chunk = chunk.dropna(subset=[ox, oy, dx, dy])
        if chunk.empty:
            continue
        origins = gpd.GeoSeries(gpd.points_from_xy(chunk[ox], chunk[oy]), crs=csv_crs)
        dests = gpd.GeoSeries(gpd.points_from_xy(chunk[dx], chunk[dy]), crs=csv_crs)
        in_munich = (
            origins.within(munich_polygon)
            | origins.touches(munich_polygon)
            | dests.within(munich_polygon)
            | dests.touches(munich_polygon)
        ).to_numpy()
        if in_munich.any():
            kept_chunks.append(chunk.loc[in_munich].copy())
            total_kept += int(in_munich.sum())
        print(f"[{label}]   chunk read={len(chunk):,} kept={int(in_munich.sum()):,}"
              f" running={total_kept:,}")
    return pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=cols)


def percentile_table(s: pd.Series, label: str) -> pd.Series:
    out = s.describe(percentiles=PERCENTILES)
    return out.rename(label)


def section_coordinate_shift(join: pd.DataFrame) -> None:
    banner("(1) COORDINATE SHIFT BETWEEN A AND B")
    n = len(join)
    ox_diff = join["origin_x"] - join["start_x"]
    oy_diff = join["origin_y"] - join["start_y"]
    dx_diff = join["destination_x"] - join["end_x"]
    dy_diff = join["destination_y"] - join["end_y"]
    o_dist = np.hypot(ox_diff, oy_diff)
    d_dist = np.hypot(dx_diff, dy_diff)

    print(f"matched trips                : {n:,}")
    print("\nOrigin coord distance |A - B| (m):")
    print(percentile_table(o_dist, "origin_dist"))
    print("\nDestination coord distance |A - B| (m):")
    print(percentile_table(d_dist, "dest_dist"))

    # Distance bins
    bins = [-0.001, 1, 5, 25, 50, 100, 250, 500, 1000, 5000, np.inf]
    labels = ["<1", "1-5", "5-25", "25-50", "50-100", "100-250", "250-500",
              "500-1k", "1k-5k", ">5k"]
    o_bin = pd.cut(o_dist, bins=bins, labels=labels)
    d_bin = pd.cut(d_dist, bins=bins, labels=labels)
    bin_df = pd.DataFrame(
        {
            "origin_count":  o_bin.value_counts().reindex(labels, fill_value=0),
            "origin_pct":    100 * o_bin.value_counts(normalize=True).reindex(labels, fill_value=0),
            "dest_count":    d_bin.value_counts().reindex(labels, fill_value=0),
            "dest_pct":      100 * d_bin.value_counts(normalize=True).reindex(labels, fill_value=0),
        }
    )
    print("\nDistance distribution (origin and destination):")
    print(bin_df.round(2).to_string())

    # Direction of shift (signed mean and median; should be ~0 if random)
    print("\nSigned A-B coordinate diff (m):")
    print(pd.DataFrame({
        "origin_dx": [ox_diff.mean(), ox_diff.median()],
        "origin_dy": [oy_diff.mean(), oy_diff.median()],
        "dest_dx":   [dx_diff.mean(), dx_diff.median()],
        "dest_dy":   [dy_diff.mean(), dy_diff.median()],
    }, index=["mean", "median"]).round(3).to_string())

    # Are A's coordinates ever exactly equal to B's? Or always shifted?
    o_zero = (o_dist < 1e-6).sum()
    print(f"\nexact origin coord match (<1e-6 m): {o_zero:,} ({100*o_zero/n:.3f}%)")

    # Stratify origin distance by the start activity type (proxy for facility type)
    print("\nOrigin shift by start_activity_type (median m, count):")
    grp = join.assign(o_dist=o_dist).groupby("start_activity_type")["o_dist"]
    summary = pd.concat({"median_m": grp.median(), "count": grp.size()}, axis=1)
    print(summary.sort_values("count", ascending=False).round(2).to_string())

    # Stratify by mode (A.mode); link snapping might depend on which network is used
    print("\nOrigin shift by A.mode (median m, count):")
    grp = join.assign(o_dist=o_dist).groupby("mode")["o_dist"]
    summary = pd.concat({"median_m": grp.median(), "count": grp.size()}, axis=1)
    print(summary.sort_values("count", ascending=False).round(2).to_string())

    # Test the link-snapping hypothesis: in B, do start coords match the start_link
    # representative point? We can check whether B's start_x/start_y equal end of
    # the previous trip's end_x/end_y for the same agent (within trip chain),
    # since MATSim should chain link-snapped points.
    chain = join.sort_values(["person_id", "trip_index"]).copy()
    chain["prev_end_x"] = chain.groupby("person_id")["end_x"].shift(1)
    chain["prev_end_y"] = chain.groupby("person_id")["end_y"].shift(1)
    chain["chain_diff"] = np.hypot(
        chain["start_x"] - chain["prev_end_x"], chain["start_y"] - chain["prev_end_y"]
    )
    valid = chain["prev_end_x"].notna()
    cd = chain.loc[valid, "chain_diff"]
    print("\nB-internal trip chain consistency (start_xy of trip k vs end_xy of trip k-1):")
    print(percentile_table(cd, "chain_diff_m").round(3))
    print(f"  >1 m: {(cd > 1).sum():,} of {len(cd):,}"
          f"  ({100*(cd>1).sum()/max(len(cd),1):.2f}%)")


def section_missing_agents(df_a: pd.DataFrame, agents_b: set) -> None:
    banner("(2) AGENTS WITH MUNICH TRIPS IN A BUT NOT IN B")
    missing = sorted(set(df_a["person_id"].unique()) - agents_b)
    print(f"missing agents: {len(missing):,}")
    sub = df_a[df_a["person_id"].isin(missing)]
    print(f"their Munich trips in A: {len(sub):,}"
          f"  (avg {len(sub)/max(len(missing),1):.2f} per agent)")

    print("\nTrips per missing agent (distribution):")
    counts = sub.groupby("person_id").size()
    print(percentile_table(counts, "trips_per_agent").round(2))

    print("\nMode share among their trips (A.mode):")
    mc = sub["mode"].value_counts(dropna=False)
    print(pd.DataFrame({"count": mc, "pct": 100 * mc / mc.sum()}).round(2).to_string())

    print("\nFollowing-purpose share among their trips:")
    pc = sub["following_purpose"].value_counts(dropna=False)
    print(pd.DataFrame({"count": pc, "pct": 100 * pc / pc.sum()}).round(2).to_string())

    # Are these agents inside the Munich population that was input to B?
    # We don't load the population here, but compare counts: 167,939 in pop, 166,424
    # produced trips => 1,515 produced no trips. Cross-check against missing_count.
    print(f"\nmissing_count vs (population - agents_with_trips):"
          f" 1515 expected, observed {len(missing)}")


def mode_share_table(df: pd.DataFrame, mode_col: str, label: str) -> pd.DataFrame:
    s = df[mode_col].value_counts(dropna=False)
    return pd.DataFrame({label + "_count": s, label + "_pct": 100 * s / s.sum()}).round(3)


def section_modes(df_a: pd.DataFrame, df_b: pd.DataFrame, join: pd.DataFrame) -> None:
    banner("(3a) MODE SHARE COMPARISON (Munich-area trips)")
    a = mode_share_table(df_a, "mode", "A")
    b = mode_share_table(df_b, "main_mode", "B").rename(
        index={mn: mn for mn in df_b["main_mode"].unique()}
    )
    cmp = pd.concat([a, b], axis=1).fillna(0)
    cmp["diff_pct_pts"] = (cmp["B_pct"] - cmp["A_pct"]).round(3)
    cmp = cmp.sort_values("A_count", ascending=False)
    print(cmp.to_string())

    banner("(3b) MODE SHARE BY PURPOSE (B vs A; following_purpose / end_activity_type)")
    pivots = []
    for label, df, mode_col, purp_col in [
        ("A", df_a, "mode", "following_purpose"),
        ("B", df_b, "main_mode", "end_activity_type"),
    ]:
        p = (
            df.groupby([purp_col, mode_col])
            .size()
            .unstack(fill_value=0)
        )
        p_pct = p.div(p.sum(axis=1), axis=0) * 100
        p_pct.columns = [f"{label}_{c}" for c in p_pct.columns]
        pivots.append(p_pct.round(2))
    # Align purposes as union
    full = pd.concat(pivots, axis=1).fillna(0).round(2)
    print(full.to_string())

    banner("(3c) MODE TRANSITIONS ON MATCHED TRIPS (A.mode -> B.main_mode)")
    trans = (
        join.groupby(["mode", "main_mode"]).size().rename("count").reset_index()
    )
    trans["pct_of_intersection"] = (100 * trans["count"] / trans["count"].sum()).round(3)
    print(trans.sort_values("count", ascending=False).head(25).to_string(index=False))

    same = (join["mode"] == join["main_mode"]).sum()
    print(f"\nmatched trips with same mode: {same:,} of {len(join):,}"
          f" ({100*same/max(len(join),1):.2f}%)")


def section_travel_times(df_a: pd.DataFrame, df_b: pd.DataFrame, join: pd.DataFrame) -> None:
    banner("(4a) TRAVEL TIME DISTRIBUTION (seconds) -- ALL Munich-area trips")
    tt_a = df_a["travel_time"].astype(float)
    tt_b = df_b["trav_time_sec"].astype(float)
    print("A.travel_time:")
    print(percentile_table(tt_a, "A_travel_time").round(1))
    print("\nB.trav_time:")
    print(percentile_table(tt_b, "B_trav_time").round(1))
    print(f"\nA total person-seconds: {tt_a.sum():,.0f}  "
          f"({tt_a.sum()/3600:,.1f} person-hours)")
    print(f"B total person-seconds: {tt_b.sum():,.0f}  "
          f"({tt_b.sum()/3600:,.1f} person-hours)")
    print(f"B / A ratio (total time)  : {tt_b.sum()/max(tt_a.sum(),1):.4f}")

    banner("(4b) TRAVEL TIME BY MODE (mean / median / p90 / count)")
    a_by_mode = (
        df_a.groupby("mode")["travel_time"].agg(
            count="size", mean="mean", median="median", p90=lambda s: s.quantile(0.9)
        )
    )
    b_by_mode = (
        df_b.groupby("main_mode")["trav_time_sec"].agg(
            count="size", mean="mean", median="median", p90=lambda s: s.quantile(0.9)
        )
    )
    a_by_mode.columns = [f"A_{c}" for c in a_by_mode.columns]
    b_by_mode.columns = [f"B_{c}" for c in b_by_mode.columns]
    full = a_by_mode.join(b_by_mode, how="outer").fillna(0).round(1)
    full = full.sort_values("A_count", ascending=False)
    print(full.to_string())

    banner("(4c) PER-TRIP TRAVEL TIME DIFFERENCE (B - A) on matched intersection")
    diff = join["trav_time_sec"] - join["travel_time"]
    print(percentile_table(diff, "B_minus_A_sec").round(1))
    print(f"|diff| <= 60s: {(diff.abs() <= 60).sum():,}"
          f" ({100*(diff.abs()<=60).sum()/len(join):.2f}%)")
    print(f"B faster (diff < -60s): {(diff < -60).sum():,}"
          f" ({100*(diff<-60).sum()/len(join):.2f}%)")
    print(f"B slower (diff >  60s): {(diff >  60).sum():,}"
          f" ({100*(diff> 60).sum()/len(join):.2f}%)")

    print("\nMean/median of (B - A) by A.mode (s):")
    by_mode = join.assign(diff=diff).groupby("mode")["diff"].agg(
        count="size", mean="mean", median="median"
    ).round(1).sort_values("count", ascending=False)
    print(by_mode.to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--file-a", type=Path, default=DEFAULT_FILE_A)
    p.add_argument("--file-b", type=Path, default=DEFAULT_FILE_B)
    p.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    p.add_argument("--csv-crs", type=str, default="EPSG:25832")
    p.add_argument("--delimiter", type=str, default=";")
    p.add_argument("--chunk-size", type=int, default=500_000)
    args = p.parse_args()

    munich_polygon = load_munich_polygon(args.boundary, args.csv_crs)

    cols_a = [
        "person_id", "person_trip_id",
        "origin_x", "origin_y", "destination_x", "destination_y",
        "departure_time", "travel_time",
        "mode", "preceding_purpose", "following_purpose",
        "euclidean_distance", "routed_distance", "vehicle_distance",
    ]
    df_a = stream_munich(
        args.file_a, munich_polygon, args.csv_crs, args.chunk_size, args.delimiter,
        label="A", cols=cols_a, ox="origin_x", oy="origin_y",
        dx="destination_x", dy="destination_y",
    )
    df_a["person_id"] = df_a["person_id"].astype("int64")
    df_a["trip_index"] = df_a["person_trip_id"].astype("int64")

    cols_b = [
        "person", "trip_number",
        "dep_time", "trav_time", "wait_time",
        "traveled_distance", "euclidean_distance",
        "main_mode", "start_activity_type", "end_activity_type",
        "start_x", "start_y", "end_x", "end_y", "start_link", "end_link",
    ]
    df_b = stream_munich(
        args.file_b, munich_polygon, args.csv_crs, args.chunk_size, args.delimiter,
        label="B", cols=cols_b, ox="start_x", oy="start_y",
        dx="end_x", dy="end_y",
    )
    df_b["person_id"] = df_b["person"].astype("int64")
    df_b["trip_index"] = df_b["trip_number"].astype("int64") - 1
    df_b["dep_time_sec"] = hms_to_seconds(df_b["dep_time"])
    df_b["trav_time_sec"] = hms_to_seconds(df_b["trav_time"])

    print(f"\nA: {len(df_a):,} trips for {df_a['person_id'].nunique():,} agents")
    print(f"B: {len(df_b):,} trips for {df_b['person_id'].nunique():,} agents")

    # Inner join for attribute analysis
    join = df_a.merge(df_b, on=["person_id", "trip_index"], how="inner",
                      validate="one_to_one")
    print(f"matched intersection: {len(join):,}")

    section_coordinate_shift(join)
    section_missing_agents(df_a, set(df_b["person_id"].unique()))
    section_modes(df_a, df_b, join)
    section_travel_times(df_a, df_b, join)

    print("\nDone.")


if __name__ == "__main__":
    main()
