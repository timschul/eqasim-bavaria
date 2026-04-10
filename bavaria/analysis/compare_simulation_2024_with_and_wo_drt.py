#!/usr/bin/env python3
"""
Analysis script to compare two simulations:
1. fleetpy_drt (with ride-pooling/DRT)
2. munich_10pct (without ride-pooling/DRT)

Fragen: Wer nutzt DRT (Zweck, vorherige Modi)? Verschiebungen im Netz (PT, Pkw)?
FLEETPY_DIR / MUNICH_DIR oben anpassen für andere Läufe (z.B. munich_2024_10pct).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Paths – relativ zum Skriptverzeichnis, bei Bedarf anpassen
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_ROOT = _SCRIPT_DIR / "analysis_data"

# If True: compare SC1 (DRT) vs SQ (2024).
# If False: compare SC3 (DRT) vs SQ (2040).
is_for_2024_sq = True

if is_for_2024_sq:
    FLEETPY_DIR = _DATA_ROOT / "MINGA_MATSIM_SC1_10pct_fs100"
    MUNICH_DIR = _DATA_ROOT / "simulation_output_sq"
    _REPORT_NAME = "compare_SC1_with_sq.txt"
    _DRT_SCENARIO_LABEL = "SC1"
    _SQ_SCENARIO_LABEL = "SQ"
else:
    FLEETPY_DIR = _DATA_ROOT / "MINGA_MATSIM_SC3_10pct_fs100"
    MUNICH_DIR = _DATA_ROOT / "simulation_output_2040"
    _REPORT_NAME = "compare_SC3_with_2040_sq.txt"
    _DRT_SCENARIO_LABEL = "SC3"
    _SQ_SCENARIO_LABEL = "SQ (2040)"

print("=" * 80)
print("SIMULATION COMPARISON ANALYSIS")
print("=" * 80)

# Keep key outputs for a compact text report
comparison = None
merged = None
car_comparison = None
sc1_mode_pct = None
sq_mode_pct = None

# ============================================================================
# QUESTION 1: Who are the DRT users?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: Who are the users for autonomous driving (DRT)?")
print("=" * 80)

# Load fleetpy trips data
print("\nLoading fleetpy_drt trips data...")
try:
    trips_fleetpy = pd.read_csv(
        FLEETPY_DIR / "eqasim_trips.csv",
        sep=';',
        low_memory=False
    )
    print(f"Loaded {len(trips_fleetpy):,} trips from fleetpy_drt")
except Exception as e:
    print(f"Error loading fleetpy trips: {e}")
    trips_fleetpy = None

if trips_fleetpy is not None:
    # Filter DRT trips
    drt_trips = trips_fleetpy[trips_fleetpy['mode'] == 'drt'].copy()
    print(f"\nTotal DRT trips: {len(drt_trips):,}")
    print(f"Percentage of all trips: {len(drt_trips)/len(trips_fleetpy)*100:.2f}%")
    
    # Analyze trip purposes
    print("\n--- DRT Trip Purposes ---")
    print("\nPreceding purpose (origin activity):")
    preceding_purpose = drt_trips['preceding_purpose'].value_counts()
    print(preceding_purpose)
    print("\nPercentages:")
    print(preceding_purpose / len(drt_trips) * 100)
    
    print("\nFollowing purpose (destination activity):")
    following_purpose = drt_trips['following_purpose'].value_counts()
    print(following_purpose)
    print("\nPercentages:")
    print(following_purpose / len(drt_trips) * 100)
    
    # Analyze trip purposes combination
    print("\n--- Purpose Combinations ---")
    purpose_comb = drt_trips.groupby(['preceding_purpose', 'following_purpose']).size().sort_values(ascending=False)
    print(purpose_comb.head(20))
    
    # Analyze departure times
    print("\n--- Departure Time Analysis ---")
    drt_trips['departure_hour'] = drt_trips['departure_time'] / 3600
    drt_trips['departure_hour'] = drt_trips['departure_hour'].round().astype(int)
    drt_trips['departure_hour'] = drt_trips['departure_hour'].clip(0, 23)
    
    print("\nDRT trips by hour of day:")
    hourly_drt = drt_trips['departure_hour'].value_counts().sort_index()
    print(hourly_drt)
    
    # Analyze trip distances
    print("\n--- Trip Distance Analysis ---")
    print(f"Mean euclidean distance: {drt_trips['euclidean_distance'].mean():.2f} m")
    print(f"Median euclidean distance: {drt_trips['euclidean_distance'].median():.2f} m")
    print(f"Mean vehicle distance: {drt_trips['vehicle_distance'].mean():.2f} m")
    print(f"Median vehicle distance: {drt_trips['vehicle_distance'].median():.2f} m")
    
    # Analyze returning trips
    print("\n--- Returning Trips Analysis ---")
    returning_counts = drt_trips['returning'].value_counts()
    print(returning_counts)
    print(f"\nReturning trips: {returning_counts.get(True, 0) / len(drt_trips) * 100:.2f}%")
    
    # To identify previous mode, we need to compare with munich_10pct scenario
    # For now, let's analyze what modes people use in similar trips
    print("\n--- Mode Share Analysis (All trips in fleetpy_drt) ---")
    mode_share_fleetpy = trips_fleetpy['mode'].value_counts()
    print(mode_share_fleetpy)
    print("\nPercentages:")
    sc1_mode_pct = mode_share_fleetpy / len(trips_fleetpy) * 100
    print(sc1_mode_pct)

# ============================================================================
# QUESTION 2: Network shifts - Mode comparison
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: Are there shifts in the network?")
print("=" * 80)

# Try to load munich_10pct data
print("\nAttempting to load munich_10pct trips data...")
munich_trips = None

# Check if there's a trips file in munich_10pct
possible_paths = [
    MUNICH_DIR / "eqasim_trips.csv",
    MUNICH_DIR / "ITERS" / "it.0" / "0.eqasim_trips.csv",
]

for path in possible_paths:
    if path.exists():
        try:
            print(f"Found trips file at: {path}")
            munich_trips = pd.read_csv(path, sep=';', low_memory=False)
            print(f"Loaded {len(munich_trips):,} trips from munich_10pct")
            break
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            continue

if munich_trips is None:
    print("Could not find munich_10pct trips data. Checking for alternative files...")
    # Check for other files that might contain trip information
    for file in MUNICH_DIR.glob("*.csv"):
        print(f"Found CSV: {file.name} ({file.stat().st_size} bytes)")
    
    # Check ITERS directory
    iters_dir = MUNICH_DIR / "ITERS"
    if iters_dir.exists():
        for iter_dir in iters_dir.iterdir():
            if iter_dir.is_dir():
                print(f"\nFiles in {iter_dir.name}:")
                for f in iter_dir.iterdir():
                    print(f"  - {f.name}")

# Compare mode shares if both datasets are available
if trips_fleetpy is not None and munich_trips is not None:
    print("\n--- Mode Share Comparison ---")
    
    mode_share_fleetpy = trips_fleetpy['mode'].value_counts()
    mode_share_munich = munich_trips['mode'].value_counts()
    sq_mode_pct = mode_share_munich / len(munich_trips) * 100
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'fleetpy_drt': mode_share_fleetpy,
        'munich_10pct': mode_share_munich
    }).fillna(0)
    
    comparison['fleetpy_pct'] = comparison['fleetpy_drt'] / len(trips_fleetpy) * 100
    comparison['munich_pct'] = comparison['munich_10pct'] / len(munich_trips) * 100
    comparison['difference'] = comparison['fleetpy_pct'] - comparison['munich_pct']
    comparison['relative_change'] = (comparison['difference'] / comparison['munich_pct'] * 100).fillna(0)
    
    print("\nMode Share Comparison:")
    print(comparison.sort_values('difference', ascending=False))

    # Transparent absolute deltas by mode (DRT scenario - SQ)
    comparison['count_delta'] = comparison['fleetpy_drt'] - comparison['munich_10pct']
    print(f"\nAbsolute trip count deltas by mode ({_DRT_SCENARIO_LABEL} - {_SQ_SCENARIO_LABEL}):")
    print(comparison['count_delta'].sort_values(ascending=False))

    print(f"\n--- {_SQ_SCENARIO_LABEL} Mode shares ---")
    print(sq_mode_pct.sort_values(ascending=False))
    print(f"\n--- {_DRT_SCENARIO_LABEL} Mode shares ---")
    if sc1_mode_pct is None:
        sc1_mode_pct = mode_share_fleetpy / len(trips_fleetpy) * 100
    print(sc1_mode_pct.sort_values(ascending=False))

    print("\n--- Key Findings ---")
    if 'drt' in comparison.index:
        print(f"DRT trips in fleetpy_drt: {comparison.loc['drt', 'fleetpy_drt']:.0f}")
    if 'car' in comparison.index:
        print(f"Car trips - fleetpy: {comparison.loc['car', 'fleetpy_pct']:.2f}%, munich: {comparison.loc['car', 'munich_pct']:.2f}%")
        print(f"Car difference: {comparison.loc['car', 'difference']:.2f} percentage points")
    if 'pt' in comparison.index:
        print(f"PT trips - fleetpy: {comparison.loc['pt', 'fleetpy_pct']:.2f}%, munich: {comparison.loc['pt', 'munich_pct']:.2f}%")

# ============================================================================
# Network Analysis - Link volumes
# ============================================================================
print("\n" + "=" * 80)
print("NETWORK ANALYSIS: Link volumes and traffic patterns")
print("=" * 80)

# Load link data
print("\nLoading link data...")
try:
    links_fleetpy = pd.read_csv(
        FLEETPY_DIR / "output_links.csv",
        sep=';',
        low_memory=False
    )
    print(f"Loaded {len(links_fleetpy):,} links from fleetpy_drt")
    print(f"Columns: {list(links_fleetpy.columns)}")
except Exception as e:
    print(f"Error loading fleetpy links: {e}")
    links_fleetpy = None

# Check for munich links
munich_links = None
if (MUNICH_DIR / "output_links.csv").exists():
    try:
        munich_links = pd.read_csv(
            MUNICH_DIR / "output_links.csv",
            sep=';',
            low_memory=False
        )
        print(f"Loaded {len(munich_links):,} links from munich_10pct")
    except Exception as e:
        print(f"Error loading munich links: {e}")

if links_fleetpy is not None:
    # Analyze link volumes
    if 'vol_car' in links_fleetpy.columns:
        print("\n--- Car Volume Analysis (fleetpy_drt) ---")
        print(f"Total car volume: {links_fleetpy['vol_car'].sum():,.0f}")
        print(f"Mean car volume per link: {links_fleetpy['vol_car'].mean():.2f}")
        print(f"Median car volume per link: {links_fleetpy['vol_car'].median():.2f}")
        print(f"Max car volume: {links_fleetpy['vol_car'].max():,.0f}")
        
        # Top links by volume
        print("\nTop 10 links by car volume:")
        print(links_fleetpy.nlargest(10, 'vol_car')[['link', 'vol_car', 'length', 'modes']])
    
    # Compare with munich if available
    if munich_links is not None and 'vol_car' in munich_links.columns:
        print("\n--- Car Volume Comparison ---")
        print(f"Fleetpy total car volume: {links_fleetpy['vol_car'].sum():,.0f}")
        print(f"Munich total car volume: {munich_links['vol_car'].sum():,.0f}")
        
        # Merge for comparison
        merged = links_fleetpy[['link', 'vol_car']].merge(
            munich_links[['link', 'vol_car']],
            on='link',
            suffixes=('_fleetpy', '_munich')
        )
        merged['difference'] = merged['vol_car_fleetpy'] - merged['vol_car_munich']
        
        print(f"\nLinks with increased car volume (fleetpy > munich): {(merged['difference'] > 0).sum():,}")
        print(f"Links with decreased car volume (fleetpy < munich): {(merged['difference'] < 0).sum():,}")
        print(f"Mean difference: {merged['difference'].mean():.2f}")
        
        print("\nTop 10 links with largest increase in car volume:")
        print(merged.nlargest(10, 'difference')[['link', 'vol_car_fleetpy', 'vol_car_munich', 'difference']])
        
        print("\nTop 10 links with largest decrease in car volume:")
        print(merged.nsmallest(10, 'difference')[['link', 'vol_car_fleetpy', 'vol_car_munich', 'difference']])

# ============================================================================
# Temporal Analysis
# ============================================================================
if trips_fleetpy is not None:
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS: When are shifts happening?")
    print("=" * 80)
    
    trips_fleetpy['departure_hour'] = trips_fleetpy['departure_time'] / 3600
    trips_fleetpy['departure_hour'] = trips_fleetpy['departure_hour'].round().astype(int)
    trips_fleetpy['departure_hour'] = trips_fleetpy['departure_hour'].clip(0, 23)
    
    # Hourly mode shares for fleetpy
    print("\n--- Hourly Mode Shares (fleetpy_drt) ---")
    hourly_modes_fleetpy = trips_fleetpy.groupby(['departure_hour', 'mode']).size().unstack(fill_value=0)
    hourly_modes_fleetpy_pct = hourly_modes_fleetpy.div(hourly_modes_fleetpy.sum(axis=1), axis=0) * 100
    
    print("\nCar share by hour:")
    if 'car' in hourly_modes_fleetpy_pct.columns:
        print(hourly_modes_fleetpy_pct['car'])
    
    print("\nPT share by hour:")
    if 'pt' in hourly_modes_fleetpy_pct.columns:
        print(hourly_modes_fleetpy_pct['pt'])
    
    print("\nDRT share by hour:")
    if 'drt' in hourly_modes_fleetpy_pct.columns:
        print(hourly_modes_fleetpy_pct['drt'])
    
    # Compare with munich if available
    if munich_trips is not None:
        munich_trips['departure_hour'] = munich_trips['departure_time'] / 3600
        munich_trips['departure_hour'] = munich_trips['departure_hour'].round().astype(int)
        munich_trips['departure_hour'] = munich_trips['departure_hour'].clip(0, 23)
        
        hourly_modes_munich = munich_trips.groupby(['departure_hour', 'mode']).size().unstack(fill_value=0)
        hourly_modes_munich_pct = hourly_modes_munich.div(hourly_modes_munich.sum(axis=1), axis=0) * 100
        
        print("\n--- Hourly Mode Share Comparison ---")
        if 'car' in hourly_modes_fleetpy_pct.columns and 'car' in hourly_modes_munich_pct.columns:
            car_comparison = pd.DataFrame({
                'fleetpy': hourly_modes_fleetpy_pct['car'],
                'munich': hourly_modes_munich_pct['car'],
            })
            car_comparison['difference'] = car_comparison['fleetpy'] - car_comparison['munich']
            print("\nCar share difference by hour:")
            print(car_comparison)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# ============================================================================
# Write compact report
# ============================================================================
report_path = _SCRIPT_DIR / _REPORT_NAME
report_lines = [
    f"{_DRT_SCENARIO_LABEL} vs {_SQ_SCENARIO_LABEL} - Key Points",
    "=" * 28,
]

if trips_fleetpy is not None:
    drt_count = int((trips_fleetpy["mode"] == "drt").sum())
    drt_share = drt_count / len(trips_fleetpy) * 100
    report_lines.append(f"- Trips {_DRT_SCENARIO_LABEL} total: {len(trips_fleetpy):,}")
    report_lines.append(f"- DRT trips in {_DRT_SCENARIO_LABEL}: {drt_count:,} ({drt_share:.3f}%)")

if "preceding_purpose" in locals() and len(preceding_purpose) > 0:
    report_lines.append(
        f"- Top DRT origin purpose: {preceding_purpose.index[0]} ({int(preceding_purpose.iloc[0]):,})"
    )
if "following_purpose" in locals() and len(following_purpose) > 0:
    report_lines.append(
        f"- Top DRT destination purpose: {following_purpose.index[0]} ({int(following_purpose.iloc[0]):,})"
    )
if "hourly_drt" in locals() and len(hourly_drt) > 0:
    peak_hour = int(hourly_drt.idxmax())
    peak_count = int(hourly_drt.max())
    report_lines.append(f"- Peak DRT departure hour: {peak_hour}:00 ({peak_count:,} trips)")

if munich_trips is not None:
    report_lines.append(f"- Trips {_SQ_SCENARIO_LABEL} total: {len(munich_trips):,}")

if comparison is not None:
    if "car" in comparison.index:
        report_lines.append(
            f"- Car share: {_DRT_SCENARIO_LABEL} {comparison.loc['car', 'fleetpy_pct']:.2f}% vs {_SQ_SCENARIO_LABEL} {comparison.loc['car', 'munich_pct']:.2f}% "
            f"(delta {comparison.loc['car', 'difference']:+.2f} pp)"
        )
    if "pt" in comparison.index:
        report_lines.append(
            f"- PT share: {_DRT_SCENARIO_LABEL} {comparison.loc['pt', 'fleetpy_pct']:.2f}% vs {_SQ_SCENARIO_LABEL} {comparison.loc['pt', 'munich_pct']:.2f}% "
            f"(delta {comparison.loc['pt', 'difference']:+.2f} pp)"
        )
    if "count_delta" in comparison.columns:
        report_lines.append(f"- Absolute trip deltas by mode ({_DRT_SCENARIO_LABEL} - {_SQ_SCENARIO_LABEL}):")
        for mode, delta in comparison["count_delta"].sort_values(ascending=False).items():
            report_lines.append(f"  - {mode}: {int(delta):+,.0f}")
if sq_mode_pct is not None:
    report_lines.append(f"- {_SQ_SCENARIO_LABEL} Mode shares (%):")
    for mode, pct in sq_mode_pct.sort_values(ascending=False).items():
        report_lines.append(f"  - {mode}: {pct:.3f}%")
if sc1_mode_pct is not None:
    report_lines.append(f"- {_DRT_SCENARIO_LABEL} Mode shares (%):")
    for mode, pct in sc1_mode_pct.sort_values(ascending=False).items():
        sq_pct = float(sq_mode_pct.get(mode, 0.0)) if sq_mode_pct is not None else 0.0
        report_lines.append(f"  - {mode}: {pct:.3f}% ({pct - sq_pct:+.3f}% vs {_SQ_SCENARIO_LABEL})")

if comparison is not None and "car" in comparison.index:
    car_delta_pct = float(comparison.loc["car", "difference"])
    drt_delta_pct = float(comparison.loc["drt", "difference"]) if "drt" in comparison.index else 0.0
    pt_delta_pct = float(comparison.loc["pt", "difference"]) if "pt" in comparison.index else 0.0
    car_passenger_delta = float(comparison.loc["car_passenger", "difference"]) if "car_passenger" in comparison.index else 0.0

    report_lines.append("- Possible reasons why car trips increased:")
    report_lines.append(
        "  - DRT uptake is small, so most mode shifts still happen among non-DRT modes."
    )
    report_lines.append(
        f"  - In this run, DRT gains (+{drt_delta_pct:.3f}%) are accompanied by larger drops in PT ({pt_delta_pct:.3f}%) and "
        f"car passenger ({car_passenger_delta:.3f}%), not by a drop in car."
    )
    report_lines.append(
        "  - Adding ride-pooling can change generalized costs/network conditions slightly, which can make car more attractive for some trips."
    )
    report_lines.append(
        f"  - Net effect observed here: car share changes by {car_delta_pct:+.3f}%."
    )

if merged is not None:
    report_lines.append(f"- Links with higher car volume in {_DRT_SCENARIO_LABEL}: {(merged['difference'] > 0).sum():,}")
    report_lines.append(f"- Links with lower car volume in {_DRT_SCENARIO_LABEL}: {(merged['difference'] < 0).sum():,}")
    report_lines.append(f"- Mean link car-volume delta ({_DRT_SCENARIO_LABEL} - {_SQ_SCENARIO_LABEL}): {merged['difference'].mean():.2f}")

if car_comparison is not None and len(car_comparison) > 0:
    max_increase_hour = int(car_comparison["difference"].idxmax())
    max_decrease_hour = int(car_comparison["difference"].idxmin())
    report_lines.append(
        f"- Largest hourly car-share increase: {max_increase_hour}:00 ({car_comparison.loc[max_increase_hour, 'difference']:+.2f} pp)"
    )
    report_lines.append(
        f"- Largest hourly car-share decrease: {max_decrease_hour}:00 ({car_comparison.loc[max_decrease_hour, 'difference']:+.2f} pp)"
    )

report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print(f"\nSaved key-point report to: {report_path}")
