#!/usr/bin/env python3
"""
Create filtered population and households directly from a trips CSV.

Inputs:
1) Trips CSV with an agent/person id column (default column: person_id).
2) Population file (.xml.gz) with <person id="..."> blocks.
3) Households file (.xml.gz) with <personId refId="..."/> members.

Rule:
- Keep agents that appear in the trips CSV (unique IDs from the configured
  trips id column).
- Filter population to those person IDs.
- Filter households to members in that filtered population.
- Drop households that end up with zero members.

Outputs:
1) Filtered population .xml.gz
2) Filtered households .xml.gz
"""

import argparse
import gzip
import hashlib
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "analysis_data" / "simulation_output_sq"
DEFAULT_TRIPS = BASE / "munich_trips.csv"
DEFAULT_POPULATION_IN = BASE / "munich_population.xml.gz"
DEFAULT_HOUSEHOLDS_IN = BASE / "munich_households.xml.gz"
DEFAULT_POPULATION_OUT = BASE / "munich_population_agents_from_munich_trips.xml.gz"
DEFAULT_HOUSEHOLDS_OUT = BASE / "munich_households_agents_from_munich_trips.xml.gz"

DEFAULT_COMPARE_POPULATION = BASE / "munich_population_agents_in_muc.xml.gz"
DEFAULT_COMPARE_HOUSEHOLDS = BASE / "munich_households_agents_in_muc.xml.gz"

PERSON_RE = re.compile(r'<person\s+id="([^"]+)"')
MEMBER_RE = re.compile(r'<personId\s+refId="([^"]+)"\s*/>')
HOUSEHOLD_OPEN_RE = re.compile(r"<household\b")
HOUSEHOLD_CLOSE_RE = re.compile(r"</household>")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_ids_from_trips(
    trips_file: Path, delimiter: str, chunk_size: int, id_column: str
) -> set[str]:
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_file}")
    selected_ids: set[str] = set()
    rows = 0
    for chunk in pd.read_csv(trips_file, sep=delimiter, chunksize=chunk_size, low_memory=False):
        if id_column not in chunk.columns:
            raise ValueError(f"Trips file must contain column '{id_column}'.")
        rows += len(chunk)
        pid = pd.to_numeric(chunk[id_column], errors="coerce")
        valid = pid.notna()
        selected_ids.update(pid[valid].astype("int64").astype(str).tolist())
    if rows == 0:
        raise ValueError("Trips file has no rows.")
    if not selected_ids:
        raise ValueError("No valid person_id values found in trips file.")
    return selected_ids


def filter_population(
    input_population: Path, output_population: Path, selected_ids: set[str]
) -> tuple[int, int, set[str], set[str]]:
    if not input_population.exists():
        raise FileNotFoundError(f"Population file not found: {input_population}")

    output_population.parent.mkdir(parents=True, exist_ok=True)
    if output_population.exists():
        output_population.unlink()

    total_persons = 0
    kept_persons = 0
    source_ids: set[str] = set()
    written_ids: set[str] = set()

    inside_person = False
    keep_person = False
    person_buffer: list[str] = []

    with gzip.open(input_population, "rt", encoding="utf-8", errors="strict") as fin, gzip.open(
        output_population, "wt", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not inside_person:
                m = PERSON_RE.search(line)
                if m:
                    person_id = m.group(1)
                    inside_person = True
                    keep_person = person_id in selected_ids
                    person_buffer = [line]
                    total_persons += 1
                    source_ids.add(person_id)
                    if keep_person:
                        written_ids.add(person_id)
                else:
                    fout.write(line)
            else:
                person_buffer.append(line)
                if "</person>" in line:
                    if keep_person:
                        fout.writelines(person_buffer)
                        kept_persons += 1
                    inside_person = False
                    keep_person = False
                    person_buffer = []

    if inside_person:
        raise ValueError("Malformed population XML: unclosed <person> block.")

    return total_persons, kept_persons, source_ids, written_ids


def filter_households(
    input_households: Path, output_households: Path, allowed_person_ids: set[str]
) -> tuple[int, int, int, int, set[str], set[str]]:
    if not input_households.exists():
        raise FileNotFoundError(f"Households file not found: {input_households}")

    output_households.parent.mkdir(parents=True, exist_ok=True)
    if output_households.exists():
        output_households.unlink()

    source_households = 0
    kept_households = 0
    source_member_rows = 0
    kept_member_rows = 0
    source_member_ids: set[str] = set()
    kept_member_ids: set[str] = set()

    inside_household = False
    household_lines: list[str] = []
    kept_members_in_household = 0

    with gzip.open(input_households, "rt", encoding="utf-8", errors="strict") as fin, gzip.open(
        output_households, "wt", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not inside_household:
                if HOUSEHOLD_OPEN_RE.search(line):
                    inside_household = True
                    source_households += 1
                    household_lines = [line]
                    kept_members_in_household = 0
                else:
                    fout.write(line)
                continue

            m = MEMBER_RE.search(line)
            if m:
                member_id = m.group(1)
                source_member_rows += 1
                source_member_ids.add(member_id)
                if member_id in allowed_person_ids:
                    household_lines.append(line)
                    kept_members_in_household += 1
                    kept_member_rows += 1
                    kept_member_ids.add(member_id)
                continue

            household_lines.append(line)
            if HOUSEHOLD_CLOSE_RE.search(line):
                if kept_members_in_household > 0:
                    fout.writelines(household_lines)
                    kept_households += 1
                inside_household = False
                household_lines = []
                kept_members_in_household = 0

    if inside_household:
        raise ValueError("Malformed households XML: unclosed <household> block.")

    return (
        source_households,
        kept_households,
        source_member_rows,
        kept_member_rows,
        source_member_ids,
        kept_member_ids,
    )


def count_persons(population_file: Path) -> tuple[int, set[str]]:
    count = 0
    ids = set()
    with gzip.open(population_file, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            m = PERSON_RE.search(line)
            if m:
                count += 1
                ids.add(m.group(1))
    return count, ids


def count_household_members(households_file: Path) -> tuple[int, set[str]]:
    count = 0
    ids = set()
    with gzip.open(households_file, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            m = MEMBER_RE.search(line)
            if m:
                count += 1
                ids.add(m.group(1))
    return count, ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create filtered population and households directly from a trips CSV "
            "(no intermediate trips filtering step)."
        )
    )
    parser.add_argument(
        "--trips-file",
        type=Path,
        default=DEFAULT_TRIPS,
        help="Input trips CSV containing agent IDs (default: munich_trips.csv).",
    )
    parser.add_argument(
        "--trips-id-column",
        type=str,
        default="person_id",
        help="Column name in trips CSV that identifies agents/persons.",
    )
    parser.add_argument("--population-in", type=Path, default=DEFAULT_POPULATION_IN)
    parser.add_argument("--households-in", type=Path, default=DEFAULT_HOUSEHOLDS_IN)
    parser.add_argument("--population-out", type=Path, default=DEFAULT_POPULATION_OUT)
    parser.add_argument("--households-out", type=Path, default=DEFAULT_HOUSEHOLDS_OUT)
    parser.add_argument("--delimiter", type=str, default=";")
    parser.add_argument("--chunk-size", type=int, default=300_000)
    parser.add_argument(
        "--compare-population",
        type=Path,
        default=DEFAULT_COMPARE_POPULATION,
        help="Reference population file for equality check.",
    )
    parser.add_argument(
        "--compare-households",
        type=Path,
        default=DEFAULT_COMPARE_HOUSEHOLDS,
        help="Reference households file for equality check.",
    )
    args = parser.parse_args()

    selected_ids = load_ids_from_trips(
        args.trips_file, args.delimiter, args.chunk_size, args.trips_id_column
    )

    total_persons, kept_persons, source_person_ids, written_person_ids = filter_population(
        args.population_in, args.population_out, selected_ids
    )
    expected_written_ids = source_person_ids.intersection(selected_ids)
    if written_person_ids != expected_written_ids:
        raise ValueError("Population verification failed: written person ID set mismatch.")
    if kept_persons != len(written_person_ids):
        raise ValueError("Population verification failed: duplicate/invalid person counting.")

    (
        src_hh,
        kept_hh,
        src_member_rows,
        kept_member_rows,
        src_member_ids,
        kept_member_ids,
    ) = filter_households(args.households_in, args.households_out, written_person_ids)
    expected_member_ids = src_member_ids.intersection(written_person_ids)
    if kept_member_ids != expected_member_ids:
        raise ValueError("Households verification failed: kept member ID set mismatch.")

    # Re-read outputs for strict post-write verification.
    out_pop_count, out_pop_ids = count_persons(args.population_out)
    out_hh_member_count, out_hh_member_ids = count_household_members(args.households_out)
    if out_pop_ids != written_person_ids or out_pop_count != len(written_person_ids):
        raise ValueError("Post-check failed: output population inconsistent.")
    if out_hh_member_ids != kept_member_ids:
        raise ValueError("Post-check failed: output households inconsistent.")

    print(f"Selected IDs from trips file ({args.trips_file.name}): {len(selected_ids):,}")
    print(f"Population source persons: {total_persons:,}")
    print(f"Population output persons: {kept_persons:,}")
    print(f"Households source households: {src_hh:,}")
    print(f"Households output households: {kept_hh:,}")
    print(f"Households source member rows: {src_member_rows:,}")
    print(f"Households output member rows: {kept_member_rows:,}")
    print(f"Population output file: {args.population_out}")
    print(f"Households output file: {args.households_out}")
    print("--- Internal verification ---")
    print(f"Population ID set match expected: {written_person_ids == expected_written_ids}")
    print(f"Household member ID set match expected: {kept_member_ids == expected_member_ids}")
    print(f"Post-check population IDs/count: {out_pop_ids == written_person_ids and out_pop_count == len(written_person_ids)}")
    print(f"Post-check household member IDs: {out_hh_member_ids == kept_member_ids}")

    # Compare with existing files generated earlier.
    if args.compare_population.exists() and args.compare_households.exists():
        cmp_pop_count, cmp_pop_ids = count_persons(args.compare_population)
        cmp_hh_member_count, cmp_hh_member_ids = count_household_members(args.compare_households)
        pop_same = out_pop_ids == cmp_pop_ids and out_pop_count == cmp_pop_count
        hh_same = out_hh_member_ids == cmp_hh_member_ids and out_hh_member_count == cmp_hh_member_count
        pop_hash_same = hash_file(args.population_out) == hash_file(args.compare_population)
        hh_hash_same = hash_file(args.households_out) == hash_file(args.compare_households)

        print("--- Comparison with existing files ---")
        print(f"Population ID-set/count equal: {pop_same}")
        print(f"Household member ID-set/count equal: {hh_same}")
        print(f"Population file byte-hash equal: {pop_hash_same}")
        print(f"Households file byte-hash equal: {hh_hash_same}")
    else:
        print("--- Comparison with existing files ---")
        print("Skipped (reference files not found).")


if __name__ == "__main__":
    main()
