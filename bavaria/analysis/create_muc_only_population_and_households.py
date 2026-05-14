#!/usr/bin/env python3
"""
Create filtered population and households directly from trips (CSV or GeoPackage).

#Call it like (e.g. for Munich 2040 100%): python create_muc_only_population_and_households.py \
  --selection-mode any_endpoint_in_munich \
  --trips-file analysis_data/munich_2040_100pct/munich_trips.gpkg \
  --population-in analysis_data/munich_2040_100pct/munich_population.xml.gz \
  --households-in analysis_data/munich_2040_100pct/munich_households.xml.gz \
  --munich-boundary analysis_data/munich_2040_100pct/munich_city_boundary.geojson \
  --population-out analysis_data/munich_2040_100pct/munich_population_agents_in_muc.xml.gz \
  --households-out analysis_data/munich_2040_100pct/munich_households_agents_in_muc.xml.gz

Inputs:
1) Trips file:
   - CSV with an agent/person id column (default: person_id), and for spatial
     mode also origin/destination coordinate columns; OR
   - GeoPackage (.gpkg) with a trips layer: person id column + LineString
     geometry (first and last vertices = trip endpoints).
2) Population file (.xml.gz) with <person id="..."> blocks.
3) Households file (.xml.gz) with <personId refId="..."/> members.

Rule:
- Determine selected agents either by:
  1) taking all unique IDs present in the trips file, OR
  2) spatially evaluating trips and selecting agents that have at least one
     trip with start OR end point inside the Munich boundary.
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
import sqlite3
import struct
from pathlib import Path

import geopandas as gpd
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
DEFAULT_MUNICH_BOUNDARY = BASE / "munich_city_boundary.geojson"

PERSON_RE = re.compile(r'<person\s+id="([^"]+)"')
MEMBER_RE = re.compile(r'<personId\s+refId="([^"]+)"\s*/>')
HOUSEHOLD_OPEN_RE = re.compile(r"<household\b")
HOUSEHOLD_CLOSE_RE = re.compile(r"</household>")


class UnsupportedGeoPackageGeometry(ValueError):
    """Raised for GeoPackage geometry encodings this script cannot parse (e.g. extended binary)."""


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


def load_ids_from_spatial_rule(
    trips_file: Path,
    boundary_file: Path,
    delimiter: str,
    chunk_size: int,
    id_column: str,
    origin_x_col: str,
    origin_y_col: str,
    destination_x_col: str,
    destination_y_col: str,
    csv_crs: str,
) -> set[str]:
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips file not found: {trips_file}")
    if not boundary_file.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_file}")

    boundary = gpd.read_file(boundary_file)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {boundary_file}")
    if boundary.crs is None:
        raise ValueError("Boundary CRS is missing.")
    munich_geom = gpd.GeoSeries([boundary.geometry.union_all()], crs=boundary.crs).to_crs(csv_crs).iloc[0]

    selected_ids: set[str] = set()
    rows = 0
    required = {id_column, origin_x_col, origin_y_col, destination_x_col, destination_y_col}
    for chunk in pd.read_csv(trips_file, sep=delimiter, chunksize=chunk_size, low_memory=False):
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"Trips file is missing required columns: {sorted(missing)}")

        rows += len(chunk)
        pid = pd.to_numeric(chunk[id_column], errors="coerce")
        ox = pd.to_numeric(chunk[origin_x_col], errors="coerce")
        oy = pd.to_numeric(chunk[origin_y_col], errors="coerce")
        dx = pd.to_numeric(chunk[destination_x_col], errors="coerce")
        dy = pd.to_numeric(chunk[destination_y_col], errors="coerce")
        valid = pid.notna() & ox.notna() & oy.notna() & dx.notna() & dy.notna()
        if not valid.any():
            continue

        valid_ids = pid[valid].astype("int64").reset_index(drop=True)
        valid_ox = ox[valid].reset_index(drop=True)
        valid_oy = oy[valid].reset_index(drop=True)
        valid_dx = dx[valid].reset_index(drop=True)
        valid_dy = dy[valid].reset_index(drop=True)
        origins = gpd.GeoSeries(gpd.points_from_xy(valid_ox, valid_oy), crs=csv_crs)
        destinations = gpd.GeoSeries(gpd.points_from_xy(valid_dx, valid_dy), crs=csv_crs)
        in_munich = (origins.within(munich_geom) | origins.touches(munich_geom)) | (
            destinations.within(munich_geom) | destinations.touches(munich_geom)
        )
        if in_munich.any():
            selected_ids.update(valid_ids[in_munich].astype(str).tolist())

    if rows == 0:
        raise ValueError("Trips file has no rows.")
    if not selected_ids:
        raise ValueError("No selected IDs found with the spatial rule.")
    return selected_ids


_GPKG_ENVELOPE_BYTES = (0, 32, 48, 48, 64)


def _gpkg_validate_ident(name: str, kind: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid GeoPackage {kind} name {name!r} (use letters, digits, underscore only).")


def _gpkg_blob_inner_wkb(blob: bytes | None) -> bytes:
    if blob is None or len(blob) == 0:
        raise ValueError("Empty geometry BLOB")
    if len(blob) >= 8 and blob[0:2] == b"GP":
        flags = blob[3]
        if (flags >> 5) & 1:
            raise UnsupportedGeoPackageGeometry(
                "GeoPackage geometry uses ExtendedGeoPackageBinary; this script only supports "
                "StandardGeoPackageBinary."
            )
        if (flags >> 4) & 1:
            raise ValueError("Empty GeoPackage geometry")
        env_type = (flags >> 1) & 0x07
        if env_type >= len(_GPKG_ENVELOPE_BYTES):
            raise ValueError(f"Invalid GeoPackage envelope indicator: {env_type}")
        return blob[8 + _GPKG_ENVELOPE_BYTES[env_type] :]
    return blob


def _wkb_linestring_endpoints_xy(wkb: bytes) -> tuple[float, float, float, float]:
    if len(wkb) < 9:
        raise ValueError("WKB buffer too short")
    bo = "<" if wkb[0] == 1 else ">"
    if wkb[0] not in (0, 1):
        raise ValueError(f"Invalid WKB endian byte: {wkb[0]}")
    geom_type = struct.unpack(bo + "I", wkb[1:5])[0]
    has_srid = bool(geom_type & 0x20000000)
    z_dim = bool(geom_type & 0x80000000)
    m_dim = bool(geom_type & 0x40000000)
    base = geom_type & 0xFF
    if base != 2:
        raise ValueError(
            f"Expected a LineString geometry (WKB type 2); found type {base}. "
            "Multi-part or other types are not supported."
        )
    coord_bytes = 16 + (8 if z_dim else 0) + (8 if m_dim else 0)
    off = 5
    if has_srid:
        if len(wkb) < off + 4:
            raise ValueError("Truncated WKB (SRID)")
        off += 4
    if len(wkb) < off + 4:
        raise ValueError("Truncated WKB (point count)")
    n_pts = struct.unpack(bo + "I", wkb[off : off + 4])[0]
    off += 4
    if n_pts < 2:
        raise ValueError("LineString must have at least two points")
    if len(wkb) < off + n_pts * coord_bytes:
        raise ValueError("Truncated WKB for LineString")
    x0 = struct.unpack(bo + "d", wkb[off : off + 8])[0]
    y0 = struct.unpack(bo + "d", wkb[off + 8 : off + 16])[0]
    last = off + (n_pts - 1) * coord_bytes
    x1 = struct.unpack(bo + "d", wkb[last : last + 8])[0]
    y1 = struct.unpack(bo + "d", wkb[last + 8 : last + 16])[0]
    return x0, y0, x1, y1


def _gpkg_linestring_endpoints(blob: bytes | None) -> tuple[float, float, float, float]:
    return _wkb_linestring_endpoints_xy(_gpkg_blob_inner_wkb(blob))


def _resolve_gpkg_feature_table(con: sqlite3.Connection, layer: str | None) -> str:
    cur = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE lower(data_type) = 'features'"
    )
    tables = [r[0] for r in cur.fetchall()]
    if layer:
        _gpkg_validate_ident(layer, "layer")
        if layer not in tables:
            raise ValueError(
                f"Layer {layer!r} not found in GeoPackage. Available feature tables: {sorted(tables)}"
            )
        return layer
    if not tables:
        raise ValueError("GeoPackage contains no vector feature tables (gpkg_contents).")
    if len(tables) > 1:
        raise ValueError(
            "GeoPackage has multiple feature tables; set --trips-gpkg-layer. "
            f"Available: {sorted(tables)}"
        )
    _gpkg_validate_ident(tables[0], "layer")
    return tables[0]


def _gpkg_geometry_meta(con: sqlite3.Connection, table: str) -> tuple[str, int | None]:
    row = con.execute(
        "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?",
        (table,),
    ).fetchone()
    if row is None:
        raise ValueError(
            f"No row in gpkg_geometry_columns for table {table!r}; cannot find geometry column."
        )
    col_name, srs_id = row[0], row[1]
    _gpkg_validate_ident(col_name, "geometry column")
    sid: int | None
    if srs_id is None:
        sid = None
    else:
        sid = int(srs_id)
    return col_name, sid


def _trips_geom_crs(srs_id: int | None, crs_fallback: str) -> str:
    if srs_id is None or srs_id <= 0:
        return crs_fallback
    return f"EPSG:{srs_id}"


def load_ids_from_trips_gpkg(
    trips_file: Path, chunk_size: int, id_column: str, layer: str | None
) -> set[str]:
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips GeoPackage not found: {trips_file}")
    _gpkg_validate_ident(id_column, "person id column")

    selected_ids: set[str] = set()
    rows = 0
    con = sqlite3.connect(f"file:{trips_file.as_posix()}?mode=ro", uri=True)
    try:
        table = _resolve_gpkg_feature_table(con, layer)
        info_cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        if id_column not in info_cols:
            raise ValueError(
                f"Trips table {table!r} has no column {id_column!r}. Columns: {sorted(info_cols)}"
            )

        last_rowid = 0
        while True:
            batch = con.execute(
                f'SELECT rowid, "{id_column}" FROM "{table}" '
                f"WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, chunk_size),
            ).fetchall()
            if not batch:
                break
            last_rowid = int(batch[-1][0])
            rows += len(batch)
            for _rid, pid in batch:
                if pid is None:
                    continue
                try:
                    ip = int(pid)
                except (TypeError, ValueError):
                    continue
                selected_ids.add(str(ip))
    finally:
        con.close()

    if rows == 0:
        raise ValueError("Trips GeoPackage has no rows.")
    if not selected_ids:
        raise ValueError("No valid person id values found in trips GeoPackage.")
    return selected_ids


def load_ids_from_spatial_rule_gpkg(
    trips_file: Path,
    boundary_file: Path,
    chunk_size: int,
    id_column: str,
    layer: str | None,
    crs_fallback: str,
) -> set[str]:
    if not trips_file.exists():
        raise FileNotFoundError(f"Trips GeoPackage not found: {trips_file}")
    if not boundary_file.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_file}")
    _gpkg_validate_ident(id_column, "person id column")

    boundary = gpd.read_file(boundary_file)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {boundary_file}")
    if boundary.crs is None:
        raise ValueError("Boundary CRS is missing.")

    con = sqlite3.connect(f"file:{trips_file.as_posix()}?mode=ro", uri=True)
    try:
        table = _resolve_gpkg_feature_table(con, layer)
        geom_col, srs_id = _gpkg_geometry_meta(con, table)
        trips_crs = _trips_geom_crs(srs_id, crs_fallback)
        info_cols = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        if id_column not in info_cols:
            raise ValueError(
                f"Trips table {table!r} has no column {id_column!r}. Columns: {sorted(info_cols)}"
            )
        if geom_col not in info_cols:
            raise ValueError(
                f"Geometry column {geom_col!r} from gpkg_geometry_columns not in table {table!r}."
            )

        munich_geom = (
            gpd.GeoSeries([boundary.geometry.union_all()], crs=boundary.crs).to_crs(trips_crs).iloc[0]
        )

        selected_ids: set[str] = set()
        rows = 0
        last_rowid = 0
        skipped_trip_rows = 0
        while True:
            batch = con.execute(
                f'SELECT rowid, "{id_column}", CAST("{geom_col}" AS BLOB) FROM "{table}" '
                f"WHERE rowid > ? ORDER BY rowid LIMIT ?",
                (last_rowid, chunk_size),
            ).fetchall()
            if not batch:
                break
            last_rowid = int(batch[-1][0])
            rows += len(batch)

            pids: list[int] = []
            ox: list[float] = []
            oy: list[float] = []
            dx: list[float] = []
            dy: list[float] = []
            for _rid, pid, blob in batch:
                if pid is None:
                    skipped_trip_rows += 1
                    continue
                try:
                    ip = int(pid)
                except (TypeError, ValueError):
                    skipped_trip_rows += 1
                    continue
                try:
                    x0, y0, x1, y1 = _gpkg_linestring_endpoints(blob)
                except UnsupportedGeoPackageGeometry:
                    raise
                except (ValueError, struct.error):
                    skipped_trip_rows += 1
                    continue
                pids.append(ip)
                ox.append(x0)
                oy.append(y0)
                dx.append(x1)
                dy.append(y1)

            if not pids:
                continue

            pid_series = pd.Series(pids, dtype="int64")
            origins = gpd.GeoSeries(gpd.points_from_xy(ox, oy), crs=trips_crs)
            destinations = gpd.GeoSeries(gpd.points_from_xy(dx, dy), crs=trips_crs)
            in_munich = (origins.within(munich_geom) | origins.touches(munich_geom)) | (
                destinations.within(munich_geom) | destinations.touches(munich_geom)
            )
            if in_munich.any():
                selected_ids.update(pid_series[in_munich].astype(str).tolist())
    finally:
        con.close()

    if rows == 0:
        raise ValueError("Trips GeoPackage has no rows.")
    if skipped_trip_rows:
        print(
            f"GeoPackage note: skipped {skipped_trip_rows:,} trip rows "
            "(missing person id, empty/null geometry, or unreadable LineString WKB)."
        )
    if not selected_ids:
        raise ValueError("No selected IDs found with the spatial rule (GeoPackage).")
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
            "Create filtered population and households from trips (CSV or GeoPackage .gpkg), "
            "without an intermediate trips-filtering step."
        )
    )
    parser.add_argument(
        "--trips-file",
        type=Path,
        default=DEFAULT_TRIPS,
        help="Input trips file: CSV (default munich_trips.csv) or GeoPackage .gpkg with trip LineStrings.",
    )
    parser.add_argument(
        "--trips-gpkg-layer",
        type=str,
        default=None,
        help=(
            "GeoPackage feature table name when the file has multiple layers "
            "(default: the only feature table, or error if ambiguous)."
        ),
    )
    parser.add_argument(
        "--trips-id-column",
        type=str,
        default="person_id",
        help="Column name in trips CSV/GeoPackage that identifies agents/persons.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        choices=["ids_from_trips", "any_endpoint_in_munich"],
        default="ids_from_trips",
        help=(
            "How to select agents: "
            "'ids_from_trips' uses all IDs present in trips file; "
            "'any_endpoint_in_munich' applies start/end-in-Munich spatial rule "
            "(CSV: coordinate columns; GeoPackage: first/last vertex of each LineString)."
        ),
    )
    parser.add_argument(
        "--munich-boundary",
        type=Path,
        default=DEFAULT_MUNICH_BOUNDARY,
        help="Boundary file used by any_endpoint_in_munich mode.",
    )
    parser.add_argument("--origin-x-col", type=str, default="origin_x")
    parser.add_argument("--origin-y-col", type=str, default="origin_y")
    parser.add_argument("--destination-x-col", type=str, default="destination_x")
    parser.add_argument("--destination-y-col", type=str, default="destination_y")
    parser.add_argument(
        "--csv-crs",
        type=str,
        default="EPSG:25832",
        help=(
            "CRS of trip coordinates in CSV mode. For .gpkg, the file's gpkg_geometry_columns "
            "srs_id is used when present; this value is the fallback if srs_id is missing or <= 0."
        ),
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
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")

    use_gpkg = args.trips_file.suffix.lower() == ".gpkg"
    if args.selection_mode == "ids_from_trips":
        if use_gpkg:
            selected_ids = load_ids_from_trips_gpkg(
                args.trips_file, args.chunk_size, args.trips_id_column, args.trips_gpkg_layer
            )
        else:
            selected_ids = load_ids_from_trips(
                args.trips_file, args.delimiter, args.chunk_size, args.trips_id_column
            )
    elif use_gpkg:
        selected_ids = load_ids_from_spatial_rule_gpkg(
            trips_file=args.trips_file,
            boundary_file=args.munich_boundary,
            chunk_size=args.chunk_size,
            id_column=args.trips_id_column,
            layer=args.trips_gpkg_layer,
            crs_fallback=args.csv_crs,
        )
    else:
        selected_ids = load_ids_from_spatial_rule(
            trips_file=args.trips_file,
            boundary_file=args.munich_boundary,
            delimiter=args.delimiter,
            chunk_size=args.chunk_size,
            id_column=args.trips_id_column,
            origin_x_col=args.origin_x_col,
            origin_y_col=args.origin_y_col,
            destination_x_col=args.destination_x_col,
            destination_y_col=args.destination_y_col,
            csv_crs=args.csv_crs,
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

    print(f"Selection mode: {args.selection_mode}")
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
