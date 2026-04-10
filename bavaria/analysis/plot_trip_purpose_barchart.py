#!/usr/bin/env python3
"""
Compute trip purpose distribution (Zweck) in MiD style and plot a horizontal bar chart.

Categories: Wegekette ohne Arbeit, Wegekette mit Arbeit, Freizeit, Einkaufen,
Ausbildung, Arbeit, Anderer Zweck.

Mapping (following_purpose → MiD-Zweck):
  work      → Arbeit
  education → Ausbildung
  shop      → Einkaufen
  leisure   → Freizeit
  other     → Anderer Zweck (Sonstiges)
  home      → Wegekette mit/ohne Arbeit (je nachdem, ob die Person an dem Tag Arbeit hat)
  unknown   → Anderer Zweck

Usage:
  python plot_trip_purpose_barchart.py
  python plot_trip_purpose_barchart.py --path simulation_output/fleetpy_drt
  python plot_trip_purpose_barchart.py --path simulation_output/munich_2024_10pct simulation_output/fleetpy_drt
  python plot_trip_purpose_barchart.py --path simulation_output/munich_2024_10pct --chunk-size 200000

Requires: pandas, matplotlib (e.g. in your project venv or conda env).
"""

import gzip
import shutil
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# =============================================================================
# Pfad zum Simulationsoutput – hier anpassen oder per --path übergeben
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[5]  # eqasim-java-bavaria
SIMULATION_OUTPUT_DIR = REPO_ROOT / "simulation_output" / "munich_2024_10pct"
# Alternativ z.B.:
# SIMULATION_OUTPUT_DIR = REPO_ROOT / "simulation_output" / "fleetpy_drt"

# Reihenfolge und Labels wie im MiD-Balkendiagramm
MID_PURPOSE_ORDER = [
    "Wegekette ohne Arbeit",
    "Wegekette mit Arbeit",
    "Freizeit",
    "Einkaufen",
    "Ausbildung",
    "Arbeit",
    "Anderer Zweck",
]


def normalize_purpose(s: str) -> str:
    """Aktivitätstyp normalisieren (z.B. work_1 -> work)."""
    if pd.isna(s):
        return "other"
    s = str(s).strip().lower()
    # Suffix _1, _2 etc. entfernen (wie in eqasim TripWriter)
    while s and s[-1].isdigit():
        s = s[:-1]
    if s.endswith("_"):
        s = s[:-1]
    return s if s else "other"


def find_trips_file(output_dir: Path) -> Path:
    """eqasim_trips.csv oder eqasim_trips_drt.csv im Ordner finden (auch .gz)."""
    for base in ("eqasim_trips_drt", "eqasim_trips"):
        for ext in (".csv", ".csv.gz"):
            p = output_dir / f"{base}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(
        f"Weder eqasim_trips.csv noch eqasim_trips_drt.csv in {output_dir} gefunden."
    )


def detect_delimiter(csv_path: Path) -> str:
    """Delimiter aus erster Zeile erkennen (; oder ,)."""
    open_fn = gzip.open if csv_path.suffix == ".gz" else open
    mode = "rt" if csv_path.suffix == ".gz" else "r"
    with open_fn(csv_path, mode, encoding="utf-8", errors="replace") as f:
        header = f.readline()
    if ";" in header and "," not in header:
        return ";"
    if "," in header and ";" not in header:
        return ","
    return ";"


def _open_csv(path: Path) -> Path:
    """CSV aus .gz entpacken; gibt Pfad zur temporären CSV-Datei zurück."""
    import tempfile
    safe = path.name.replace(".", "_")
    tmp = Path(tempfile.gettempdir()) / f"eqasim_trips_{safe}.csv"
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f_in:
        with open(tmp, "w", encoding="utf-8") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return tmp


def load_trips(path: Path) -> pd.DataFrame:
    """Trips-CSV laden (Delimiter wird automatisch erkannt). Nur bei kleinen Dateien."""
    delimiter = detect_delimiter(path)
    to_remove = None
    read_path = path
    if path.suffix == ".gz":
        read_path = _open_csv(path)
        to_remove = read_path
    try:
        df = pd.read_csv(read_path, sep=delimiter, low_memory=False)
    finally:
        if to_remove is not None and to_remove.exists():
            try:
                to_remove.unlink()
            except OSError:
                pass
    for col in ("preceding_purpose", "following_purpose"):
        if col not in df.columns:
            raise ValueError(f"Spalte '{col}' fehlt in {path}. Vorhanden: {list(df.columns)}")
    return df


# Chunk-Größe für Speicher-effiziente Verarbeitung (Anpassung bei Bedarf oder per --chunk-size)
CHUNK_SIZE = 500_000


def _persons_with_work(path: Path, delimiter: str, read_path: Path, chunk_size: int) -> set:
    """Erster Durchlauf: Ermittle alle person_id, die mindestens eine Arbeitstätigkeit haben."""
    persons_with_work = set()
    for chunk in pd.read_csv(
        read_path, sep=delimiter, chunksize=chunk_size, low_memory=False,
        usecols=["person_id", "following_purpose"],
    ):
        chunk["_f"] = chunk["following_purpose"].map(normalize_purpose)
        persons_with_work |= set(chunk.loc[chunk["_f"] == "work", "person_id"].unique())
    return persons_with_work


def _count_mid_purposes_chunked(
    path: Path, delimiter: str, read_path: Path, persons_with_work: set, chunk_size: int,
) -> tuple[dict[str, int], int]:
    """Zweiter Durchlauf: Zähle Fahrten pro MiD-Zweck (chunkweise, vektorisiert)."""
    from collections import defaultdict
    counts = defaultdict(int)
    total = 0
    for chunk in pd.read_csv(
        read_path, sep=delimiter, chunksize=chunk_size, low_memory=False,
        usecols=["person_id", "following_purpose"],
    ):
        f = chunk["following_purpose"].map(normalize_purpose)
        has_work = chunk["person_id"].isin(persons_with_work)
        mid = _vectorized_mid_category(f, has_work)
        for cat, n in mid.value_counts().items():
            counts[cat] += n
        total += len(chunk)
    return dict(counts), total


def _vectorized_mid_category(purpose: pd.Series, has_work: pd.Series) -> pd.Series:
    """Vektorisierte Zuordnung normalisierter Zweck + has_work -> MiD-Kategorie."""
    out = pd.Series(index=purpose.index, data="Anderer Zweck", dtype=object)
    out.loc[purpose == "work"] = "Arbeit"
    out.loc[purpose.isin(("education", "ausbildung"))] = "Ausbildung"
    out.loc[purpose.isin(("shop", "shopping", "einkaufen"))] = "Einkaufen"
    out.loc[purpose == "leisure"] = "Freizeit"
    home_mask = purpose == "home"
    out.loc[home_mask & has_work] = "Wegekette mit Arbeit"
    out.loc[home_mask & ~has_work] = "Wegekette ohne Arbeit"
    return out


def compute_purpose_counts_chunked(path: Path, chunk_size: int = CHUNK_SIZE) -> tuple[pd.Series, int]:
    """
    Zweckverteilung aus großer CSV per Chunk-Reading (speicherarm).
    Gibt (Prozent-Series, Gesamtanzahl Fahrten) zurück.
    """
    delimiter = detect_delimiter(path)
    to_remove = None
    read_path = path
    if path.suffix == ".gz":
        read_path = _open_csv(path)
        to_remove = read_path
    try:
        persons_with_work = _persons_with_work(path, delimiter, read_path, chunk_size)
        counts, total = _count_mid_purposes_chunked(
            path, delimiter, read_path, persons_with_work, chunk_size
        )
    finally:
        if to_remove is not None and to_remove.exists():
            try:
                to_remove.unlink()
            except OSError:
                pass
    if total == 0:
        return pd.Series(index=MID_PURPOSE_ORDER, data=0.0), 0
    pct = pd.Series(index=MID_PURPOSE_ORDER, data=0.0)
    for cat, n in counts.items():
        if cat in pct.index:
            pct[cat] = n / total * 100.0
        else:
            pct["Anderer Zweck"] += n / total * 100.0
    return pct, total


def classify_trip_purpose_mid(df: pd.DataFrame) -> pd.Series:
    """
    Jede Fahrt in eine der 7 MiD-Zweckkategorien einteilen.

    - Arbeit, Ausbildung, Einkaufen, Freizeit: direkt aus following_purpose.
    - other / unbekannte Aktivitätstypen (z.B. outside, freight) → Anderer Zweck.
    - home (Rückweg): je nach Wegekette (ob die Person an dem Tag Arbeit hat)
      → Wegekette mit Arbeit bzw. Wegekette ohne Arbeit.
    """
    df = df.copy()
    df["_following"] = df["following_purpose"].map(normalize_purpose)

    # Pro Person: hat die Person an dem Tag mindestens eine Fahrt mit Zweck "work"?
    person_has_work = (
        df.groupby("person_id")["_following"]
        .transform(lambda x: (x == "work").any())
    )

    def map_to_mid(row):
        purpose = row["_following"]
        has_work = row["person_has_work"]

        if purpose == "work":
            return "Arbeit"
        if purpose in ("education", "ausbildung"):
            return "Ausbildung"
        if purpose in ("shop", "shopping", "einkaufen"):
            return "Einkaufen"
        if purpose == "leisure":
            return "Freizeit"
        if purpose == "other" or purpose not in ("home", "work", "education", "shop", "leisure"):
            return "Anderer Zweck"
        # nur home → Wegekette mit/ohne Arbeit
        if has_work:
            return "Wegekette mit Arbeit"
        return "Wegekette ohne Arbeit"

    df["person_has_work"] = person_has_work
    mid_purpose = df.apply(map_to_mid, axis=1)
    return mid_purpose


def compute_percentages(series: pd.Series) -> pd.Series:
    """Anteile in Prozent (Summe 100%)."""
    counts = series.value_counts()
    total = counts.sum()
    if total == 0:
        return pd.Series(index=MID_PURPOSE_ORDER, data=0.0)
    pct = counts / total * 100.0
    # Alle Kategorien in fester Reihenfolge, fehlende mit 0
    result = pd.Series(index=MID_PURPOSE_ORDER, data=0.0)
    for k, v in pct.items():
        if k in result.index:
            result[k] = v
        else:
            result["Anderer Zweck"] = result["Anderer Zweck"] + v
    return result


def plot_mid_style_barchart(
    pct: pd.Series,
    title: str,
    output_path: Path,
    total_trips: int | None = None,
) -> None:
    """Horizontales Balkendiagramm im MiD-Stil (graue Balken, Prozent am Ende)."""
    # Immer alle 7 MiD-Kategorien in fester Reihenfolge (auch bei 0 %)
    order = MID_PURPOSE_ORDER
    values = [pct.get(c, 0.0) for c in order]
    labels = order

    if total_trips is not None:
        title = f"{title}\n(n = {total_trips:,} Fahrten)"

    fig, ax = plt.subplots(figsize=(8, 4))
    # Erste Kategorie oben (wie im MiD-Diagramm)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, values, color="grey", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.15 if values and max(values) > 0 else 30)
    ax.set_xlabel("")
    ax.set_ylabel("Zweck", labelpad=8)
    ax.set_title(title)

    # Prozentwerte mit einer Dezimalstelle, damit Szenarien unterscheidbar sind
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(
            val + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagramm gespeichert: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Zweckverteilung (MiD-Stil) aus Simulationsoutput erstellen.")
    parser.add_argument(
        "--path",
        type=str,
        nargs="*",
        default=None,
        help="Ein oder mehrere Simulationsoutput-Ordner (z.B. simulation_output/fleetpy_drt). "
             "Mehrere nacheinander verarbeiten. Ohne Angabe: SIMULATION_OUTPUT_DIR.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Zeilen pro Chunk beim Einlesen (Standard: {CHUNK_SIZE:,}). Bei wenig RAM verkleinern.",
    )
    args = parser.parse_args()

    if args.path:
        output_dirs = [Path(p) for p in args.path]
    else:
        output_dirs = [Path(SIMULATION_OUTPUT_DIR)]

    script_dir = Path(__file__).resolve().parent
    script_dir.joinpath("analysis_data").mkdir(parents=True, exist_ok=True)

    for output_dir in output_dirs:
        if not output_dir.is_absolute():
            output_dir = REPO_ROOT / output_dir
        if not output_dir.exists():
            print(f"Überspringe (nicht gefunden): {output_dir}")
            continue

        trips_path = find_trips_file(output_dir)
        trips_path_abs = trips_path.resolve()
        print(f"\n[{output_dir.name}] Datei: {trips_path_abs.name}")
        print(f"  Pfad: {trips_path_abs}")
        print(f"  Chunk-Größe: {args.chunk_size:,}")
        pct, total = compute_purpose_counts_chunked(trips_path, chunk_size=args.chunk_size)
        print(f"Anzahl Fahrten: {total:,}")

        print("Zweckverteilung (in %):")
        for cat in MID_PURPOSE_ORDER:
            print(f"  {cat}: {pct.get(cat, 0):.1f}%")

        scenario_name = output_dir.name.replace("_", " ").title()
        title = f"{scenario_name} – Alle Fahrten"

        plot_path = script_dir / "analysis_data" / f"trip_purpose_{output_dir.name}.png"
        plot_mid_style_barchart(pct, title, plot_path, total_trips=total)

        csv_path = script_dir / "analysis_data" / f"trip_purpose_{output_dir.name}.csv"
        pct.to_csv(csv_path, header=["percent"])
        print(f"Gespeichert: {plot_path.name}, {csv_path.name}")


if __name__ == "__main__":
    main()
