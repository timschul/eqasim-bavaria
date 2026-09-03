"""Standalone-Test des Ae-112-Fixes in model.py gegen erfundene Daten.

Kein synpp, kein echter Lauf -- ruft execute() end-to-end mit einem Mock-Context und
kleinen, von Hand nachrechenbaren Zahlen auf. D_matsim-magdeburg Ae-112, sechstes
Code-Review, Entscheidung Logbuch 03.09.2026.

Braucht die eqasim-Laufumgebung (Python 3.10, pandas 1.5.3), NICHT das .venv von
D_matsim-magdeburg (dessen pandas 3.x macht `.values` auf einer DataFrame-Spalte
schreibgeschuetzt -- ValueError beim Raking, unabhaengig von diesem Fix):

    C:\\Users\\timschul\\AppData\\Local\\Programs\\Python310-eqasim\\python.exe bavaria/ipf/test_model.py

Was dieser Test NICHT zeigt: eine toy-grosse, von Hand gebaute Bevoelkerung mit nur
einer Gemeinde ist so ueberbestimmt (mehr Nebenbedingungen als freie Zellen), dass
sich das reale Ausmass der 18-19-Verdopplung darin nicht sauber nachstellen laesst --
mehrere Versuche mit verschaerften Zielwerten zeigten keinen belastbaren Vorher-
Nachher-Unterschied. Was er zeigt: der Fix laeuft durch, konvergiert, trifft die
True-Seiten-Zielwerte weiterhin exakt, und die harte Mindestalter-Schranke haelt. Die
eigentliche Kreuzprobe -- verschwindet die 18-19-Verdopplung an den echten Daten --
ist Sache von N-03 (Populationsneuzug), nicht dieses Tests.
"""
import pandas as pd

try:
    from model import execute  # gestartet mit cwd = bavaria/ipf
except ModuleNotFoundError:
    from bavaria.ipf.model import execute  # eqasim-Laufumgebung: nur der Repo-Root ist auf sys.path (_pth)


class FortschrittMock:
    def __call__(self, iterable, total=None, label=None, minimum_interval=1.0):
        return iterable


class ContextMock:
    def __init__(self, daten, config):
        self._daten = daten
        self._config = config
        self.progress = FortschrittMock()

    def stage(self, name):
        return self._daten[name]

    def config(self, name, default=None):
        return self._config.get(name, default)


def bau_szenario():
    """Eine Kommune C1 in einem Departement D1, zwei Geschlechter. Bevoelkerungs-
    klassen [0,15,20) - deutlich groeber als die Erwerbstaetigen- ([0,16)) und
    Fuehrerschein-Klassen ([0,18)): die 15-19-Jaehrigen-Klasse ueberspannt beide
    Grenzen, genau das Muster, das Ae-112 in echt gefunden hat (Lizenzklasse 18-20
    ueberspannt die Bevoelkerungsklassengrenze 18/21)."""
    df_population = pd.DataFrame([
        # commune_index, departement_index, sex, age_class, weight
        ("C1", "D1", 1, 0, 50.0),
        ("C1", "D1", 1, 15, 30.0),
        ("C1", "D1", 1, 20, 100.0),
        ("C1", "D1", 2, 0, 50.0),
        ("C1", "D1", 2, 15, 30.0),
        ("C1", "D1", 2, 20, 100.0),
    ], columns=["commune_index", "departement_index", "sex", "age_class", "weight"])
    # execute() merged die Bezeichner (commune_id/departement_id) am Ende zurueck
    # an - im echten Datensatz eigene Spalten, hier gleich dem *_index.
    df_population["commune_id"] = df_population["commune_index"]
    df_population["departement_id"] = df_population["departement_index"]

    df_employment = pd.DataFrame([
        # departement_index, sex, age_class, weight
        ("D1", 1, 16, 60.0),
        ("D1", 2, 16, 50.0),
    ], columns=["departement_index", "sex", "age_class", "weight"])

    df_licenses_country = pd.DataFrame([
        # sex, age_class, weight
        (1, 18, 20.0),
        (2, 18, 15.0),
    ], columns=["sex", "age_class", "weight"])

    df_licenses_kreis = pd.DataFrame([
        # departement_index, weight
        ("D1", 35.0),
    ], columns=["departement_index", "weight"])

    return df_population, df_employment, df_licenses_country, df_licenses_kreis


def test_konvergiert_und_haelt_die_zielwerte():
    context = ContextMock(
        daten={"bavaria.ipf.prepare": bau_szenario()},
        config={"bavaria.minimum_age.employment": 16},
    )
    df_model = execute(context)

    gesamtgewicht = df_model["weight"].sum()
    erwartetes_gesamt = 360.0  # 6 Zeilen der Bevoelkerung aufsummiert
    assert abs(gesamtgewicht - erwartetes_gesamt) < 1.0, \
        f"Gesamtgewicht {gesamtgewicht} weicht von {erwartetes_gesamt} ab"

    erwerbstaetig_D1 = df_model[
        (df_model["departement_id"] == "D1") & (df_model["employed"])
    ]["weight"].sum()
    assert abs(erwerbstaetig_D1 - 110.0) < 1.0, \
        f"Erwerbstaetige D1 (Ziel 60+50=110) erreicht: {erwerbstaetig_D1}"

    fuehrerschein_D1 = df_model[
        (df_model["departement_id"] == "D1") & (df_model["license"])
    ]["weight"].sum()
    assert abs(fuehrerschein_D1 - 35.0) < 1.0, \
        f"Fuehrerscheininhaber D1 (Ziel 35, Kreisebene) erreicht: {fuehrerschein_D1}"

    unter_16_erwerbstaetig = df_model[
        (df_model["age_class"] < 16) & (df_model["employed"])
    ]["weight"].sum()
    assert unter_16_erwerbstaetig < 1.0, \
        f"unter 16 duerfen nicht erwerbstaetig sein, ist aber {unter_16_erwerbstaetig}"

    print("test_konvergiert_und_haelt_die_zielwerte: OK "
          f"(gesamt={gesamtgewicht:.1f}, erwerbstaetig_D1={erwerbstaetig_D1:.1f}, "
          f"fuehrerschein_D1={fuehrerschein_D1:.1f})")


def test_gesamtgewicht_je_combined_age_class_bleibt_erhalten():
    """Kein Regressionstest fuer die Groessenordnung der echten 18-19-Verdopplung
    (siehe Modul-Docstring, warum ein Toy-Szenario dafuer nicht taugt) -- nur eine
    Erhaltungspruefung: die drei combined_age_class-Unterklassen, in die die
    Bevoelkerungszeile Alter 15 (Gewicht 60, beide Geschlechter) zerlegt wird,
    muessen in Summe wieder 60 ergeben. Wuerde diese Summe abweichen, waere das ein
    Fehler in der Aufteilungslogik selbst (`combined_by_population_age`/
    `df_population_expanded`), unabhaengig vom eigentlichen Ae-112-Befund."""
    context = ContextMock(
        daten={"bavaria.ipf.prepare": bau_szenario()},
        config={"bavaria.minimum_age.employment": 16},
    )
    df_model = execute(context)

    je_altersklasse = df_model.groupby("age_class")["weight"].sum()
    summe_15_16_18 = je_altersklasse[15] + je_altersklasse[16] + je_altersklasse[18]
    assert abs(summe_15_16_18 - 60.0) < 1.0, \
        f"Summe der drei Unterklassen von Bevoelkerungszeile 15 (Ziel 60): {summe_15_16_18}"

    print("test_gesamtgewicht_je_combined_age_class_bleibt_erhalten: OK "
          f"(15={je_altersklasse[15]:.1f}, 16={je_altersklasse[16]:.1f}, "
          f"18={je_altersklasse[18]:.1f}, Summe={summe_15_16_18:.1f})")


if __name__ == "__main__":
    test_konvergiert_und_haelt_die_zielwerte()
    test_gesamtgewicht_je_combined_age_class_bleibt_erhalten()
    print("Alle Tests OK")
