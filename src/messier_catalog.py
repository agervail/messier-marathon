"""
Messier catalogue — 110 Messier objects fetched from SIMBAD via
astroquery, cached locally after first download.

Public API
----------
MESSIER_OBJECTS : list of (number: int, ra_deg: float, dec_deg: float,
                            otype: str, name: str)
  otype codes: GC, OC, Gx, EN, PN, SNR, SC, DS
"""

import os

_CACHE = os.path.join(os.path.dirname(__file__), "messier_cache.csv")

# Simbad otype → our type codes
_OTYPE_MAP = {
    # Globular clusters
    "GlC": "GC",
    # Open clusters / stellar associations
    "OpC": "OC", "As*": "OC", "Cl*": "OC",
    # Galaxies (all subtypes)
    "G":   "Gx", "AGN": "Gx", "Sy1": "Gx", "Sy2": "Gx",
    "EmG": "Gx", "GiG": "Gx", "GiP": "Gx", "IG":  "Gx",
    "PaG": "Gx", "SBG": "Gx", "LSB": "Gx", "BClG": "Gx",
    "LIN": "Gx", "H2G": "Gx",
    # Emission / reflection nebulae
    "HII": "EN", "RNe": "EN", "MoC": "EN", "Cld": "EN",
    "SFR": "EN", "bub": "EN",
    # Interstellar medium features (M24 star cloud)
    "ISM": "SC",
    # Planetary nebulae
    "PN":  "PN",
    # Supernova remnants
    "SNR": "SNR",
    # Double / multiple stars (M40)
    "**":  "DS", "SB*": "DS",
}

# Well-known common names, keyed by Messier number
_COMMON_NAMES = {
    1:   "Crab Nebula",
    6:   "Papillon",
    7:   "Ptolemy Cluster",
    8:   "Lagoon Nebula",
    11:  "Wild Duck",
    13:  "Hercules Cluster",
    16:  "Eagle Nebula",
    17:  "Omega Nebula",
    20:  "Trifid Nebula",
    24:  "Milky Way Patch",
    27:  "Dumbbell Nebula",
    31:  "Andromeda Galaxy",
    33:  "Triangulum Galaxy",
    40:  "Winnecke 4",
    42:  "Orion Nebula",
    43:  "De Mairan's Nebula",
    44:  "Beehive",
    45:  "Pleiades",
    51:  "Whirlpool Galaxy",
    57:  "Ring Nebula",
    63:  "Sunflower Galaxy",
    64:  "Black Eye Galaxy",
    74:  "Phantom Galaxy",
    76:  "Little Dumbbell",
    81:  "Bode's Galaxy",
    82:  "Cigar Galaxy",
    83:  "Southern Pinwheel",
    87:  "Virgo A",
    97:  "Owl Nebula",
    101: "Pinwheel Galaxy",
    102: "Spindle Galaxy",
    104: "Sombrero Galaxy",
}


def _fetch_from_simbad():
    """Query SIMBAD for M 1 … M 110 and return the result table."""
    from astroquery.simbad import Simbad

    print("Téléchargement du catalogue Messier depuis SIMBAD…")
    s = Simbad()
    s.add_votable_fields("otype")
    identifiers = [f"M {i}" for i in range(1, 111)]
    result = s.query_objects(identifiers)
    return result


def _table_to_list(table):
    """Convert the SIMBAD result table to [(num, ra_deg, dec_deg, otype, name), ...]."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    cols = table.colnames

    # Simbad column names vary across astroquery versions — detect dynamically
    ra_col = next(
        (c for c in cols if c.upper() in ("RA", "RA_D", "RAJ2000") or c.lower() == "ra"),
        None,
    )
    dec_col = next(
        (c for c in cols if c.upper() in ("DEC", "DE", "DEJ2000") or c.lower() in ("dec", "de")),
        None,
    )
    otype_col = next(
        (c for c in cols if "OTYPE" in c.upper() or c.lower() in ("otype", "main_type")),
        None,
    )
    if ra_col is None or dec_col is None or otype_col is None:
        raise RuntimeError(
            f"Colonnes SIMBAD introuvables parmi {cols}. "
            f"ra={ra_col}, dec={dec_col}, otype={otype_col}"
        )

    objects = []
    for i, row in enumerate(table):
        num = i + 1
        ra_val  = str(row[ra_col]).strip()
        dec_val = str(row[dec_col]).strip()

        # Coordinates may be decimal degrees or sexagesimal depending on version
        try:
            coord = SkyCoord(ra=float(ra_val) * u.deg,
                             dec=float(dec_val) * u.deg, frame="icrs")
        except ValueError:
            coord = SkyCoord(ra=ra_val, dec=dec_val,
                             unit=(u.hourangle, u.deg), frame="icrs")

        otype_raw = str(row[otype_col]).strip()
        otype = _OTYPE_MAP.get(otype_raw, "Gx")   # unknown subtypes → galaxy

        name = _COMMON_NAMES.get(num, "")
        objects.append((num, float(coord.ra.deg), float(coord.dec.deg), otype, name))

    return objects


def _save_cache(objects):
    with open(_CACHE, "w", encoding="utf-8") as f:
        for num, ra, dec, otype, name in objects:
            safe_name = name.replace(",", " ")
            f.write(f"{num},{ra:.6f},{dec:.6f},{otype},{safe_name}\n")
    print(f"Cache sauvegardé : {_CACHE}  ({len(objects)} objets)")


def _load_cache():
    objects = []
    with open(_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 4)
            num  = int(parts[0])
            ra   = float(parts[1])
            dec  = float(parts[2])
            otype = parts[3]
            name  = parts[4] if len(parts) == 5 else ""
            objects.append((num, ra, dec, otype, name))
    return objects


def load_messier():
    if os.path.exists(_CACHE):
        return _load_cache()
    table   = _fetch_from_simbad()
    objects = _table_to_list(table)
    _save_cache(objects)
    return objects


MESSIER_OBJECTS = load_messier()
