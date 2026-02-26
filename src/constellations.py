"""
Constellation boundaries — fetched from Vizier VI/49
(Davenhall & Leggett 1989, IAU boundaries based on Delporte 1930).
Coordinates are in B1875 FK4 in the catalog; converted to ICRS J2000.
Cached locally after first download.

Public API
----------
get_boundaries() → (segments, labels)
  segments : list of (ra_deg, dec_deg) numpy arrays  [ICRS J2000]
  labels   : list of (ra_deg, dec_deg, abbr)          [ICRS J2000]

FRENCH_NAMES : dict  abbr → French name
"""

import os
import numpy as np

_CACHE = os.path.join(os.path.dirname(__file__), "constellations_cache.csv")

# ── French names for all 88 IAU constellations ──────────────────────────────
FRENCH_NAMES = {
    "And": "Andromède",      "Ant": "Machine Pneumatique", "Aps": "Oiseau de Paradis",
    "Aqr": "Verseau",        "Aql": "Aigle",               "Ara": "Autel",
    "Ari": "Bélier",         "Aur": "Cocher",              "Boo": "Bouvier",
    "Cae": "Burin",          "Cam": "Girafe",              "Cnc": "Cancer",
    "CVn": "Chiens de Chasse","CMa": "Grand Chien",        "CMi": "Petit Chien",
    "Cap": "Capricorne",     "Car": "Carène",              "Cas": "Cassiopée",
    "Cen": "Centaure",       "Cep": "Céphée",              "Cet": "Baleine",
    "Cha": "Caméléon",       "Cir": "Compas",              "Col": "Colombe",
    "Com": "Chevelure de Bérénice", "CrA": "Couronne Australe", "CrB": "Couronne Boréale",
    "Crv": "Corbeau",        "Crt": "Coupe",               "Cru": "Croix du Sud",
    "Cyg": "Cygne",          "Del": "Dauphin",             "Dor": "Dorade",
    "Dra": "Dragon",         "Equ": "Petit Cheval",        "Eri": "Éridan",
    "For": "Fourneau",       "Gem": "Gémeaux",             "Gru": "Grue",
    "Her": "Hercule",        "Hor": "Horloge",             "Hya": "Hydre",
    "Hyi": "Hydre Mâle",     "Ind": "Indien",              "Lac": "Lézard",
    "Leo": "Lion",           "LMi": "Petit Lion",          "Lep": "Lièvre",
    "Lib": "Balance",        "Lup": "Loup",                "Lyn": "Lynx",
    "Lyr": "Lyre",           "Men": "Table",               "Mic": "Microscope",
    "Mon": "Licorne",        "Mus": "Mouche",              "Nor": "Règle",
    "Oct": "Octant",         "Oph": "Ophiuchus",           "Ori": "Orion",
    "Pav": "Paon",           "Peg": "Pégase",              "Per": "Persée",
    "Phe": "Phénix",         "Pic": "Peintre",             "PsA": "Poisson Austral",
    "Psc": "Poissons",       "Pup": "Poupe",               "Pyx": "Boussole",
    "Ret": "Réticule",       "Sge": "Flèche",              "Sgr": "Sagittaire",
    "Sco": "Scorpion",       "Scl": "Sculpteur",           "Sct": "Écu de Sobieski",
    "Ser": "Serpent",        "Sex": "Sextant",             "Tau": "Taureau",
    "Tel": "Télescope",      "TrA": "Triangle Austral",    "Tri": "Triangle",
    "Tuc": "Toucan",         "UMa": "Grande Ourse",        "UMi": "Petite Ourse",
    "Vel": "Voiles",         "Vir": "Vierge",              "Vol": "Poisson Volant",
    "Vul": "Petit Renard",
}


def _fetch_from_vizier():
    """Download constellation boundary vertices from Vizier VI/49."""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord, FK4
    import astropy.units as u

    print("Téléchargement des frontières de constellations (Vizier VI/49)…")
    v = Vizier(columns=["*"], row_limit=-1)
    tables = v.get_catalogs("VI/49")
    cat = tables[0]

    cols = cat.colnames
    # Detect column names (RA, Dec, Constellation abbreviation)
    ra_col  = next(c for c in cols if "RA"  in c.upper())
    dec_col = next(c for c in cols if c.upper().startswith("DE") and c != ra_col)
    cst_col = next(c for c in cols if c.lower() in ("cst", "const", "abb", "name", "con"))

    ra_raw  = np.array([float(r) for r in cat[ra_col]])
    dec_raw = np.array([float(d) for d in cat[dec_col]])
    csts    = [str(c).strip() for c in cat[cst_col]]

    # RA in VI/49 is in decimal hours (B1875) → convert to degrees
    ra_deg_b1875 = ra_raw * 15 if ra_raw.max() < 25 else ra_raw

    sc_fk4  = SkyCoord(ra=ra_deg_b1875 * u.deg, dec=dec_raw * u.deg,
                       frame=FK4(equinox="B1875"))
    sc_icrs = sc_fk4.icrs

    return csts, sc_icrs.ra.deg, sc_icrs.dec.deg


def _save_cache(csts, ra_icrs, dec_icrs):
    with open(_CACHE, "w", encoding="utf-8") as f:
        for c, r, d in zip(csts, ra_icrs, dec_icrs):
            f.write(f"{c},{r:.6f},{d:.6f}\n")
    print(f"Cache sauvegardé : {_CACHE}  ({len(csts)} sommets)")


def _load_cache():
    csts, ras, decs = [], [], []
    with open(_CACHE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            csts.append(parts[0])
            ras.append(float(parts[1]))
            decs.append(float(parts[2]))
    return csts, np.array(ras), np.array(decs)


def _build_segments(csts, ra_icrs, dec_icrs):
    """
    Group consecutive same-constellation vertices into polyline segments.
    Returns (segments, labels).
    """
    segments = []
    cent = {}   # abbr → ([ra…], [dec…])

    prev    = None
    seg_ra  = []
    seg_dec = []

    def _flush(abbr, seg_ra, seg_dec):
        if len(seg_ra) < 2:
            return
        ra_arr  = np.array(seg_ra,  dtype=float)
        dec_arr = np.array(seg_dec, dtype=float)
        # Insert NaN at RA wrap-around jumps so plot() doesn't draw across the map
        jumps = np.where(np.abs(np.diff(ra_arr)) > 180)[0]
        if len(jumps):
            ra_arr  = np.insert(ra_arr,  jumps + 1, np.nan)
            dec_arr = np.insert(dec_arr, jumps + 1, np.nan)
        segments.append((ra_arr, dec_arr))
        if abbr not in cent:
            cent[abbr] = ([], [])
        cent[abbr][0].extend(seg_ra)
        cent[abbr][1].extend(seg_dec)

    for i, cst in enumerate(csts):
        if cst != prev:
            if prev is not None:
                _flush(prev, seg_ra, seg_dec)
            seg_ra, seg_dec = [], []
            prev = cst
        seg_ra.append(ra_icrs[i])
        seg_dec.append(dec_icrs[i])

    if prev is not None:
        _flush(prev, seg_ra, seg_dec)

    labels = [(np.mean(v[0]), np.mean(v[1]), abbr)
              for abbr, v in cent.items()]

    return segments, labels


def get_boundaries():
    """
    Return constellation boundary segments and label positions (ICRS J2000).

    Returns
    -------
    segments : list of (ra_deg_array, dec_deg_array)
    labels   : list of (ra_deg, dec_deg, abbr)
    """
    if os.path.exists(_CACHE):
        csts, ra_icrs, dec_icrs = _load_cache()
    else:
        csts, ra_icrs, dec_icrs = _fetch_from_vizier()
        _save_cache(csts, ra_icrs, dec_icrs)

    return _build_segments(csts, ra_icrs, dec_icrs)
