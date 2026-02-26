"""
Bright star catalog — 1000 brightest stars from the Yale Bright Star
Catalogue 5th edition (Vizier V/50), cached locally after first download.

Public API
----------
BRIGHT_STARS : list of (name: str, ra_deg: float, dec_deg: float, vmag: float)
"""

import os
import numpy as np

_CACHE = os.path.join(os.path.dirname(__file__), "bright_stars_cache.csv")
N_STARS = 1000


def _fetch_from_vizier():
    """Download Yale BSC5 from Vizier and return a sorted astropy Table."""
    from astroquery.vizier import Vizier
    print("Téléchargement du Yale Bright Star Catalogue (Vizier V/50)…")
    v = Vizier(columns=["HR", "Name", "RAJ2000", "DEJ2000", "Vmag"],
               row_limit=-1)
    tables = v.get_catalogs("V/50")
    cat = tables[0]
    # Keep only rows with a valid Vmag
    mask = ~cat["Vmag"].mask if hasattr(cat["Vmag"], "mask") else np.ones(len(cat), bool)
    cat = cat[mask]
    cat.sort("Vmag")
    return cat[:N_STARS]


def _table_to_list(cat):
    """Convert a BSC5 Vizier table to [(name, ra_deg, dec_deg, vmag), ...]."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # RAJ2000 and DEJ2000 arrive as sexagesimal strings ("HH MM SS.S", "+DD MM SS")
    ra_str  = [str(r).strip() for r in cat["RAJ2000"]]
    dec_str = [str(d).strip() for d in cat["DEJ2000"]]

    coords = SkyCoord(ra=ra_str, dec=dec_str,
                      unit=(u.hourangle, u.deg), frame="icrs")

    stars = []
    for coord, row in zip(coords, cat):
        name = str(row["Name"]).strip() if row["Name"] else ""
        vmag = float(row["Vmag"])
        stars.append((name, float(coord.ra.deg), float(coord.dec.deg), vmag))
    return stars


def _save_cache(stars):
    with open(_CACHE, "w", encoding="utf-8") as f:
        for name, ra, dec, mag in stars:
            # Escape commas in names (rare but possible)
            safe_name = name.replace(",", " ")
            f.write(f"{ra:.6f},{dec:.6f},{mag:.3f},{safe_name}\n")
    print(f"Cache sauvegardé : {_CACHE}  ({len(stars)} étoiles)")


def _load_cache():
    stars = []
    with open(_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 3)
            ra, dec, mag = float(parts[0]), float(parts[1]), float(parts[2])
            name = parts[3] if len(parts) == 4 else ""
            stars.append((name, ra, dec, mag))
    return stars


def load_bright_stars():
    if os.path.exists(_CACHE):
        return _load_cache()
    cat   = _fetch_from_vizier()
    stars = _table_to_list(cat)
    _save_cache(stars)
    return stars


BRIGHT_STARS = load_bright_stars()
