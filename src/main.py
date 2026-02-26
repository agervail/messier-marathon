import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timezone

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astropy.units as u

from bright_stars import BRIGHT_STARS
from messier_catalog import MESSIER_OBJECTS
from constellations import get_boundaries, FRENCH_NAMES
from constellation_lines import get_lines

LON_DEG = 5.86980    # Longitude de l'observateur (degrés, positif vers l'Est)
LAT_DEG = 44.56410   # Latitude de l'observateur (degrés, positif vers le Nord)
TIME_UTC = "2025-03-21 20:00:00"  # Heure d'observation en UTC

# Visual style per object type  (colours chosen for white background)
TYPE_STYLE = {
    "GC":  {"color": "#cc9900", "marker": "o",  "size": 60,  "label": "Amas globulaire"},
    "OC":  {"color": "#0088bb", "marker": "^",  "size": 55,  "label": "Amas ouvert"},
    "Gx":  {"color": "#cc5500", "marker": "s",  "size": 55,  "label": "Galaxie"},
    "EN":  {"color": "#cc2288", "marker": "*",  "size": 100, "label": "Nébuleuse diffuse"},
    "PN":  {"color": "#338800", "marker": "D",  "size": 55,  "label": "Nébuleuse planétaire"},
    "SNR": {"color": "#dd1111", "marker": "P",  "size": 80,  "label": "Reste de supernova"},
    "SC":  {"color": "#555555", "marker": "+",  "size": 80,  "label": "Nuage stellaire"},
    "DS":  {"color": "#777777", "marker": "x",  "size": 60,  "label": "Étoile double"},
}

# Objects to label explicitly (notable ones)
LABEL_OBJECTS = {
    1, 8, 13, 16, 17, 27, 31, 33, 42, 44, 45, 51, 57, 81, 82, 83, 87, 97, 101, 104
}


def equatorial_to_altaz(ra_deg, dec_deg, lon_deg, lat_deg, utc_time, elevation_m=0.0):
    """
    Convertit des coordonnées équatoriales (AD/Dec, J2000) en coordonnées
    horizontales (Altitude/Azimut) pour un observateur et une heure UTC donnés.

    Paramètres
    ----------
    ra_deg : float ou array-like
        Ascension(s) droite(s) en degrés (J2000).
    dec_deg : float ou array-like
        Déclinaison(s) en degrés (J2000).
    lon_deg : float
        Longitude de l'observateur en degrés (positif vers l'Est).
    lat_deg : float
        Latitude de l'observateur en degrés (positif vers le Nord).
    utc_time : str ou datetime
        Heure UTC d'observation, ex. "2025-06-21 22:00:00".
    elevation_m : float, optionnel
        Altitude de l'observateur en mètres (défaut 0).

    Retourne
    --------
    list of dict avec les clés :
        'ra_deg'   – ascension droite originale (degrés)
        'dec_deg'  – déclinaison originale (degrés)
        'alt_deg'  – altitude en degrés  (> 0 : au-dessus de l'horizon)
        'az_deg'   – azimut en degrés   (0 = Nord, 90 = Est, 180 = Sud, 270 = Ouest)
        'visible'  – True si l'objet est au-dessus de l'horizon

    Exemple
    -------
    >>> # Paris (48.85°N, 2.35°E) le 21 juin 2025 à 22 h UTC
    >>> results = equatorial_to_altaz(
    ...     ra_deg=83.82,   # Nébuleuse d'Orion M42
    ...     dec_deg=-5.39,
    ...     lon_deg=2.35,
    ...     lat_deg=48.85,
    ...     utc_time="2025-06-21 22:00:00",
    ... )
    >>> print(results)
    """
    ra_arr = np.atleast_1d(np.asarray(ra_deg,  dtype=float))
    dec_arr = np.atleast_1d(np.asarray(dec_deg, dtype=float))

    if isinstance(utc_time, str):
        obs_time = Time(utc_time, scale="utc")
    elif isinstance(utc_time, datetime):
        obs_time = Time(utc_time, scale="utc")
    else:
        obs_time = Time(utc_time)

    location = EarthLocation(
        lon=lon_deg * u.deg,
        lat=lat_deg * u.deg,
        height=elevation_m * u.m,
    )
    frame_altaz = AltAz(obstime=obs_time, location=location)

    coords = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")
    coords_altaz = coords.transform_to(frame_altaz)

    results = []
    for ra, dec, alt, az in zip(
        ra_arr, dec_arr,
        coords_altaz.alt.deg,
        coords_altaz.az.deg,
    ):
        results.append({
            "ra_deg":  float(ra),
            "dec_deg": float(dec),
            "alt_deg": float(alt),
            "az_deg":  float(az),
            "visible": bool(alt > 0),
        })

    return results if len(results) > 1 else results[0]


def catalog_altaz(lon_deg, lat_deg, utc_time, elevation_m=0.0,
                  include_stars=True, mag_limit=4.5):
    """
    Convertit TOUS les objets dessinés sur la carte (catalogue Messier +
    étoiles brillantes) en coordonnées Alt/Az pour un lieu et une heure donnés.

    Paramètres
    ----------
    lon_deg, lat_deg : float
        Longitude et latitude de l'observateur (degrés).
    utc_time : str ou datetime
        Heure UTC d'observation.
    elevation_m : float
        Altitude de l'observateur en mètres.
    include_stars : bool
        Si True, inclut aussi les étoiles brillantes (défaut : True).
    mag_limit : float
        Magnitude limite pour les étoiles incluses (défaut 4.5).

    Retourne
    --------
    dict avec deux clés :
        'messier' : liste de dict  {'number', 'name', 'type',
                                    'ra_deg', 'dec_deg',
                                    'alt_deg', 'az_deg', 'visible'}
        'stars'   : liste de dict  {'name', 'mag',
                                    'ra_deg', 'dec_deg',
                                    'alt_deg', 'az_deg', 'visible'}
                    (vide si include_stars=False)

    Exemple
    -------
    >>> # Observatoire de Paris, nuit du 21 juin 2025
    >>> data = catalog_altaz(
    ...     lon_deg=2.3364, lat_deg=48.8362,
    ...     utc_time="2025-06-21 22:00:00",
    ... )
    >>> visible = [o for o in data['messier'] if o['visible']]
    >>> print(f"{len(visible)} objets Messier visibles ce soir")
    >>> for obj in sorted(visible, key=lambda o: -o['alt_deg'])[:5]:
    ...     print(f"  M{obj['number']:>3} {obj['name']:<25} "
    ...           f"alt={obj['alt_deg']:+.1f}°  az={obj['az_deg']:.1f}°")
    """
    # ── Catalogue Messier ──────────────────────────────────────────────────
    m_ra = [r[1] for r in MESSIER_OBJECTS]
    m_dec = [r[2] for r in MESSIER_OBJECTS]

    m_altaz = equatorial_to_altaz(m_ra, m_dec, lon_deg, lat_deg,
                                  utc_time, elevation_m)
    if isinstance(m_altaz, dict):          # single-object edge case
        m_altaz = [m_altaz]

    messier_results = []
    for (num, ra, dec, otype, name), altaz in zip(MESSIER_OBJECTS, m_altaz):
        altaz["number"] = num
        altaz["name"] = name
        altaz["type"] = otype
        messier_results.append(altaz)

    # ── Étoiles brillantes ─────────────────────────────────────────────────
    stars_results = []
    if include_stars:
        filtered = [(n, ra, dec, mag) for n, ra, dec, mag in BRIGHT_STARS
                    if mag <= mag_limit]
        if filtered:
            s_ra = [s[1] for s in filtered]
            s_dec = [s[2] for s in filtered]
            s_altaz = equatorial_to_altaz(s_ra, s_dec, lon_deg, lat_deg,
                                          utc_time, elevation_m)
            if isinstance(s_altaz, dict):
                s_altaz = [s_altaz]
            for (sname, ra, dec, mag), altaz in zip(filtered, s_altaz):
                altaz["name"] = sname
                altaz["mag"] = mag
                stars_results.append(altaz)

    return {"messier": messier_results, "stars": stars_results}


def _star_size(mag):
    """Return scatter marker size from visual magnitude (brighter → larger)."""
    return np.clip(10 * np.power(10, (4.5 - mag) / 5), 2, 300)


def load_horizon(path=None):
    """
    Charge un fichier d'horizon au format Stellarium (deux colonnes : Azimut Altitude).
    Retourne (az_deg, alt_deg) en numpy arrays, triés par azimut (0–360°).
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "horizon.txt")

    az_list, alt_list = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line[0].isdigit():
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    az_list.append(float(parts[0]))
                    alt_list.append(float(parts[1]))
                except ValueError:
                    continue

    az = np.array(az_list)
    alt = np.array(alt_list)
    order = np.argsort(az)
    return az[order], alt[order]


def make_sky_map(out="messier_sky_map.png"):
    numbers = [r[0] for r in MESSIER_OBJECTS]
    ra_deg = np.array([r[1] for r in MESSIER_OBJECTS])
    dec_deg = np.array([r[2] for r in MESSIER_OBJECTS])
    types = [r[3] for r in MESSIER_OBJECTS]
    names = [r[4] for r in MESSIER_OBJECTS]

    # ── Dédoublonnage des étoiles (RA/Dec arrondis à 2 décimales) ─────────────
    seen = set()
    stars_unique = []
    for entry in BRIGHT_STARS:
        name, ra, dec, mag = entry
        key = (round(ra, 2), round(dec, 2))
        if key not in seen:
            seen.add(key)
            stars_unique.append(entry)

    s_ra = np.array([s[1] for s in stars_unique])
    s_dec = np.array([s[2] for s in stars_unique])
    s_mag = np.array([s[3] for s in stars_unique])

    # ── Figure — equirectangular, RA on X, Dec on Y ───────────────────────────
    fig, ax = plt.subplots(figsize=(22, 10), facecolor="white")
    ax.set_facecolor("white")

    # RA increases to the LEFT (astronomical convention)
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")

    # ── Frontières de constellations ─────────────────────────────────────────
    const_segs, const_labels = get_boundaries()

    for ra_arr, dec_arr in const_segs:
        ax.plot(ra_arr, dec_arr,
                color="#ddeeff", linewidth=0.3, alpha=0.5,
                linestyle="-", zorder=1)

    for ra_lbl, dec_lbl, abbr in const_labels:
        if not (0 <= ra_lbl <= 360 and -90 <= dec_lbl <= 90):
            continue
        fr = FRENCH_NAMES.get(abbr, abbr)
        ax.text(ra_lbl, dec_lbl, fr.upper(),
                fontsize=3.8, color="#7799bb", alpha=0.65,
                ha="center", va="center", zorder=1,
                fontfamily="sans-serif")

    # ── Lignes de constellations (figures traditionnelles) ────────────────────
    const_lines = get_lines()

    for ra1, dec1, ra2, dec2 in const_lines:
        # Skip segments that cross the RA=0/360 wrap-around
        if abs(ra1 - ra2) > 180:
            continue
        ax.plot([ra1, ra2], [dec1, dec2],
                color="#4477aa", linewidth=0.9, alpha=0.55,
                solid_capstyle="round", zorder=2)

    # ── Étoiles brillantes ───────────────────────────────────────────────────
    sizes = _star_size(s_mag)
    ax.scatter(s_ra, s_dec,
               s=sizes, c="#222244", alpha=0.75,
               linewidths=0, zorder=3)

    # Noms des étoiles les plus brillantes (V < 2.0)
    for name, ra, dec, mag in stars_unique:
        if name and mag < 2.0:
            ax.annotate(name, xy=(ra, dec),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=5, color="#334466", alpha=0.8, zorder=3)

    # ── Équateur céleste (Dec = 0) ────────────────────────────────────────────
    ax.axhline(0, color="#aaaacc", linewidth=0.8, linestyle="--",
               alpha=0.6, zorder=3)

    # ── Grid ──────────────────────────────────────────────────────────────────
    # Major grid every 2 h / 30°
    ax.set_xticks(np.arange(0, 361, 30))          # every 30° = 2 h
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, color="#dde2ee", linewidth=0.5, linestyle=":", zorder=0)
    # Minor grid every 15° = 1 h / 10°
    ax.set_xticks(np.arange(0, 361, 15), minor=True)
    ax.set_yticks(np.arange(-90, 91, 10), minor=True)
    ax.grid(True, which="minor", color="#eef0f8", linewidth=0.3,
            linestyle=":", zorder=0)

    # ── RA tick labels in hours ───────────────────────────────────────────────
    ra_major = np.arange(0, 361, 30)   # 0°, 30°, … 360°
    ax.set_xticklabels([f"{int(d / 15)}h" for d in ra_major],
                       fontsize=9, color="#334466")
    ax.set_yticklabels([f"{d:+d}°" for d in np.arange(-90, 91, 30)],
                       fontsize=9, color="#334466")

    for spine in ax.spines.values():
        spine.set_edgecolor("#6688bb")
    ax.tick_params(colors="#334466", which="both")

    # ── Messier objects ───────────────────────────────────────────────────────
    plotted_types = set()
    for num, ra, dec, otype, name in zip(numbers, ra_deg, dec_deg, types, names):
        style = TYPE_STYLE.get(otype, TYPE_STYLE["DS"])
        label_arg = style["label"] if otype not in plotted_types else "_nolegend_"
        plotted_types.add(otype)

        ax.scatter(ra, dec,
                   c=style["color"], marker=style["marker"],
                   s=style["size"] * 1.3, zorder=5, alpha=0.95,
                   linewidths=0.4,
                   edgecolors="#333333" if style["marker"] not in (
                       "+", "x") else "none",
                   label=label_arg)

        ax.annotate(f"M{num}", xy=(ra, dec),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=4.5, color="#444444", alpha=0.8, zorder=6)

        if num in LABEL_OBJECTS:
            display = name if name else f"M{num}"
            ax.annotate(display, xy=(ra, dec),
                        xytext=(5, -9), textcoords="offset points",
                        fontsize=6.5, color=style["color"],
                        fontweight="bold", alpha=1.0, zorder=7)

    # ── Axis labels & title ───────────────────────────────────────────────────
    ax.set_xlabel("Ascension droite  α", fontsize=11,
                  color="#334466", labelpad=8)
    ax.set_ylabel("Déclinaison  δ", fontsize=11, color="#334466", labelpad=8)
    ax.set_title(
        "Catalogue de Messier — Carte du ciel (coordonnées équatoriales J2000)",
        fontsize=14, color="#111133", fontweight="bold", pad=14,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    obj_handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in TYPE_STYLE.values()
    ]
    extra_handles = [
        plt.scatter([], [], s=60,  c="#222244", label="Étoile (V < 1)"),
        plt.scatter([], [], s=20,  c="#222244", label="Étoile (V 1–3)"),
        plt.scatter([], [], s=5,   c="#222244", label="Étoile (V 3–4.5)"),
        plt.Line2D([0], [0], color="#aaaacc", linewidth=1.0,
                   linestyle="--", label="Équateur céleste (δ = 0°)"),
    ]
    ax.legend(
        handles=obj_handles + extra_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.13),
        ncol=6, fontsize=8.5,
        facecolor="white", edgecolor="#6688bb",
        labelcolor="#222222", framealpha=0.95,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    type_counts = {}
    for r in MESSIER_OBJECTS:
        type_counts[r[3]] = type_counts.get(r[3], 0) + 1
    summary = "  |  ".join(
        f"{TYPE_STYLE[t]['label']}: {n}"
        for t, n in sorted(type_counts.items(), key=lambda x: -x[1])
    )
    fig.text(0.5, 0.005, f"{len(MESSIER_OBJECTS)} objets  ·  {summary}",
             ha="center", fontsize=7.5, color="#334466")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Image sauvegardée : {out}")
    plt.close(fig)


def make_altaz_map(lon_deg=LON_DEG, lat_deg=LAT_DEG,
                   utc_time=TIME_UTC,
                   elevation_m=0.0,
                   out="messier_altaz_map.png"):
    """
    Génère une carte rectangulaire Alt/Az : azimut en X (0–360°), altitude en Y (0–90°).
    Seuls les objets au-dessus de l'horizon sont tracés.

    Paramètres
    ----------
    lon_deg : float   – longitude observateur (degrés, positif Est)
    lat_deg : float   – latitude observateur (degrés, positif Nord)
    utc_time : str    – heure UTC, ex. "2025-06-21 22:00:00"
    elevation_m : float – altitude du site en mètres
    out : str         – nom du fichier image de sortie
    """
    # ── 1. Conversion de tous les objets ──────────────────────────────────────
    data = catalog_altaz(lon_deg, lat_deg, utc_time,
                         elevation_m, mag_limit=4.5)
    visible_m = [o for o in data["messier"] if o["visible"]]
    visible_s = [o for o in data["stars"] if o["visible"]]

    # ── 2. Conversion des lignes de constellations ────────────────────────────
    raw_lines = get_lines()                        # [(ra1,dec1,ra2,dec2), …]
    if raw_lines:
        pts_ra = [seg[0] for seg in raw_lines] + [seg[2] for seg in raw_lines]
        pts_dec = [seg[1] for seg in raw_lines] + [seg[3] for seg in raw_lines]
        pts_altaz = equatorial_to_altaz(pts_ra, pts_dec,
                                        lon_deg, lat_deg, utc_time, elevation_m)
        n = len(raw_lines)
        ends1 = pts_altaz[:n]
        ends2 = pts_altaz[n:]
    else:
        ends1, ends2 = [], []

    # ── 3. Figure rectangulaire : Az en X, Alt en Y ───────────────────────────
    fig, ax = plt.subplots(figsize=(22, 8), facecolor="#08081a")
    ax.set_facecolor("#08081a")

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)

    # ── Grille ────────────────────────────────────────────────────────────────
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_yticks(np.arange(0, 91, 15))
    ax.grid(True, color="#1e2d55", linewidth=0.5, linestyle=":", zorder=0)
    ax.set_xticks(np.arange(0, 361, 15), minor=True)
    ax.set_yticks(np.arange(0, 91, 5), minor=True)
    ax.grid(True, which="minor", color="#141e3a", linewidth=0.3,
            linestyle=":", zorder=0)

    # Labels azimut avec points cardinaux
    cardinal = {0: "N", 45: "NE", 90: "E", 135: "SE",
                180: "S", 225: "SO", 270: "O", 315: "NO", 360: "N"}
    ax.set_xticklabels(
        [f"{cardinal[d]}  {d}°" for d in np.arange(0, 361, 45)],
        fontsize=9, color="#aabbee",
    )
    ax.set_yticklabels([f"{d}°" for d in np.arange(0, 91, 15)],
                       fontsize=9, color="#aabbee")

    # Ligne d'horizon astronomique (alt = 0)
    ax.axhline(0, color="#3355aa", linewidth=1.0, alpha=0.5, zorder=2,
               linestyle="--")

    # ── Horizon local (fichier horizon.txt) ───────────────────────────────────
    hz_path = os.path.join(os.path.dirname(__file__), "horizon.txt")
    if os.path.exists(hz_path):
        hz_az, hz_alt = load_horizon(hz_path)
        # Fermer le polygone à 0° et 360° pour le fill
        hz_az_c = np.concatenate([[0.0], hz_az, [360.0]])
        hz_alt_c = np.concatenate([[hz_alt[0]], hz_alt, [hz_alt[-1]]])
        ax.fill_between(hz_az_c, 0, hz_alt_c,
                        color="#3a1500", alpha=0.55, zorder=2)
        ax.plot(hz_az_c, hz_alt_c,
                color="#ff7733", linewidth=1.4, alpha=0.95, zorder=11,
                label="Horizon local")

    # ── Lignes de constellations (segments entièrement au-dessus horizon) ─────
    for a1, a2 in zip(ends1, ends2):
        if a1["alt_deg"] > 0 and a2["alt_deg"] > 0:
            if abs(a1["az_deg"] - a2["az_deg"]) > 180:
                continue
            ax.plot(
                [a1["az_deg"], a2["az_deg"]],
                [a1["alt_deg"], a2["alt_deg"]],
                color="#2255aa", linewidth=0.9, alpha=0.55,
                solid_capstyle="round", zorder=3,
            )

    # ── Étoiles ───────────────────────────────────────────────────────────────
    if visible_s:
        s_az = np.array([s["az_deg"] for s in visible_s])
        s_alt = np.array([s["alt_deg"] for s in visible_s])
        s_mag = np.array([s["mag"] for s in visible_s])
        ax.scatter(s_az, s_alt,
                   s=_star_size(s_mag), c="#d0d8ff", alpha=0.80,
                   linewidths=0, zorder=4)
        for star in visible_s:
            if star["name"] and star["mag"] < 1.5:
                ax.annotate(star["name"],
                            xy=(star["az_deg"], star["alt_deg"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=5.5, color="#aabbee", alpha=0.9, zorder=5)

    # ── Objets Messier ────────────────────────────────────────────────────────
    plotted_types = set()
    for obj in visible_m:
        style = TYPE_STYLE.get(obj["type"], TYPE_STYLE["DS"])
        az, alt = obj["az_deg"], obj["alt_deg"]
        lbl = style["label"] if obj["type"] not in plotted_types else "_nolegend_"
        plotted_types.add(obj["type"])

        ax.scatter(az, alt,
                   c=style["color"], marker=style["marker"],
                   s=style["size"] * 1.3, zorder=6, alpha=0.95,
                   linewidths=0.4,
                   edgecolors="#ffffff" if style["marker"] not in (
                       "+", "x") else "none",
                   label=lbl)

        ax.annotate(f"M{obj['number']}", xy=(az, alt),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=4.5, color="#cccccc", alpha=0.85, zorder=7)

        if obj["number"] in LABEL_OBJECTS:
            display = obj["name"] if obj["name"] else f"M{obj['number']}"
            ax.annotate(display, xy=(az, alt),
                        xytext=(5, -9), textcoords="offset points",
                        fontsize=6.5, color=style["color"],
                        fontweight="bold", alpha=1.0, zorder=8)

    # ── Labels des axes ───────────────────────────────────────────────────────
    ax.set_xlabel("Azimut  Az  (0° = Nord · 90° = Est · 180° = Sud · 270° = Ouest)",
                  fontsize=11, color="#8899cc", labelpad=8)
    ax.set_ylabel("Altitude  Alt", fontsize=11, color="#8899cc", labelpad=8)

    for spine in ax.spines.values():
        spine.set_edgecolor("#2244aa")
    ax.tick_params(colors="#8899cc", which="both")

    # ── Titre ─────────────────────────────────────────────────────────────────
    loc_str = f"lat {lat_deg:+.2f}°  lon {lon_deg:+.2f}°  alt {elevation_m:.0f} m"
    time_str = utc_time if isinstance(utc_time, str) else str(utc_time)
    ax.set_title(
        f"Catalogue de Messier — Coordonnées Alt/Az\n"
        f"{loc_str}  ·  {time_str} UTC  ·  "
        f"{len(visible_m)} objets Messier visibles sur {len(data['messier'])}",
        fontsize=13, color="#ccd8ff", fontweight="bold", pad=12,
    )

    # ── Légende ───────────────────────────────────────────────────────────────
    obj_handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in TYPE_STYLE.values()
    ]
    star_handles = [
        plt.scatter([], [], s=60, c="#d0d8ff", label="Étoile (V < 1)"),
        plt.scatter([], [], s=20, c="#d0d8ff", label="Étoile (V 1–3)"),
        plt.scatter([], [], s=5,  c="#d0d8ff", label="Étoile (V 3–4.5)"),
    ]
    ax.legend(
        handles=obj_handles + star_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.15),
        ncol=6, fontsize=8.5,
        facecolor="#0d1030", edgecolor="#3355aa",
        labelcolor="#ccddff", framealpha=0.95,
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Carte Alt/Az sauvegardée : {out}")
    plt.close(fig)


def make_altaz_map_polar(lon_deg=LON_DEG, lat_deg=LAT_DEG,
                         utc_time=TIME_UTC,
                         elevation_m=0.0,
                         out="messier_altaz_map_polar.png"):
    """
    Génère une carte du ciel polaire en coordonnées Alt/Az (vue depuis le sol).

    Le centre du cercle = zénith (alt 90°), le bord = horizon (alt 0°).
    Nord en haut, azimut croissant dans le sens horaire (N→E→S→O).

    Paramètres
    ----------
    lon_deg : float   – longitude observateur (degrés, positif Est)
    lat_deg : float   – latitude observateur (degrés, positif Nord)
    utc_time : str    – heure UTC, ex. "2025-06-21 22:00:00"
    elevation_m : float – altitude du site en mètres
    out : str         – nom du fichier image de sortie
    """
    # ── 1. Conversion de tous les objets ──────────────────────────────────────
    data = catalog_altaz(lon_deg, lat_deg, utc_time,
                         elevation_m, mag_limit=4.5)
    visible_m = [o for o in data["messier"] if o["visible"]]
    visible_s = [o for o in data["stars"] if o["visible"]]

    # ── 2. Conversion des lignes de constellations ────────────────────────────
    raw_lines = get_lines()
    if raw_lines:
        pts_ra = [seg[0] for seg in raw_lines] + [seg[2] for seg in raw_lines]
        pts_dec = [seg[1] for seg in raw_lines] + [seg[3] for seg in raw_lines]
        pts_altaz = equatorial_to_altaz(pts_ra, pts_dec,
                                        lon_deg, lat_deg, utc_time, elevation_m)
        n = len(raw_lines)
        ends1 = pts_altaz[:n]
        ends2 = pts_altaz[n:]
    else:
        ends1, ends2 = [], []

    # ── 3. Figure polaire ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw={"projection": "polar"},
                           facecolor="#08081a")
    ax.set_facecolor("#08081a")

    # Nord en haut, azimut croissant dans le sens horaire (N→E→S→O)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # r = 90 − altitude  →  zénith au centre (r=0), horizon au bord (r=90)
    ax.set_rlim(0, 90)

    # ── Grille et anneaux d'altitude ──────────────────────────────────────────
    ax.set_rticks([0, 30, 60, 90])
    ax.yaxis.set_tick_params(labelcolor="#8899cc", labelsize=8)
    ax.set_yticklabels(["Zénith 90°", "60°", "30°", "0° Horizon"],
                       color="#8899cc", fontsize=8)
    ax.grid(True, color="#1e2d55", linewidth=0.6, linestyle=":", alpha=0.8)

    # Cercle horizon astronomique (alt = 0)
    theta_full = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta_full, np.full(360, 90),
            color="#3355aa", linewidth=1.0, alpha=0.5, zorder=2,
            linestyle="--")

    # ── Horizon local (fichier horizon.txt) ───────────────────────────────────
    hz_path = os.path.join(os.path.dirname(__file__), "horizon.txt")
    if os.path.exists(hz_path):
        hz_az, hz_alt = load_horizon(hz_path)
        # Interpoler sur une grille régulière pour un tracé polaire lisse
        theta_smooth = np.linspace(0, 2 * np.pi, 720)
        # Dupliquer de part et d'autre pour gérer le wrap-around
        az_ext = np.concatenate([hz_az - 360, hz_az, hz_az + 360])
        alt_ext = np.concatenate([hz_alt,       hz_alt, hz_alt])
        r_smooth = 90 - np.interp(np.degrees(theta_smooth), az_ext, alt_ext)
        # Zone bloquée : remplissage entre l'horizon et le bord (r=90)
        ax.fill_between(theta_smooth, r_smooth, 90,
                        color="#3a1500", alpha=0.55, zorder=2)
        ax.plot(theta_smooth, r_smooth,
                color="#ff7733", linewidth=1.4, alpha=0.95, zorder=11)

    # Points cardinaux
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
                       fontsize=11, fontweight="bold", color="#ccd8ff")

    # ── Lignes de constellations (segments entièrement au-dessus horizon) ─────
    for a1, a2 in zip(ends1, ends2):
        if a1["alt_deg"] > 1 and a2["alt_deg"] > 1:
            ax.plot(
                [np.radians(a1["az_deg"]), np.radians(a2["az_deg"])],
                [90 - a1["alt_deg"],       90 - a2["alt_deg"]],
                color="#2255aa", linewidth=0.9, alpha=0.55,
                solid_capstyle="round", zorder=3,
            )

    # ── Étoiles ───────────────────────────────────────────────────────────────
    for star in visible_s:
        theta = np.radians(star["az_deg"])
        r = 90 - star["alt_deg"]
        ax.scatter(theta, r, s=_star_size(star["mag"]), c="#d0d8ff",
                   alpha=0.85, linewidths=0, zorder=4)
        if star["name"] and star["mag"] < 1.5:
            ax.annotate(star["name"], (theta, r),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6.5, color="#aabbee", alpha=0.9, zorder=5)

    # ── Objets Messier ────────────────────────────────────────────────────────
    plotted_types = set()
    for obj in visible_m:
        style = TYPE_STYLE.get(obj["type"], TYPE_STYLE["DS"])
        theta = np.radians(obj["az_deg"])
        r = 90 - obj["alt_deg"]
        lbl = style["label"] if obj["type"] not in plotted_types else "_nolegend_"
        plotted_types.add(obj["type"])

        ax.scatter(theta, r,
                   c=style["color"], marker=style["marker"],
                   s=style["size"] * 1.4, zorder=6, alpha=0.95,
                   linewidths=0.5,
                   edgecolors="#ffffff" if style["marker"] not in (
                       "+", "x") else "none",
                   label=lbl)

        ax.annotate(f"M{obj['number']}", (theta, r),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=4.5, color="#dddddd", alpha=0.85, zorder=7)

        if obj["number"] in LABEL_OBJECTS:
            display = obj["name"] if obj["name"] else f"M{obj['number']}"
            ax.annotate(display, (theta, r),
                        xytext=(5, -10), textcoords="offset points",
                        fontsize=6.5, color=style["color"],
                        fontweight="bold", alpha=1.0, zorder=8)

    # ── Titre ─────────────────────────────────────────────────────────────────
    loc_str = f"lat {lat_deg:+.2f}°  lon {lon_deg:+.2f}°  alt {elevation_m:.0f} m"
    time_str = utc_time if isinstance(utc_time, str) else str(utc_time)
    ax.set_title(
        f"Catalogue de Messier — Vue Alt/Az (polaire)\n"
        f"{loc_str}  ·  {time_str} UTC\n"
        f"{len(visible_m)} objets Messier visibles sur {len(data['messier'])}",
        fontsize=11, color="#ccd8ff", fontweight="bold", pad=20,
    )

    # ── Légende ───────────────────────────────────────────────────────────────
    obj_handles = [
        mpatches.Patch(color=v["color"], label=v["label"])
        for v in TYPE_STYLE.values()
    ]
    star_handles = [
        plt.scatter([], [], s=60, c="#d0d8ff", label="Étoile (V < 1)"),
        plt.scatter([], [], s=20, c="#d0d8ff", label="Étoile (V 1–3)"),
        plt.scatter([], [], s=5,  c="#d0d8ff", label="Étoile (V 3–4.5)"),
    ]
    ax.legend(
        handles=obj_handles + star_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=5, fontsize=8,
        facecolor="#0d1030", edgecolor="#3355aa",
        labelcolor="#ccddff", framealpha=0.95,
    )

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Carte Alt/Az polaire sauvegardée : {out}")
    plt.close(fig)


def compute_visibility(date_str, lon_deg=LON_DEG, lat_deg=LAT_DEG,
                       elevation_m=0.0, step_min=5, twilight_deg=-12.0):
    """
    Pour chaque objet Messier, calcule la fenêtre de visibilité sur la nuit donnée.

    Paramètres
    ----------
    date_str     : str   – date d'observation "YYYY-MM-DD"
    lon_deg      : float – longitude observateur (degrés Est)
    lat_deg      : float – latitude observateur (degrés Nord)
    elevation_m  : float – altitude du site en mètres
    step_min     : int   – pas d'échantillonnage en minutes (défaut 5)
    twilight_deg : float – seuil solaire pour la nuit (défaut -12° = crépuscule nautique)

    Retourne
    --------
    list of dict avec les clés :
        'number', 'name', 'type',
        'rise'         : heure UTC de première visibilité "HH:MM" (ou None)
        'transit'      : heure UTC d'altitude maximale "HH:MM" (ou None)
        'set'          : heure UTC de dernière visibilité "HH:MM" (ou None)
        'alt_max'      : altitude max en degrés pendant la visibilité
        'az_transit'   : azimut au transit en degrés
        'duration_min' : durée totale de visibilité en minutes
        'visible_tonight' : bool
    """
    from astropy.coordinates import get_sun

    t0 = Time(f"{date_str} 12:00:00", scale="utc")
    n_t = int(24 * 60 / step_min)
    times = t0 + np.arange(n_t) * step_min * u.min

    loc = EarthLocation(lon=lon_deg * u.deg, lat=lat_deg * u.deg,
                        height=elevation_m * u.m)
    frame_all = AltAz(obstime=times, location=loc)

    # Nuit astronomique/nautique
    sun_alt = get_sun(times).transform_to(frame_all).alt.deg
    is_night = sun_alt < twilight_deg
    if not np.any(is_night):           # nuit blanche (été à haute latitude)
        is_night = sun_alt < -6.0

    # Horizon local
    hz_path = os.path.join(os.path.dirname(__file__), "horizon.txt")
    has_hz = os.path.exists(hz_path)
    if has_hz:
        hz_az_h, hz_alt_h = load_horizon(hz_path)
        az_ext = np.concatenate([hz_az_h - 360, hz_az_h, hz_az_h + 360])
        alt_ext = np.concatenate([hz_alt_h,       hz_alt_h, hz_alt_h])

    results = []
    for num, ra, dec, otype, name in MESSIER_OBJECTS:
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        aa = coord.transform_to(frame_all)
        oalt = aa.alt.deg   # shape (n_t,)
        oaz = aa.az.deg

        above = (oalt > np.interp(oaz, az_ext, alt_ext)
                 ) if has_hz else (oalt > 0)
        vis = above & is_night

        # Transit = meilleure altitude pendant la nuit (avec horizon local)
        vis_alt = np.where(vis, oalt, -np.inf)
        i_tr = int(np.argmax(vis_alt)) if np.any(vis) else \
            int(np.argmax(np.where(is_night, oalt, -np.inf)))

        if not np.any(vis):
            results.append({
                "number": num, "name": name, "type": otype,
                "rise": None, "transit": None, "set": None,
                "alt_max": round(float(oalt[i_tr]), 1),
                "az_transit": round(float(oaz[i_tr]), 1),
                "duration_min": 0, "visible_tonight": False,
            })
            continue

        idx = np.where(vis)[0]
        results.append({
            "number": num, "name": name, "type": otype,
            "rise":         times[idx[0]].iso[11:16],
            "transit":      times[i_tr].iso[11:16],
            "set":          times[idx[-1]].iso[11:16],
            "alt_max":      round(float(oalt[i_tr]), 1),
            "az_transit":   round(float(oaz[i_tr]), 1),
            "duration_min": int(len(idx) * step_min),
            "visible_tonight": True,
        })

    return results


_SHORT_TYPE = {
    "GC": "Amas glob.", "OC": "Amas ouvert", "Gx": "Galaxie",
    "EN": "Nébul. diff.", "PN": "Nébul. plan.",
    "SNR": "SNR", "SC": "Nuage stell.", "DS": "Étoile dble",
}


def make_visibility_table(date_str=TIME_UTC[:10], lon_deg=LON_DEG, lat_deg=LAT_DEG,
                          elevation_m=0.0, step_min=5, twilight_deg=-12.0,
                          out="messier_visibility.png"):
    """
    Génère une image-tableau de visibilité des 110 objets Messier pour une nuit donnée.

    Colonnes : M# · Nom · Type · Lever · Transit · Coucher · Durée · Alt max · Azimut
    Triés : visibles d'abord (par heure de lever), puis non-visibles (par alt max).
    """
    print(f"Calcul des visibilités pour le {date_str}…")
    rows = compute_visibility(date_str, lon_deg, lat_deg, elevation_m,
                              step_min, twilight_deg)

    vis = sorted([r for r in rows if r["visible_tonight"]],
                 key=lambda r: r["rise"])
    nv = sorted([r for r in rows if not r["visible_tonight"]],
                key=lambda r: -r["alt_max"])
    all_rows = vis + nv
    n_rows = len(all_rows)

    row_h = 0.20
    hdr_h = 0.36
    ttl_h = 0.72
    fig_h = n_rows * row_h + hdr_h + ttl_h + 0.45
    fig_w = 20.0

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#08081a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_facecolor("#08081a")
    ax.axis("off")

    # Colonnes : (label, x_centre, alignement)
    COL = [
        ("M#",       0.80, "right"),
        ("Nom",      2.40, "left"),
        ("Type",     5.80, "left"),
        ("Lever",    8.60, "center"),
        ("Transit", 10.20, "center"),
        ("Coucher", 11.80, "center"),
        ("Durée",   13.20, "center"),
        ("Alt max", 14.80, "center"),
        ("Azimut",  16.30, "center"),
    ]

    y_ttl = fig_h - 0.22 - ttl_h / 2
    y_hdr = fig_h - 0.22 - ttl_h - hdr_h / 2
    y_data0 = fig_h - 0.22 - ttl_h - hdr_h

    # Titre
    ax.text(fig_w / 2, y_ttl,
            f"Visibilité des objets Messier — nuit du {date_str}\n"
            f"lat {lat_deg:+.4f}°  lon {lon_deg:+.4f}°  "
            f"crépuscule nautique (soleil < {twilight_deg}°)  ·  "
            f"{len(vis)} visibles / {n_rows} objets",
            ha="center", va="center", color="#ccd8ff",
            fontsize=11, fontweight="bold")

    # En-tête
    ax.add_patch(plt.Rectangle((0.15, y_hdr - hdr_h / 2),
                               fig_w - 0.3, hdr_h, color="#1a2040", zorder=1))
    for lbl, xc, ha in COL:
        ax.text(xc, y_hdr, lbl, ha=ha, va="center",
                color="#aabbee", fontsize=8, fontweight="bold",
                fontfamily="monospace")

    # Lignes de données
    for i, row in enumerate(all_rows):
        y = y_data0 - i * row_h - row_h / 2

        # Séparateur visible / non-visible
        if i == len(vis) and len(nv) > 0:
            sep = y + row_h / 2
            ax.plot([0.15, fig_w - 0.15], [sep, sep],
                    color="#334466", linewidth=0.7, zorder=5)
            ax.text(fig_w / 2, sep + 0.03,
                    "── objets non visibles cette nuit ──",
                    ha="center", va="bottom", color="#445577",
                    fontsize=6.5, fontstyle="italic")

        # Fond de ligne
        if row["visible_tonight"]:
            bg = "#0e1535" if i % 2 == 0 else "#121a3a"
        else:
            bg = "#090914" if i % 2 == 0 else "#0b0b1a"
        ax.add_patch(plt.Rectangle((0.15, y - row_h / 2),
                                   fig_w - 0.3, row_h, color=bg, zorder=1))

        # Couleur texte
        tc = (TYPE_STYLE.get(row["type"], TYPE_STYLE["DS"])["color"]
              if row["visible_tonight"] else "#3d3d55")

        # Formatage durée
        dur = row["duration_min"]
        if dur >= 60:
            dur_s = f"{dur // 60}h{dur % 60:02d}"
        elif dur > 0:
            dur_s = f"{dur}min"
        else:
            dur_s = "—"

        vals = [
            f"M{row['number']:>3}",
            (row["name"] or "—")[:18],
            _SHORT_TYPE.get(row["type"], row["type"]),
            row["rise"] or "—",
            row["transit"] or "—",
            row["set"] or "—",
            dur_s,
            f"{row['alt_max']:+.1f}°",
            f"{row['az_transit']:.1f}°" if row["az_transit"] is not None else "—",
        ]

        for (lbl, xc, ha), val in zip(COL, vals):
            ax.text(xc, y, val, ha=ha, va="center",
                    color=tc, fontsize=7, fontfamily="monospace")

    # Légende types en bas
    lx = 0.5
    for otype, style in TYPE_STYLE.items():
        ax.text(lx, 0.15, f"■ {_SHORT_TYPE.get(otype, otype)} ({otype})",
                ha="left", va="center", color=style["color"],
                fontsize=6, fontfamily="monospace")
        lx += 2.45

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Tableau sauvegardé : {out}")
    plt.close(fig)


def main():
    import os
    result_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(result_dir, exist_ok=True)

    make_sky_map(out=os.path.join(result_dir, "messier_sky_map.png"))
    make_altaz_map(
        lon_deg=LON_DEG, lat_deg=LAT_DEG,
        utc_time=TIME_UTC,
        out=os.path.join(result_dir, "messier_altaz_map.png"),
    )
    make_altaz_map_polar(
        lon_deg=LON_DEG, lat_deg=LAT_DEG,
        utc_time=TIME_UTC,
        out=os.path.join(result_dir, "messier_altaz_map_polar.png"),
    )
    make_visibility_table(
        date_str=TIME_UTC[:10],
        lon_deg=LON_DEG, lat_deg=LAT_DEG,
        out=os.path.join(result_dir, "messier_visibility.png"),
    )


if __name__ == "__main__":
    main()
