"""
card_chart.py — Génère une vue oculaire pour les objets Messier.
Télécharge les étoiles du champ via Vizier (Tycho-2) à chaque exécution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

from messier_catalog import MESSIER_OBJECTS

# ── Configuration ─────────────────────────────────────────────────────────────
FOV_DEG = 1.8           # Champ de vue de l'oculaire (degrés)
MAG_LIMIT = 11.5        # Magnitude limite des étoiles affichées

TYPE_LABEL = {
    "GC": "Globular Cluster", "OC": "Open Cluster", "Gx": "Galaxy",
    "EN": "Diffuse Nebula", "PN": "Planetary Nebula",
    "SNR": "Supernova Remnant", "SC": "Star Cloud", "DS": "Double Star",
}


def fetch_field_stars(center_ra, center_dec):
    """Télécharge les étoiles du champ via Vizier (catalogue Tycho-2)."""
    center = SkyCoord(ra=center_ra, dec=center_dec, unit="deg", frame="icrs")
    radius = FOV_DEG * u.deg

    v = Vizier(columns=["_RAJ2000", "_DEJ2000", "VTmag"],
               column_filters={"VTmag": f"<{MAG_LIMIT}"},
               row_limit=-1)
    tables = v.query_region(center, radius=radius, catalog="I/259/tyc2")

    if not tables:
        return []

    cat = tables[0]
    stars = []
    for row in cat:
        try:
            ra = float(row["_RAJ2000"])
            dec = float(row["_DEJ2000"])
            mag = float(row["VTmag"])
            stars.append((ra, dec, mag))
        except (ValueError, KeyError, TypeError):
            continue
    return stars


def make_eyepiece_view(target_num, out=None):
    """Génère la vue simulée dans l'oculaire centrée sur M<target_num>."""
    obj = next(o for o in MESSIER_OBJECTS if o[0] == target_num)
    _, center_ra, center_dec, otype, obj_name = obj

    if out is None:
        out = os.path.join(os.path.dirname(__file__), "..", "result",
                           f"m{target_num}_eyepiece.png")

    print(f"M{target_num} — récupération des étoiles…")
    stars = fetch_field_stars(center_ra, center_dec)
    print(f"  → {len(stars)} étoiles")

    half_fov = FOV_DEG / 2.0
    cos_dec = np.cos(np.radians(center_dec))

    # ── Projeter les étoiles en offsets angulaires ────────────────────────────
    field_stars = []
    for ra, dec, mag in stars:
        dra = (ra - center_ra) * cos_dec
        ddec = dec - center_dec
        if dra**2 + ddec**2 <= half_fov**2:
            field_stars.append((dra, ddec, mag))

    # ── Objets Messier dans le champ ──────────────────────────────────────────
    messier_in_fov = []
    for num, ra, dec, ot, name in MESSIER_OBJECTS:
        dra = (ra - center_ra) * cos_dec
        ddec = dec - center_dec
        if dra**2 + ddec**2 <= half_fov**2:
            messier_in_fov.append((num, dra, ddec, ot, name))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white",
                           subplot_kw={"aspect": "equal"})
    ax.set_facecolor("white")
    lim = half_fov * 1.02
    ax.set_xlim(lim, -lim)    # RA croissant vers la gauche (Est)
    ax.set_ylim(-lim, lim)
    ax.axis("off")

    # Fond blanc du champ circulaire
    sky = Circle((0, 0), half_fov, facecolor="white", edgecolor="none",
                 zorder=0)
    ax.add_patch(sky)

    # Masque blanc autour du cercle
    outer = lim * 2
    theta = np.linspace(0, 2 * np.pi, 300)
    circle_x = half_fov * np.cos(theta)
    circle_y = half_fov * np.sin(theta)

    rect_verts = [(-outer, -outer), (outer, -outer),
                  (outer, outer), (-outer, outer), (-outer, -outer)]
    circ_verts = list(zip(circle_x[::-1], circle_y[::-1]))
    all_verts = rect_verts + circ_verts + [rect_verts[0]]
    codes = ([Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * (len(circ_verts) - 1)
             + [Path.CLOSEPOLY])
    mask_patch = PathPatch(Path(all_verts, codes),
                           facecolor="white", edgecolor="none", zorder=8)
    ax.add_patch(mask_patch)

    # Bord de l'oculaire
    edge = Circle((0, 0), half_fov, fill=False, edgecolor="#888888",
                  linewidth=2, zorder=9)
    ax.add_patch(edge)

    # ── Étoiles ───────────────────────────────────────────────────────────────
    if field_stars:
        dra_arr = np.array([s[0] for s in field_stars])
        ddec_arr = np.array([s[1] for s in field_stars])
        mag_arr = np.array([s[2] for s in field_stars])

        x, y = dra_arr, ddec_arr

        # Taille et luminosité proportionnelles à la magnitude
        sizes = np.clip(8 * np.power(10, (MAG_LIMIT - mag_arr) / 3.5),
                        0.5, 120)
        alphas = np.clip(0.25 + 0.75 * (MAG_LIMIT - mag_arr) / MAG_LIMIT,
                         0.15, 1.0)

        # Couleurs RGBA individuelles pour alpha par étoile
        colors = np.zeros((len(x), 4))
        colors[:, :3] = 0.0       # noir
        colors[:, 3] = alphas     # transparence individuelle

        ax.scatter(x, y, s=sizes, c=colors, linewidths=0, zorder=5)

    # ── Objets Messier dans le champ ──────────────────────────────────────────
    for num, dra, ddec, ot, name in messier_in_fov:
        color = "#333333" if num == target_num else "#666666"
        r = 0.08
        obj_circle = Circle(
            (dra, ddec), r,
            facecolor="none", edgecolor=color,
            linewidth=1.5, linestyle="--", alpha=0.7, zorder=6)
        ax.add_patch(obj_circle)

        label = f"M{num}"
        if name:
            label += f" ({name})"
        ax.text(dra, ddec + r + 0.04, label,
                fontsize=11 if num == target_num else 9,
                color=color, ha="center", va="bottom",
                fontweight="bold", zorder=7)

    # ── Type de l'objet (centré en haut à l'intérieur du cercle) ─────────────
    type_fr = TYPE_LABEL.get(otype, otype)
    ax.text(0, half_fov * 0.88, type_fr,
            ha="center", va="top", fontsize=11, color="#555555", zorder=10)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


if __name__ == "__main__":
    for n in range(1, 11):
        make_eyepiece_view(n)
