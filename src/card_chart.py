"""
card_chart.py — Génère une vue oculaire pour les objets Messier.
Télécharge les étoiles du champ via Vizier (Tycho-2) et une image DSS
de l'objet depuis SkyView à chaque exécution.
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
from astropy.visualization import ZScaleInterval

from messier_catalog import MESSIER_OBJECTS
from dss_fetch import fetch_object_size, fetch_dss_image, DSS_PADDING

# ── Configuration ─────────────────────────────────────────────────────────────
FOV_DEG = 2           # Champ de vue de l'oculaire (degrés)
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


def make_eyepiece_view(target_num, out=None, show_dss_stars=False):
    """Génère la vue simulée dans l'oculaire centrée sur M<target_num>."""
    obj = next(o for o in MESSIER_OBJECTS if o[0] == target_num)
    _, center_ra, center_dec, otype, obj_name = obj

    if out is None:
        out = os.path.join(os.path.dirname(__file__), "..", "img",
                           f"m{target_num}_eyepiece.png")

    print(f"  — récupération des étoiles…")
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

    # ── Objets Messier dans le champ + leur image DSS ────────────────────────
    messier_in_fov = []
    for num, ra, dec, ot, name in MESSIER_OBJECTS:
        dra = (ra - center_ra) * cos_dec
        ddec = dec - center_dec
        if dra**2 + ddec**2 <= half_fov**2:
            obj_size = max(fetch_object_size(num), 5.0)
            print(f"  M{num} — taille : {obj_size:.1f}'")
            print(
                f"    — récupération image DSS2 ({obj_size * DSS_PADDING:.0f}')…")
            dss_data, dss_r = fetch_dss_image(ra, dec, obj_size)
            if dss_data is not None:
                print(
                    f"    → image {dss_data.shape[1]}x{dss_data.shape[0]} px")
            else:
                print("    → pas d'image DSS disponible")
            messier_in_fov.append((num, dra, ddec, ot, name, dss_data, dss_r))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10), facecolor="white",
                           subplot_kw={"aspect": "equal"})
    ax.set_facecolor("white")
    lim = half_fov * 1.02
    top_margin = lim * 0.08   # petite marge en haut pour le titre
    ax.set_xlim(lim, -lim)    # RA croissant vers la gauche (Est)
    ax.set_ylim(-lim, lim + top_margin)
    ax.axis("off")

    # Fond blanc du champ circulaire
    sky = Circle((0, 0), half_fov, facecolor="white", edgecolor="none",
                 zorder=0)
    ax.add_patch(sky)

    # ── Étoiles ───────────────────────────────────────────────────────────────
    if field_stars:
        dra_arr = np.array([s[0] for s in field_stars])
        ddec_arr = np.array([s[1] for s in field_stars])
        mag_arr = np.array([s[2] for s in field_stars])

        x, y = dra_arr, ddec_arr

        sizes = np.clip(8 * np.power(10, (MAG_LIMIT - mag_arr) / 3.5),
                        0.5, 120)
        alphas = np.clip(0.8 + 0.2 * (MAG_LIMIT - mag_arr) / MAG_LIMIT,
                         0.8, 1.0)

        colors = np.zeros((len(x), 4))
        colors[:, :3] = 0.0
        colors[:, 3] = alphas

        ax.scatter(x, y, s=sizes, c=colors, linewidths=0, zorder=2)

    # ── Images DSS incrustées pour chaque objet Messier du champ ────────────
    for num, dra, ddec, ot, name, dss_data, dss_r in messier_in_fov:
        if dss_data is None:
            continue
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(dss_data)
        r = dss_r
        img = ax.imshow(dss_data, cmap="gray_r", origin="lower",
                        extent=[dra + r, dra - r, ddec - r, ddec + r],
                        vmin=vmin, vmax=vmax, zorder=3,
                        interpolation="bicubic")
        # Clip au cercle DSS
        dss_clip = Circle((dra, ddec), r, transform=ax.transData)
        img.set_clip_path(dss_clip)

        # Contour pointillé autour de l'objet
        dss_border = Circle((dra, ddec), r, fill=False,
                            edgecolor="black", linewidth=2,
                            linestyle="--", zorder=5)
        ax.add_patch(dss_border)

        # Étoiles dans le cercle DSS → rouge par-dessus l'image
        if show_dss_stars and field_stars:
            in_dss = [s for s in field_stars
                      if (s[0] - dra)**2 + (s[1] - ddec)**2 <= r**2]
            if in_dss:
                dx = np.array([s[0] for s in in_dss])
                dy = np.array([s[1] for s in in_dss])
                dm = np.array([s[2] for s in in_dss])
                sz = np.clip(8 * np.power(10, (MAG_LIMIT - dm) / 3.5),
                             0.5, 120)
                al = np.clip(0.25 + 0.75 * (MAG_LIMIT - dm) / MAG_LIMIT,
                             0.15, 1.0)
                rc = np.zeros((len(dx), 4))
                rc[:, 0] = 1.0    # rouge
                rc[:, 3] = al
                ax.scatter(dx, dy, s=sz, c=rc, linewidths=0, zorder=4)

    # ── Masque blanc autour du cercle ─────────────────────────────────────────
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

    # ── Labels Messier ────────────────────────────────────────────────────────
    for num, dra, ddec, ot, name, _, dss_r_label in messier_in_fov:
        color = "#333333" if num == target_num else "#666666"
        label_y = ddec + dss_r_label + 0.02
        va = "bottom"
        # Si le label sort du cercle de l'oculaire, on le colle au bord intérieur
        dist = np.sqrt(dra**2 + label_y**2)
        if dist > half_fov:
            label_y = np.sqrt(max(half_fov**2 - dra**2, 0)) - 0.02
            va = "top"
        ax.text(dra, label_y, f"M{num}",
                fontsize=13 if num == target_num else 11,
                color="black", ha="center", va=va,
                fontweight="bold", zorder=10)

    # ── Titre : nom en haut à gauche, type en haut à droite ─────────────────
    title = f"M{target_num}"
    if obj_name:
        title += f"  ({obj_name})"
    type_en = TYPE_LABEL.get(otype, otype)
    ax.text(0.02, 0.99, title, fontsize=20, fontweight="bold",
            color="#333333", ha="left", va="top", transform=ax.transAxes,
            zorder=10)
    ax.text(0.98, 0.99, type_en, fontsize=16,
            color="#555555", ha="right", va="top", transform=ax.transAxes,
            zorder=10)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"  → {out}")
    plt.close(fig)


if __name__ == "__main__":
    for n in range(1, 111):
        make_eyepiece_view(n)
    # make_eyepiece_view(42)
    # make_eyepiece_view(106)
