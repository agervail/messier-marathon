"""
dss_fetch.py — Récupère la taille angulaire et l'image DSS d'un objet Messier.

Utilisable en standalone :
    python dss_fetch.py 1          # sauvegarde M1 en FITS + PNG
    python dss_fetch.py 1 42 57    # plusieurs objets

Ou en import :
    from dss_fetch import fetch_object_size, fetch_dss_image
"""

import sys
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad
from astropy.io import fits
from astropy.visualization import ZScaleInterval

from messier_catalog import MESSIER_OBJECTS

# ── Configuration ─────────────────────────────────────────────────────────────
DSS_PIXELS = 500
DSS_PADDING = 1.3
DEFAULT_SIZE_ARCMIN = 10


def fetch_object_size(target_num):
    """Récupère la taille angulaire (arcmin) d'un objet Messier via SIMBAD.

    Retourne DEFAULT_SIZE_ARCMIN si SIMBAD ne renvoie rien d'exploitable.
    """
    s = Simbad()
    s.add_votable_fields("dim")
    result = s.query_object(f"M {target_num}")
    if result is None:
        return DEFAULT_SIZE_ARCMIN
    try:
        maj = float(result["galdim_majaxis"][0])
        if np.isnan(maj) or maj <= 0:
            return DEFAULT_SIZE_ARCMIN
        return maj
    except (KeyError, TypeError, ValueError, IndexError):
        return DEFAULT_SIZE_ARCMIN


ALL_SURVEYS = ["DSS", "DSS1 Blue", "DSS1 Red", "DSS2 Blue", "DSS2 Red", "DSS2 IR"]


def fetch_dss_image(center_ra, center_dec, size_arcmin,
                    padding=DSS_PADDING, pixels=DSS_PIXELS, survey=None):
    """Télécharge une image DSS depuis SkyView.

    Paramètres
    ----------
    center_ra, center_dec : float
        Coordonnées J2000 en degrés.
    size_arcmin : float
        Taille angulaire de l'objet en arcmin.
    padding : float
        Facteur de marge autour de l'objet (défaut 1.3).
    pixels : int
        Résolution de l'image en pixels (défaut 500).
    survey : str ou None
        Survey spécifique (ex: "DSS2 Blue"). Si None, essaie tous les
        surveys dans l'ordre DSS → DSS2 Blue → DSS2 Red.

    Retourne
    --------
    (data, radius_deg) : (np.ndarray, float) ou (None, 0)
        L'image 2D et le rayon du cutout en degrés.
    """
    coord = SkyCoord(ra=center_ra, dec=center_dec, unit="deg", frame="icrs")
    pos_str = coord.to_string("hmsdms")

    max_radius = 0.5
    radius_deg = min(size_arcmin * padding / 60.0 / 2.0, max_radius)

    surveys = [survey] if survey else ALL_SURVEYS
    for s in surveys:
        # Certaines tailles exactes tombent sur un bug serveur SkyView
        # (limite de plaque) → on essaie avec un léger jitter du rayon.
        for r in [radius_deg, radius_deg * 1.03, radius_deg * 0.97]:
            try:
                images = SkyView.get_images(position=pos_str,
                                            survey=[s],
                                            radius=r * u.deg,
                                            pixels=pixels)
                if not images:
                    continue
                hdu = images[0][0]
                if _is_mosaic(hdu.header):
                    break  # multi-plaque → passer au survey suivant
                return hdu.data, r
            except Exception:
                continue
    return None, 0


def _is_mosaic(header):
    """Vérifie dans les headers FITS si l'image est une mosaïque multi-plaques."""
    n_plates = sum(1 for k, v in header.items()
                   if k == "HISTORY" and "Used image" in str(v))
    return n_plates > 1


def fetch_messier_dss(target_num, **kwargs):
    """Raccourci : récupère la taille puis l'image DSS d'un objet Messier.

    Retourne (data, radius_deg, size_arcmin).
    """
    obj = next((o for o in MESSIER_OBJECTS if o[0] == target_num), None)
    if obj is None:
        raise ValueError(f"M{target_num} introuvable dans le catalogue")

    _, ra, dec, _, _ = obj
    size = fetch_object_size(target_num)
    data, radius_deg = fetch_dss_image(ra, dec, size, **kwargs)
    return data, radius_deg, size


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Récupère les images DSS d'objets Messier.")
    parser.add_argument("nums", type=int, nargs="+",
                        help="Numéros Messier (ex: 1 42 57)")
    parser.add_argument("--compare", action="store_true",
                        help="Affiche les 3 surveys côte à côte")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "result")
    os.makedirs(out_dir, exist_ok=True)

    for num in args.nums:
        obj = next((o for o in MESSIER_OBJECTS if o[0] == num), None)
        if obj is None:
            print(f"M{num} : introuvable dans le catalogue")
            continue

        _, ra, dec, _, name = obj
        size = fetch_object_size(num)
        label = f"M{num}"
        if name:
            label += f" ({name})"
        print(f"{label} — taille : {size:.1f}'")

        if args.compare:
            n = len(ALL_SURVEYS)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            for ax, survey in zip(axes.flat, ALL_SURVEYS):
                data, r = fetch_dss_image(ra, dec, size, survey=survey)
                if data is None:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            fontsize=20, transform=ax.transAxes)
                    ax.set_title(survey, fontsize=14)
                    ax.axis("off")
                    continue
                interval = ZScaleInterval()
                vmin, vmax = interval.get_limits(data)
                ax.imshow(data, cmap="gray", origin="lower",
                          vmin=vmin, vmax=vmax)
                ax.set_title(survey, fontsize=14)
                ax.axis("off")
            for ax in axes.flat[len(ALL_SURVEYS):]:
                ax.axis("off")
            plt.suptitle(f"{label} — {size:.1f}'", fontsize=16,
                         fontweight="bold")
            plt.tight_layout()
            out = os.path.join(out_dir, f"m{num}_dss_compare.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {out}")
        else:
            data, radius_deg, size = fetch_messier_dss(num)
            if data is None:
                print("  pas d'image DSS disponible")
                continue
            print(f"  image {data.shape[1]}x{data.shape[0]} px, "
                  f"rayon {radius_deg:.4f}°")

            fits_path = os.path.join(out_dir, f"m{num}_dss.fits")
            hdu = fits.PrimaryHDU(data)
            hdu.writeto(fits_path, overwrite=True)
            print(f"  → {fits_path}")

            png_path = os.path.join(out_dir, f"m{num}_dss.png")
            interval = ZScaleInterval()
            vmin, vmax = interval.get_limits(data)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data, cmap="gray", origin="lower",
                      vmin=vmin, vmax=vmax)
            ax.set_title(f"{label} — DSS ({size:.1f}')")
            ax.axis("off")
            plt.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  → {png_path}")
