import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bright_stars import BRIGHT_STARS
from messier_catalog import MESSIER_OBJECTS
from constellations import get_boundaries, FRENCH_NAMES
from constellation_lines import get_lines

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


def _star_size(mag):
    """Return scatter marker size from visual magnitude (brighter → larger)."""
    return np.clip(10 * np.power(10, (4.5 - mag) / 5), 2, 300)


def make_sky_map():
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

    out = "messier_sky_map.png"
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Image sauvegardée : {out}")
    plt.close(fig)


def main():
    make_sky_map()


if __name__ == "__main__":
    main()
