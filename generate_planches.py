#!/usr/bin/env python3
"""
Génère des planches A4 (300 DPI) regroupant les images du dossier img/
en grilles de 6 lignes x 4 colonnes (24 images par planche).
"""

import os
import re
from PIL import Image, ImageDraw, ImageFont

# --- Configuration ---
IMG_DIR = os.path.join(os.path.dirname(__file__), "img")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "planches")
COLS = 4
ROWS = 6
DPI = 300
# A4 en pixels à 300 DPI (210mm x 297mm)
A4_WIDTH = int(210 / 25.4 * DPI)   # 2480
A4_HEIGHT = int(297 / 25.4 * DPI)  # 3508
MARGIN = 40  # marge extérieure en pixels
PADDING = 20  # espace entre images
LABEL_HEIGHT = 30  # espace pour le nom sous chaque image
BG_COLOR = (255, 255, 255)


def natural_sort_key(filename):
    """Tri naturel : m1, m2, ... m10, m11, ..."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', filename)]


def generate_planches():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=natural_sort_key,
    )
    print(f"{len(files)} images trouvées dans {IMG_DIR}/")

    # Zone utile pour la grille
    grid_w = A4_WIDTH - 2 * MARGIN
    grid_h = A4_HEIGHT - 2 * MARGIN

    # Taille de chaque cellule
    cell_w = (grid_w - (COLS - 1) * PADDING) // COLS
    cell_h = (grid_h - (ROWS - 1) * PADDING) // ROWS
    thumb_h = cell_h - LABEL_HEIGHT  # hauteur dispo pour l'image

    per_page = ROWS * COLS
    num_pages = (len(files) + per_page - 1) // per_page

    for page_idx in range(num_pages):
        batch = files[page_idx * per_page : (page_idx + 1) * per_page]
        planche = Image.new("RGB", (A4_WIDTH, A4_HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(planche)

        for i, fname in enumerate(batch):
            row, col = divmod(i, COLS)
            x = MARGIN + col * (cell_w + PADDING)
            y = MARGIN + row * (cell_h + PADDING)

            img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
            # Redimensionner en conservant le ratio
            img.thumbnail((cell_w, thumb_h), Image.LANCZOS)
            # Centrer dans la cellule
            offset_x = x + (cell_w - img.width) // 2
            offset_y = y + (thumb_h - img.height) // 2
            planche.paste(img, (offset_x, offset_y))

            # Label sous l'image
            label = fname.replace("_eyepiece.png", "").replace("_eyepiece.jpg", "").upper()
            bbox = draw.textbbox((0, 0), label)
            tw = bbox[2] - bbox[0]
            draw.text((x + (cell_w - tw) // 2, y + thumb_h + 4), label, fill=(0, 0, 0))

        out_path = os.path.join(OUTPUT_DIR, f"planche_{page_idx + 1:02d}.png")
        planche.save(out_path, dpi=(DPI, DPI))
        print(f"  -> {out_path} ({len(batch)} images)")

    print(f"\n{num_pages} planche(s) générée(s) dans {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_planches()
