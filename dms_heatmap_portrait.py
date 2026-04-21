"""
DMS-style heatmap from a portrait photo — PORTRAIT ORIENTATION.

X-axis (top): 20 amino acids + deletion + 2 frameshifts + stop codon (24 total)
Y-axis: residue positions (48, scaled to keep cells square)

Usage:
    python dms_heatmap_portrait.py                  # uses synthetic portrait
    python dms_heatmap_portrait.py your_photo.jpg   # uses your actual image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFilter
import os
import sys

# ── CONFIG ──────────────────────────────────────────────────────────────
GRID_AA = 24          # 20 AAs + del + 2 frameshifts + stop
GRID_POSITIONS = 48   # scaled up from 40 to maintain square cells (48/24 = 40/20)
COLORMAP = "RdBu_r"

# Full x-axis labels: 20 standard AAs, deletion (Δ), frameshifts (+1, -1), stop (*)
AA_LABELS = list("ACDEFGHIKLMNPQRSTVWY") + ["Δ", "+1", "-1", "*"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── LOAD OR GENERATE IMAGE ─────────────────────────────────────────────
def load_image(path=None):
    if path and os.path.exists(path):
        print(f"Loading image from: {path}")
        img = Image.open(path).convert("L")
        return np.array(img, dtype=float)

    print("No image provided — generating synthetic portrait luminance map...")
    return build_synthetic_portrait(size=500)


def build_synthetic_portrait(size=500):
    h = w = size
    img = np.ones((h, w)) * 195

    y, x = np.mgrid[0:h, 0:w]
    cy, cx = h * 0.44, w * 0.50
    yn = (y - cy) / (h * 0.5)
    xn = (x - cx) / (w * 0.5)

    head_r = ((xn / 0.32)**2 + ((yn + 0.02) / 0.42)**2)
    head = np.where(head_r < 1.0, 200 + 30 * (1 - head_r), 0)
    img = np.where(head_r < 1.0, head, img)

    forehead_mask = (yn > -0.32) & (yn < -0.08) & (head_r < 0.7)
    img = np.where(forehead_mask, 225 + 15 * np.exp(-xn**2 / 0.06), img)

    hair_top = (yn < -0.22) & (np.abs(xn) < 0.34) & (yn > -0.50)
    img = np.where(hair_top, 110 + 20 * np.exp(-xn**2 / 0.08), img)

    for sign in [-1, 1]:
        temple = (yn > -0.38) & (yn < -0.22) & (xn * sign > 0.08) & (xn * sign < 0.28)
        img = np.where(temple, 215 + 10 * np.exp(-yn**2 / 0.01), img)
        side = (yn > -0.25) & (yn < 0.05) & (xn * sign > 0.28) & (xn * sign < 0.38)
        img = np.where(side, 115, img)

    thin = (yn < -0.42) & (yn > -0.52) & (np.abs(xn) < 0.25)
    img = np.where(thin, 140 + 30 * np.abs(xn) / 0.25, img)

    gy = -0.06
    gh, gw, gg = 0.055, 0.14, 0.025
    for sign in [-1, 1]:
        lens = (np.abs(yn - gy) < gh) & (xn * sign > gg / 2) & (xn * sign < gw * 2 + gg / 2)
        frame = lens & ((np.abs(yn - gy) > gh * 0.7) |
                        (np.abs(xn - sign * (gw + gg / 4)) > gw * 0.85))
        img = np.where(lens & ~frame, img * 0.85, img)
        img = np.where(frame, 80, img)
    bridge = (np.abs(yn - gy) < 0.015) & (np.abs(xn) < gg)
    img = np.where(bridge, 90, img)

    for ex in [-0.14, 0.14]:
        eye_r = ((xn - ex)**2 / 0.003 + (yn - gy)**2 / 0.001)
        img = np.where(eye_r < 1.0, 100, img)
        brow = (np.abs(yn - (gy - 0.08)) < 0.012) & (np.abs(xn - ex) < 0.09)
        img = np.where(brow, 130, img)

    nose = np.exp(-xn**2 / 0.004 - (yn - 0.06)**2 / 0.015) * 25
    img = np.clip(img + nose, 0, 255)
    for sign in [-1, 1]:
        shadow = np.exp(-(xn - sign * 0.04)**2 / 0.001 - (yn - 0.09)**2 / 0.003) * 25
        img = np.clip(img - shadow, 0, 255)

    lip_mask = (np.abs(yn - 0.16) < 0.018) & (np.abs(xn) < 0.12)
    img = np.where(lip_mask, 140, img)
    teeth = (np.abs(yn - 0.165) < 0.012) & (np.abs(xn) < 0.09)
    img = np.where(teeth, 245, img)

    for sign in [-1, 1]:
        crease = np.exp(-((xn - sign * 0.12)**2 + (yn - 0.10)**2) / 0.005) * 20
        img = np.clip(img - crease, 0, 255)

    chin = (yn > 0.22) & (yn < 0.32) & (np.abs(xn) < 0.18)
    img = np.where(chin, 190 + 15 * np.exp(-xn**2 / 0.02), img)
    neck = (yn > 0.30) & (yn < 0.45) & (np.abs(xn) < 0.12)
    img = np.where(neck, 175, img)

    jacket = (yn > 0.38) & ((np.abs(xn) > 0.10) | (yn > 0.50))
    img = np.where(jacket, 45 + 15 * np.exp(-xn**2 / 0.1), img)
    shoulders = (yn > 0.35) & (yn < 0.55) & (np.abs(xn) > 0.25) & (np.abs(xn) < 0.55)
    img = np.where(shoulders, 50, img)

    collar = (yn > 0.35) & (yn < 0.55) & (np.abs(xn) < 0.05 + (yn - 0.35) * 0.5) & (np.abs(xn) < 0.15)
    img = np.where(collar, 210, img)
    for sign in [-1, 1]:
        lapel = (yn > 0.36) & (yn < 0.48) & (xn * sign > 0.06) & (xn * sign < 0.18) & \
                (yn < 0.36 + (xn * sign - 0.06) * 2.5)
        img = np.where(lapel, 200, img)

    img_pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L')
    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=3))
    img = np.array(img_pil, dtype=float)
    img = np.clip(img + np.random.normal(0, 3, (h, w)), 0, 255)

    return img


# ── DOWNSAMPLE INTO GRID ───────────────────────────────────────────────
def image_to_grid(img, rows, cols):
    h, w = img.shape
    block_h, block_w = h // rows, w // cols
    grid = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = np.mean(img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w])
    return grid


# ── NORMALIZE TO DMS SCORES ───────────────────────────────────────────
def normalize_to_dms_scores(grid):
    g = (grid - grid.min()) / (grid.max() - grid.min())
    return (g - 0.5) * 6


# ── RENDER DMS HEATMAP (PORTRAIT ORIENTATION) ──────────────────────────
def render_dms_heatmap(scores, output_path):
    """
    Portrait orientation:
      x-axis (top): 24 substitution types (20 AAs + del + 2 frameshifts + stop)
      y-axis: residue positions (48 rows)
    """
    rows, cols = scores.shape

    fig, ax = plt.subplots(figsize=(7, 14), dpi=200)

    vmax = np.max(np.abs(scores))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(scores, aspect='auto', cmap=COLORMAP, norm=norm,
                   interpolation='nearest')

    # X-axis (top): all 24 substitution labels
    ax.set_xticks(range(cols))
    ax.set_xticklabels(AA_LABELS[:cols], fontsize=7, fontfamily='monospace',
                       rotation=0)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel("Substitution", fontsize=10, fontweight='bold', labelpad=8)

    # Y-axis: residue positions (every position labeled)
    positions = np.arange(1, rows + 1)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(positions, fontsize=6)
    ax.set_ylabel("Residue Position", fontsize=10, fontweight='bold')

    # Colorbar (horizontal, below)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.04, location='bottom',
                        orientation='horizontal')
    cbar.set_label(r"log$_2$ enrichment score", fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)

    # Title
    ax.set_title("Deep Mutational Scan\nFunctional Landscape",
                 fontsize=13, fontweight='bold', pad=30)

    # Subtle grid
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.3, alpha=0.5)
    ax.tick_params(which='minor', length=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved heatmap: {output_path}")


# ── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    if img_path and os.path.exists(img_path):
        out_dir = os.path.dirname(os.path.abspath(img_path))
    else:
        out_dir = SCRIPT_DIR

    img = load_image(img_path)

    # Save grayscale reference
    ref_path = os.path.join(out_dir, "portrait_grayscale.png")
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L').save(ref_path)
    print(f"Saved grayscale reference: {ref_path}")

    # Downsample: 48 rows (positions) x 24 cols (substitutions)
    grid = image_to_grid(img, GRID_POSITIONS, GRID_AA)
    scores = normalize_to_dms_scores(grid)

    # Render
    heatmap_path = os.path.join(out_dir, "dms_heatmap.png")
    render_dms_heatmap(scores, heatmap_path)

    print(f"\nGrid: {scores.shape}, Score range: [{scores.min():.2f}, {scores.max():.2f}]")
