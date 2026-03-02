#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate architecture diagram for the decoder-only fine-tuning paper.

Shows two paths:
  Top:    Training pipeline (pre-computed embeddings → decoder → loss)
  Bottom: Inference pipeline (image → encoder → decoder → mask)

Run: python scripts/generate_architecture_diagram.py
Output: paper/figures/architecture_diagram.png (and .pdf)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ── Colour palette ──────────────────────────────────────────────────
C_FROZEN   = "#B0C4DE"   # light steel blue  – frozen modules
C_TRAIN    = "#FF8C42"   # warm orange        – trainable modules
C_DATA     = "#90EE90"   # light green        – data / inputs
C_LOSS     = "#FF6B6B"   # coral red          – loss
C_CACHE    = "#DDA0DD"   # plum               – cached embeddings
C_OUTPUT   = "#87CEEB"   # sky blue           – outputs
C_BG       = "#FAFAFA"   # near-white background
C_TEXT     = "#1A1A1A"
C_ARROW    = "#555555"


def rounded_box(ax, xy, w, h, label, color, fontsize=9, bold=False,
                text_color=C_TEXT, alpha=0.92, sublabel=None, icon=None):
    """Draw a rounded rectangle with centred text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="#444444",
        linewidth=1.2, alpha=alpha, zorder=2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    # Main label
    ty = y + h / 2 if sublabel is None else y + h * 0.6
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color, zorder=3)
    # Sublabel (smaller, below main)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                fontsize=fontsize - 2.5, fontstyle="italic", color="#555555", zorder=3)
    # Icon/emoji-style marker top-left
    if icon:
        ax.text(x + 0.12, y + h - 0.18, icon, ha="left", va="top",
                fontsize=fontsize - 2, color="#777777", zorder=3)


def arrow(ax, start, end, color=C_ARROW, style="-|>", lw=1.5, ls="-"):
    """Draw an arrow between two points."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, linestyle=ls,
                                connectionstyle="arc3,rad=0"),
                zorder=4)


def curved_arrow(ax, start, end, color=C_ARROW, rad=0.3, style="-|>", lw=1.3):
    """Draw a curved arrow."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=4)


def dashed_arrow(ax, start, end, color=C_ARROW, lw=1.2):
    arrow(ax, start, end, color=color, lw=lw, ls="--")


def section_label(ax, x, y, text, fontsize=11):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=C_TEXT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#AAAAAA", linewidth=0.8),
            zorder=5)


def legend_entry(ax, x, y, color, label, fontsize=8):
    box = FancyBboxPatch(
        (x, y - 0.12), 0.35, 0.24,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="#666666", linewidth=0.8, alpha=0.9, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + 0.5, y, label, ha="left", va="center",
            fontsize=fontsize, color=C_TEXT, zorder=3)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(14, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ================================================================
    # SECTION A: TRAINING PIPELINE  (y ~ 4.2 – 6.8)
    # ================================================================
    section_label(ax, 0.0, 6.9, "(a) Training Pipeline")

    # Row 1: Pre-compute path (top, dashed = offline)
    rounded_box(ax, (0.3, 5.8), 1.8, 0.8, "Training\nImages", C_DATA, fontsize=8.5,
                sublabel="6,000 + flips")
    rounded_box(ax, (3.0, 5.8), 2.2, 0.8, "Image Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="ViT-H · 632M · frozen")
    rounded_box(ax, (6.2, 5.8), 2.0, 0.8, "Embedding\nCache", C_CACHE, fontsize=9,
                sublabel=".npy files")

    dashed_arrow(ax, (2.1, 6.2), (3.0, 6.2))
    dashed_arrow(ax, (5.2, 6.2), (6.2, 6.2))

    ax.text(4.1, 5.55, "offline pre-compute (run once)", ha="center", va="top",
            fontsize=7, fontstyle="italic", color="#888888")

    # Row 2: Training forward pass
    # Cached embeddings feed into decoder
    rounded_box(ax, (0.3, 3.8), 1.8, 0.8, "Prompt\n(point / box)", C_DATA, fontsize=8.5,
                sublabel="50/50 random")
    rounded_box(ax, (3.0, 3.8), 2.2, 0.8, "Prompt Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="frozen")
    rounded_box(ax, (6.2, 3.8), 2.2, 0.8, "Mask Decoder", C_TRAIN, fontsize=9.5,
                bold=True, sublabel="~4M params · trainable")

    # Arrows: prompt → prompt encoder → decoder
    arrow(ax, (2.1, 4.2), (3.0, 4.2))
    arrow(ax, (5.2, 4.2), (6.2, 4.2))

    # Arrow: cache → decoder (from above)
    arrow(ax, (7.2, 5.8), (7.2, 4.6))

    # Decoder → predicted mask
    rounded_box(ax, (9.2, 3.8), 1.6, 0.8, "Predicted\nMask", C_OUTPUT, fontsize=8.5)
    arrow(ax, (8.4, 4.2), (9.2, 4.2))

    # GT mask
    rounded_box(ax, (9.2, 5.8), 1.6, 0.8, "Ground\nTruth", C_DATA, fontsize=8.5)

    # Loss
    rounded_box(ax, (11.5, 4.8), 1.8, 0.8, "Loss", C_LOSS, fontsize=10,
                bold=True, sublabel="BCE + Dice")
    arrow(ax, (10.8, 4.2), (11.5, 5.0))    # pred → loss
    arrow(ax, (10.8, 6.2), (11.5, 5.6))    # GT → loss

    # Backprop arrow (curved, back to decoder)
    curved_arrow(ax, (11.5, 5.2), (8.4, 4.5), color=C_LOSS, rad=-0.25, lw=1.5)
    ax.text(10.3, 3.4, "backprop", ha="center", va="top",
            fontsize=7.5, fontstyle="italic", color=C_LOSS)

    # Snowflake symbols for frozen
    ax.text(3.15, 6.45, "\u2744", fontsize=10, color="#4477AA", zorder=5)
    ax.text(3.15, 4.45, "\u2744", fontsize=10, color="#4477AA", zorder=5)
    # Fire symbol for trainable
    ax.text(6.35, 4.45, "\U0001F525", fontsize=9, zorder=5)

    # ================================================================
    # SECTION B: INFERENCE PIPELINE  (y ~ 0 – 2.5)
    # ================================================================
    section_label(ax, 0.0, 2.7, "(b) Inference Pipeline")

    rounded_box(ax, (0.3, 1.4), 1.8, 0.8, "Test Image", C_DATA, fontsize=9)
    rounded_box(ax, (3.0, 1.4), 2.2, 0.8, "Image Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="ViT-H · frozen")
    ax.text(3.15, 2.05, "\u2744", fontsize=10, color="#4477AA", zorder=5)

    rounded_box(ax, (0.3, 0.0), 1.8, 0.8, "User Prompt", C_DATA, fontsize=8.5,
                sublabel="point / box")
    rounded_box(ax, (3.0, 0.0), 2.2, 0.8, "Prompt Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="frozen")
    ax.text(3.15, 0.65, "\u2744", fontsize=10, color="#4477AA", zorder=5)

    rounded_box(ax, (6.2, 0.6), 2.2, 0.9, "Specialized\nMask Decoder", C_TRAIN, fontsize=9,
                bold=True, sublabel="fine-tuned · 16MB")
    ax.text(6.35, 1.32, "\U0001F525", fontsize=9, zorder=5)

    rounded_box(ax, (9.2, 0.6), 1.8, 0.9, "Segmentation\nMask", C_OUTPUT, fontsize=9)

    # Arrows
    arrow(ax, (2.1, 1.8), (3.0, 1.8))
    arrow(ax, (5.2, 1.8), (6.6, 1.5))   # encoder → decoder (angled)
    arrow(ax, (2.1, 0.4), (3.0, 0.4))
    arrow(ax, (5.2, 0.4), (6.2, 0.8))   # prompt enc → decoder (angled)
    arrow(ax, (8.4, 1.05), (9.2, 1.05))

    # ================================================================
    # LEGEND
    # ================================================================
    legend_entry(ax, 12.0, 2.5, C_FROZEN, "Frozen", fontsize=8)
    legend_entry(ax, 12.0, 2.0, C_TRAIN, "Trainable", fontsize=8)
    legend_entry(ax, 12.0, 1.5, C_DATA, "Data / Input", fontsize=8)
    legend_entry(ax, 12.0, 1.0, C_CACHE, "Cached", fontsize=8)
    legend_entry(ax, 12.0, 0.5, C_LOSS, "Loss", fontsize=8)

    # ================================================================
    # Divider line
    # ================================================================
    ax.plot([-0.3, 14.3], [3.2, 3.2], color="#CCCCCC", lw=1.0, ls="--", zorder=1)

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ["png", "pdf"]:
        path = f"paper/figures/architecture_diagram.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
