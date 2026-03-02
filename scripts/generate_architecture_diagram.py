#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate architecture diagram for the Special-SAM paper.

Shows three sections:
  (a) Decoder-Only Training Pipeline (pre-computed embeddings → decoder → loss)
  (b) LLPM + Decoder Training Pipeline (image → LLPM → encoder → decoder → loss)
  (c) Inference Pipeline (with optional LLPM path)

Run: python scripts/generate_architecture_diagram.py
Output: paper/figures/architecture_diagram.png (and .pdf)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# ── Colour palette ──────────────────────────────────────────────────
C_FROZEN   = "#B0C4DE"   # light steel blue  – frozen modules
C_TRAIN    = "#FF8C42"   # warm orange        – trainable modules
C_DATA     = "#90EE90"   # light green        – data / inputs
C_LOSS     = "#FF6B6B"   # coral red          – loss
C_CACHE    = "#DDA0DD"   # plum               – cached embeddings
C_OUTPUT   = "#87CEEB"   # sky blue           – outputs
C_LLPM     = "#FFD700"   # gold               – LLPM module (novel)
C_TEXT     = "#1A1A1A"
C_ARROW    = "#555555"


def rounded_box(ax, xy, w, h, label, color, fontsize=9, bold=False,
                text_color=C_TEXT, alpha=0.92, sublabel=None):
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
    ty = y + h / 2 if sublabel is None else y + h * 0.6
    ax.text(x + w / 2, ty, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=text_color, zorder=3)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.28, sublabel, ha="center", va="center",
                fontsize=fontsize - 2.5, fontstyle="italic", color="#555555", zorder=3)


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
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(-0.5, 11.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # ================================================================
    # SECTION A: DECODER-ONLY TRAINING  (y ~ 8.5 – 11)
    # ================================================================
    section_label(ax, 0.0, 11.0, "(a) Decoder-Only Training")

    # Row 1: Pre-compute path (top, dashed = offline)
    rounded_box(ax, (0.3, 9.8), 1.8, 0.8, "Training\nImages", C_DATA, fontsize=8.5,
                sublabel="6,000 + flips")
    rounded_box(ax, (3.0, 9.8), 2.2, 0.8, "Image Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="ViT-H \u00b7 632M \u00b7 frozen")
    rounded_box(ax, (6.2, 9.8), 2.0, 0.8, "Embedding\nCache", C_CACHE, fontsize=9,
                sublabel=".npy files")

    dashed_arrow(ax, (2.1, 10.2), (3.0, 10.2))
    dashed_arrow(ax, (5.2, 10.2), (6.2, 10.2))

    ax.text(4.1, 9.55, "offline pre-compute (run once)", ha="center", va="top",
            fontsize=7, fontstyle="italic", color="#888888")

    # Row 2: Training forward pass
    rounded_box(ax, (0.3, 7.8), 1.8, 0.8, "Prompt\n(point / box)", C_DATA, fontsize=8.5,
                sublabel="50/50 random")
    rounded_box(ax, (3.0, 7.8), 2.2, 0.8, "Prompt Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="frozen")
    rounded_box(ax, (6.2, 7.8), 2.2, 0.8, "Mask Decoder", C_TRAIN, fontsize=9.5,
                bold=True, sublabel="~4M params \u00b7 trainable")

    arrow(ax, (2.1, 8.2), (3.0, 8.2))
    arrow(ax, (5.2, 8.2), (6.2, 8.2))
    arrow(ax, (7.2, 9.8), (7.2, 8.6))  # cache → decoder

    # Decoder → predicted mask
    rounded_box(ax, (9.2, 7.8), 1.6, 0.8, "Predicted\nMask", C_OUTPUT, fontsize=8.5)
    arrow(ax, (8.4, 8.2), (9.2, 8.2))

    # GT mask
    rounded_box(ax, (9.2, 9.8), 1.6, 0.8, "Ground\nTruth", C_DATA, fontsize=8.5)

    # Loss
    rounded_box(ax, (11.5, 8.8), 1.8, 0.8, "Loss", C_LOSS, fontsize=10,
                bold=True, sublabel="BCE + Dice")
    arrow(ax, (10.8, 8.2), (11.5, 9.0))    # pred → loss
    arrow(ax, (10.8, 10.2), (11.5, 9.6))   # GT → loss

    # Backprop arrow (curved, back to decoder only)
    curved_arrow(ax, (11.5, 9.2), (8.4, 8.5), color=C_LOSS, rad=-0.25, lw=1.5)
    ax.text(10.3, 7.55, "backprop (decoder only)", ha="center", va="top",
            fontsize=7, fontstyle="italic", color=C_LOSS)

    # Snowflake / fire symbols
    ax.text(3.15, 10.45, "\u2744", fontsize=10, color="#4477AA", zorder=5)
    ax.text(3.15, 8.45, "\u2744", fontsize=10, color="#4477AA", zorder=5)
    ax.text(6.35, 8.45, "\U0001F525", fontsize=9, zorder=5)

    # ── Divider ──
    ax.plot([-0.3, 14.8], [7.2, 7.2], color="#CCCCCC", lw=1.0, ls="--", zorder=1)

    # ================================================================
    # SECTION B: LLPM + DECODER TRAINING  (y ~ 4.0 – 7.0)
    # ================================================================
    section_label(ax, 0.0, 6.9, "(b) LLPM + Decoder Training (Ours)")

    # Main forward path: Image → LLPM → Normalize → Encoder → Decoder → Mask
    rounded_box(ax, (0.3, 5.5), 1.5, 0.8, "Training\nImages", C_DATA, fontsize=8,
                sublabel="6,000")
    rounded_box(ax, (2.3, 5.5), 1.6, 0.8, "LLPM", C_LLPM, fontsize=9.5,
                bold=True, sublabel="~31K \u00b7 trainable")
    rounded_box(ax, (4.5, 5.5), 1.6, 0.8, "Normalize\n& Clamp", C_DATA, fontsize=7.5)
    rounded_box(ax, (6.7, 5.5), 2.0, 0.8, "Image Encoder", C_FROZEN, fontsize=8.5,
                bold=True, sublabel="ViT-H \u00b7 frozen")
    rounded_box(ax, (9.3, 5.5), 2.0, 0.8, "Mask Decoder", C_TRAIN, fontsize=9,
                bold=True, sublabel="~4M \u00b7 trainable")

    arrow(ax, (1.8, 5.9), (2.3, 5.9))
    arrow(ax, (3.9, 5.9), (4.5, 5.9))
    arrow(ax, (6.1, 5.9), (6.7, 5.9))
    arrow(ax, (8.7, 5.9), (9.3, 5.9))

    # Prompt path below
    rounded_box(ax, (0.3, 4.2), 1.5, 0.8, "Prompt\n(point / box)", C_DATA, fontsize=7.5,
                sublabel="50/50 random")
    rounded_box(ax, (2.5, 4.2), 2.0, 0.8, "Prompt Encoder", C_FROZEN, fontsize=8.5,
                bold=True, sublabel="frozen")
    arrow(ax, (1.8, 4.6), (2.5, 4.6))
    arrow(ax, (4.5, 4.6), (9.3, 5.5))  # prompt enc → decoder (angled up)

    # Decoder → Predicted Mask
    rounded_box(ax, (11.8, 5.5), 1.4, 0.8, "Predicted\nMask", C_OUTPUT, fontsize=8)
    arrow(ax, (11.3, 5.9), (11.8, 5.9))

    # GT mask
    rounded_box(ax, (11.8, 4.2), 1.4, 0.8, "Ground\nTruth", C_DATA, fontsize=8)

    # Loss
    rounded_box(ax, (13.5, 4.8), 1.3, 0.8, "Loss", C_LOSS, fontsize=9.5,
                bold=True, sublabel="BCE+Dice")
    arrow(ax, (13.2, 5.9), (13.5, 5.4))   # pred → loss
    arrow(ax, (13.2, 4.6), (13.5, 5.0))   # GT → loss

    # Backprop arrow — goes all the way back to LLPM through frozen encoder
    curved_arrow(ax, (13.5, 5.2), (3.9, 5.7), color=C_LOSS, rad=-0.35, lw=1.5)
    ax.text(8.5, 4.0, "backprop through frozen encoder to LLPM + decoder",
            ha="center", va="top", fontsize=7, fontstyle="italic", color=C_LOSS)

    # Symbols
    ax.text(2.45, 6.15, "\U0001F525", fontsize=9, zorder=5)   # LLPM trainable
    ax.text(6.85, 6.15, "\u2744", fontsize=10, color="#4477AA", zorder=5)  # encoder frozen
    ax.text(9.45, 6.15, "\U0001F525", fontsize=9, zorder=5)   # decoder trainable
    ax.text(2.65, 4.85, "\u2744", fontsize=10, color="#4477AA", zorder=5)  # prompt frozen

    # LLPM internal detail annotation
    ax.text(3.1, 6.55, "Edge Branch + Enhancement Branch + Residual",
            ha="center", va="center", fontsize=6.5, fontstyle="italic",
            color="#666666", zorder=5)

    # ── Divider ──
    ax.plot([-0.3, 14.8], [3.5, 3.5], color="#CCCCCC", lw=1.0, ls="--", zorder=1)

    # ================================================================
    # SECTION C: INFERENCE PIPELINE  (y ~ 0 – 3.2)
    # ================================================================
    section_label(ax, 0.0, 3.2, "(c) Inference Pipeline")

    # LLPM inference path (full)
    rounded_box(ax, (0.3, 1.6), 1.5, 0.8, "Test Image", C_DATA, fontsize=9)
    rounded_box(ax, (2.3, 1.6), 1.6, 0.8, "LLPM", C_LLPM, fontsize=9.5,
                bold=True, sublabel="~31K params")
    rounded_box(ax, (4.5, 1.6), 2.0, 0.8, "Image Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="ViT-H \u00b7 frozen")
    rounded_box(ax, (7.1, 1.6), 2.2, 0.8, "Specialized\nMask Decoder", C_TRAIN, fontsize=8.5,
                bold=True, sublabel="fine-tuned")
    rounded_box(ax, (10.0, 1.6), 1.8, 0.8, "Segmentation\nMask", C_OUTPUT, fontsize=8.5)

    arrow(ax, (1.8, 2.0), (2.3, 2.0))
    arrow(ax, (3.9, 2.0), (4.5, 2.0))
    arrow(ax, (6.5, 2.0), (7.1, 2.0))
    arrow(ax, (9.3, 2.0), (10.0, 2.0))

    # Prompt path
    rounded_box(ax, (0.3, 0.2), 1.5, 0.8, "User Prompt", C_DATA, fontsize=8.5,
                sublabel="point / box")
    rounded_box(ax, (2.5, 0.2), 2.0, 0.8, "Prompt Encoder", C_FROZEN, fontsize=9,
                bold=True, sublabel="frozen")
    arrow(ax, (1.8, 0.6), (2.5, 0.6))
    arrow(ax, (4.5, 0.6), (7.1, 1.6))  # prompt enc → decoder (angled up)

    # Symbols
    ax.text(2.45, 2.25, "\U0001F525", fontsize=9, zorder=5)
    ax.text(4.65, 2.25, "\u2744", fontsize=10, color="#4477AA", zorder=5)
    ax.text(7.25, 2.25, "\U0001F525", fontsize=9, zorder=5)
    ax.text(2.65, 0.85, "\u2744", fontsize=10, color="#4477AA", zorder=5)

    # ================================================================
    # LEGEND
    # ================================================================
    legend_entry(ax, 12.5, 3.0, C_FROZEN, "Frozen", fontsize=8)
    legend_entry(ax, 12.5, 2.5, C_TRAIN, "Trainable", fontsize=8)
    legend_entry(ax, 12.5, 2.0, C_LLPM, "LLPM (novel)", fontsize=8)
    legend_entry(ax, 12.5, 1.5, C_DATA, "Data / Input", fontsize=8)
    legend_entry(ax, 12.5, 1.0, C_CACHE, "Cached", fontsize=8)
    legend_entry(ax, 12.5, 0.5, C_LOSS, "Loss", fontsize=8)

    # ── Save ────────────────────────────────────────────────────────
    os.makedirs("paper/figures", exist_ok=True)
    for ext in ["png", "pdf"]:
        path = f"paper/figures/architecture_diagram.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
