#!/usr/bin/env python
"""Figure 1: SOPNet architecture schematic (encoder--decoder signed field)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURE_DIR = Path("outputs/paper/figures")

ENCODER_COLOR = "#dce9f7"
BOTTLENECK_COLOR = "#f7e6d5"
DECODER_COLOR = "#dff2e1"
HEAD_COLOR = "#f2d9e6"
TEXT_KW = dict(ha="center", va="center", fontsize=5.9)


def wave(start: float, end: float, y_center: float, height: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, 400)
    onset = 0.55
    mask = (t >= onset).astype(float)
    bump = -mask * np.exp(-((t - onset) * 12.0)) * np.sin((t - onset) * 55.0)
    peak = np.max(np.abs(bump))
    trace = 0.06 * rng.standard_normal(t.size) + 0.9 * bump / peak
    x = start + t * (end - start)
    return x, y_center + height * trace


def field_curve(start: float, end: float, y_center: float, height: float):
    t = np.linspace(0.0, 1.0, 400)
    g = -np.exp(-((t - 0.45) ** 2) / (2 * 0.05**2))
    x = start + t * (end - start)
    return x, y_center + height * g


def box(ax, x, y, w, h, text, color=ENCODER_COLOR, fontsize=5.9):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=0.8,
            edgecolor="#444444",
            facecolor=color,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, dashed=False, color="#333333"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.8,
            linestyle="--" if dashed else "-",
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def skip_arc(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle="arc3,rad=-0.28",
            arrowstyle="-|>",
            mutation_scale=5,
            linewidth=0.7,
            linestyle="--",
            color="#999999",
            shrinkA=0,
            shrinkB=0,
        )
    )


def main() -> None:
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=200)
    ax.set_xlim(0, 19.3)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    y_row, h = 3.95, 0.95
    y_dec = 2.05

    # input waveform
    x_wave, y_wave = wave(0.15, 1.85, y_row + h / 2, 0.55)
    ax.plot(x_wave, y_wave, color="#1f4e79", linewidth=0.8)
    ax.text(1.0, y_row + h + 0.35, "input waveform $x$ (400 samples)", ha="center", fontsize=5.9)

    # encoder
    arrow(ax, 1.9, y_row + h / 2, 2.15, y_row + h / 2)
    box(ax, 2.15, y_row, 1.75, h, "multi-scale\nstem, 48 ch", fontsize=5.5)
    stages = [
        "down $\\times2$\n+ res block\n64 ch",
        "down $\\times2$\n+ res block\n96 ch",
        "down $\\times2$\n+ res block\n128 ch",
    ]
    stage_x = [4.1, 6.05, 8.0]
    for x, text in zip(stage_x, stages):
        arrow(ax, x - 0.2, y_row + h / 2, x, y_row + h / 2)
        box(ax, x, y_row, 1.75, h, text, fontsize=5.5)

    # bottleneck
    bx, bw = 9.95, 2.9
    arrow(ax, 9.75, y_row + h / 2, bx, y_row + h / 2)
    box(
        ax,
        bx,
        y_row,
        bw,
        h,
        "4$\\times$ dilated residual\n192 ch, $d=1,2,4,8$",
        BOTTLENECK_COLOR,
        fontsize=5.5,
    )
    ax.text(bx - 0.45, y_row - 0.30, "resolution $T/8$", ha="center", fontsize=5.5, color="#555555")

    # decoder
    dec_x = [10.15, 11.95, 13.75]
    dec_texts = ["up + fuse\n96 ch", "up + fuse\n64 ch", "up + fuse\n48 ch"]
    arrow(ax, 10.45, y_row, 10.45, y_dec + h)
    for i, (x, text) in enumerate(zip(dec_x, dec_texts)):
        if i > 0:
            arrow(ax, x - 0.25, y_dec + h / 2, x, y_dec + h / 2)
        box(ax, x, y_dec, 1.55, h, text, DECODER_COLOR)

    # skip connections
    skip_arc(ax, 3.03, y_row, 10.15 + 0.78, y_dec + h)
    skip_arc(ax, 4.98, y_row, 11.95 + 0.78, y_dec + h)
    skip_arc(ax, 6.93, y_row, 13.75 + 0.78, y_dec + h)
    ax.text(7.6, 3.05, "skip connections", fontsize=5.5, color="#888888", ha="center")

    # head + output field
    arrow(ax, 13.75 + 1.55, y_dec + h / 2, 15.75, y_dec + h / 2)
    box(ax, 15.75, y_dec, 1.45, h, "1$\\times1$ conv\n+ $\\tanh$", HEAD_COLOR)
    arrow(ax, 17.2, y_dec + h / 2, 17.5, y_dec + h / 2)
    f_x, f_y = field_curve(17.5, 19.2, y_dec + h / 2, 0.85)
    ax.axhline(
        y_dec + h / 2,
        xmin=17.5 / 19.3,
        xmax=19.2 / 19.3,
        color="#cccccc",
        linewidth=0.5,
        linestyle=":",
    )
    ax.plot(f_x, f_y, color="#7b2d26", linewidth=0.9)
    ax.text(18.35, y_dec - 0.28, "signed field $f(t)$", ha="center", fontsize=5.9)

    # row labels removed; boxes are self-describing

    # decoding annotation
    ax.text(
        9.4,
        0.55,
        "decode:  $i^{\\star}=\\arg\\max_t |f(t)|$,   "
        "$\\hat{y}=\\mathrm{sign}\\,f(i^{\\star})$,   "
        "confidence $=|f(i^{\\star})|$",
        fontsize=6.8,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f5f5", edgecolor="#bbbbbb"),
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURE_DIR / "fig1_architecture"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.with_suffix(".pdf"), "and", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
