#!/usr/bin/env python
"""Render the two-panel SOPNet robustness heatmap from measured CSV results.

Panel (a) shows strict U/D accuracy under additive white Gaussian noise at
fixed SNR levels. Panel (b) shows the same metric as the reference P-arrival
window is shifted. Both panels share one colour scale, and every annotated
value is read directly from ``snr_all.csv`` / ``p_shift_all.csv``. Missing
model-condition combinations are rendered as ``NA`` rather than filled.

Example
-------
python scripts/plot_paper_robustness_heatmaps.py \
    --snr paper/results/tables/snr_all.csv \
    --p-shift paper/results/tables/p_shift_all.csv \
    --output-dir paper/results/figures
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

MODEL_ORDER: tuple[str, ...] = (
    "sopnet",
    "cfm",
    "rpnet",
    "ross",
    "eqpolarity",
    "diting_motion",
)
MODEL_LABELS: dict[str, str] = {
    "sopnet": "SOPNet",
    "cfm": "CFM",
    "rpnet": "RPNet",
    "ross": "Ross",
    "eqpolarity": "EQPolarity",
    "diting_motion": "DiTingMotion",
}

# Publication grids. The fine P-shift grid is a superset of the coarse grid.
REQUIRED_SNR_DB: tuple[float, ...] = (float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0)
REQUIRED_SHIFT_SECONDS: tuple[float, ...] = (
    -0.20,
    -0.10,
    -0.05,
    -0.03,
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
    0.03,
    0.05,
    0.10,
    0.20,
)

# Times New Roman is preferred; Nimbus Roman is its metric-compatible URW
# clone and is available in this environment. DejaVu Serif is the last resort.
FONT_FAMILIES: tuple[str, ...] = ("Nimbus Roman", "Times New Roman", "DejaVu Serif")

# Muted blue-green sequential ramp: monotonically darkening, low saturation,
# and safe for the common forms of colour-vision deficiency.
CMAP_COLORS: tuple[str, ...] = (
    "#f4f3ee",
    "#dfe8e3",
    "#c3d8d3",
    "#9ac3c0",
    "#6ca5aa",
    "#4a8292",
    "#356475",
    "#244857",
)
FIGURE_WIDTH_CM = 17.0
FIGURE_HEIGHT_IN = 6.1


@dataclass(frozen=True)
class HeatmapData:
    """Accuracy matrices aligned to a common model and condition order."""

    models: tuple[str, ...]
    snr_db: tuple[float, ...]
    shift_seconds: tuple[float, ...]
    snr_accuracy: np.ndarray
    shift_accuracy: np.ndarray


def _ordered_snr(values: set[float]) -> tuple[float, ...]:
    infinite = sorted(value for value in values if np.isposinf(value))
    finite = sorted((value for value in values if np.isfinite(value)), reverse=True)
    return tuple(infinite + finite)


def _read_accuracy_table(
    path: Path,
    condition_column: str,
    required_conditions: Sequence[float],
) -> tuple[tuple[float, ...], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"result table not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"model", "accuracy", condition_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path}: missing columns {missing_columns}")
    if frame.empty:
        raise ValueError(f"{path}: table is empty")

    frame = frame.copy()
    frame["model"] = frame["model"].astype(str).str.strip().str.lower()
    unknown_models = sorted(set(frame["model"]) - set(MODEL_ORDER))
    if unknown_models:
        raise ValueError(f"{path}: unexpected models {unknown_models}")

    duplicated = frame.duplicated(["model", condition_column])
    if duplicated.any():
        duplicates = frame.loc[duplicated, ["model", condition_column]].to_dict("records")
        raise ValueError(f"{path}: duplicate model-condition rows: {duplicates}")

    accuracy = pd.to_numeric(frame["accuracy"], errors="coerce")
    invalid = accuracy.isna() | (accuracy < 0.0) | (accuracy > 1.0)
    if invalid.any():
        raise ValueError(f"{path}: accuracy must be a fraction in [0, 1]")
    frame["accuracy"] = accuracy

    values = {float(value) for value in frame[condition_column]}
    missing_conditions = [value for value in required_conditions if value not in values]
    if missing_conditions:
        raise ValueError(f"{path}: missing grid conditions {missing_conditions}")

    if condition_column == "snr_db":
        conditions = _ordered_snr(values)
    else:
        conditions = tuple(sorted(values))

    pivot = frame.pivot(index="model", columns=condition_column, values="accuracy")
    pivot = pivot.reindex(index=list(MODEL_ORDER), columns=list(conditions))
    return conditions, pivot.to_numpy(dtype=float)


def load_heatmap_data(snr_csv: Path, p_shift_csv: Path) -> HeatmapData:
    """Load and validate the two robustness tables used by the figure."""

    snr_db, snr_accuracy = _read_accuracy_table(Path(snr_csv), "snr_db", REQUIRED_SNR_DB)
    shift_seconds, shift_accuracy = _read_accuracy_table(
        Path(p_shift_csv), "shift_seconds", REQUIRED_SHIFT_SECONDS
    )
    return HeatmapData(
        models=MODEL_ORDER,
        snr_db=snr_db,
        shift_seconds=shift_seconds,
        snr_accuracy=snr_accuracy,
        shift_accuracy=shift_accuracy,
    )


def _configure_style() -> str:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": list(FONT_FAMILIES),
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    return font_manager.findfont(font_manager.FontProperties(family="serif"))


def _relative_luminance(rgb: Sequence[float]) -> float:
    def linearise(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (linearise(float(value)) for value in rgb[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _text_colour(rgba: Sequence[float]) -> str:
    luminance = _relative_luminance(rgba)
    white_contrast = 1.05 / (luminance + 0.05)
    dark_contrast = (luminance + 0.05) / 0.05
    return "#ffffff" if white_contrast >= dark_contrast else "#1b1b1b"


def _snr_labels(values: Sequence[float]) -> list[str]:
    return ["Clean" if np.isposinf(value) else f"{value:g}" for value in values]


def _shift_labels(values: Sequence[float]) -> list[str]:
    return [f"{int(round(value * 1000.0))}" for value in values]


def _accuracy_scale(snr_percent: np.ndarray, shift_percent: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate(
        [
            snr_percent[np.isfinite(snr_percent)],
            shift_percent[np.isfinite(shift_percent)],
        ]
    )
    if finite.size == 0:
        raise ValueError("no finite accuracy values to plot")
    lower = max(0.0, 5.0 * np.floor(finite.min() / 5.0))
    upper = min(100.0, 5.0 * np.ceil(finite.max() / 5.0))
    if upper <= lower:
        upper = min(100.0, lower + 5.0)
    return float(lower), float(upper)


def _draw_panel(
    axis,
    values_percent: np.ndarray,
    *,
    x_labels: Sequence[str],
    title: str,
    xlabel: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
):
    masked = np.ma.masked_invalid(values_percent)
    image = axis.imshow(
        masked,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    rows, columns = values_percent.shape
    axis.set_xticks(range(columns), x_labels, fontsize=9.0)
    axis.set_yticks(
        range(rows),
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        fontsize=9.0,
    )
    axis.tick_params(axis="both", which="major", length=0, pad=3)
    axis.set_title(title, loc="left", fontsize=10.0, fontweight="bold", pad=6)
    axis.set_xlabel(xlabel, fontsize=9.5, labelpad=5)
    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.set_xticks(np.arange(-0.5, columns, 1.0), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1.0), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.9)
    axis.tick_params(which="minor", length=0)

    for row in range(rows):
        for column in range(columns):
            value = values_percent[row, column]
            if not np.isfinite(value):
                label, colour = "NA", "#555555"
            else:
                label = f"{value:.1f}"
                colour = _text_colour(cmap(norm(value)))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9.0,
                color=colour,
            )
    return image


def _strip_trailing_whitespace(path: Path) -> None:
    """Remove insignificant trailing blanks from a generated text file.

    Matplotlib wraps long SVG path coordinates across lines and leaves a space
    before some newlines.  Those spaces are valid separators but create noisy
    git diffs, so normalise them without changing the rendered geometry.
    """

    text = path.read_text(encoding="utf-8")
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(cleaned, encoding="utf-8")


def build_figure(
    data: HeatmapData,
    output_dir: Path,
    *,
    width_cm: float = FIGURE_WIDTH_CM,
    dpi: int = 600,
) -> dict[str, Path]:
    """Render and save the combined figure in PDF, SVG and 600 dpi PNG."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()

    snr_percent = data.snr_accuracy * 100.0
    shift_percent = data.shift_accuracy * 100.0
    lower, upper = _accuracy_scale(snr_percent, shift_percent)
    norm = Normalize(vmin=lower, vmax=upper)
    cmap = LinearSegmentedColormap.from_list("sopnet_muted_teal", CMAP_COLORS, N=256)
    cmap.set_bad("#d9d9d9")

    figure = plt.figure(figsize=(width_cm / 2.54, FIGURE_HEIGHT_IN))
    figure.patch.set_facecolor("white")
    top_axis = figure.add_axes([0.18, 0.56, 0.70, 0.355])
    bottom_axis = figure.add_axes([0.18, 0.085, 0.70, 0.355])
    colour_axis = figure.add_axes([0.897, 0.085, 0.016, 0.83])

    image = _draw_panel(
        top_axis,
        snr_percent,
        x_labels=_snr_labels(data.snr_db),
        title="(a) Noise robustness",
        xlabel="Signal-to-noise ratio (dB)",
        cmap=cmap,
        norm=norm,
    )
    _draw_panel(
        bottom_axis,
        shift_percent,
        x_labels=_shift_labels(data.shift_seconds),
        title="(b) Sensitivity to P-arrival shifts",
        xlabel="P-arrival reference shift (ms)",
        cmap=cmap,
        norm=norm,
    )

    colourbar = figure.colorbar(image, cax=colour_axis)
    colourbar.set_label("Strict U/D accuracy (%)", fontsize=9.5, labelpad=8)
    colourbar.ax.tick_params(labelsize=9.0, length=2, width=0.6)
    colourbar.outline.set_linewidth(0.5)
    ticks = np.arange(np.ceil(lower / 10.0) * 10.0, upper + 0.001, 10.0)
    if len(ticks) >= 2:
        colourbar.set_ticks(ticks)

    paths = {
        "pdf": output_dir / "robustness_heatmaps.pdf",
        "svg": output_dir / "robustness_heatmaps.svg",
        "png": output_dir / "robustness_heatmaps.png",
    }
    figure.savefig(
        paths["pdf"],
        facecolor="white",
        metadata={
            "Title": "SOPNet robustness heatmaps",
            "Subject": "Strict U/D accuracy under additive noise and P-arrival shifts",
            "Creator": "scripts/plot_paper_robustness_heatmaps.py",
        },
    )
    figure.savefig(paths["svg"], facecolor="white")
    _strip_trailing_whitespace(paths["svg"])
    figure.savefig(paths["png"], dpi=dpi, facecolor="white")
    plt.close(figure)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snr",
        type=Path,
        default=Path("paper/results/tables/snr_all.csv"),
        help="noise-robustness table (strict U/D accuracy)",
    )
    parser.add_argument(
        "--p-shift",
        type=Path,
        default=Path("paper/results/tables/p_shift_all.csv"),
        help="P-arrival shift table (strict U/D accuracy)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/results/figures"),
        help="directory for the PDF, SVG and PNG outputs",
    )
    parser.add_argument("--width-cm", type=float, default=FIGURE_WIDTH_CM)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    data = load_heatmap_data(args.snr, args.p_shift)
    paths = build_figure(
        data,
        args.output_dir,
        width_cm=args.width_cm,
        dpi=args.dpi,
    )

    font_path = font_manager.findfont(font_manager.FontProperties(family="serif"))
    print(f"serif font: {Path(font_path).name}")
    print(f"SNR columns: {', '.join(_snr_labels(data.snr_db))} dB")
    print("P-shift columns (ms): " + ", ".join(_shift_labels(data.shift_seconds)))
    for path in paths.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
