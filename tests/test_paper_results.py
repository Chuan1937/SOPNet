"""Verification tests for the published SOPNet paper results.

These tests treat ``paper/results/`` as the publication artefact: the seven
CSVs must be internally consistent, the heatmap must be reproducible from the
two robustness tables alone, and every published file must match the SHA-256
recorded in ``provenance/experiment_manifest.json``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "paper" / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MANIFEST_PATH = RESULTS_DIR / "provenance" / "experiment_manifest.json"

MODEL_KEYS = ("sopnet", "cfm", "rpnet", "ross", "eqpolarity", "diting_motion")
DISPLAY_NAMES = {
    "sopnet": "SOPNet",
    "cfm": "CFM",
    "rpnet": "RPNet",
    "ross": "Ross",
    "eqpolarity": "EQPolarity",
    "diting_motion": "DiTingMotion",
}
SNR_LEVELS = (float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0)
P_SHIFT_SECONDS = (
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
TABLE_NAMES = (
    "main_results.csv",
    "per_source_results.csv",
    "baseline_protocol.csv",
    "ablation_results.csv",
    "bootstrap.csv",
    "snr_all.csv",
    "p_shift_all.csv",
)
FIGURE_NAMES = (
    "robustness_heatmaps.pdf",
    "robustness_heatmaps.svg",
    "robustness_heatmaps.png",
)


def _load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_plot_module():
    path = REPO_ROOT / "scripts" / "plot_paper_robustness_heatmaps.py"
    spec = importlib.util.spec_from_file_location("plot_paper_robustness_heatmaps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def plot_module():
    return _load_plot_module()


@pytest.mark.parametrize("name", TABLE_NAMES)
def test_published_tables_are_readable(name):
    frame = _load_table(name)
    assert not frame.empty, name
    assert frame.columns.is_unique, name


def test_main_results():
    frame = _load_table("main_results.csv")
    assert len(frame) == 6
    assert set(frame["model"]) == set(DISPLAY_NAMES.values())
    assert frame["n_known"].eq(327231).all()
    assert frame["n"].eq(777720).all()
    for column in ("accuracy", "macro_f1", "precision", "recall"):
        assert frame[column].between(0.0, 1.0).all(), column
    assert frame["mcc"].between(-1.0, 1.0).all()

    sopnet = frame.set_index("model").loc["SOPNet"]
    assert sopnet["accuracy"] == pytest.approx(0.9781958310795738, abs=1e-12)
    assert sopnet["macro_f1"] == pytest.approx(0.978041719859299, abs=1e-12)
    assert sopnet["mcc"] == pytest.approx(0.9560845516040444, abs=1e-12)


def test_per_source_results():
    frame = _load_table("per_source_results.csv")
    assert len(frame) == 30
    assert set(frame["source"]) == {"scsn", "diting", "pnw", "instance", "txed"}
    assert frame.groupby("model").size().eq(5).all()
    assert frame.groupby("model")["n_known"].sum().eq(327231).all()
    assert frame["accuracy"].between(0.0, 1.0).all()
    assert frame["mcc"].between(-1.0, 1.0).all()

    for source in frame["source"].unique():
        best = frame[frame["source"] == source].sort_values("accuracy").iloc[-1]
        assert best["model"] == "SOPNet", source


def test_baseline_protocol_is_honest_about_missing_metadata():
    frame = _load_table("baseline_protocol.csv")
    assert len(frame) == 6
    assert set(frame["model"]) == set(DISPLAY_NAMES.values())
    assert frame["seed"].dropna().eq(36).all()
    assert not frame["checkpoint"].dropna().str.startswith("/").any()

    cfm = frame.set_index("model").loc["CFM"]
    assert pd.isna(cfm["epochs_run"]), "CFM training epochs were never recorded"
    eqpolarity = frame.set_index("model").loc["EQPolarity"]
    assert eqpolarity["epochs_run"] == 5
    assert eqpolarity["best_epoch"] == 5

    text = frame.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    assert "/home/" not in text


def test_ablation_uses_converged_c_and_full_model():
    frame = _load_table("ablation_results.csv").set_index("experiment")
    assert set(frame.index) == {"A_cls_ud", "B_field", "C_polarity", "D_full"}
    assert frame["n_known"].eq(327231).all()

    assert frame.loc["A_cls_ud", "accuracy"] == pytest.approx(0.9746325989897046, abs=1e-12)
    assert frame.loc["B_field", "accuracy"] == pytest.approx(0.9752315642466637, abs=1e-12)
    assert frame.loc["C_polarity", "accuracy"] == pytest.approx(0.9780766492172196, abs=1e-12)
    assert frame.loc["C_polarity", "macro_f1"] == pytest.approx(0.9779220710825787, abs=1e-12)
    assert frame.loc["C_polarity", "mcc"] == pytest.approx(0.9558449745639014, abs=1e-12)

    accuracies = [
        frame.loc["A_cls_ud", "accuracy"],
        frame.loc["B_field", "accuracy"],
        frame.loc["C_polarity", "accuracy"],
        frame.loc["D_full", "accuracy"],
    ]
    assert accuracies == sorted(accuracies)

    main = _load_table("main_results.csv").set_index("model").loc["SOPNet"]
    for column in ("accuracy", "macro_f1", "precision", "recall", "mcc"):
        assert frame.loc["D_full", column] == pytest.approx(main[column], abs=1e-12)


def test_bootstrap_matches_main_results():
    frame = _load_table("bootstrap.csv")
    assert len(frame) == 5
    assert set(frame["baseline"]) == set(MODEL_KEYS) - {"sopnet"}
    assert frame["n_resamples"].eq(1000).all()
    assert frame["confidence"].eq(0.95).all()
    assert frame["n_groups"].eq(63044).all()
    assert (frame["delta"] > 0).all()
    assert (frame["ci_low"] > 0).all()
    assert (frame["ci_high"] > frame["delta"]).all()
    assert frame["p_improvement"].eq(1.0).all()

    main = _load_table("main_results.csv").set_index("model")
    assert frame["sopnet_macro_f1"].eq(main.loc["SOPNet", "macro_f1"]).all()
    for row in frame.itertuples():
        display = DISPLAY_NAMES[row.baseline]
        assert row.baseline_macro_f1 == pytest.approx(main.loc[display, "macro_f1"], abs=1e-12)


def test_robustness_grids_are_complete_and_matched():
    snr = _load_table("snr_all.csv")
    shift = _load_table("p_shift_all.csv")

    for frame, condition, expected in (
        (snr, "snr_db", set(SNR_LEVELS)),
        (shift, "shift_seconds", set(P_SHIFT_SECONDS)),
    ):
        assert set(frame["model"]) == set(MODEL_KEYS)
        assert set(frame[condition].astype(float)) == expected
        assert not frame.duplicated(["model", condition]).any()
        assert len(frame) == len(MODEL_KEYS) * len(expected)
        assert frame.groupby(["model", condition]).size().eq(1).all()
        assert frame["n"].eq(30000).all()
        assert frame["n_known"].eq(12520).all()
        assert frame["support"].eq(frame["n_known"]).all()
        assert frame["accuracy"].between(0.0, 1.0).all()
        assert frame["macro_f1_ud"].between(0.0, 1.0).all()
        assert frame["mcc"].between(-1.0, 1.0).all()

    clean = snr[np.isposinf(snr["snr_db"])].set_index("model")["accuracy"].sort_index()
    zero = shift[shift["shift_seconds"] == 0.0].set_index("model")["accuracy"].sort_index()
    pd.testing.assert_series_equal(clean, zero, check_names=False)


def test_heatmap_values_are_traceable_to_csv(plot_module):
    snr = _load_table("snr_all.csv")
    shift = _load_table("p_shift_all.csv")
    data = plot_module.load_heatmap_data(
        TABLES_DIR / "snr_all.csv",
        TABLES_DIR / "p_shift_all.csv",
    )

    assert data.models == plot_module.MODEL_ORDER
    assert data.snr_db == SNR_LEVELS
    assert data.shift_seconds == P_SHIFT_SECONDS
    assert data.snr_accuracy.shape == (6, 6)
    assert data.shift_accuracy.shape == (6, 13)

    for row, model in enumerate(data.models):
        model_snr = snr[snr["model"] == model]
        for column, level in enumerate(data.snr_db):
            rows = (
                model_snr[np.isposinf(model_snr["snr_db"])]
                if np.isposinf(level)
                else model_snr[model_snr["snr_db"] == level]
            )
            assert data.snr_accuracy[row, column] == pytest.approx(rows["accuracy"].item(), abs=1e-15)
        model_shift = shift[shift["model"] == model]
        for column, seconds in enumerate(data.shift_seconds):
            row_data = model_shift[model_shift["shift_seconds"] == seconds]
            assert data.shift_accuracy[row, column] == pytest.approx(row_data["accuracy"].item(), abs=1e-15)


def test_heatmap_is_reproducible(plot_module, tmp_path):
    data = plot_module.load_heatmap_data(
        TABLES_DIR / "snr_all.csv",
        TABLES_DIR / "p_shift_all.csv",
    )
    paths = plot_module.build_figure(data, tmp_path, width_cm=17.0, dpi=100)

    for name in FIGURE_NAMES:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 1000

    png = Image.open(paths["png"])
    assert png.width == round(17.0 / 2.54 * 100.0)
    assert png.height > 0

    svg_text = paths["svg"].read_text()
    svg_width = re.search(r'width="([0-9.]+)pt"', svg_text)
    assert svg_width is not None
    assert float(svg_width.group(1)) / 72.0 * 2.54 == pytest.approx(17.0, abs=0.02)
    assert not any(line != line.rstrip() for line in svg_text.splitlines())


def test_missing_cell_is_rendered_as_na(plot_module, tmp_path):
    snr = _load_table("snr_all.csv")
    keep = ~((snr["model"] == "sopnet") & np.isposinf(snr["snr_db"]))
    partial = snr[keep].copy()
    partial_path = tmp_path / "snr_partial.csv"
    partial.to_csv(partial_path, index=False)

    data = plot_module.load_heatmap_data(partial_path, TABLES_DIR / "p_shift_all.csv")
    assert np.isnan(data.snr_accuracy[0, 0])

    paths = plot_module.build_figure(data, tmp_path / "figure", dpi=80)
    assert paths["png"].exists()


def test_manifest_hashes_and_metadata():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"])
    publication_commit = manifest.get("publication_commit")
    assert publication_commit is None or re.fullmatch(r"[0-9a-f]{40}", publication_commit)
    assert manifest["repository"] == "https://github.com/Chuan1937/SOPNet"

    tables = {entry["name"]: entry for entry in manifest["tables"]}
    assert set(tables) == set(TABLE_NAMES)
    for name, entry in tables.items():
        path = REPO_ROOT / entry["path"]
        assert path == TABLES_DIR / name
        assert entry["sha256"] == _sha256(path)
        assert entry["bytes"] == path.stat().st_size
        frame = _load_table(name)
        assert entry["rows"] == len(frame)
        assert entry["columns"] == list(frame.columns)

    figures = {Path(entry["path"]).name: entry for entry in manifest["figures"]}
    assert set(figures) == set(FIGURE_NAMES)
    for name, entry in figures.items():
        path = FIGURES_DIR / name
        assert entry["sha256"] == _sha256(path)
        assert entry["bytes"] == path.stat().st_size

    models = {entry["key"]: entry for entry in manifest["models"]}
    assert set(models) == set(MODEL_KEYS)
    for entry in models.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["checkpoint_sha256"])
        assert entry["checkpoint_bytes"] > 0
        checkpoint = REPO_ROOT / entry["checkpoint"]
        assert not Path(entry["checkpoint"]).is_absolute()
        if checkpoint.exists():
            assert _sha256(checkpoint) == entry["checkpoint_sha256"]

    experiment = manifest["experiment"]
    assert experiment["random_seeds"] == {
        "training": 36,
        "ablation": 36,
        "data_split": 20261004,
        "robustness_subset": 20261004,
        "bootstrap": 20261004,
    }
    assert experiment["test_split"]["n"] == 777720
    assert experiment["test_split"]["n_known_ud"] == 327231
    assert experiment["robustness_subset"]["n"] == 30000
    assert experiment["robustness_subset"]["n_known_ud"] == 12520
    assert set(experiment["noise"]["levels_db"]) == {"clean", 20, 10, 5, 0, -5}
    assert experiment["p_shift"]["grid_seconds"] == list(P_SHIFT_SECONDS)
    assert len(experiment["p_shift"]["grid_ms"]) == 13


def test_published_figures_and_readme_exist():
    for name in FIGURE_NAMES:
        path = FIGURES_DIR / name
        assert path.exists()
        assert path.stat().st_size > 0
    assert (RESULTS_DIR / "README.md").exists()
