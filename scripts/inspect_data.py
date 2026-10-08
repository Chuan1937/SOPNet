#!/usr/bin/env python
"""Audit the five raw datasets: counts, encodings, structures and label conventions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sopnet.data.manifest import (  # noqa: E402
    COMPONENT_INDEX,
    SOURCE_PATHS,
    SOURCES,
    manifest_summary,
    scan_diting,
    scan_instance,
    scan_pnw,
    scan_scsn,
    scan_txed,
)
from sopnet.utils.logging import get_logger  # noqa: E402

SCANNERS = {
    "scsn": scan_scsn,
    "txed": scan_txed,
    "instance": scan_instance,
    "pnw": scan_pnw,
    "diting": scan_diting,
}


def probe_hdf5(path: Path, reader) -> dict:
    import h5py

    info: dict = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return info
    try:
        with h5py.File(path, "r") as handle:
            info["keys"] = list(handle.keys())
            if "data_format" in handle:
                fmt = {}
                for key in handle["data_format"].keys():
                    value = handle["data_format"][key][()]
                    fmt[key] = value.decode() if isinstance(value, bytes) else value
                info["data_format"] = fmt
            if "data" in handle:
                total = 0
                shapes = {}
                for bucket in handle["data"].keys():
                    shape = str(handle["data"][bucket].shape[1:])
                    total += handle["data"][bucket].shape[0]
                    shapes[shape] = shapes.get(shape, 0) + 1
                info["n_rows"] = total
                info["bucket_shapes"] = shapes
            if "X" in handle:
                info["X_shape"] = list(handle["X"].shape)
                info["X_dtype"] = str(handle["X"].dtype)
                if "Y" in handle:
                    values, counts = np.unique(handle["Y"][:], return_counts=True)
                    info["Y_counts"] = {str(v): int(c) for v, c in zip(values, counts)}
    except Exception as error:  # noqa: BLE001
        info["error"] = f"{type(error).__name__}: {error}"
    return info


def verify_scsn(root: Path, n: int = 200_000) -> dict:
    import h5py

    path = root / SOURCE_PATHS["scsn"]["flat"]
    with h5py.File(path, "r") as handle:
        n = min(n, handle["X"].shape[0])
        x = handle["X"][:n]
        y = handle["Y"][:n]
    result = {}
    for label in np.unique(y):
        mask = y == label
        result[str(int(label))] = {
            "n": int(mask.sum()),
            "signed_post_p_mean": float(x[mask, 300:380].mean()),
        }
    return result


def verify_diting(root: Path, per_class: int = 800) -> dict:
    import h5py
    import pandas as pd
    from scipy.signal import butter, detrend, sosfiltfilt

    directory = root / SOURCE_PATHS["diting"]["dir"]
    csv_path = directory / "DiTing330km_part_0.csv"
    h5_path = directory / "DiTing330km_part_0.hdf5"
    if not csv_path.exists():
        return {}
    frame = pd.read_csv(csv_path, usecols=["key", "p_pick", "p_motion"])
    frame = frame[frame["p_motion"].astype(str).str.strip().isin(["U", "D", "R", "C"])]
    chosen = []
    for label in ["U", "D", "R", "C"]:
        pool = frame[frame["p_motion"].astype(str).str.strip() == label]
        if len(pool):
            chosen.append(pool.sample(min(per_class, len(pool)), random_state=0))
    frame = pd.concat(chosen) if chosen else frame
    sos = butter(3, [1.0, 20.0], btype="bandpass", fs=50, output="sos")

    result: dict = {}
    with h5py.File(h5_path, "r") as handle:
        group = handle["earthquake"]
        for _, row in frame.iterrows():
            key_float = float(row["key"])
            fraction = str(key_float).split(".")[1] if "." in str(key_float) else "0"
            key = f"{int(key_float):06d}.{fraction.ljust(4, '0')[:4]}"
            if key not in group:
                continue
            waveform = detrend(group[key][:, 0].astype(np.float64), type="linear")
            waveform = sosfiltfilt(sos, waveform)
            pick = int(row["p_pick"])
            pre = waveform[max(0, pick - 250) : max(1, pick - 50)]
            noise = float(np.median(np.abs(pre - np.median(pre))) * 1.4826) + 1e-12
            segment = waveform[max(0, pick - 10) : pick + 60]
            crossings = np.where(np.abs(segment) > 4 * noise)[0]
            if not len(crossings):
                continue
            label = str(row["p_motion"]).strip()
            entry = result.setdefault(label, {"n": 0, "positive": 0})
            entry["n"] += 1
            entry["positive"] += int(segment[crossings[0]] > 0)
    for label, entry in result.items():
        entry["fraction_positive"] = entry["positive"] / entry["n"] if entry["n"] else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/d/AI_Seismic_Data")
    parser.add_argument("--output", default="outputs/audit")
    parser.add_argument("--datasets", nargs="+", default=SOURCES)
    parser.add_argument("--verify", action="store_true", help="run empirical label checks")
    args = parser.parse_args()

    logger = get_logger("sopnet.audit")
    root = Path(args.data_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    report: dict = {"data_root": str(root), "sources": {}}
    summaries = []
    for name in args.datasets:
        logger.info("scanning %s ...", name)
        frame = SCANNERS[name](root)
        summary = manifest_summary(frame).loc[[name]]
        summaries.append(summary)
        report["sources"][name] = {
            "n_samples": int(len(frame)),
            "label_counts": {
                str(k): int(v) for k, v in frame["canonical_label"].value_counts().sort_index().items()
            },
            "unique_events": int(frame["event_key"].nunique()),
            "sampling_rate": int(frame["sampling_rate"].iloc[0]) if len(frame) else None,
            "component_index": COMPONENT_INDEX[name],
            "p_pick_range": [float(frame["p_pick"].min()), float(frame["p_pick"].max())]
            if len(frame)
            else None,
        }
        logger.info("%s: %d samples, %d unique events", name, len(frame), frame["event_key"].nunique())

    import pandas as pd

    distribution = pd.concat(summaries)
    distribution.to_csv(output / "label_distribution.csv")
    print(distribution.to_string())

    structures = {
        "scsn": probe_hdf5(root / SOURCE_PATHS["scsn"]["flat"], None),
        "txed": probe_hdf5(root / SOURCE_PATHS["txed"]["hdf5"], None),
        "instance": probe_hdf5(root / SOURCE_PATHS["instance"]["hdf5"], None),
        "pnw": probe_hdf5(root / SOURCE_PATHS["pnw"]["hdf5"], None),
    }
    report["structures"] = structures

    if args.verify:
        logger.info("verifying SCSN numeric label convention ...")
        report["verification"] = {
            "scsn_signed_post_p": verify_scsn(root),
            "diting_first_break": verify_diting(root),
        }
        logger.info("verification: %s", json.dumps(report["verification"], indent=2))

    with open(output / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    logger.info("audit written to %s", output)


if __name__ == "__main__":
    main()
