"""Unified polarity manifest: logical merge of SCSN, TXED, INSTANCE, DiTing and PNW.

The manifest stores *metadata only*; original waveforms are never copied.
``source`` is retained exclusively for label mapping, trace parsing and auditing
and must never be used as a model input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import pandas as pd

from sopnet.data.canonical import (
    DOWN,
    UNKNOWN,
    UP,
    SCSN_LABELS,
    map_raw_label,
)

MANIFEST_COLUMNS: List[str] = [
    "sample_id",
    "source",
    "source_path",
    "trace_id",
    "row_index",
    "p_pick",
    "sampling_rate",
    "component",
    "raw_label",
    "canonical_label",
    "event_key",
    "split_native",
    "waveform_hash",
]

SOURCES: List[str] = ["scsn", "txed", "instance", "pnw", "diting"]

SOURCE_PATHS: Dict[str, Dict[str, str]] = {
    "scsn": {"flat": "scsn/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5"},
    "txed": {"csv": "txed/Texd_filtered.csv", "hdf5": "txed/Texd.hdf5"},
    "instance": {"csv": "Instance/INSTANCE_filtered.csv", "hdf5": "Instance/INSTANCE.hdf5"},
    "pnw": {"csv": "pnw/pnw.csv", "hdf5": "pnw/pnw.hdf5"},
    "diting": {"dir": "谛听1.0-50HZ/Diting50hz", "n_parts": "28"},
}

DITING_N_PARTS = 28

# component index used to read the vertical channel from each bucket file,
# taken from the file's own ``data_format/component_order`` attribute:
#   TXED / INSTANCE -> ZNE  (Z index 0)
#   PNW             -> ENZ  (Z index 2)
#   DiTing          -> Z index 0 (verified empirically)
COMPONENT_INDEX: Dict[str, int] = {
    "scsn": 0,
    "txed": 0,
    "instance": 0,
    "pnw": 2,
    "diting": 0,
}

NATIVE_SAMPLING_RATE: Dict[str, float] = {
    "scsn": 100.0,
    "txed": 100.0,
    "instance": 100.0,
    "pnw": 100.0,
    "diting": 50.0,
}


def parse_trace_id(trace_id: str, default_length: Optional[int] = None):
    """Parse a SeisBench-style trace id ``bucketN$i,:3,:L``.

    Returns ``(group, row, component_end, sample_end)``.
    """
    text = str(trace_id)
    if "$" not in text:
        raise ValueError(f"Invalid trace id: {trace_id}")
    group, spec = text.split("$", 1)
    fields = spec.split(",")
    if len(fields) != 3:
        raise ValueError(f"Invalid trace id spec: {trace_id}")

    def _int(token: str, fallback: int) -> int:
        token = token.strip()
        if token in ("", ":"):
            return fallback
        return int(token.lstrip(":"))

    row = _int(fields[0], 0)
    comp_end = _int(fields[1], 3)
    sample_end = _int(fields[2], default_length if default_length is not None else 0)
    return group, row, comp_end, sample_end


class SourceReader:
    """Reusable, handle-caching waveform reader for one data source."""

    def __init__(self, source: str, data_root: Path):
        self.source = source
        self.data_root = Path(data_root)
        self._handles: Dict[str, h5py.File] = {}
        self._diting_parts: Dict[str, h5py.File] = {}

    def _path(self, source_path: str) -> Path:
        return self.data_root / source_path

    def _handle(self, source_path: str) -> h5py.File:
        key = str(source_path)
        if key not in self._handles:
            self._handles[key] = h5py.File(self._path(source_path), "r")
        return self._handles[key]

    def _diting_handle(self, part: int) -> h5py.File:
        key = str(part)
        if key not in self._diting_parts:
            directory = self.data_root / SOURCE_PATHS["diting"]["dir"]
            path = directory / f"DiTing330km_part_{part}.hdf5"
            self._diting_parts[key] = h5py.File(path, "r")
        return self._diting_parts[key]

    def read(self, row) -> np.ndarray:
        """Read the vertical-component waveform for one manifest row."""
        source = self.source
        if source == "scsn":
            handle = self._handle(row["source_path"])
            return np.asarray(handle["X"][int(row["row_index"])], dtype=np.float64)

        if source == "diting":
            part_text, key = str(row["trace_id"]).split(":", 1)
            handle = self._diting_handle(int(part_text))
            return np.asarray(handle["earthquake"][key][:, COMPONENT_INDEX[source]], dtype=np.float64)

        handle = self._handle(row["source_path"])
        group, index, _, sample_end = parse_trace_id(row["trace_id"])
        length = row.get("trace_length")
        if sample_end == 0:
            sample_end = int(length) if length is not None else handle["data"][group].shape[-1]
        return np.asarray(
            handle["data"][group][index, COMPONENT_INDEX[source], :sample_end],
            dtype=np.float64,
        )

    def close(self) -> None:
        for handle in list(self._handles.values()):
            handle.close()
        for handle in list(self._diting_parts.values()):
            handle.close()
        self._handles.clear()
        self._diting_parts.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def read_waveform(source: str, row, data_root: Path) -> np.ndarray:
    """One-shot convenience reader (opens files per call; prefer SourceReader)."""
    reader = SourceReader(source, data_root)
    try:
        return reader.read(row)
    finally:
        reader.close()


def _finalize(frame: pd.DataFrame, source: str, source_path: str) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    frame["source"] = source
    frame["source_path"] = source_path
    frame["component"] = "Z"
    frame["sampling_rate"] = int(NATIVE_SAMPLING_RATE[source])
    frame["waveform_hash"] = np.uint64(0)
    frame["sample_id"] = [f"{source[:1]}:{i:09d}" for i in range(len(frame))]
    frame["row_index"] = frame["row_index"].astype(np.int32)
    frame["p_pick"] = frame["p_pick"].astype(np.float32)
    frame["canonical_label"] = frame["canonical_label"].astype(np.int8)
    return frame[MANIFEST_COLUMNS]


def _canonical_from_map(raws, source: str) -> np.ndarray:
    return np.array([map_raw_label(source, raw) for raw in raws], dtype=np.int8)


def scan_scsn(data_root: Path, limit: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    source_path = SOURCE_PATHS["scsn"]["flat"]
    with h5py.File(Path(data_root) / source_path, "r") as handle:
        labels = np.asarray(handle["Y"][:])
        events = np.asarray(handle["evids"][:])
    lut = np.array(
        [SCSN_LABELS[0], SCSN_LABELS[1], SCSN_LABELS[2]], dtype=np.int8
    )
    frame = pd.DataFrame(
        {
            "trace_id": "",
            "row_index": np.arange(labels.size, dtype=np.int32),
            "p_pick": np.full(labels.size, 300.0, dtype=np.float32),
            "raw_label": labels.astype(str),
            "canonical_label": lut[labels.astype(np.int64)],
            "event_key": np.array([f"scsn:{event}" for event in events]),
            "split_native": pd.NA,
        }
    )
    frame = _finalize(frame, "scsn", source_path)
    return _limit(frame, limit, seed)


def _event_from_trace_name(trace_name: object, fallback: str) -> str:
    text = str(trace_name)
    if text.endswith("_EV"):
        parts = text.split("_")
        if len(parts) >= 3:
            return "txed:" + "_".join(parts[:-2])
    return "txed:" + fallback


def scan_txed(data_root: Path, limit: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    source_path = SOURCE_PATHS["txed"]["csv"]
    usecols = [
        "trace_name",
        "trace_p_arrival_sample",
        "trace_polarity",
        "trace_name_original",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "split",
    ]
    frame = pd.read_csv(Path(data_root) / source_path, usecols=usecols, low_memory=False)
    frame = frame[frame["trace_p_arrival_sample"].notna() & frame["trace_name"].notna()].copy()
    fallback = (
        frame["source_origin_time"].astype(str)
        + "|"
        + frame["source_latitude_deg"].astype(str)
        + "|"
        + frame["source_longitude_deg"].astype(str)
        + "|"
        + frame["source_depth_km"].astype(str)
    )
    frame["event_key"] = [
        _event_from_trace_name(tn, fb)
        for tn, fb in zip(frame["trace_name_original"], fallback)
    ]
    frame = pd.DataFrame(
        {
            "trace_id": frame["trace_name"].astype(str).values,
            "row_index": np.arange(len(frame), dtype=np.int32),
            "p_pick": frame["trace_p_arrival_sample"].astype(float).values,
            "raw_label": frame["trace_polarity"].astype(str).values,
            "canonical_label": _canonical_from_map(frame["trace_polarity"].values, "txed"),
            "event_key": frame["event_key"].values,
            "split_native": frame["split"].astype(str).values,
        }
    )
    # row_index refers to the CSV position of the filtered frame; the HDF5 index
    # lives inside trace_id, which is what SourceReader uses.
    frame = _finalize(frame, "txed", SOURCE_PATHS["txed"]["hdf5"])
    return _limit(frame, limit, seed)


def scan_instance(data_root: Path, limit: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    source_path = SOURCE_PATHS["instance"]["csv"]
    usecols = [
        "trace_name",
        "trace_P_arrival_sample",
        "trace_polarity",
        "source_id",
        "split",
    ]
    frame = pd.read_csv(Path(data_root) / source_path, usecols=usecols, low_memory=False)
    frame = frame[frame["trace_P_arrival_sample"].notna() & frame["trace_name"].notna()].copy()
    frame = pd.DataFrame(
        {
            "trace_id": frame["trace_name"].astype(str).values,
            "row_index": np.arange(len(frame), dtype=np.int32),
            "p_pick": frame["trace_P_arrival_sample"].astype(float).values,
            "raw_label": frame["trace_polarity"].astype(str).values,
            "canonical_label": _canonical_from_map(frame["trace_polarity"].values, "instance"),
            "event_key": ["instance:" + str(s) for s in frame["source_id"].values],
            "split_native": frame["split"].astype(str).values,
        }
    )
    frame = _finalize(frame, "instance", SOURCE_PATHS["instance"]["hdf5"])
    return _limit(frame, limit, seed)


def scan_pnw(data_root: Path, limit: Optional[int] = None, seed: int = 0) -> pd.DataFrame:
    source_path = SOURCE_PATHS["pnw"]["csv"]
    usecols = [
        "trace_name",
        "trace_P_arrival_sample",
        "trace_P_polarity",
        "event_id",
    ]
    frame = pd.read_csv(Path(data_root) / source_path, usecols=usecols, low_memory=False)
    frame = frame[frame["trace_P_arrival_sample"].notna() & frame["trace_name"].notna()].copy()
    frame = pd.DataFrame(
        {
            "trace_id": frame["trace_name"].astype(str).values,
            "row_index": np.arange(len(frame), dtype=np.int32),
            "p_pick": frame["trace_P_arrival_sample"].astype(float).values,
            "raw_label": frame["trace_P_polarity"].fillna("unknown").astype(str).values,
            "canonical_label": _canonical_from_map(frame["trace_P_polarity"].values, "pnw"),
            "event_key": ["pnw:" + str(e) for e in frame["event_id"].values],
            "split_native": pd.NA,
        }
    )
    frame = _finalize(frame, "pnw", SOURCE_PATHS["pnw"]["hdf5"])
    return _limit(frame, limit, seed)


def diting_hdf5_key(raw_key: object) -> str:
    value = float(raw_key)
    fraction = str(value).split(".")[1] if "." in str(value) else "0"
    return f"{int(value):06d}.{fraction.ljust(4, '0')[:4]}"


def scan_diting(
    data_root: Path,
    limit: Optional[int] = None,
    seed: int = 0,
    n_parts: int = DITING_N_PARTS,
) -> pd.DataFrame:
    directory = Path(data_root) / SOURCE_PATHS["diting"]["dir"]
    frames = []
    for part in range(n_parts):
        csv_path = directory / f"DiTing330km_part_{part}.csv"
        if not csv_path.exists():
            continue
        part_frame = pd.read_csv(
            csv_path, usecols=["key", "p_pick", "p_motion", "ev_id"], low_memory=False
        )
        part_frame = part_frame[part_frame["p_pick"].notna() & part_frame["key"].notna()].copy()
        part_frame = pd.DataFrame(
            {
                "trace_id": [
                    f"{part}:{diting_hdf5_key(key)}" for key in part_frame["key"].values
                ],
                "row_index": np.arange(len(part_frame), dtype=np.int32),
                "p_pick": part_frame["p_pick"].astype(float).values,
                "raw_label": part_frame["p_motion"].fillna("unknown").astype(str).values,
                "canonical_label": _canonical_from_map(part_frame["p_motion"].values, "diting"),
                "event_key": [
                    f"diting:{part}:{event}" for event in part_frame["ev_id"].values
                ],
                "split_native": pd.NA,
            }
        )
        frames.append(part_frame)
    if not frames:
        raise FileNotFoundError(f"No DiTing part CSVs found in {directory}")
    frame = pd.concat(frames, ignore_index=True)
    frame = _finalize(frame, "diting", SOURCE_PATHS["diting"]["dir"])
    return _limit(frame, limit, seed)


def _limit(frame: pd.DataFrame, limit: Optional[int], seed: int) -> pd.DataFrame:
    if limit is None or limit >= len(frame):
        return frame
    return frame.sample(n=int(limit), random_state=seed).reset_index(drop=True)


SCANNERS = {
    "scsn": scan_scsn,
    "txed": scan_txed,
    "instance": scan_instance,
    "pnw": scan_pnw,
    "diting": scan_diting,
}


def build_manifest(
    data_root: Path,
    datasets: Optional[Iterable[str]] = None,
    limit_per_source: Optional[int] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build the unified manifest (metadata only)."""
    names = list(datasets) if datasets is not None else SOURCES
    unknown = set(names) - set(SCANNERS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}; known: {SOURCES}")
    frames = [
        SCANNERS[name](Path(data_root), limit=limit_per_source, seed=seed) for name in names
    ]
    manifest = pd.concat(frames, ignore_index=True)
    for column in ("source", "source_path", "component"):
        manifest[column] = manifest[column].astype("category")
    return manifest[MANIFEST_COLUMNS]


def save_manifest(manifest: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(path, index=False)
    return path


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def manifest_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    """Label distribution per source and in total."""
    grouped = (
        manifest.groupby(["source", "canonical_label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(columns={UP: "U", DOWN: "D", UNKNOWN: "X"})
    )
    grouped["total"] = grouped.sum(axis=1)
    total = grouped.sum(axis=0).to_frame("ALL").T
    return pd.concat([grouped, total])
