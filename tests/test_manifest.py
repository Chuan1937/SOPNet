from __future__ import annotations

from sopnet.data.canonical import DOWN, UNKNOWN, UP
from sopnet.data.manifest import (
    SourceReader,
    read_waveform,
    scan_diting,
    scan_instance,
    scan_pnw,
    scan_scsn,
    scan_txed,
)


def test_scan_scsn_mapping(fake_root):
    frame = scan_scsn(fake_root)
    assert len(frame) == 8
    assert frame["canonical_label"].tolist() == [DOWN, UP, UNKNOWN, DOWN, UP, UNKNOWN, DOWN, UP]
    assert frame["sampling_rate"].iloc[0] == 100
    assert frame["p_pick"].iloc[0] == 300
    assert frame["event_key"].nunique() == 4


def test_scan_txed(fake_root):
    frame = scan_txed(fake_root)
    assert len(frame) == 2
    assert frame["canonical_label"].tolist() == [UP, UNKNOWN]
    assert frame["event_key"].nunique() == 1
    assert all(key.startswith("txed:") for key in frame["event_key"])


def test_scan_instance(fake_root):
    frame = scan_instance(fake_root)
    assert frame["canonical_label"].tolist() == [UP, UNKNOWN]
    assert frame["event_key"].iloc[0] == "instance:100"


def test_scan_pnw(fake_root):
    frame = scan_pnw(fake_root)
    assert frame["canonical_label"].tolist() == [UP, DOWN]


def test_scan_diting(fake_root):
    frame = scan_diting(fake_root, n_parts=1)
    assert frame["canonical_label"].tolist() == [UP, DOWN]
    assert frame["sampling_rate"].iloc[0] == 50
    assert frame["trace_id"].iloc[0] == "0:000001.0001"


def test_readers_return_correct_components(fake_root):
    scsn = scan_scsn(fake_root).iloc[0]
    assert read_waveform("scsn", scsn, fake_root).shape == (600,)

    txed = scan_txed(fake_root).iloc[0]
    assert read_waveform("txed", txed, fake_root).shape == (6000,)

    instance = scan_instance(fake_root).iloc[0]
    assert read_waveform("instance", instance, fake_root).shape == (12000,)

    pnw = scan_pnw(fake_root).iloc[0]
    assert read_waveform("pnw", pnw, fake_root).shape == (15001,)

    diting = scan_diting(fake_root, n_parts=1).iloc[0]
    assert read_waveform("diting", diting, fake_root).shape == (9000,)


def test_source_reader_reuses_handles(fake_root):
    frame = scan_pnw(fake_root)
    with SourceReader("pnw", fake_root) as reader:
        for _, row in frame.iterrows():
            reader.read(row)
        assert len(reader._handles) == 1
