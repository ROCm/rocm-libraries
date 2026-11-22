# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Legacy tune/search/bench trace parsing (bench/log.py) ignores the extended mask-128 keys.

Fixtures: a real producer record before/after the append-only extension, and real
extended-profile excerpts. Break: admit pointer_mode/activation args to LOG_FIELDS,
lose call weights, or share --specialize's no-default / record-local UNKNOWN rules.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.bench import log as blog

DATA = Path(__file__).parent / "test_data" / "specialize_calls"
OLD_DATA = Path(__file__).parent / "test_data" / "trace_logging"
NEW_KEYS = {"pointer_mode", "activation_arg0", "activation_arg1"}


@pytest.mark.parametrize("mode", ["tune", "search", "bench"])
@pytest.mark.parametrize("sample,total", [
    ("existing_fixture", 1), ("producer_pointer_pair.yaml", 5),
    ("producer_arg0_pair.yaml", 5), ("producer_arg1_pair.yaml", 5),
], ids=["existing", "pointer_pair", "arg0_pair", "arg1_pair"])
def test_legacy_modes_keep_old_grouping_with_new_trace_fields(mode, sample, total, tmp_path):
    if sample == "existing_fixture":
        old = OLD_DATA / "mask128-old.yaml"
        new = OLD_DATA / "mask128-new.yaml"
    else:
        new = DATA / sample
        # Derived old-format comparison, not a claimed real/model trace.
        old = tmp_path / "old-format.yaml"
        old.write_text(yaml.safe_dump([
            {key: value for key, value in row.items() if key not in NEW_KEYS}
            for row in blog.read(new)
        ], sort_keys=False))
    old_parsed = blog.parse(old, as_df=True)
    new_parsed = blog.parse(new, as_df=True)
    pd.testing.assert_frame_equal(new_parsed, old_parsed)
    assert len(new_parsed) == 1
    assert new_parsed.iloc[0]["call_count"] == total
    assert NEW_KEYS.isdisjoint(new_parsed.columns)
    assert (new_parsed.iloc[0]["M"], new_parsed.iloc[0]["N"], new_parsed.iloc[0]["K"]) == (128, 128, 128)
    if mode == "bench":
        assert blog.update(blog.parse(new))[0] == blog.update(blog.parse(old))[0]
    else:
        # keep_thr=0 exercises real parse/update/summarize without GPU timing.
        old_summary = blog.summarize(DATA, old, output_dir=tmp_path / "old", keep_thr=0)
        new_summary = blog.summarize(DATA, new, output_dir=tmp_path / "new", keep_thr=0)
        assert len(old_summary) == len(new_summary)
        for got, want in zip(new_summary, old_summary):
            pd.testing.assert_frame_equal(got, want)
        assert (tmp_path / "new" / "gemms.csv").read_bytes() == (tmp_path / "old" / "gemms.csv").read_bytes()


# Break: lose the legacy defaults (count 1, matmul, scale type, compute-type prefix).
def test_legacy_parser_retains_its_existing_defaults(tmp_path):
    row = dict(blog.read(DATA / "producer_default.yaml")[0])
    for key in ("call_count", "function", "scale_type"):
        del row[key]
    row["compute_type"] = "f32_r"
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump([row], sort_keys=False))
    parsed = blog.parse(path)
    assert len(parsed) == 1
    assert parsed[0]["call_count"] == 1
    assert parsed[0]["function"] == "matmul"
    assert parsed[0]["scale_type"] == "f32_r"
    assert parsed[0]["compute_type"] == "c_f32_r"


# Break: replace the legacy global drop of an incomplete column with record-local UNKNOWN.
def test_legacy_parser_retains_global_missing_column_drop(tmp_path):
    row = dict(blog.read(DATA / "producer_default.yaml")[0], call_count=2)
    missing = dict(row, call_count=3)
    del missing["beta"]
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump([row, missing], sort_keys=False))
    parsed = blog.parse(path)
    assert len(parsed) == 1
    assert parsed[0]["call_count"] == 5
    assert "beta" not in parsed[0]
