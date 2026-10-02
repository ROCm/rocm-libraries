# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opcode-specific advice from the source-tree LDS counter/ISA analyzer."""

from pathlib import Path
import runpy

import pytest


@pytest.fixture(scope="module")
def analyzer():
    path = (
        Path(__file__).resolve().parents[2]
        / "dsl_docs/optimization/utilities/tools/stage4_analyze/analyze_lds_conflicts.py"
    )
    return runpy.run_path(str(path))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    ("reads", "writes"), [(51, 0), (0, 51), (51, 51), (0, 0), (50, 50)]
)
def test_b128_advice_distinguishes_read_and_write_periods(
    analyzer, arch, reads, writes
):
    recommendations = analyzer["generate_recommendations"](
        counters=analyzer["LDSCounters"](),
        isa_stats=analyzer["LDSInstructionStats"](
            ds_read_b128=reads, ds_write_b128=writes
        ),
        severity="low",
        has_xor=False,
        has_padding=False,
        power_of_2_stride=False,
        arch=arch,
    )
    advice = [text for text in recommendations if "period" in text.lower()]
    assert len(advice) == int(reads > 50) + int(writes > 50)
    for opcode, count, period in (
        ("ds_read_b128", reads, 64 if arch == "gfx950" else 32),
        ("ds_write_b128", writes, 32),
    ):
        matches = [text for text in advice if opcode in text]
        if count <= 50:
            assert not matches
            continue
        assert len(matches) == 1
        assert f"{period} dwords ({period * 4} bytes)" in matches[0]
        assert "wave64" in matches[0]
        assert "phase" in matches[0]
        assert "active lanes" in matches[0]
        assert "can still conflict" in matches[0]
    assert all("Ensure stride >" not in text for text in recommendations)
