# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Synchronization contract for the generated multi-wave LayerNorm reduction."""

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if TENSILE_ROOT not in sys.path:
    sys.path.insert(0, TENSILE_ROOT)

from gpu_test_helpers import init_rocisa  # noqa: E402

from Tensile.Common.Architectures import gfxToIsa  # noqa: E402
from Tensile.Common.DataType import DataType  # noqa: E402

import LayerNormGenerator  # noqa: E402


def _layer_norm_generator():
    target = "gfx90a"
    isa = gfxToIsa(target)
    init_rocisa(target=target, wavesize=64)
    return LayerNormGenerator.LayerNormKernelGenerator(
        io_type=DataType("S"),
        num_workitems=256,
        num_load_count=4,
        num_load_size=4,
        sweep_once=0,
        wavefront_size=64,
        arch=target,
        isa=isa,
    )


@pytest.mark.unit
def test_inter_wave_reduction_synchronizes_before_lds_reuse():
    text = str(_layer_norm_generator().inter_wave_reduction())

    upper = text.split("label_upper:", 1)[1].split("label_lower:", 1)[0]
    lower = text.split("label_lower:", 1)[1].split("label_empty:", 1)[0]
    empty = text.split("label_empty:", 1)[1].split("label_inter_sync:", 1)[0]
    sync = text.split("label_inter_sync:", 1)[1].split("label_end:", 1)[0]

    assert upper.count("s_barrier") == 1
    assert lower.count("s_barrier") == 1
    assert empty.count("s_barrier") == 1
    assert sync.count("s_barrier") == 1

    assert "ds_write" in upper
    assert "s_branch label_inter_sync" in upper

    assert lower.index("s_barrier") < lower.index("ds_read")
    assert "s_branch label_inter_sync" in lower

    assert "s_branch label_inter" in sync
