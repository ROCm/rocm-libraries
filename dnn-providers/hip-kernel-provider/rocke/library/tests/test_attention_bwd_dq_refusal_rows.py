# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every row of the dQ resource-gate tables the C builder ports has a
parity config that Python refuses at that row.

The C validator copies DQ_SPILLING_COMBOS and DQ_SCHED_SPILLING by value, and
the parity configs of attention_bwd_dq_emit.py are what check it (both
golden checks require the C builder to refuse a config Python refuses). A
row added in Python without a config would leave the C builder building that
kernel with every gate green; this test fails instead.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from kernels.common.attention_bwd_dq import (
    DQ_SCHED_SPILLING,
    DQ_SPILLING_COMBOS,
    AttnBwdDqSpec,
    validate_attn_bwd_dq_spec,
)

_PARITY = Path(__file__).resolve().parent / "parity"
_COMBO = (
    "head_size",
    "waves",
    "block_m",
    "block_n",
    "kv_prefetch",
    "head_loop",
    "edge_tiles",
)
_SCHED = _COMBO + ("kv_pad", "transpose_source", "sched")


def _emitter():
    if str(_PARITY) not in sys.path:
        sys.path.insert(0, str(_PARITY))
    spec = importlib.util.spec_from_file_location(
        "_dq_emit_for_refusal_rows", _PARITY / "attention_bwd_dq_emit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dq_configs():
    """(arch, AttnBwdDqSpec) of every from-scratch dQ parity config."""
    m = _emitter()
    out = []
    for idx in range(len(m._CONFIGS)):
        (kind, fields, _knobs), arch = m._spec(idx)
        if kind != "dq":
            continue
        try:
            out.append((arch, AttnBwdDqSpec(**dict(fields))))
        except ValueError:
            continue  # refused at construction, before any table
    return out


def _refused_at(arch, spec, row, names, table):
    if arch != row[0] or tuple(getattr(spec, n) for n in names) != tuple(row[1:]):
        return False
    with pytest.raises(ValueError, match=table):
        validate_attn_bwd_dq_spec(spec, arch)
    return True


@pytest.mark.parametrize("row", sorted(DQ_SPILLING_COMBOS), ids=str)
def test_every_spilling_combo_has_a_parity_config(dq_configs, row):
    hits = [s for a, s in dq_configs if _refused_at(a, s, row, _COMBO, "spills VGPRs")]
    assert hits, f"no attention_bwd_dq_emit.py config for DQ_SPILLING_COMBOS row {row}"


@pytest.mark.parametrize("row", sorted(DQ_SCHED_SPILLING), ids=str)
def test_every_sched_spilling_row_has_a_parity_config(dq_configs, row):
    hits = [
        s for a, s in dq_configs if _refused_at(a, s, row, _SCHED, "DQ_SCHED_SPILLING")
    ]
    assert hits, f"no attention_bwd_dq_emit.py config for DQ_SCHED_SPILLING row {row}"
