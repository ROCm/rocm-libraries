# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every LAYOUT_SPILLING_COMBOS row has a parity config, in Python and in C.

The C main-kernel validator copies LAYOUT_SPILLING_COMBOS by value
(attention_bwd.cpp). The parity configs of attention_bwd_spec_emit are what
check it: both golden checks require the C builder to refuse a config Python
refuses. A row added in Python without a config, or a config missing from the
C table (attention_bwd_spec_emit.py --print-c-configs not re-run), would leave
the C validator unchecked on that row with every gate green; this test fails
instead.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from kernels.common.attention_bwd import (
    LAYOUT_SPILLING_COMBOS,
    AttnBwdSpec,
    _layout_spill_key,
    validate_attn_bwd_spec,
)

_PARITY = Path(__file__).resolve().parent / "parity"


def _emitter():
    if str(_PARITY) not in sys.path:
        sys.path.insert(0, str(_PARITY))
    spec = importlib.util.spec_from_file_location(
        "_spec_emit_for_layout_rows", _PARITY / "attention_bwd_spec_emit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _c_configs() -> set:
    """The config strings of the C twin's CONFIGS table."""
    text = (_PARITY / "attention_bwd_spec_emit.c").read_text()
    body = text[text.index("CONFIGS[] = {") :]
    body = body[: body.index("\n};")]
    out, cur = set(), None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("/*"):
            if cur is not None:
                out.add(cur)
            cur = ""
            continue
        for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', line):
            cur = (cur or "") + lit
    if cur is not None:
        out.add(cur)
    return out


def _row_hit(arch, fields):
    """The LAYOUT_SPILLING_COMBOS key a config is refused at, or None."""
    try:
        spec = AttnBwdSpec(**fields)
        validate_attn_bwd_spec(spec, arch, main_kernel_refusals=False)
    except ValueError as e:
        if "LAYOUT_SPILLING_COMBOS" in str(e):
            return _layout_spill_key(spec.resolved(arch), arch)
    return None


@pytest.fixture(scope="module")
def hits():
    """row -> the config strings refused at it."""
    m = _emitter()
    out = {}
    for _comment, config in m._CONFIGS:
        arch, fields = m._parse(config)
        row = _row_hit(arch, fields)
        if row is not None:
            out.setdefault(row, []).append(config)
    return out


@pytest.mark.parametrize("row", sorted(LAYOUT_SPILLING_COMBOS, key=str), ids=str)
def test_every_layout_spill_row_has_a_parity_config(hits, row):
    configs = hits.get(row)
    assert (
        configs
    ), f"no attention_bwd_spec_emit.py config for LAYOUT_SPILLING_COMBOS row {row}"
    in_c = [c for c in configs if c in _c_configs()]
    assert in_c, (
        f"the config for row {row} is not in attention_bwd_spec_emit.c: "
        "re-run attention_bwd_spec_emit.py --print-c-configs"
    )
