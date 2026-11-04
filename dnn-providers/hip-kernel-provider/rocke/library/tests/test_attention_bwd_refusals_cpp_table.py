# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ refusal table is the Python one.

The C++ twin of the recorded resource-gate refusals
(platform/cpp/instances/common/_attention_bwd_refusals.cpp) looks the
spec_digest key up in _attention_bwd_refusals_table.inc, which
``attention_bwd_refusals_emit.py --print-c-table`` generates from
RESOURCE_REFUSED and DQ_RESOURCE_REFUSED. A row added to (or dropped from)
either dict without regenerating the file would let the C++ validator build a
kernel Python refuses (or refuse one Python builds) with every IR gate green;
this test fails instead. The parity pair attention_bwd_refusals_emit checks
the C++ digest and verdicts on its configs.
"""

import importlib.util
import sys
from pathlib import Path

from kernels.common._attention_bwd_refusals import RESOURCE_REFUSED
from kernels.common.attention_bwd_dq import DQ_RESOURCE_REFUSED

_PARITY = Path(__file__).resolve().parent / "parity"
_TABLE = (
    Path(__file__).resolve().parents[2]
    / "platform"
    / "cpp"
    / "instances"
    / "common"
    / "_attention_bwd_refusals_table.inc"
)


def _emitter():
    if str(_PARITY) not in sys.path:
        sys.path.insert(0, str(_PARITY))
    spec = importlib.util.spec_from_file_location(
        "_refusals_emit_for_table", _PARITY / "attention_bwd_refusals_emit.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_cpp_table_is_generated_from_the_python_dicts():
    expected = _emitter().c_table()
    assert _TABLE.read_text() == expected, (
        f"{_TABLE.name} differs from RESOURCE_REFUSED / DQ_RESOURCE_REFUSED: "
        "regenerate it with `python library/tests/parity/"
        "attention_bwd_refusals_emit.py --print-c-table > "
        "platform/cpp/instances/common/_attention_bwd_refusals_table.inc`"
    )


def test_the_cpp_table_holds_every_row_once():
    text = _TABLE.read_text()
    for (arch, digest), _reason in (
        *RESOURCE_REFUSED.items(),
        *DQ_RESOURCE_REFUSED.items(),
    ):
        assert f'{{"{arch}", "{digest}", ' in text, (arch, digest)
    rows = [line for line in text.splitlines() if line.startswith('    {"')]
    assert len(rows) == len(RESOURCE_REFUSED) + len(DQ_RESOURCE_REFUSED)
