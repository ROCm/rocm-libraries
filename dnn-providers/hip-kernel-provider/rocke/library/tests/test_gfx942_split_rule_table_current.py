# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Drift guard for hipDNN's gfx942 3D decode split-rule table.

The hip-kernel-provider ingestor engine restates ``_num_segments`` in C++ and
checks it against ``Gfx942AttentionUnified3dSplitRuleTable.inc``, a table
generated from this library. Re-running the generator here and comparing it to
the committed table byte-for-byte makes a change to ``_num_segments`` (or the
problem it is fed) fail rocKE CI, not only the provider build. Fix a failure by
regenerating the table (command in the generator docstring) and updating the
C++ restatement in ``Gfx942AttentionUnified3dGeometry.hpp`` to match.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
# In-tree: <provider>/rocke/library/tests/<this>. The installed CTest layout
# stages library/tests as <install>/tests/library/tests, without the provider
# sources, so the table is unreachable there.
_LIBRARY = _HERE.parents[1]
_PROVIDER = _HERE.parents[3]
_IN_TREE = (
    _LIBRARY.name == "library"
    and _LIBRARY.parent.name == "rocke"
    and (_LIBRARY.parent / "platform" / "CMakeLists.txt").is_file()
)
_PACKS = _PROVIDER / "src" / "tests" / "engines" / "kernel_ingestor_engine" / "packs"
_GENERATOR = _PACKS / "gen_gfx942_attention_unified_3d_split_rule.py"
_TABLE = _PACKS / "Gfx942AttentionUnified3dSplitRuleTable.inc"


def test_gfx942_split_rule_table_is_current():
    if not _IN_TREE:
        pytest.skip(
            "installed test layout: the hip-kernel-provider sources holding "
            "Gfx942AttentionUnified3dSplitRuleTable.inc are not staged; "
            "this guard runs from the source tree only"
        )
    assert _GENERATOR.is_file(), f"missing generator: {_GENERATOR}"
    assert _TABLE.is_file(), f"missing table: {_TABLE}"

    spec = importlib.util.spec_from_file_location("_gen_gfx942_split_rule", _GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen.main()

    assert out.getvalue() == _TABLE.read_text(), (
        f"{_TABLE.name} is stale against rocKE's _num_segments: regenerate it with "
        f"the command in {_GENERATOR.name} and update the C++ split rule to match"
    )
