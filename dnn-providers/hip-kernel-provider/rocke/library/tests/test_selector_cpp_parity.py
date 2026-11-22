# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ gfx950 tiled-2D selector refuses problems it cannot represent.

``rocke_engine.gfx950_attn_tiled_spec_from_problem`` takes a
``UnifiedAttentionProblem`` as a dict, and its C struct has no member for some
of the dataclass's fields. A problem that sets one of those is refused rather
than selected as if the field were at its default.

Needs the ``rocke_engine`` binding (a source build with ``-DROCKE_BUILD_PYBIND=ON``
on ``PYTHONPATH``); skipped without it.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from unittest import mock

import kernels.common.attention_unified as _au
import pytest
from kernels import UnifiedAttentionProblem

rocke_engine = pytest.importorskip("rocke_engine")

ARCH = "gfx950"

# The combo cohort: bf16, d64, block size 32, GQA-8, max_seqlen_q > 256.
COMBO = UnifiedAttentionProblem(
    total_q=1024,
    num_seqs=2,
    num_query_heads=64,
    num_kv_heads=8,
    head_size=64,
    block_size=32,
    max_seqlen_q=512,
    max_seqlen_k=4096,
    dtype="bf16",
)


def _python_spec(problem: UnifiedAttentionProblem) -> dict:
    # The selector resolves the arch through this module handle; pin it so the
    # host's GPU does not leak into the selection.
    with mock.patch.object(_au, "_resolve_attention_arch", return_value=ARCH):
        return asdict(_au._tiled_spec_from_problem(problem, ARCH))


def _cpp_spec(problem: UnifiedAttentionProblem) -> dict:
    return rocke_engine.gfx950_attn_tiled_spec_from_problem(asdict(problem), ARCH)


@pytest.mark.parametrize(
    "field, value",
    [
        ("use_additive_bias", True),
        ("emit_lse", True),
        ("causal_top_left", True),
        ("right_bound", -1),
        ("right_bound", 7),
        ("waves_per_eu", 3),
    ],
)
def test_binding_refuses_fields_the_c_problem_cannot_carry(field, value):
    """Each of these changes the Python spec, so selecting as if it were at
    its default would hand back a different kernel without saying so."""
    problem = replace(COMBO, **{field: value})
    assert _python_spec(problem) != _python_spec(COMBO)
    with pytest.raises(ValueError, match=rf"{field}={value}: .*cannot carry this"):
        _cpp_spec(problem)
