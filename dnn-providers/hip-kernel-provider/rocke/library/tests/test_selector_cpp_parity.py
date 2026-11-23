# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ gfx950 tiled-2D selector picks the spec the Python selector picks.

``rocke_engine.gfx950_attn_tiled_spec_from_problem`` runs the C++ selector and
its LDS-budget resolver; ``attention_unified._tiled_spec_from_problem`` runs the
Python selector and ``_resolve_lds_budget``. The binding returns the spec fields
the C++ selector sets, and each must equal the Python spec's field of the same
name.

The binding takes a ``UnifiedAttentionProblem`` as a dict, and its C struct has
no member for some of the dataclass's fields. A problem that sets one of those
is refused rather than selected as if the field were at its default.

Needs the ``rocke_engine`` binding (a source build with ``-DROCKE_BUILD_PYBIND=ON``
on ``PYTHONPATH``); skipped without it.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace
from typing import NamedTuple
from unittest import mock

import kernels.common.attention_unified as _au
import pytest
from kernels import UnifiedAttentionProblem
from rocke.tests.lanes import BINDING_LANE, import_or_skip_until

rocke_engine = import_or_skip_until("rocke_engine", BINDING_LANE)

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


class Geometry(NamedTuple):
    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    num_seqs: int
    max_seqlen_q: int


# The score modifiers each selector gate tests: ALiBi, QQ bias and softcap ride
# the transposed 32x32 path (the combo among them), sinks force its scalar
# state, and every one of them closes the D256 fast route.
SCORE_MODIFIERS = (
    {},
    {"use_alibi": True},
    {"use_qq_bias": True},
    {"softcap": 30.0},
    {"use_sinks": True},
    {"use_sinks": True, "use_alibi": True},
    {"use_alibi": True, "use_qq_bias": True, "softcap": 30.0},
)

# fp8 is a bf16 problem with an fp8 KV cache.
DTYPES = ({"dtype": "bf16"}, {"dtype": "fp16"}, {"dtype": "bf16", "use_fp8": True})

SLIDING_WINDOWS = (0, 256)

# One geometry per selector cohort and its nearest boundary.
COMPACT_GEOMETRIES = (
    Geometry(64, 32, 64, 8, 2, 512),  # combo
    Geometry(64, 32, 32, 4, 2, 512),  # combo, other GQA-8 head counts
    Geometry(64, 32, 64, 8, 2, 256),  # combo boundary: max_seqlen_q <= 256
    Geometry(64, 16, 64, 8, 2, 512),  # multi-batch transposed 32x32
    Geometry(128, 32, 32, 4, 8, 512),  # multi-batch d128
    Geometry(64, 32, 32, 32, 1, 512),  # single-batch combo d64, short
    Geometry(64, 64, 64, 8, 1, 2048),  # single-batch combo d64, long (early V)
    Geometry(128, 16, 64, 8, 1, 512),  # single-batch combo d128, short
    Geometry(128, 64, 32, 32, 1, 2048),  # single-batch combo d128, long
    Geometry(192, 16, 32, 32, 1, 2048),  # single-batch d192
    Geometry(256, 32, 64, 8, 2, 512),  # D256 prefill
    Geometry(256, 64, 64, 8, 8, 1),  # D256 decode, register-PV LDS budget
    Geometry(64, 16, 64, 8, 8, 1),  # decode
    Geometry(128, 32, 16, 4, 2, 64),  # short prefill
)

# Boolean spec fields no compact-grid problem sets, and why.
_CHOICES_THE_GRID_CANNOT_TAKE = {
    "emit_lse": "the binding refuses an LSE request",
    "use_i64_kv_addr": "needs a large num_kv_blocks; see the large-cache test",
    "use_sched_barrier": "needs the V double buffer (d64 only) at d128",
}


def _full_geometries():
    """Every head size, block size, head pairing and sequence shape the
    selectors branch on."""
    head_sizes = (64, 128, 192, 256)
    block_sizes = (16, 32, 64)
    head_pairs = ((64, 8), (32, 32), (16, 4), (32, 4))
    seq_shapes = tuple(itertools.product((1, 2, 8), (1, 64, 257, 512, 2048)))
    for head_size, block_size, heads, seqs in itertools.product(
        head_sizes, block_sizes, head_pairs, seq_shapes
    ):
        yield Geometry(head_size, block_size, *heads, *seqs)


def _problems(geometries):
    """Each geometry crossed with every dtype, window and score modifier."""
    for geometry, dtype, window, modifier in itertools.product(
        geometries, DTYPES, SLIDING_WINDOWS, SCORE_MODIFIERS
    ):
        yield replace(
            COMBO,
            total_q=geometry.num_seqs * geometry.max_seqlen_q,
            sliding_window=window,
            **geometry._asdict(),
            **dtype,
            **modifier,
        )


def _python_spec(problem: UnifiedAttentionProblem) -> dict:
    # The selector resolves the arch through this module handle; pin it so the
    # host's GPU does not leak into the selection.
    with mock.patch.object(_au, "_resolve_attention_arch", return_value=ARCH):
        return asdict(_au._tiled_spec_from_problem(problem, ARCH))


def _cpp_spec(problem: UnifiedAttentionProblem) -> dict:
    return rocke_engine.gfx950_attn_tiled_spec_from_problem(asdict(problem), ARCH)


def _differing_fields(python_spec: dict, cpp_spec: dict) -> dict:
    """``{field: (python, cpp)}`` for every field the C++ spec sets that the
    Python spec does not have, or has with another value."""
    missing = object()
    return {
        name: (python_spec.get(name, missing), value)
        for name, value in cpp_spec.items()
        if python_spec.get(name, missing) != value
    }


def _unported_d256_fast_route(problem: UnifiedAttentionProblem) -> bool:
    """Expected divergence: the gfx950 bf16 D256 prefill fast route.

    The Python selector narrows the launch geometry for this cohort and
    overrides the codegen knobs with ``_d256_gfx950_spec_overrides``; the C++
    selector narrows only the tile size and applies none of the knobs, and the
    C++ tiled-2D twin cannot build the route's Q-direct-register path
    (``platform/dsl_docs/development/known_gaps.md``). Until both are ported,
    these problems must differ, so the check fails as soon as the C++ selector
    starts matching and this exception can be deleted.
    """
    return _au._d256_gfx950_fast(problem, ARCH)


def _check_parity(problems) -> dict:
    """Assert C++ / Python spec parity over ``problems``; return the Python
    specs, keyed by problem, for coverage checks."""
    unexpected, expected_divergence_now_matches, python_specs = [], [], {}
    for problem in problems:
        python_spec = _python_spec(problem)
        differences = _differing_fields(python_spec, _cpp_spec(problem))
        python_specs[problem] = python_spec
        if _unported_d256_fast_route(problem):
            if not differences:
                expected_divergence_now_matches.append(problem)
        elif differences:
            unexpected.append((problem, differences))
    assert not unexpected, (
        f"{len(unexpected)} problems where the C++ selector picks another spec; "
        f"first: {unexpected[0]}"
    )
    assert not expected_divergence_now_matches, (
        "the C++ selector now matches the D256 fast route on "
        f"{expected_divergence_now_matches[0]}; delete _unported_d256_fast_route"
    )
    return python_specs


def test_compact_grid_matches_python():
    """Every selector cohort crossed with each dtype, window and score modifier
    (a few hundred problems), with the D256 fast route as the one named
    divergence."""
    python_specs = _check_parity(_problems(COMPACT_GEOMETRIES))
    # The grid must reach the routes it claims to cover: each boolean choice
    # the C++ selector makes is taken both ways, and the biased transposed
    # 32x32 cohort and the expected divergence are both present.
    for name, value in _cpp_spec(COMBO).items():
        if isinstance(value, bool) and name not in _CHOICES_THE_GRID_CANNOT_TAKE:
            taken = {spec[name] for spec in python_specs.values()}
            assert taken == {False, True}, f"{name} is always {taken}"
    assert any(
        spec["use_transposed_qk_32x32"] and problem.use_alibi
        for problem, spec in python_specs.items()
    )
    assert any(_unported_d256_fast_route(problem) for problem in python_specs)


def test_large_kv_cache_takes_i64_addressing_in_both():
    """A paged KV cache past the i32 buffer range switches both selectors to
    i64 addressing; below it neither does."""
    small_cache = replace(COMBO, num_kv_blocks=64)
    large_cache = replace(COMBO, num_kv_blocks=1 << 16)  # 2 GiB of bf16 K
    _check_parity((small_cache, large_cache))
    assert not _cpp_spec(small_cache)["use_i64_kv_addr"]
    assert _cpp_spec(large_cache)["use_i64_kv_addr"]


@pytest.mark.exhaustive
def test_full_grid_matches_python():
    """The full cross product of head size, block size, head pairing,
    sequence shape, dtype, window and score modifier (about 30,000 problems)."""
    _check_parity(_problems(_full_geometries()))


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
