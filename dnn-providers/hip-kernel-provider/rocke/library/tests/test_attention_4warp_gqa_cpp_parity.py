# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ gfx942 4-warp GQA builder and its host predicates agree with Python.

- Every spec dispatch builds for a problem routed to ``build_gfx942_4warp_gqa``
  lowers to the same LLVM-IR bytes through the C++ builder (D256 lean, D128
  double-buffer pipe with and without the GQA head-fold, D128 block_size 64),
  plus specs the builder admits off-route.
- The builder rejects what Python rejects, with Python's message.
- ``gfx942_gqa_fold_eligible``, ``gfx942_4warp_launch_grid`` and both spec-shaped
  support wrappers return Python's answer.

The byte-identity gate pins a fixed sample of these kernels at every LLVM
flavor (``parity/gfx942_attention_4warp_gqa_emit.{py,c}``); this sweeps more
shapes at one flavor."""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace

import pytest
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _gfx942_4warp_fast,
    _tiled_spec_from_problem,
    gfx942_4warp_launch_grid,
    gfx942_gqa_fold_eligible,
)
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_gfx942_4warp_gqa,
    supports_gfx942_4warp_gqa_spec,
    supports_tiled_2d_spec,
)

_ARCH = "gfx942"


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
    return pytest.importorskip("rocke_engine")


def _lower_python(kernel, arch):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(kernel, arch=arch)


def _problem(**kw) -> UnifiedAttentionProblem:
    base = {
        "total_q": 4096,
        "num_seqs": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 4096,
        "max_seqlen_k": 4096,
        "dtype": "bf16",
    }
    base.update(kw)
    return UnifiedAttentionProblem(**base)


def _prefill_problems():
    for dtype, head_size, block_size, (nq, nkv), window, num_seqs in itertools.product(
        ("bf16", "fp16"),
        (64, 128, 256),
        (16, 32, 64),
        ((32, 8), (40, 8), (16, 2), (8, 8)),
        (0, 1024, 4096),
        (1, 4),
    ):
        yield _problem(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_query_heads=nq,
            num_kv_heads=nkv,
            sliding_window=window,
            num_seqs=num_seqs,
            total_q=num_seqs * 2048,
            max_seqlen_q=2048,
        )


def _routed_specs():
    seen = {}
    for p in _prefill_problems():
        if _gfx942_4warp_fast(p, _ARCH):
            spec = _tiled_spec_from_problem(p, _ARCH)
            seen.setdefault(repr(sorted(asdict(spec).items())), spec)
    return list(seen.values())


def _off_route_specs():
    """Specs the builder admits that dispatch does not route to it today: D128
    causal (the window == 0 branches of the general body) and no waves_per_eu."""
    base = _tiled_spec_from_problem(_problem(sliding_window=4096), _ARCH)
    return [
        replace(base, sliding_window=0),
        replace(base, sliding_window=0, block_size=64, tile_size=64),
        replace(base, waves_per_eu=None),
        replace(base, num_seqs=0),
    ]


def test_routed_specs_cover_every_body():
    specs = _routed_specs()
    names = {build_gfx942_4warp_gqa(s, arch=_ARCH).name for s in specs}
    assert any(s.head_size == 256 for s in specs)
    assert any(s.head_size == 128 and s.block_size == 64 for s in specs)
    assert any(n.endswith("_4wgqa_fold") for n in names)
    assert any(
        s.head_size == 128 and n.endswith("_4wgqa") for s, n in zip(specs, names)
    )


@pytest.mark.parametrize("which", ["routed", "off_route"])
def test_cpp_builder_matches_python(engine, which):
    specs = _routed_specs() if which == "routed" else _off_route_specs()
    assert specs
    for spec in specs:
        expected = _lower_python(build_gfx942_4warp_gqa(spec, arch=_ARCH), _ARCH)
        actual = engine.gfx942_attention_4warp_gqa_lower_llvm(asdict(spec), _ARCH)
        assert actual == expected, spec


def _reject_cases():
    d128 = _tiled_spec_from_problem(_problem(sliding_window=4096), _ARCH)
    # The dispatched D256 spec sets register-PV, whose spec validation would
    # reject fp16 / sliding window before the builder runs.
    d256 = replace(
        _tiled_spec_from_problem(
            _problem(head_size=256, num_query_heads=16, num_kv_heads=2), _ARCH
        ),
        use_register_pv=False,
    )
    return [
        ("i64", d128, _ARCH, replace(d128, use_i64_kv_addr=True)),
        ("head_size", d128, _ARCH, replace(d128, head_size=64)),
        ("lean_fp16", d256, _ARCH, replace(d256, dtype="fp16")),
        ("lean_window", d256, _ARCH, replace(d256, sliding_window=1024)),
        ("arch", d128, "gfx90a", d128),
    ]


@pytest.mark.parametrize("case", _reject_cases(), ids=lambda c: c[0])
def test_cpp_builder_rejects_like_python(engine, case):
    _, _, arch, spec = case
    with pytest.raises(NotImplementedError) as py_err:
        build_gfx942_4warp_gqa(spec, arch=arch)
    with pytest.raises(RuntimeError) as cpp_err:
        engine.gfx942_attention_4warp_gqa_lower_llvm(asdict(spec), arch)
    assert str(cpp_err.value).endswith(str(py_err.value))


def test_fold_eligible_matches_python(engine):
    for args in itertools.product(
        (64, 128, 256),
        (1, 2, 4, 8),
        (0, 1, 4096),
        ("bf16", "fp16"),
        (16, 32, 64),
    ):
        assert engine.gfx942_gqa_fold_eligible(*args) == gfx942_gqa_fold_eligible(
            *args
        ), args


def test_launch_grid_matches_python(engine):
    for p in _prefill_problems():
        got = tuple(engine.gfx942_4warp_launch_grid(asdict(p)))
        assert got == gfx942_4warp_launch_grid(p), p
    bad = asdict(_problem(num_kv_heads=3))
    with pytest.raises(ValueError):
        engine.gfx942_4warp_launch_grid(bad)


@pytest.mark.parametrize("arch", [_ARCH, "gfx90a"])
def test_spec_wrappers_match_python(engine, arch):
    seen = set()
    compared = 0
    for p in _prefill_problems():
        try:
            spec = _tiled_spec_from_problem(p, _ARCH)
        except (RuntimeError, ValueError):
            continue
        key = repr(sorted(asdict(spec).items()))
        if key in seen:
            continue
        seen.add(key)
        d = asdict(spec)
        for py_fn, cpp_fn in (
            (supports_tiled_2d_spec, engine.gfx942_attention_tiled_2d_supports_spec),
            (
                supports_gfx942_4warp_gqa_spec,
                engine.gfx942_attention_4warp_gqa_supports_spec,
            ),
        ):
            assert tuple(cpp_fn(d, arch)) == py_fn(spec, arch=arch), (arch, spec)
        compared += 1
    assert compared


def test_spec_wrappers_see_spec_fields(engine):
    """A spec field the wrapper forwards changes the verdict on both sides."""
    spec = UnifiedAttention2DTiledSpec(
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        dtype="bf16",
        num_warps=8,
        tile_size=128,
    )
    want = supports_tiled_2d_spec(spec, arch=_ARCH)
    got = tuple(engine.gfx942_attention_tiled_2d_supports_spec(asdict(spec), _ARCH))
    assert got == want
    assert not want[0]
    assert tuple(
        engine.gfx942_attention_4warp_gqa_supports_spec(asdict(spec), _ARCH)
    ) == supports_gfx942_4warp_gqa_spec(spec, arch=_ARCH)
