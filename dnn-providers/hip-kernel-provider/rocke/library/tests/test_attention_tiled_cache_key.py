# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cache-key contract for the tiled 2D / 3D attention launchers.

``_tiled_cache_key`` and ``_tiled_3d_cache_key`` are computed on every launch
without building the spec, so they must carry the resolved output of every
selector the spec builder consults. Two failure directions:

  * Under-keying: two problems share a key but build different kernels, and
    the cache serves whichever compiled first.
  * Over-keying: a raw shape field splits problems that build the same kernel,
    so every batch count or seqlen compiles its own copy.

Layers:
  * Invariant sweep: across every selector threshold, an equal key implies an
    equal spec (plus builder route and compile backend).
  * Targeted pairs: shapes that resolve to different selector outputs get
    different keys.
  * ``_binary_search_iters`` mirrors every spec class's ``binary_search_iters``.

No GPU required.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from kernels.common import attention_unified as au

P = au.UnifiedAttentionProblem

ARCHES = ("gfx942", "gfx950", "gfx1250")

# Env knobs that steer selectors. Cleared so the sweep sees production routing.
_ROUTING_ENV = (
    "HIPDNN_GFX942_K_SLICE_HD",
    "HIPDNN_GFX942_K_SLICED_RING",
    "HIPDNN_GFX942_K_LDSSEQ",
    "HIPDNN_GFX942_FLASH_WIDE",
    "HIPDNN_GFX942_FLASH_MLIM",
    "HIPDNN_GFX942_BF16_CFVST",
    "HIPDNN_GFX942_D128_SMALLTILE_DK",
    "HIPDNN_GFX942_3D_HOIST",
    "HIPDNN_GFX942_3D_WKV",
    "HIPDNN_GFX950_D128_SMALL_TILE",
    "HIPDNN_GFX950_D128_SOFTMAX_INTERLEAVE",
)


@pytest.fixture
def pin_arch(monkeypatch):
    """Pin the memoized arch (conftest restores it) and clear routing env."""
    for var in _ROUTING_ENV:
        monkeypatch.delenv(var, raising=False)

    def _pin(arch):
        au._RESOLVED_ATTENTION_ARCH = arch

    return _pin


def _norm_spec(spec):
    """Spec identity as the emitter sees it: ``num_seqs`` only reaches IR
    through ``binary_search_iters``."""
    fields = dataclasses.asdict(spec)
    fields.pop("num_seqs")
    fields["binary_search_iters"] = spec.binary_search_iters
    return fields


def _built_2d(problem):
    route = au._gfx942_4warp_route(problem)
    return dict(
        spec=_norm_spec(au._tiled_spec_from_problem(problem)),
        builder=route.cohort if route is not None else "default",
        backend=au._select_2d_compile_backend(problem),
    )


def _built_3d(problem):
    return dict(
        spec=_norm_spec(au._tiled_3d_spec_from_problem(problem)),
        reduce=(
            problem.head_size,
            problem.num_query_heads,
            problem.num_kv_heads,
            problem.dtype,
            au._num_segments(problem),
            au._select_3d_waves_per_eu(problem),
        ),
    )


# ---------------------------------------------------------------------------
# Invariant sweep
# ---------------------------------------------------------------------------

# Dynamic axes straddle every num_seqs / max_seqlen_q / max_seqlen_k threshold
# the selectors read: ns 1 vs >=2 and the log2 search buckets; q at 1, 64, 128,
# 256, 512, 768, 1024, 1536, 2048; k across 16 * T for T in {32, 64}.
_NUM_SEQS = (1, 2, 3, 4)
_SEQLEN_Q = (1, 2, 64, 65, 256, 257, 512, 513, 768, 769, 1024, 1025, 1536, 2048)
_SEQLEN_K = ("q", 512, 513, 1024, 1025)

# (sliding_window, use_sinks, bias) combinations; bias is one feature at a time.
_FEATURES = (
    (0, False, None),
    (128, False, None),
    (0, True, None),
    (0, False, "softcap"),
    (0, False, "alibi"),
)


def _static_grid(arch):
    for dtype, fp8, hd, bs, heads, (sw, sinks, bias) in itertools.product(
        ("bf16", "fp16"),
        (False, True),
        (64, 128, 256),
        (16, 32, 64),
        ((8, 8), (32, 8), (64, 8)),
        _FEATURES,
    ):
        yield dict(
            dtype=dtype,
            use_fp8=fp8,
            fp8_fnuz=fp8 and arch == "gfx942",
            head_size=hd,
            block_size=bs,
            num_query_heads=heads[0],
            num_kv_heads=heads[1],
            sliding_window=sw,
            use_sinks=sinks,
            softcap=30.0 if bias == "softcap" else 0.0,
            use_alibi=bias == "alibi",
        )


def _sweep(arch):
    for static in _static_grid(arch):
        for ns, q, k in itertools.product(_NUM_SEQS, _SEQLEN_Q, _SEQLEN_K):
            k = q if k == "q" else k
            if k < q:
                continue
            yield P(
                total_q=ns * q, num_seqs=ns, max_seqlen_q=q, max_seqlen_k=k, **static
            )


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("path", ("2d", "3d"))
def test_equal_key_implies_equal_build(pin_arch, arch, path):
    """No two problems may share a key while building different kernels.

    This is the guard for the next selector: one added to the spec builder but
    not to the key fails here as soon as the sweep straddles its threshold.
    """
    pin_arch(arch)
    if path == "2d":
        supported = au.supports_native_unified_attention_tiled
        key_fn, built_fn = au._tiled_cache_key, _built_2d
    else:
        supported = au.supports_native_unified_attention_3d_tiled
        key_fn, built_fn = au._tiled_3d_cache_key, _built_3d

    first = {}
    checked = 0
    for problem in _sweep(arch):
        if not supported(problem)[0]:
            continue
        try:
            # gfx1250's 3D key builds the spec itself, so it raises too.
            key = key_fn(problem)
            built = built_fn(problem)
        except (ValueError, NotImplementedError):
            # Admitted by the gate but rejected by the spec; never launches.
            continue
        checked += 1
        seen = first.setdefault(key, (problem, built))
        if seen[1] != built:
            other, other_built = seen
            diff = {
                name: (other_built["spec"].get(name), built["spec"].get(name))
                for name in built["spec"]
                if other_built["spec"].get(name) != built["spec"].get(name)
            }
            for name in built:
                if name != "spec" and other_built[name] != built[name]:
                    diff[name] = (other_built[name], built[name])
            pytest.fail(
                f"{arch} {path}: one key, two kernels\n  {other}\n  {problem}\n"
                f"  differing: {diff}"
            )
    assert checked > 0


# ---------------------------------------------------------------------------
# Different selector outputs -> different keys
# ---------------------------------------------------------------------------


def _assert_split(key_fn, built_fn, a, b, field):
    assert built_fn(a)["spec"][field] != built_fn(b)["spec"][field]
    assert key_fn(a) != key_fn(b)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_search_bucket_edge_splits_2d_key(pin_arch, arch):
    # num_seqs 3 and 4 fall in different log2 search buckets (2 vs 3 iterations).
    # d64 GQA-8 (+ fp8 K/V on gfx1250): a geometry every arch's tiled 2D supports.
    pin_arch(arch)
    base = dict(
        num_query_heads=64,
        num_kv_heads=8,
        head_size=64,
        block_size=32,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        dtype="bf16",
        use_fp8=arch == "gfx1250",
    )
    a = P(total_q=3 * 1024, num_seqs=3, **base)
    b = P(total_q=4 * 1024, num_seqs=4, **base)
    _assert_split(au._tiled_cache_key, _built_2d, a, b, "binary_search_iters")


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_search_bucket_edge_splits_3d_key(pin_arch, arch):
    pin_arch(arch)
    base = dict(
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=8192,
        dtype="bf16",
    )
    a = P(total_q=3, num_seqs=3, **base)
    b = P(total_q=4, num_seqs=4, **base)
    _assert_split(au._tiled_3d_cache_key, _built_3d, a, b, "binary_search_iters")


def test_fp8_mfma_qk_kv_threshold_splits_key(pin_arch):
    # T = 2 * block_size = 32 -> the K-in-LDS fp8 path turns off above
    # max_seqlen_k = 16 * T = 512. Nothing else in the key reads max_seqlen_k.
    pin_arch("gfx950")
    base = dict(
        total_q=1,
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=64,
        block_size=16,
        max_seqlen_q=1,
        dtype="bf16",
        use_fp8=True,
    )
    a = P(max_seqlen_k=512, **base)
    b = P(max_seqlen_k=513, **base)
    assert au._enable_fp8_mfma_qk(a) and not au._enable_fp8_mfma_qk(b)
    _assert_split(au._tiled_cache_key, _built_2d, a, b, "use_fp8_mfma_qk")


def test_single_batch_interleave_splits_key(pin_arch):
    # The d128 single-batch cohort turns on the softmax/MFMA interleave; the
    # same shape at num_seqs=2 does not.
    pin_arch("gfx950")
    base = dict(
        num_query_heads=8,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=257,
        max_seqlen_k=257,
        dtype="bf16",
    )
    a = P(total_q=257, num_seqs=1, **base)
    b = P(total_q=514, num_seqs=2, **base)
    _assert_split(au._tiled_cache_key, _built_2d, a, b, "use_softmax_mfma_interleave")


# ---------------------------------------------------------------------------
# _binary_search_iters mirrors the spec classes
# ---------------------------------------------------------------------------


def _spec_classes():
    from kernels.gfx942.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec as Gfx942Spec2D,
    )
    from kernels.gfx942.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx942Spec3D,
    )
    from kernels.gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec as Gfx950Spec2D,
    )
    from kernels.gfx950.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec as Gfx950Spec3D,
    )
    from kernels.gfx1250.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec as Gfx1250Spec2D,
    )

    return (Gfx942Spec2D, Gfx942Spec3D, Gfx950Spec2D, Gfx950Spec3D, Gfx1250Spec2D)


@pytest.mark.parametrize("num_seqs", (0, 1, 2, 3, 4, 7, 8, 15, 16, 450, 1 << 20))
@pytest.mark.parametrize("spec_cls", _spec_classes(), ids=lambda c: c.__module__)
def test_binary_search_iters_mirrors_spec(spec_cls, num_seqs):
    # binary_search_iters is a pure function of num_seqs; bypass __init__
    # validation so the comparison needs no full geometry.
    spec = object.__new__(spec_cls)
    object.__setattr__(spec, "num_seqs", num_seqs)
    problem = P(
        total_q=1,
        num_seqs=num_seqs,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=1,
        dtype="bf16",
    )
    assert au._binary_search_iters(problem) == spec.binary_search_iters
