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
  * Invariant sweep: across every selector threshold, two problems share a key
    exactly when they build the same spec, builder route and compile backend.
  * Targeted pairs: shapes that resolve to different selector outputs get
    different keys; shapes that resolve to the same outputs share a key and
    lower to byte-identical IR.
  * Launch geometry: the grid caches still separate batch counts.
  * ``_binary_search_iters`` mirrors every spec class's ``binary_search_iters``.

No GPU required.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from kernels.common import attention_unified as au
from kernels.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx1250Spec2D,
)
from kernels.gfx1250.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec as Gfx1250Spec3D,
)
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
from rocke.core.lower_llvm import LLVM_FLAVORS, _lower_kernel_to_llvm_python

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
def routing_env(monkeypatch):
    """Clear the routing env knobs so selectors see production routing. The arch
    is passed explicitly to every selector, never read from the device."""
    for var in _ROUTING_ENV:
        monkeypatch.delenv(var, raising=False)


def _norm_spec(spec):
    """Spec identity as the emitter sees it: ``num_seqs`` only reaches IR
    through ``binary_search_iters``, which the 3D decode grid does not emit."""
    fields = dataclasses.asdict(spec)
    fields.pop("num_seqs")
    if not getattr(spec, "use_decode_grid", False):
        fields["binary_search_iters"] = spec.binary_search_iters
    return fields


def _built_2d(problem, arch):
    route = au._gfx942_4warp_route(problem, arch)
    return dict(
        spec=_norm_spec(au._tiled_spec_from_problem(problem, arch)),
        builder=route.cohort if route is not None else "default",
        backend=au._select_2d_compile_backend(problem, arch),
    )


def _built_3d(problem, arch):
    return dict(
        spec=_norm_spec(au._tiled_3d_spec_from_problem(problem, arch)),
        reduce=(
            problem.head_size,
            problem.num_query_heads,
            problem.num_kv_heads,
            problem.dtype,
            au._num_segments(problem, arch),
            au._select_3d_waves_per_eu(problem, arch),
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


# Env states to pool per arch. Routing knobs are read on every launch and can
# flip mid-process; pooling them catches a selector whose output only changes
# when a knob flips.
_ENV_STATES = {
    "gfx942": (
        {},
        {"HIPDNN_GFX942_FLASH_WIDE": "0"},
        {"HIPDNN_GFX942_FLASH_WIDE": "2"},
    ),
    "gfx950": (
        {},
        {
            "HIPDNN_GFX950_D128_SMALL_TILE": "0",
            "HIPDNN_GFX950_D128_SOFTMAX_INTERLEAVE": "off",
        },
    ),
}


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("path", ("2d", "3d"))
def test_key_matches_build_identity(routing_env, monkeypatch, arch, path):
    """Key equality and build equality must coincide across the sweep.

    Equal key, different build: under-keying. This is the guard for the next
    selector; one added to the spec builder but not to the key fails here as
    soon as the sweep straddles its threshold.

    Equal build, different key: over-keying. A raw shape field, or a selector
    the arch's builder never reads, compiles duplicate binaries.

    Pooling the env states also catches a selector whose output only changes
    when a knob flips.
    """
    if path == "2d":
        supported = au.supports_native_unified_attention_tiled
        key_fn, built_fn = au._tiled_cache_key, _built_2d
    else:
        supported = au.supports_native_unified_attention_3d_tiled
        key_fn, built_fn = au._tiled_3d_cache_key, _built_3d

    first = {}
    key_of_build = {}
    checked = 0
    for env in _ENV_STATES.get(arch, ({},)):
        for var, val in env.items():
            monkeypatch.setenv(var, val)
        for problem in _sweep(arch):
            if not supported(problem, arch)[0]:
                continue
            try:
                # gfx1250's 3D key builds the spec itself, so it raises too.
                key = key_fn(problem, arch)
                built = built_fn(problem, arch)
            except (ValueError, NotImplementedError):
                # Admitted by the gate but rejected by the spec; never launches.
                continue
            checked += 1
            launch = (problem, env)
            seen = first.setdefault(key, (launch, built))
            if seen[1] != built:
                other_launch, other_built = seen
                diff = {
                    name: (other_built["spec"].get(name), built["spec"].get(name))
                    for name in built["spec"]
                    if other_built["spec"].get(name) != built["spec"].get(name)
                }
                for name in built:
                    if name != "spec" and other_built[name] != built[name]:
                        diff[name] = (other_built[name], built[name])
                pytest.fail(
                    f"{arch} {path}: one key, two kernels\n"
                    f"  {other_launch}\n  {launch}\n"
                    f"  differing: {diff}"
                )
            build_id = repr(
                sorted((name, repr(value)) for name, value in built.items())
            )
            other_key, other_launch = key_of_build.setdefault(build_id, (key, launch))
            if other_key != key:
                diff = [
                    (i, a, b) for i, (a, b) in enumerate(zip(other_key, key)) if a != b
                ]
                pytest.fail(
                    f"{arch} {path}: one kernel, two keys\n"
                    f"  {other_launch}\n  {launch}\n"
                    f"  differing key entries (index, a, b): {diff}"
                )
        for var in env:
            monkeypatch.delenv(var, raising=False)
    assert checked > 0


# ---------------------------------------------------------------------------
# Different selector outputs -> different keys
# ---------------------------------------------------------------------------


def _assert_split(key_fn, built_fn, a, b, field, arch):
    assert built_fn(a, arch)["spec"][field] != built_fn(b, arch)["spec"][field]
    assert key_fn(a, arch) != key_fn(b, arch)


@pytest.mark.parametrize("arch", ("gfx942", "gfx950", "gfx1250"))
def test_search_bucket_edge_splits_2d_key(routing_env, arch):
    # num_seqs 3 and 4 fall in different log2 search buckets (2 vs 3 iterations).
    # d64 GQA-8 (+ fp8 K/V on gfx1250): a geometry every arch's tiled 2D supports.
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
    _assert_split(au._tiled_cache_key, _built_2d, a, b, "binary_search_iters", arch)
    # The trip count is a literal in the IR, so the binaries really differ.
    assert _lower_2d(a, arch, "llvm22") != _lower_2d(b, arch, "llvm22")


@pytest.mark.parametrize("arch", ("gfx942", "gfx950"))
def test_search_bucket_edge_splits_3d_key(routing_env, arch):
    # Two query tokens per sequence, so the launch takes the searching prefill
    # grid rather than the decode grid.
    base = dict(
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=2,
        max_seqlen_k=8192,
        dtype="bf16",
    )
    a = P(total_q=6, num_seqs=3, **base)
    b = P(total_q=8, num_seqs=4, **base)
    assert not au._use_decode_grid(a, arch)
    _assert_split(au._tiled_3d_cache_key, _built_3d, a, b, "binary_search_iters", arch)


def test_fp8_mfma_qk_kv_threshold_splits_key(routing_env):
    # T = 2 * block_size = 32 -> the K-in-LDS fp8 path turns off above
    # max_seqlen_k = 16 * T = 512. Nothing else in the key reads max_seqlen_k.
    arch = "gfx950"
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
    assert au._enable_fp8_mfma_qk(a, arch) and not au._enable_fp8_mfma_qk(b, arch)
    _assert_split(au._tiled_cache_key, _built_2d, a, b, "use_fp8_mfma_qk", arch)


def test_single_batch_interleave_splits_key(routing_env):
    # The d128 single-batch cohort turns on the softmax/MFMA interleave; the
    # same shape at num_seqs=2 does not. The search bucket also separates this
    # pair, so the tail entry itself is pinned by the env-flip test below.
    arch = "gfx950"
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
    _assert_split(
        au._tiled_cache_key, _built_2d, a, b, "use_softmax_mfma_interleave", arch
    )


def test_env_knob_flip_splits_single_batch_key(routing_env, monkeypatch):
    # Routing env knobs are read on every launch and can change mid-process.
    # q=257 under default env and q=1025 with both d128 knobs off resolve the
    # same geometry; only the single-batch tail separates their keys.
    arch = "gfx950"
    base = dict(
        num_seqs=1,
        num_query_heads=8,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        dtype="bf16",
    )
    a = P(total_q=257, max_seqlen_q=257, max_seqlen_k=257, **base)
    key_a = au._tiled_cache_key(a, arch)
    spec_a = _built_2d(a, arch)["spec"]
    monkeypatch.setenv("HIPDNN_GFX950_D128_SMALL_TILE", "0")
    monkeypatch.setenv("HIPDNN_GFX950_D128_SOFTMAX_INTERLEAVE", "off")
    b = P(total_q=1025, max_seqlen_q=1025, max_seqlen_k=1025, **base)
    field = "use_softmax_mfma_interleave"
    assert spec_a[field] != _built_2d(b, arch)["spec"][field]
    assert key_a != au._tiled_cache_key(b, arch)


# ---------------------------------------------------------------------------
# Same selector outputs -> same key and byte-identical IR
# ---------------------------------------------------------------------------


def _lower_2d(problem, arch, flavor):
    _, build, _ = au._tiled_2d_impl(arch)
    spec = au._tiled_spec_from_problem(problem, arch)
    route = au._gfx942_4warp_route(problem, arch)
    kernel = route.builder(spec, arch=arch) if route else build(spec, arch=arch)
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=flavor)


def _lower_3d(problem, arch, flavor):
    _, reduce_spec_cls, build_seg, build_red, _ = au._tiled_3d_impl(arch)
    seg = build_seg(au._tiled_3d_spec_from_problem(problem, arch), arch=arch)
    red = build_red(
        reduce_spec_cls(
            head_size=problem.head_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            num_segments=au._num_segments(problem, arch),
            waves_per_eu=au._select_3d_waves_per_eu(problem, arch),
        ),
        arch=arch,
    )
    return tuple(
        _lower_kernel_to_llvm_python(k, arch=arch, llvm_flavor=flavor)
        for k in (seg, red)
    )


_D64_GQA8 = dict(
    num_query_heads=64, num_kv_heads=8, head_size=64, block_size=32, dtype="bf16"
)
_D128_DECODE = dict(
    num_query_heads=32,
    num_kv_heads=8,
    head_size=128,
    block_size=16,
    max_seqlen_q=1,
    max_seqlen_k=8192,
    dtype="bf16",
)

# (id, arch, path, problem_a, problem_b). Each pair resolves to the same
# selector outputs, so the pair must share one binary.
_SHARED_PAIRS = [
    *(
        (
            f"{arch}-2d-ns4-vs-ns7",
            arch,
            "2d",
            P(
                total_q=4 * 1024,
                num_seqs=4,
                max_seqlen_q=1024,
                max_seqlen_k=1024,
                use_fp8=arch == "gfx1250",
                **_D64_GQA8,
            ),
            P(
                total_q=7 * 1024,
                num_seqs=7,
                max_seqlen_q=1024,
                max_seqlen_k=1024,
                use_fp8=arch == "gfx1250",
                **_D64_GQA8,
            ),
        )
        for arch in ARCHES
    ),
    # Inside one max_seqlen_q band of the gfx950 combo cohort (512 < q <= 768).
    (
        "gfx950-2d-q600-vs-q700",
        "gfx950",
        "2d",
        P(total_q=1200, num_seqs=2, max_seqlen_q=600, max_seqlen_k=600, **_D64_GQA8),
        P(total_q=1400, num_seqs=2, max_seqlen_q=700, max_seqlen_k=700, **_D64_GQA8),
    ),
    # The decode grid emits no search, so batch sizes in different search
    # buckets share a binary. The decode split rule sizes segments by
    # num_seqs * num_kv_heads, so the pair also needs a shared segment count:
    # ns 31 and 64 both clamp to 2 segments, two buckets apart.
    *(
        (
            f"{arch}-3d-decode-ns31-vs-ns64",
            arch,
            "3d",
            P(total_q=31, num_seqs=31, **_D128_DECODE),
            P(total_q=64, num_seqs=64, **_D128_DECODE),
        )
        for arch in ("gfx942", "gfx950")
    ),
    # gfx1250 3D floors the search at 32 iterations, so ns 3 and 4 share a
    # binary even though they sit in different log2 buckets.
    (
        "gfx1250-3d-ns3-vs-ns4",
        "gfx1250",
        "3d",
        P(total_q=3, num_seqs=3, max_seqlen_q=1, max_seqlen_k=8192, **_D64_GQA8),
        P(total_q=4, num_seqs=4, max_seqlen_q=1, max_seqlen_k=8192, **_D64_GQA8),
    ),
]


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize(
    "arch,path,a,b", [p[1:] for p in _SHARED_PAIRS], ids=[p[0] for p in _SHARED_PAIRS]
)
def test_same_outputs_share_key_and_ir(routing_env, arch, path, a, b, flavor):
    key_fn, lower = (
        (au._tiled_cache_key, _lower_2d)
        if path == "2d"
        else (au._tiled_3d_cache_key, _lower_3d)
    )
    assert a.num_seqs != b.num_seqs or a.max_seqlen_q != b.max_seqlen_q
    assert key_fn(a, arch) == key_fn(b, arch)
    assert lower(a, arch, flavor) == lower(b, arch, flavor)


# ---------------------------------------------------------------------------
# Launch geometry still separates batch counts
# ---------------------------------------------------------------------------


def test_shape_key_carries_batch_and_token_counts():
    base = dict(max_seqlen_q=100, max_seqlen_k=100, **_D64_GQA8)
    a = P(total_q=400, num_seqs=4, **base)
    b = P(total_q=400, num_seqs=7, **base)
    c = P(total_q=700, num_seqs=7, **base)
    key = ("k",)
    assert au._shape_key(key, a) != au._shape_key(key, b)
    assert au._shape_key(key, b) != au._shape_key(key, c)


def test_launch_meta_separates_num_seqs_under_one_key(routing_env):
    # Same binary (one cache key) and same total_q, but the grid adds one
    # padding q-block per sequence, so the launch geometry must differ.
    arch = "gfx950"
    base = dict(total_q=400, max_seqlen_q=100, max_seqlen_k=100, **_D64_GQA8)
    a = P(num_seqs=4, **base)
    b = P(num_seqs=7, **base)
    key = au._tiled_cache_key(a, arch)
    assert key == au._tiled_cache_key(b, arch)
    au._2D_LAUNCH_META.clear()
    grid_a = au._get_2d_launch_meta(a, key, arch=arch).grid
    grid_b = au._get_2d_launch_meta(b, key, arch=arch).grid
    assert grid_b[1] - grid_a[1] == b.num_seqs - a.num_seqs


# ---------------------------------------------------------------------------
# _binary_search_iters mirrors the spec classes
# ---------------------------------------------------------------------------


# (spec class, floor). gfx1250 3D floors the search at 32 iterations.
_SPEC_CLASSES = (
    (Gfx942Spec2D, 1),
    (Gfx942Spec3D, 1),
    (Gfx950Spec2D, 1),
    (Gfx950Spec3D, 1),
    (Gfx1250Spec2D, 1),
    (Gfx1250Spec3D, 32),
)


@pytest.mark.parametrize("num_seqs", (0, 1, 2, 3, 4, 7, 8, 15, 16, 450, 1 << 20))
@pytest.mark.parametrize(
    "spec_cls, floor", _SPEC_CLASSES, ids=[c.__module__ for c, _ in _SPEC_CLASSES]
)
def test_binary_search_iters_mirrors_spec(spec_cls, floor, num_seqs):
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
    assert au._binary_search_iters(problem, floor=floor) == spec.binary_search_iters
