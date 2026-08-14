# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the gfx942 sliced-K ring slice width (``k_slice_hd``).

The ring stages K as ``[ring_depth, T, k_slice_hd]``, so the width sets both
``k_groups = head_size // k_slice_hd`` and the K_lds row stride. The QK read has
32 lanes take 32 consecutive K rows at a fixed column, so the row stride in
dwords is what decides how many LDS banks the read spreads over.

This file pins the *plumbing*, not a routing choice: the width is a real spec
field with a selector, a cache-key entry, a kernel-name tag and a validator, and
it defaults to the shipped 32 so nothing moves until a selector change asks it
to. The tests are therefore mostly "this knob is inert at its default, live when
turned, and rejected when illegal".

Layers:
  * Selector / spec / validator assertions (pure Python).
  * Emitted-IR assertions: identical IR at the default, changed IR (and a
    predictable LDS delta) at width 16, on both llvm flavors.

No GPU required.
"""

from __future__ import annotations

import hashlib
import re

import pytest

from kernels import UnifiedAttentionProblem, build_unified_attention_2d_tiled
from kernels.common import attention_unified as au
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    supports_tiled_2d,
)

ENV = "HIPDNN_GFX942_K_SLICE_HD"

# ``@name = ... addrspace(3) global [N x T]`` -- the LDS allocations.
_LDS_RE = re.compile(
    r"^@[\w.$]+ = .*addrspace\(3\) global \[(\d+) x ([a-z0-9]+)\]", re.MULTILINE
)
_ELEM_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "half": 16, "float": 32}


@pytest.fixture
def gfx942(monkeypatch):
    # Set the memoized arch global directly (same pattern as the sibling ring
    # test): the resolver caches into _RESOLVED_ATTENTION_ARCH, so replacing only
    # the function leaves the cached global stale and leaks into sibling tests.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    for var in (
        ENV,
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_K_LDSSEQ",
        "HIPDNN_GFX942_BF16_WIDE",
        "HIPDNN_GFX942_D128_SMALLTILE_DK",
        "HIPDNN_GFX942_FLASH_MLIM",
        "HIPDNN_GFX942_FLASH_WIDE",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


def _problem(dtype, sq=4096, hq=32, hk=8, d=128, bs=64):
    return UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
    )


def _lds_bytes(llvm: str) -> int:
    total = 0
    for count, elem in _LDS_RE.findall(llvm):
        assert elem in _ELEM_BITS, f"unknown LDS element type {elem!r}"
        total += int(count) * _ELEM_BITS[elem] // 8
    return total


def _lower(spec, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    return _lower_kernel_to_llvm_python(kernel, arch="gfx942", llvm_flavor=flavor)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spec(dtype="fp16", d=128, **kw):
    """Build a spec directly, so depth and width are independent of what the
    production selector currently routes."""
    base = {
        "head_size": d,
        "block_size": 64,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "dtype": dtype,
        "use_sinks": False,
        "sliding_window": 0,
        "has_softcap": False,
    }
    base.update(kw)
    return UnifiedAttention2DTiledSpec(**base)


def _ring_spec(dtype="fp16", d=128, ring_depth=3, k_slice_hd=32, t=64, nw=4):
    return _spec(
        dtype=dtype,
        d=d,
        num_warps=nw,
        block_m_per_warp=32,
        tile_size=t,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_conflict_free_v_store=True,
        use_k_sliced_ring=True,
        ring_depth=ring_depth,
        k_slice_hd=k_slice_hd,
    )


def _capture_gate_kwargs(arch, dtype):
    """Record the kwargs the shared feasibility caller passes to an arch's gate.

    The caller binds ``supports_tiled_2d`` as a local from ``_tiled_2d_impl(arch)``,
    so the seam to intercept is that lookup rather than a module attribute.
    """
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return (False, "spy")

    spec_cls, build, _real = au._tiled_2d_impl(arch)
    old_arch = au._RESOLVED_ATTENTION_ARCH
    old_impl = au._tiled_2d_impl
    au._RESOLVED_ATTENTION_ARCH = arch
    au._tiled_2d_impl = lambda _a: (spec_cls, build, _spy)
    try:
        au.supports_native_unified_attention_tiled(_problem(dtype, d=128, bs=64))
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._tiled_2d_impl = old_impl
        au._2D_LAUNCH_META.clear()
    return captured


def _nonring_spec(dtype="bf16", d=128, k_slice_hd=32, nw=2, t=64):
    return _spec(
        dtype=dtype,
        d=d,
        num_warps=nw,
        block_m_per_warp=32,
        tile_size=t,
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_conflict_free_v_store=True,
        k_slice_hd=k_slice_hd,
    )


# ---------------------------------------------------------------------------
# Default is the shipped width -- nothing moves without a selector change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("d,bs", [(128, 64), (64, 16)])
def test_selector_returns_the_shipped_width(gfx942, dtype, d, bs):
    # Nothing narrows yet. This change only makes the width expressible; selecting
    # a narrower one requires mirroring it into the C++ engine first.
    p = _problem(dtype, d=d, bs=bs)
    assert au._select_gfx942_flash_k_slice_hd(p) == 32
    assert au._tiled_spec_from_problem(p).k_slice_hd == 32


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250", "gfx1151", "gfx1201"])
def test_ring_params_are_not_passed_to_other_arch_gates(arch):
    """The shared feasibility caller must not hand these to gates that lack them.

    ``supports_native_unified_attention_tiled`` dispatches to a per-arch
    ``supports_tiled_2d``, and only gfx942's declares the sliced-K ring parameters.
    Passing them unconditionally is the failure mode #10126 fixes for a different
    flag.

    Asserted by capturing the kwargs the shared caller actually passes, rather than
    by provoking a ``TypeError``: the exception route is unreliable in both
    directions, since a pre-existing gap can mask which parameter was at fault, and
    once that gap is fixed no exception fires at all and the test would silently
    stop checking anything.
    """
    captured = _capture_gate_kwargs(arch, "bf16")
    assert captured, "the shared caller never reached supports_tiled_2d"
    for param in ("ring_depth", "k_slice_hd"):
        assert param not in captured, (
            f"{arch} feasibility check was passed {param!r}; only the gfx942 gate "
            f"declares it. Passed: {sorted(captured)}"
        )


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_ring_params_are_passed_to_the_gfx942_gate(dtype):
    """The counterpart: gfx942's gate must receive them on every route into it.

    Both dtypes, because the shared caller has two call sites -- the bf16-wide
    branch returns early -- and an estimate that omits them sizes a different kernel
    than the builder emits.
    """
    captured = _capture_gate_kwargs("gfx942", dtype)
    assert captured, "the shared caller never reached supports_tiled_2d"
    assert captured.get("ring_depth") in (2, 3)
    assert captured.get("k_slice_hd") == 32


def test_spec_field_defaults_to_32():
    # A spec built without the field must be the shipped geometry, so existing
    # callers (and any serialized spec) keep their meaning.
    assert _spec().k_slice_hd == 32


@pytest.mark.parametrize("width", [8, 16, 64])
def test_env_override_selects_width(gfx942, monkeypatch, width):
    monkeypatch.setenv(ENV, str(width))
    p = _problem("fp16")
    assert au._select_gfx942_flash_k_slice_hd(p) == width
    assert au._tiled_spec_from_problem(p).k_slice_hd == width


@pytest.mark.parametrize(
    "bogus",
    [
        "",
        "0",
        "12",
        "24",
        "48",
        "999",
        "sixteen",
        "-16",
        # Numeric but not decimal: str.isdigit() accepts these and int() rejects
        # them, so an isdigit guard would raise on the dispatch path.
        "\u00b2",
        "\u2075",
    ],
)
def test_env_override_out_of_range_falls_back_to_32(gfx942, monkeypatch, bogus):
    # A malformed or illegal override must not be able to turn a supported problem
    # into a spec the validator rejects, and must never raise.
    monkeypatch.setenv(ENV, bogus)
    p = _problem("fp16")
    assert au._select_gfx942_flash_k_slice_hd(p) == 32
    assert au._tiled_spec_from_problem(p).k_slice_hd == 32


# ---------------------------------------------------------------------------
# Kernel identity: name tag and launcher cache key
# ---------------------------------------------------------------------------


def test_kernel_name_untagged_at_default_and_tagged_when_narrowed():
    # The tag is suppressed at 32 so every already-shipped ring kernel keeps its
    # current symbol name (and its cached HSACO).
    tokens = _ring_spec(k_slice_hd=32).kernel_name().split("_")
    assert not any(t.startswith("ks") and t != "ksring" for t in tokens), tokens
    name16 = _ring_spec(k_slice_hd=16).kernel_name()
    assert "ks16" in name16
    assert _ring_spec(k_slice_hd=32).kernel_name() != name16


def test_kernel_name_untagged_when_ring_off():
    # Off the ring the width is meaningless, so it must not perturb the name.
    assert _spec(k_slice_hd=32).kernel_name() == _spec(k_slice_hd=16).kernel_name()


def test_cache_key_distinguishes_width_on_the_ring(gfx942, monkeypatch):
    # Two widths are different schedules with different LDS footprints. If the key
    # did not separate them, the second width in a process would silently reuse the
    # first one's cached launcher -- the failure mode #10102 hit with batch.
    p = _problem("fp16")  # fp16 D128 is on the ring
    assert au._enable_gfx942_flash_k_sliced_ring(p)
    monkeypatch.setenv(ENV, "32")
    key32 = au._tiled_cache_key(p)
    monkeypatch.setenv(ENV, "16")
    key16 = au._tiled_cache_key(p)
    assert key32 != key16


def test_cache_key_ignores_width_off_the_ring(gfx942, monkeypatch):
    p = _problem("bf16")  # bf16 D128 is non-ring
    assert not au._enable_gfx942_flash_k_sliced_ring(p)
    monkeypatch.setenv(ENV, "32")
    key32 = au._tiled_cache_key(p)
    monkeypatch.setenv(ENV, "16")
    key16 = au._tiled_cache_key(p)
    assert key32 == key16


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width",
    [
        # Do not divide head_size=128: the last slice would read past the K_lds row.
        0,
        -16,
        12,
        24,
        48,
        96,
        # Divide 128 but are below/not a multiple of the 8-element QK k-step, so
        # k_steps_per_group floors to zero and the slice would emit no MFMA. These
        # are the cases that exercise the second validator branch; without them it
        # is dead, because every value above is already caught by the first.
        1,
        2,
        4,
    ],
)
def test_validator_rejects_illegal_width(width):
    with pytest.raises(ValueError, match="k_slice_hd"):
        _ring_spec(k_slice_hd=width)


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_validator_accepts_legal_width(width):
    assert _ring_spec(k_slice_hd=width).k_slice_hd == width


def test_validator_ignores_width_off_the_ring():
    # Off the ring the field is unused, so an odd value must not be an error --
    # only the ring schedule constrains it.
    assert _spec(k_slice_hd=12).k_slice_hd == 12


# ---------------------------------------------------------------------------
# LDS footprint: the ahead-of-time budget gate must track depth AND width
# ---------------------------------------------------------------------------


def _gate_admits(ring_depth, k_slice_hd, t=128, head_size=128):
    """Whether the ahead-of-time LDS-budget gate admits this ring configuration."""
    ok, _reason = supports_tiled_2d(
        head_size=head_size,
        block_size=64,
        dtype="fp16",
        num_queries_per_kv=4,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype=None,
        num_warps=4,
        block_m_per_warp=32,
        tile_size=t,
        arch="gfx942",
        use_mfma_32x32x8=True,
        use_transposed_qk_32x32=True,
        use_conflict_free_v_store=True,
        use_k_sliced_ring=True,
        ring_depth=ring_depth,
        k_slice_hd=k_slice_hd,
    )
    return ok


@pytest.mark.parametrize("ring_depth", [2, 3])
@pytest.mark.parametrize("k_slice_hd", [16, 32])
@pytest.mark.parametrize("t,head_size", [(64, 64), (64, 128), (128, 64), (128, 128)])
def test_budget_gate_admits_every_legal_ring_config(
    ring_depth, k_slice_hd, t, head_size
):
    """Admission must not change for any ring configuration the spec allows.

    Worth being explicit about what this does and does not prove. Teaching the gate
    the real depth and width makes its estimate match what the builder emits, but it
    cannot flip an admission decision for any currently-legal ring config: the
    largest estimate the old code could produce (3 slots of 32) already fits the
    64 KB budget at every legal T and head size, so every accurate estimate -- all
    of which are smaller -- also fits. The change is therefore a consistency fix
    rather than an unlock, and this test is a regression guard on that: it fails if
    someone inverts the arithmetic badly enough to start rejecting valid work.
    """
    assert _gate_admits(ring_depth, k_slice_hd, t=t, head_size=head_size)


# ---------------------------------------------------------------------------
# Emitted IR
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22"])
def test_width_is_inert_off_the_ring(flavor):
    # The non-ring path never reads the field, so the IR must be untouched. This is
    # what makes a future selector change safe to scope to the ring only.
    ir32 = _lower(_nonring_spec(k_slice_hd=32), flavor)
    ir16 = _lower(_nonring_spec(k_slice_hd=16), flavor)
    assert _sha(ir32) == _sha(ir16)


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22"])
@pytest.mark.parametrize("ring_depth", [2, 3])
def test_narrowing_changes_ring_ir(flavor, ring_depth):
    ir32 = _lower(_ring_spec(ring_depth=ring_depth, k_slice_hd=32), flavor)
    ir16 = _lower(_ring_spec(ring_depth=ring_depth, k_slice_hd=16), flavor)
    assert _sha(ir32) != _sha(ir16)


@pytest.mark.parametrize("ring_depth", [2, 3])
@pytest.mark.parametrize("t", [64, 128])
def test_lds_delta_is_exactly_the_k_slot_arithmetic(ring_depth, t):
    # K_lds is [ring_depth, T, width] at 2 bytes/element, and nothing else in the
    # kernel depends on the width, so halving it must free exactly
    # ring_depth * T * 16 * 2 bytes -- no more, no less.
    spec32 = _ring_spec(ring_depth=ring_depth, k_slice_hd=32, t=t)
    spec16 = _ring_spec(ring_depth=ring_depth, k_slice_hd=16, t=t)
    lds32 = _lds_bytes(_lower(spec32, "llvm20"))
    lds16 = _lds_bytes(_lower(spec16, "llvm20"))
    assert lds32 - lds16 == ring_depth * t * 16 * 2


@pytest.mark.parametrize("ring_depth", [2, 3])
def test_narrowing_lowers_lds(ring_depth):
    # Sanity in the direction that matters for occupancy: a narrower slice is never
    # more LDS.
    spec32 = _ring_spec(ring_depth=ring_depth, k_slice_hd=32)
    spec16 = _ring_spec(ring_depth=ring_depth, k_slice_hd=16)
    assert _lds_bytes(_lower(spec16, "llvm20")) < _lds_bytes(_lower(spec32, "llvm20"))
