# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ unified-attention routing and coverage gates agree with Python.

Routing (``select_path`` / ``select_2d`` / ``select_3d``) must match exactly.
The gates must reject with Python's reason wherever Python rejects. Where Python
admits a problem the C++ builders cannot build yet, the C++ gate rejects it with
a "not ported to C++" reason instead; each such reason is checked against the
condition that is meant to trigger it. For the tiled 2D gate the test derives
the exact expected answer, "not ported" reason included, from the Python
predicates, and checks that every problem C++ admits actually builds."""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, replace

import pytest
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _canonical_arch_for_selection,
    _enable_gfx942_flash_k_sliced_ring,
    _enable_gfx942_flash_mask_limit,
    _enable_v_double_buffer,
    _gfx942_4warp_fast,
    _mask_label,
    _reject_mask_on_arch,
    _reject_mask_semantics,
    _tiled_3d_spec_from_problem,
    supports_native_unified_attention,
    supports_native_unified_attention_3d_tiled,
    supports_native_unified_attention_tiled,
)

_ARCHES = (
    "gfx942",
    "gfx950",
    "gfx942:sramecc+",
    "GFX950",
    "gfx1250",
    "gfx90a",
    "gfx1100",
)

# (num_query_heads, num_kv_heads): MHA, nqpk 4 / 3 / 16 / 20.
_HEADS = ((8, 8), (32, 8), (24, 8), (32, 2), (40, 2))


@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("rocke_engine")


def _problem(**kw) -> UnifiedAttentionProblem:
    base = {
        "total_q": 4,
        "num_seqs": 4,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 8192,
        "dtype": "bf16",
    }
    base.update(kw)
    return UnifiedAttentionProblem(**base)


def _routing_problems():
    for (
        head_size,
        block_size,
        sliding_window,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        (nq, nkv),
        (num_cus, target_ctas),
        dtype,
    ) in itertools.product(
        (64, 128, 256),
        (16, 32, 64),
        (0, 128),
        (1, 7, 256, 1024),
        (1, 512, 513, 8192),
        (1, 4, 64),
        _HEADS,
        ((80, 0), (304, 0), (304, 16), (120, 100000)),
        ("bf16", "fp8"),
    ):
        yield _problem(
            total_q=num_seqs * max_seqlen_q,
            num_seqs=num_seqs,
            num_query_heads=nq,
            num_kv_heads=nkv,
            head_size=head_size,
            block_size=block_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sliding_window=sliding_window,
            num_cus=num_cus,
            target_ctas=target_ctas,
            dtype=dtype,
        )


def test_routing_matches_python(engine):
    for p in _routing_problems():
        d = asdict(p)
        assert engine.unified_attention_select_path(d) == p.select_path(), p
        assert engine.unified_attention_select_2d(d) == asdict(p.select_2d()), p
        attn, reduce = p.select_3d()
        assert engine.unified_attention_select_3d(d) == (
            asdict(attn),
            asdict(reduce),
        ), p


def test_routing_rejects_bad_head_ratio(engine):
    d = asdict(_problem(num_query_heads=30, num_kv_heads=8))
    for fn in (
        engine.unified_attention_select_path,
        engine.unified_attention_select_2d,
        engine.unified_attention_select_3d,
    ):
        with pytest.raises(ValueError):
            fn(d)


def _flag_variants(p: UnifiedAttentionProblem):
    for (
        (use_fp8, fp8_fnuz),
        q_dtype,
        bias,
        (top_left, right_bound),
    ) in itertools.product(
        ((False, False), (True, False), (True, True), (False, True)),
        (None, "fp16", "fp8"),
        (None, "use_alibi", "use_qq_bias", "use_additive_bias"),
        itertools.product((False, True), (-1, 0, 5)),
    ):
        kw = {
            "use_fp8": use_fp8,
            "fp8_fnuz": fp8_fnuz,
            "q_dtype": q_dtype,
            "causal_top_left": top_left,
            "right_bound": right_bound,
        }
        if bias is not None:
            kw[bias] = True
        yield replace(p, **kw)


def _shape_variants():
    for dtype, head_size, block_size, (nq, nkv) in itertools.product(
        ("fp16", "bf16", "fp32"),
        (64, 96, 128, 256),
        (8, 16, 64),
        ((8, 8), (24, 8), (32, 2), (40, 2), (30, 8)),
    ):
        yield _problem(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_query_heads=nq,
            num_kv_heads=nkv,
        )


def _gate_problems():
    yield from _shape_variants()
    for base in (_problem(), _problem(head_size=96)):
        yield from _flag_variants(base)


def _python_gate(fn, p, arch):
    try:
        return fn(p, arch)
    except ValueError as e:
        return False, str(e)


def test_tiled_3d_gate_matches_python(engine):
    for p, arch in itertools.product(_gate_problems(), _ARCHES):
        ok, reason = engine.unified_attention_supports_tiled_3d(asdict(p), arch)
        py_ok, py_reason = _python_gate(
            supports_native_unified_attention_3d_tiled, p, arch
        )
        canon = _canonical_arch_for_selection(arch)
        ctx = (p, arch, py_ok, py_reason)
        if reason == "gfx1250 tiled 3D is not ported to C++":
            assert not ok and canon == "gfx1250", ctx
        elif reason == f"tiled 3D {_mask_label(p)} masks are not ported to C++ yet":
            assert not ok and py_ok and canon == "gfx950" and not p.default_mask, ctx
        elif reason == "tiled 3D additive bias is not ported to C++ yet":
            assert not ok and py_ok and canon == "gfx950" and p.use_additive_bias, ctx
        else:
            assert (ok, reason) == (py_ok, py_reason), ctx


def test_tiled_3d_admitted_problems_build(engine):
    """Every problem the C++ 3D gate admits lowers through the C++ 3D segment
    builder, from the spec the C++ 3D selectors build.

    Python's gate also admits gfx942 problems whose KV cache needs 64-bit
    addressing, which the gfx942 3D spec rejects as not implemented; C++ must
    reject those the same way."""
    seen = set()
    for p, arch in itertools.product(_tiled_2d_shapes(), ("gfx942", "gfx950")):
        ok, _ = engine.unified_attention_supports_tiled_3d(asdict(p), arch)
        if not ok:
            continue
        hook = getattr(engine, f"{arch}_attn_tiled_3d_spec_from_problem")
        try:
            _tiled_3d_spec_from_problem(p, arch)
        except NotImplementedError as e:
            with pytest.raises(NotImplementedError, match=re.escape(str(e))):
                hook(asdict(p), arch)
            continue
        spec = hook(asdict(p), arch)
        key = (arch, repr(sorted(spec.items())))
        if key in seen:
            continue
        seen.add(key)
        lower = getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")
        assert lower(spec, arch), (p, arch, spec)


def test_scalar_gate_matches_python(engine):
    for p, arch in itertools.product(_gate_problems(), _ARCHES):
        ok, reason = engine.unified_attention_supports_scalar(asdict(p), arch)
        py_ok, py_reason = _python_gate(supports_native_unified_attention, p, arch)
        assert (ok, reason) == (py_ok, py_reason), (p, arch)


# One setting per gfx942 flash env knob the 2D gate reads.
_GFX942_ENVS = (
    {"HIPDNN_GFX942_FLASH_MLIM": "0"},
    {"HIPDNN_GFX942_FLASH_MLIM": " D64 "},
    {"HIPDNN_GFX942_K_SLICED_RING": "off"},
    {"HIPDNN_GFX942_FLASH_WIDE": "0"},
    {"HIPDNN_GFX942_FLASH_WIDE": "2"},
    {"HIPDNN_GFX942_BF16_CFVST": "false"},
    {"HIPDNN_GFX942_D128_SMALLTILE_DK": "Disabled"},
    {"HIPDNN_GFX942_K_SLICE_HD": "32"},
    {"HIPDNN_GFX942_K_SLICE_HD": " 008"},
)


def _tiled_2d_not_ported(p: UnifiedAttentionProblem, arch: str):
    """The "not ported" reason C++ must give for a problem Python admits."""
    if arch == "gfx942":
        if _gfx942_4warp_fast(p, arch):
            return "gfx942 4-warp GQA tiled 2D is not ported to C++ yet"
        if _enable_gfx942_flash_k_sliced_ring(p, arch):
            return "gfx942 tiled 2D sliced-K ring is not ported to C++ yet"
        if _enable_gfx942_flash_mask_limit(p, arch):
            return "gfx942 tiled 2D flash mask-limit is not ported to C++ yet"
        return None
    if p.use_additive_bias:
        return "tiled 2D additive bias is not ported to C++ yet"
    if not p.default_mask:
        return f"tiled 2D {_mask_label(p)} masks are not ported to C++ yet"
    if _enable_v_double_buffer(p, arch):
        return "tiled 2D V double-buffer is not ported to C++ yet"
    return None


def _expected_tiled_2d(p: UnifiedAttentionProblem, arch: str):
    """The exact (ok, reason) the C++ 2D gate must return."""
    canon = _canonical_arch_for_selection(arch)
    if canon == "gfx1250":
        # C++ has no gfx1250 tiled 2D: only the shared rejects run first.
        rejected = _reject_mask_semantics(p) or _reject_mask_on_arch(p, canon)
        if rejected is not None:
            return rejected
        if p.softcap > 0:
            return False, "gfx1250 tiled 2D does not support softcap yet"
        if p.use_additive_bias:
            return False, "additive bias is only implemented on gfx950"
        return False, "gfx1250 tiled 2D is not ported to C++"
    ok, reason = _python_gate(supports_native_unified_attention_tiled, p, arch)
    if not ok:
        return ok, reason
    not_ported = _tiled_2d_not_ported(p, canon)
    return (False, not_ported) if not_ported else (ok, reason)


def _tiled_2d_shapes():
    for (
        dtype,
        head_size,
        block_size,
        (nq, nkv),
        num_seqs,
        max_seqlen_q,
        sliding_window,
        extra,
    ) in itertools.product(
        ("fp16", "bf16"),
        (64, 128, 256),
        (16, 32, 64),
        ((8, 8), (32, 8), (64, 8), (40, 2)),
        (1, 4),
        (1, 512, 1024, 2048),
        (0, 256),
        (None, "use_sinks", "softcap", "use_alibi", "use_qq_bias", "num_kv_blocks"),
    ):
        kw = {}
        if extra == "softcap":
            kw["softcap"] = 30.0
        elif extra == "num_kv_blocks":
            kw["num_kv_blocks"] = 1 << 22  # past the i32 KV buffer range
        elif extra is not None:
            kw[extra] = True
        yield _problem(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_query_heads=nq,
            num_kv_heads=nkv,
            num_seqs=num_seqs,
            total_q=num_seqs * max_seqlen_q,
            max_seqlen_q=max_seqlen_q,
            sliding_window=sliding_window,
            **kw,
        )


def _check_tiled_2d(engine, problems, arches):
    for p, arch in itertools.product(problems, arches):
        got = tuple(engine.unified_attention_supports_tiled_2d(asdict(p), arch))
        assert got == _expected_tiled_2d(p, arch), (p, arch)


def test_tiled_2d_gate_matches_python(engine):
    _check_tiled_2d(engine, _tiled_2d_shapes(), ("gfx942", "gfx950"))
    _check_tiled_2d(engine, _gate_problems(), _ARCHES)


@pytest.mark.parametrize(
    "env", _GFX942_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items())
)
def test_tiled_2d_gate_follows_gfx942_env(engine, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    _check_tiled_2d(engine, _tiled_2d_shapes(), ("gfx942",))


def test_tiled_2d_admitted_problems_build(engine):
    """Every problem the C++ gate admits lowers through the C++ builder, from
    the spec the C++ selector builds."""
    seen = set()
    for p, arch in itertools.product(_tiled_2d_shapes(), ("gfx942", "gfx950")):
        ok, _ = engine.unified_attention_supports_tiled_2d(asdict(p), arch)
        if not ok:
            continue
        spec = getattr(engine, f"{arch}_attn_tiled_spec_from_problem")(asdict(p), arch)
        lower = getattr(engine, f"{arch}_attention_tiled_2d_lower_llvm")
        key = (arch, repr(sorted(spec.items())))
        if key in seen:
            continue
        seen.add(key)
        assert lower(spec, arch), (p, arch, spec)
