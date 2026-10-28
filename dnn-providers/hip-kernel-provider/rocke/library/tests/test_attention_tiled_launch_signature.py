# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch-signature contract for the tiled 2D / 3D attention kernels.

``KernelLauncher`` packs kernargs by the signature's names and order, so a kernel
whose params drift from its launch signature reads the wrong argument without
faulting. Every builder route the production path takes is built here and its
params are compared, by name and type in order, with the signature its launcher
uses (``_tiled_2d_signature`` / ``_tiled_3d_signatures``). That includes the
runtime shape kernargs a spec class declares (``RUNTIME_KERNARG_FIELDS``).

No GPU required.
"""

from __future__ import annotations

import pytest

from kernels.common import attention_unified as au

P = au.UnifiedAttentionProblem

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


@pytest.fixture(autouse=True)
def routing_env(monkeypatch):
    """Production routing: no env knob steers a selector."""
    for var in _ROUTING_ENV:
        monkeypatch.delenv(var, raising=False)


def _prefill(*, dtype="bf16", head_size=128, hq=32, hkv=8, sq=1024, **kw):
    return P(
        total_q=2 * sq,
        num_seqs=2,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=kw.pop("block_size", 16),
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
        **kw,
    )


def _decode(*, dtype="bf16", head_size=128, hq=32, hkv=8, **kw):
    return P(
        total_q=4,
        num_seqs=4,
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=head_size,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        dtype=dtype,
        **kw,
    )


# (arch, case id, problem, kernel-name suffix the route must emit or None)
_CASES_2D = (
    ("gfx942", "narrow_d64", _prefill(head_size=64, hq=8, hkv=8, sq=128), None),
    ("gfx942", "fp16_flash_d128", _prefill(dtype="fp16", sq=2048), None),
    ("gfx942", "bf16_flash_d128", _prefill(sq=2048), None),
    ("gfx942", "window", _prefill(sliding_window=256, hq=8, hkv=8), None),
    ("gfx942", "sinks", _prefill(use_sinks=True), None),
    (
        "gfx942",
        "4warp_d256",
        _prefill(head_size=256, hq=16, hkv=2, sq=4096),
        "_4wgqa",
    ),
    (
        "gfx942",
        "4warp_fold",
        _prefill(sliding_window=4096, sq=8192),
        "_4wgqa_fold",
    ),
    (
        "gfx942",
        "4warp_fp16_window",
        _prefill(dtype="fp16", sliding_window=4096, sq=8192),
        "_4wgqa",
    ),
    ("gfx950", "bf16_d128", _prefill(), None),
    (
        "gfx950",
        "fp16_d64_window",
        _prefill(dtype="fp16", head_size=64, sliding_window=256),
        None,
    ),
)


def _assert_matches(kernel, signature):
    names = [p.name for p in kernel.params]
    assert names == [a["name"] for a in signature], kernel.name
    types = [p.type.name.replace(" ", "") for p in kernel.params]
    assert types == [a["type"].replace(" ", "") for a in signature], kernel.name


@pytest.mark.parametrize(
    "arch, problem, suffix",
    [pytest.param(a, p, s, id=f"{a}-{i}") for a, i, p, s in _CASES_2D],
)
def test_2d_kernel_params_match_launch_signature(arch, problem, suffix):
    kernel = au._tiled_2d_kernel(problem, arch)
    if suffix is not None:
        assert kernel.name.endswith(suffix), kernel.name
    signature = au._tiled_2d_signature(
        arch,
        problem.dtype,
        kv_dtype=au._kv_storage_dtype(problem),
        include_additive_bias=problem.use_additive_bias,
    )
    _assert_matches(kernel, signature)


_CASES_3D = (
    ("gfx942", "decode", _decode(), False),
    ("gfx942", "decode_window", _decode(sliding_window=256), False),
    ("gfx942", "decode_lse", _decode(emit_lse=True), False),
    ("gfx942", "strided", _decode(hq=4, hkv=2), True),
    ("gfx950", "decode", _decode(), False),
    ("gfx950", "strided", _decode(hq=4, hkv=2), True),
)


@pytest.mark.parametrize(
    "arch, problem, strided_kv",
    [pytest.param(a, p, s, id=f"{a}-{i}") for a, i, p, s in _CASES_3D],
)
def test_3d_kernel_params_match_launch_signatures(arch, problem, strided_kv):
    num_segments = au._num_segments(problem, arch)
    segment, reduce = au._tiled_3d_kernels(
        problem, arch, num_segments, strided_kv=strided_kv
    )
    seg_sig, red_sig = au._tiled_3d_signatures(
        arch,
        problem.dtype,
        kv_dtype=au._kv_storage_dtype(problem),
        strided_kv=strided_kv,
        include_additive_bias=problem.use_additive_bias,
        emit_lse=problem.emit_lse,
    )
    _assert_matches(segment, seg_sig)
    _assert_matches(reduce, red_sig)


@pytest.mark.parametrize("codepath", ["narrow", "transposed_x8", "gfx942_4warp"])
def test_tuning_candidate_signature_matches_its_kernel(codepath):
    """The registry's tuning candidates launch through ``candidate.signature``."""
    from dispatch.attention import attention_candidates
    from dispatch.attention.common import AttentionMaskType, AttentionRequest
    from dispatch.attention.gfx942_unified import GFX942_TUNING_VARIANTS

    variants = [v for v in GFX942_TUNING_VARIANTS if v.codepath == codepath]
    candidates = {c.spec_id: c for c in attention_candidates()}
    head_size = 256 if codepath == "gfx942_4warp" else 128
    checked = 0
    for variant in variants:
        request = AttentionRequest(
            batch=1,
            nhead_q=16,
            nhead_k=2,
            seqlen_q=1024,
            seqlen_k=1024,
            hdim_q=head_size,
            hdim_v=head_size,
            arch="gfx942",
            dtype="bf16",
            mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
            num_cus=304,
            algorithm="unified_tuning",
            spec_id=variant.spec_id,
        )
        candidate = candidates[variant.spec_id]
        if not candidate.admits(request)[0]:
            continue
        spec = candidate.select_spec(request)
        _assert_matches(candidate.build(spec, "gfx942"), candidate.signature(spec))
        checked += 1
        break
    assert checked, f"no {codepath} tuning candidate admits the request"


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_runtime_values_cover_every_declared_field(arch):
    """The shared argument pack supplies every field a spec class can declare,
    so a launcher never raises on a runtime kernarg."""
    segment_cls, reduce_cls, _, _, _ = au._tiled_3d_impl(arch)
    spec_2d_cls, _, _ = au._tiled_2d_impl(arch)
    values = au._runtime_kernarg_values(_prefill())
    for cls in (spec_2d_cls, segment_cls, reduce_cls):
        missing = set(au._runtime_kernarg_fields(cls)) - set(values)
        assert not missing, (cls.__name__, missing)
