# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only IR tests for opt-in LSE output on the tiled 2D unified-attention
kernels (gfx942 + gfx950). No GPU: specs are built and lowered to LLVM text."""

import hashlib
import importlib
import re

import pytest

from rocke.core.lower_llvm import _resolve_llvm_flavor, lower_kernel_to_llvm

_BASE = dict(
    head_size=128,
    block_size=64,
    num_query_heads=4,
    num_kv_heads=2,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    use_alibi=False,
)

# Spec kwargs that select each epilogue branch of build_unified_attention_2d_tiled.
BRANCH_SPEC_KW = {
    "gfx950": {
        "transposed32": dict(
            use_mfma_32x32=True,
            block_m_per_warp=32,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=True,
        ),
        "mfma32": dict(use_mfma_32x32=True, block_m_per_warp=32),
        "striped16": {},
    },
    "gfx942": {
        "transposed32": dict(
            use_mfma_32x32x8=True, use_transposed_qk_32x32=True, block_m_per_warp=32
        ),
        "mfma32": dict(use_mfma_32x32x8=True, block_m_per_warp=32),
        "striped16": {},
    },
}

# One FP32 LSE store per (writer lane, softmax-state slot), Python-unrolled.
_EXPECTED_LSE_STORES = {
    "gfx950": {"transposed32": 1, "mfma32": 16, "striped16": 4},
    "gfx942": {"transposed32": 1, "mfma32": 16, "striped16": 4},
}

# sha256 of lower_kernel_to_llvm(...) text for the LSE-off build, keyed by the
# resolved LLVM flavor (the text differs per flavor, like the flavor-keyed IR
# golden), one entry per committed flavor. Recorded after the running-max carry
# fix was merged into this branch, before any LSE store existed.
_OFF_SHA256 = {
    "llvm20": {
        "gfx950": {
            "transposed32": "b9632fa4fedd6f73aeab033e682e1c7bf5e981dc5638b07a8c686d01ab0c51fe",
            "mfma32": "75c49deabc455a47c596709cab31c2434cf7dc0a9903f292f0178ab96bc2ed57",
            "striped16": "2581bfec6efe59e48aec76a867d28d53a26ef41066d7df4a9b050529b983aad5",
        },
        "gfx942": {
            "transposed32": "f529f39c3eaceaa06689456bdc96d340652d0a09b74811379e98ef9b71eb967f",
            "mfma32": "022a6b762edbc729f480577c9a123b15d79fb7226de95cfed20034c2f88d4f0e",
            "striped16": "9f3b0e780b9ed4044bc42980fcc9577002b43bc3c278bbfdf3fa481bd771ea9a",
        },
    },
    "llvm22": {
        "gfx950": {
            "transposed32": "74b11c4a3665fd25c04c27b2519f109011cd195750a30c96e5e142e47e3f474f",
            "mfma32": "240eae5f3ece173ba83de9008663f5e3be12691964ad6935acca5b4ead0947c9",
            "striped16": "2e22eae5e44a748168efb850ff7992a74c3e846855828dbd3e2f2cef94ec2293",
        },
        "gfx942": {
            "transposed32": "a7951ee9ac5147a517372396ef04bc1a31ada7b2c32e62ffbcf613fb74f96403",
            "mfma32": "e0fc1492d9ae4ba48e4eaa92b4939c41224c8f0805983135a5519471dccbcce7",
            "striped16": "af4015ea1955e1b12866ddc55b49ffe43738c34d2a942ca208e4d8deafea9a35",
        },
    },
    "llvm23": {
        "gfx950": {
            "transposed32": "d69c31d423d534ee46350166125420153c60d7aaf547fa82e3b6f684b2acafc8",
            "mfma32": "8181cc42d6d9f5db85fe7af8749aed653920e4f4c956ae2d80d85f746d0ca9f5",
            "striped16": "ceff7404264975479bd8b1a68177914a40325010fcb6fb3b4ce34ef7290ab77b",
        },
        "gfx942": {
            "transposed32": "218efdce76d55d559851aaea80e3bc1adf03d5ddbb52c83e854665ea15b4fb83",
            "mfma32": "f54c66a9ab69e0b5b54a34d3b9cd4ca305e8af9fa29e7a1ecd8f63760f30ac91",
            "striped16": "9184314ed9b3719f999427b24fe12f0214a9db058f6160ded11d091f820240a0",
        },
    },
}

_CASES = [
    (arch, branch)
    for arch in ("gfx950", "gfx942")
    for branch in ("transposed32", "mfma32", "striped16")
]
_SPEC_ARCHES = ["gfx950", "gfx942"]


def _module(arch):
    return importlib.import_module(f"kernels.{arch}.attention_tiled_2d")


def _spec(arch, branch, **kw):
    return _module(arch).UnifiedAttention2DTiledSpec(
        **_BASE, **BRANCH_SPEC_KW[arch][branch], **kw
    )


def _lower(spec, arch):
    kernel = _module(arch).build_unified_attention_2d_tiled(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch)


def _lse_store_count(ir):
    """FP32 global stores. The LSE-off 2D kernels have none (bf16/fp16 O)."""
    return sum(
        1
        for line in ir.splitlines()
        if "store float" in line and "ptr addrspace(1)" in line
    )


def _define_line(ir):
    return next(line for line in ir.splitlines() if line.startswith("define"))


@pytest.mark.parametrize("arch", _SPEC_ARCHES)
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_2d_spec_rejects_non_boolean_emit_lse(arch, bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _spec(arch, "striped16", emit_lse=bad)


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_kernel_name_gains_lse_suffix_only_when_on(arch, branch):
    default = _spec(arch, branch)
    off = _spec(arch, branch, emit_lse=False)
    on = _spec(arch, branch, emit_lse=True)
    assert default.emit_lse is False
    assert off.kernel_name() == default.kernel_name()
    assert not default.kernel_name().endswith("_lse")
    assert on.kernel_name() == default.kernel_name() + "_lse"


# gfx942 also builds branch A with one softmax-state slot (production flash
# specs set it) and the q-major grid; the name rule must hold for those too.
@pytest.mark.parametrize(
    "branch,extra",
    [
        ("transposed32", {}),
        ("transposed32", {"use_transposed_scalar_state": True}),
        ("mfma32", {}),
        ("striped16", {}),
        ("striped16", {"use_q_major_grid": True}),
    ],
)
def test_gfx942_lse_name_is_off_name_plus_lse_suffix(branch, extra):
    off = _spec("gfx942", branch, **extra)
    on = _spec("gfx942", branch, emit_lse=True, **extra)
    assert off.emit_lse is False
    assert not off.kernel_name().endswith("_lse")
    assert on.kernel_name() == off.kernel_name() + "_lse"


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_lse_off_ir_is_byte_identical_to_pre_change(arch, branch):
    """LSE off adds no param, no store, no instruction."""
    default_ir = _lower(_spec(arch, branch), arch)
    off_ir = _lower(_spec(arch, branch, emit_lse=False), arch)
    assert off_ir == default_ir
    assert "lse_ptr" not in off_ir
    assert _lse_store_count(off_ir) == 0
    flavor = _resolve_llvm_flavor()
    assert flavor in _OFF_SHA256, f"record the LSE-off hashes for LLVM flavor {flavor}"
    pinned = _OFF_SHA256[flavor][arch][branch]
    assert hashlib.sha256(off_ir.encode()).hexdigest() == pinned


@pytest.mark.parametrize("arch,branch", _CASES)
def test_2d_lse_on_appends_lse_ptr_last_and_stores_once_per_slot(arch, branch):
    on_ir = _lower(_spec(arch, branch, emit_lse=True), arch)
    define = _define_line(on_ir)
    assert re.search(
        r"ptr addrspace\(1\) noalias writeonly nocapture align 4 %lse_ptr\) #\d+ \{$",
        define,
    ), define
    assert _lse_store_count(on_ir) == _EXPECTED_LSE_STORES[arch][branch]
    # natural_lse_from_log2_stats: (m + log2 l) * ln2
    assert "@llvm.log2.f32" in on_ir


# Additive-bias params are appended after qq_bias_stride_0; branch B rejects bias.
@pytest.mark.parametrize(
    "arch,branch", [("gfx950", "transposed32"), ("gfx950", "striped16")]
)
def test_2d_lse_ptr_stays_last_with_additive_bias(arch, branch):
    define = _define_line(
        _lower(_spec(arch, branch, emit_lse=True, use_additive_bias=True), arch)
    )
    assert "%additive_bias_sq_stride, ptr addrspace(1)" in define, define
    assert re.search(r"%lse_ptr\) #\d+ \{$", define), define


@pytest.mark.parametrize(
    "branch,extra,stores",
    [
        # Production flash specs set use_transposed_scalar_state (1 state slot).
        ("transposed32", {"use_transposed_scalar_state": True}, 1),
        # Swapped launch grid: same kv_head_idx / q_block names, same index.
        ("striped16", {"use_q_major_grid": True}, 4),
        ("transposed32", {"use_q_major_grid": True}, 1),
        # Two 16-row atoms per warp: 8 state slots per lane.
        ("striped16", {"block_m_per_warp": 32}, 8),
    ],
)
def test_gfx942_lse_variants_store_once_per_row_slot(branch, extra, stores):
    off_ir = _lower(_spec("gfx942", branch, **extra), "gfx942")
    on_ir = _lower(_spec("gfx942", branch, emit_lse=True, **extra), "gfx942")
    assert "lse_ptr" not in off_ir
    assert _lse_store_count(off_ir) == 0
    assert _lse_store_count(on_ir) == stores
    assert "@llvm.log2.f32" in on_ir


def _gfx942_branch_of(spec):
    if spec.use_mfma_32x32x8:
        return "transposed32" if spec.use_transposed_qk_32x32 else "mfma32"
    return "striped16"


@pytest.mark.parametrize(
    "fields,branch",
    [
        # Mandate shape, bf16 flash (_spec_gfx942_bf16_flash).
        (
            dict(dtype="bf16", total_q=8192, num_seqs=1, max_seqlen_q=8192),
            "transposed32",
        ),
        # fp16 flash (_spec_gfx942_fp16_flash).
        (
            dict(dtype="fp16", total_q=8192, num_seqs=1, max_seqlen_q=8192),
            "transposed32",
        ),
        # Decode-like 2D shape: generic narrow (_spec_generic_2d_non_gfx950).
        (dict(dtype="bf16", total_q=4, num_seqs=4, max_seqlen_q=1), "striped16"),
        # D256 bf16 prefill: generic narrow when the 4-warp cohort is not used.
        (
            dict(
                dtype="bf16", head_size=256, total_q=8192, num_seqs=1, max_seqlen_q=8192
            ),
            "striped16",
        ),
    ],
)
def test_gfx942_production_specs_use_branches_the_numeric_gate_covers(fields, branch):
    """Production gfx942 2D specs land on branch A or C, the branches the GPU
    numeric gate runs by default. Branch B (non-transposed x8) is reachable
    only through explicit or tuning specs, which the gate pins separately."""
    from kernels.common import attention_unified as au

    base = dict(
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_k=8192,
    )
    problem = au.UnifiedAttentionProblem(**{**base, **fields})
    assert _gfx942_branch_of(au._tiled_spec_from_problem(problem, "gfx942")) == branch
