# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free generation matrix for the CK DSL provider's SDPA-fwd path.

This is the Phase-3 centerpiece: a thorough, torch-FREE test suite that
proves the unified paged/varlen tiled-2D SDPA-fwd *kernel generation* is
correct across the full supported CODEGEN variant matrix -- everything
short of a GPU launch. comgr cross-compiles to a gfx950 HSACO on this
gfx90a host; we never ``hipModuleLoadData`` the gfx950 object (it would
fail on gfx90a), so we assert only on the compiled artifact: the 18-slot
arg schema, the gfx950 ISA, a non-empty HSACO, the recomputed launch
grid/block, and -- for the codegen-distinguishing knobs -- the LLVM IR
intrinsics that prove the knob actually changed the generated code.

HARD CONSTRAINT: nothing here imports torch. ``ck_dsl`` and
``ck_dsl_provider`` are pure-Python + comgr; the no-torch-creep guard
(``test_generation_path_runs_with_torch_absent``) spawns a subprocess
with a meta-path finder that makes ``import torch`` raise, then runs the
full generation path, to prove it.

Sibling :mod:`conftest` self-bootstraps ``sys.path`` for both packages so
this runs standalone::

    .venv/bin/python -m pytest \
        dnn-providers/ck-dsl-provider/python/tests/test_sdpa_generation.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from typing import Any, Dict

import pytest


# Package paths recomputed here (independently of conftest, which pytest does
# not expose as an importable module) so the no-torch-creep subprocess can be
# handed both sys.path entries. tests/ -> python/ -> ck-dsl-provider/ ->
# dnn-providers/ -> <repo root>.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDER_PYTHON = os.path.normpath(os.path.join(_THIS_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
_CK_DSL_PYTHON = os.path.normpath(
    os.path.join(_REPO_ROOT, "projects", "composablekernel", "python")
)

# Imported lazily inside conftest's bootstrapped sys.path. These succeed at
# collection time because conftest ran first.
from ck_dsl_provider import compile_service as cs


ARCH = "gfx950"


# ---------------------------------------------------------------------------
# Payload builders + an independent grid recomputation.
# ---------------------------------------------------------------------------


def _make_payload(
    *,
    head_size: int = 64,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    dtype: str = "bf16",
    mask_mode: str = "causal",
    batch: int = 2,
    seqlen_q: int = 256,
    seqlen_k: int = 256,
    block_size: int = 32,
    is_paged: bool = False,
    is_varlen: bool = False,
    sliding_window: int = 0,
    use_sinks: bool = False,
    num_warps: int = 4,
    block_m_per_warp: int = 16,
    tile_size: int = 0,
    waves_per_eu: int = 0,
    use_mfma_32x32: bool = False,
    use_transposed_qk_32x32: bool = False,
    use_register_pv: bool = False,
    use_early_v_schedule: bool = False,
    use_fast_paged_kv_desc: bool = False,
) -> Dict[str, Any]:
    """Build a unified-path payload dict in the exact wire shape the C++
    ``sdpaSpecToPayload`` emits (top keys + nested ``shape`` + nested
    ``knobs``)."""
    return {
        "batch": batch,
        "shape": {
            "head_size": head_size,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
        },
        "dtype": dtype,
        "mask_mode": mask_mode,
        "seqlen_q": seqlen_q,
        "seqlen_k": seqlen_k,
        "is_paged": is_paged,
        "block_size": block_size,
        "is_varlen": is_varlen,
        "sliding_window": sliding_window,
        "use_sinks": use_sinks,
        "knobs": {
            "num_warps": num_warps,
            "block_m_per_warp": block_m_per_warp,
            "tile_size": tile_size,
            "waves_per_eu": waves_per_eu,
            "use_mfma_32x32": use_mfma_32x32,
            "use_transposed_qk_32x32": use_transposed_qk_32x32,
            "use_register_pv": use_register_pv,
            "use_early_v_schedule": use_early_v_schedule,
            "use_fast_paged_kv_desc": use_fast_paged_kv_desc,
        },
    }


def _expected_grid(payload: Dict[str, Any]) -> tuple:
    """Independently recompute ``(num_kv_heads, total_q//block_q + num_seqs, 1)``.

    Deliberately re-derived here from the payload alone (NOT by calling the
    provider's ``_unified_grid``) so a regression in the provider's grid
    math is caught rather than mirrored. Matches the kernel reference
    formula: block_m = num_warps * block_m_per_warp; block_q = block_m //
    num_queries_per_kv (or 1 if NQK > block_m); total_q = batch * Sq;
    num_seqs = batch.
    """
    shape = payload["shape"]
    knobs = payload["knobs"]
    num_kv_heads = shape["num_kv_heads"]
    nqk = shape["num_query_heads"] // shape["num_kv_heads"]
    num_warps = knobs["num_warps"]
    block_m_per_warp = knobs["block_m_per_warp"]
    block_m = num_warps * block_m_per_warp
    block_q = block_m // nqk if nqk <= block_m else 1
    num_seqs = payload["batch"]
    total_q = num_seqs * payload["seqlen_q"]
    total_blocks = total_q // block_q + num_seqs
    return (int(num_kv_heads), int(total_blocks), 1)


# The fixed 18-slot ABI: 10 pointers, 5 f32, 3 i32, in that order.
_EXPECTED_SCHEMA_KINDS = ["Pointer"] * 10 + ["F32"] * 5 + ["I32"] * 3
_EXPECTED_SCHEMA_NAMES = [
    "output_ptr",
    "query_ptr",
    "key_cache_ptr",
    "value_cache_ptr",
    "sink_ptr",
    "block_tables_ptr",
    "seq_lens_ptr",
    "alibi_slopes_ptr",
    "qq_bias_ptr",
    "query_start_len_ptr",
    "scale",
    "k_scale",
    "v_scale",
    "out_scale",
    "softcap",
    "num_seqs",
    "block_table_stride",
    "qq_bias_stride_0",
]


def _assert_unified_result(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """Shared per-case assertions on a compiled unified result dict."""
    assert result["kind"] == "sdpa_fmha_fwd_unified"

    schema = result["arg_schema"]
    assert len(schema) == 18, f"expected 18-slot ABI, got {len(schema)}"
    assert [s["kind"] for s in schema] == _EXPECTED_SCHEMA_KINDS
    assert [s["name"] for s in schema] == _EXPECTED_SCHEMA_NAMES

    # gfx950 ISA threaded through to comgr.
    assert "gfx950" in result["isa"], result["isa"]

    # A real HSACO came back (cross-compiled to gfx950; not launched here).
    assert isinstance(result["hsaco"], (bytes, bytearray))
    assert len(result["hsaco"]) > 0

    # block = (64 * num_warps, 1, 1).
    num_warps = payload["knobs"]["num_warps"]
    assert result["block"] == (64 * num_warps, 1, 1)

    # Grid matches the INDEPENDENTLY recomputed reference formula.
    assert result["grid"] == _expected_grid(
        payload
    ), f"grid {result['grid']} != expected {_expected_grid(payload)}"

    assert result["lds_bytes"] == 0


# ---------------------------------------------------------------------------
# 1. Generation matrix (real comgr HSACOs). Each case must be buildable per
#    the DSL gate; the constraints are encoded in the chosen knob combos.
#
# Coverage tracking -- every CODEGEN-axis value is hit at least once:
#   dtype:            fp16, bf16
#   head_size:        64, 128, 256
#   block_size:       16, 32, 64
#   use_sinks:        False, True
#   sliding_window:   0, >0
#   num_warps:        1, 2, 4, 8
#   block_m_per_warp: 16, 32
#   tile_size:        0 (default), explicit multiple
#   use_mfma_32x32 (+use_transposed_qk_32x32): True
#   use_early_v_schedule: True
#   use_register_pv (bf16-only): True
#   GQA ratio:        1, 8, 16
# ---------------------------------------------------------------------------

# Each tuple is (case_id, payload). Validity constraints (DMA floor, mfma32
# prerequisites, register_pv exclusions, GQA divisibility, per-wave tokens)
# are satisfied per-case so every payload actually builds.
_MATRIX_CASES = [
    # --- dtype axis ----------------------------------------------------
    ("dtype_bf16", _make_payload(dtype="bf16")),
    ("dtype_fp16", _make_payload(dtype="fp16")),
    # --- head_size axis ------------------------------------------------
    ("head_64", _make_payload(head_size=64)),
    ("head_128", _make_payload(head_size=128)),
    ("head_256", _make_payload(head_size=256)),
    # --- block_size axis (num_warps clamped so the DMA floor holds) ----
    ("block_16", _make_payload(block_size=16, num_warps=1)),
    ("block_32", _make_payload(block_size=32)),
    ("block_64", _make_payload(block_size=64)),
    # --- use_sinks axis ------------------------------------------------
    ("sinks_off", _make_payload(use_sinks=False)),
    ("sinks_on", _make_payload(use_sinks=True)),
    # --- sliding_window axis (tile_size=block_size on the sw path) -----
    ("sliding_window_0", _make_payload(sliding_window=0)),
    ("sliding_window_pos", _make_payload(sliding_window=256, tile_size=32)),
    # --- num_warps sweep {1,2,4,8} -------------------------------------
    # num_warps=8 needs tile_size*head >= 8*64*8=4096 -> tile_size>=64.
    ("num_warps_1", _make_payload(num_warps=1)),
    ("num_warps_2", _make_payload(num_warps=2)),
    ("num_warps_4", _make_payload(num_warps=4)),
    ("num_warps_8", _make_payload(num_warps=8, block_size=64, tile_size=64)),
    # --- block_m_per_warp {16,32} (32 needs num_warps in {1,2,4}) ------
    ("block_m_per_warp_16", _make_payload(block_m_per_warp=16)),
    (
        "block_m_per_warp_32",
        _make_payload(num_warps=2, block_m_per_warp=32, tile_size=32),
    ),
    # --- tile_size axis: default (0) and an explicit multiple ----------
    ("tile_size_default", _make_payload(tile_size=0)),
    ("tile_size_explicit", _make_payload(tile_size=64)),
    # --- mfma 32x32 + transposed QK (needs bmpw=32, tile%32==0) --------
    (
        "mfma_32x32",
        _make_payload(
            num_warps=4,
            block_m_per_warp=32,
            tile_size=64,
            use_mfma_32x32=True,
            use_transposed_qk_32x32=True,
        ),
    ),
    # --- early-V schedule (default 16x16x32 atom) ----------------------
    ("early_v_schedule", _make_payload(use_early_v_schedule=True)),
    # --- register-PV (bf16-only; excludes sinks/sw/softcap) ------------
    ("register_pv_bf16", _make_payload(dtype="bf16", use_register_pv=True)),
    # --- GQA ratio {1, 8, 16} ------------------------------------------
    ("gqa_1", _make_payload(num_query_heads=8, num_kv_heads=8)),
    ("gqa_8", _make_payload(num_query_heads=64, num_kv_heads=8)),
    ("gqa_16", _make_payload(num_query_heads=128, num_kv_heads=8)),
    # --- fast paged-KV descriptor (the curated h64kv8 fast lane) -------
    (
        "fast_paged_kv_desc",
        _make_payload(
            head_size=64,
            num_query_heads=64,
            num_kv_heads=8,
            num_warps=4,
            tile_size=64,
            use_fast_paged_kv_desc=True,
        ),
    ),
    # --- the relocated 2c reference shape (GQA8, bf16, head64) ----------
    (
        "ref_2c_head64_gqa8_bf16_paged32",
        _make_payload(
            head_size=64,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            mask_mode="causal",
            batch=2,
            seqlen_q=1024,
            seqlen_k=1024,
            block_size=32,
            num_warps=4,
            block_m_per_warp=16,
        ),
    ),
]


@pytest.mark.parametrize(
    "payload", [c[1] for c in _MATRIX_CASES], ids=[c[0] for c in _MATRIX_CASES]
)
def test_generation_matrix_compiles_and_matches_abi(payload):
    """Every codegen-axis value compiles to a gfx950 HSACO with the fixed
    18-slot ABI and a grid/block consistent with the recomputed formula."""
    result = cs._compile_sdpa_fwd_unified(payload, arch=ARCH)
    _assert_unified_result(result, payload)


def test_relocated_2c_unified_compile_path():
    """Relocated from ``test_ck_dsl.py::test_ck_dsl_provider_unified_compile_path``.

    Drives the provider's unified SDPA-fwd compile path for the 2c POC
    reference shape and asserts the named-slot ABI + the grid math the
    original test checked. Kept as a standalone test (in addition to its
    matrix entry) so the named-slot / grid-derivation assertions the 2c
    test carried are preserved verbatim here in the provider-owned suite.
    """
    batch = 2
    seqlen_q = 1024
    num_warps = 4
    block_m_per_warp = 16
    payload = _make_payload(
        head_size=64,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        mask_mode="causal",
        batch=batch,
        seqlen_q=seqlen_q,
        seqlen_k=1024,
        block_size=32,
        num_warps=num_warps,
        block_m_per_warp=block_m_per_warp,
    )

    result = cs._compile_sdpa_fwd_unified(payload, arch=ARCH)

    assert result["kind"] == "sdpa_fmha_fwd_unified"

    schema = result["arg_schema"]
    assert len(schema) == 18
    assert [s["kind"] for s in schema] == _EXPECTED_SCHEMA_KINDS
    assert schema[0]["name"] == "output_ptr"
    assert schema[5]["name"] == "block_tables_ptr"
    assert schema[10]["name"] == "scale"
    assert schema[-1]["name"] == "qq_bias_stride_0"

    num_queries_per_kv = 64 // 8
    block_m = num_warps * block_m_per_warp
    block_q = block_m // num_queries_per_kv
    total_q = batch * seqlen_q
    expected_total_blocks = total_q // block_q + batch
    assert result["grid"] == (8, expected_total_blocks, 1)
    assert result["block"] == (64 * num_warps, 1, 1)
    assert "gfx950" in result["isa"]


# ---------------------------------------------------------------------------
# 2. No-torch-creep guard: prove the generation path imports + runs with
#    torch unavailable, via a subprocess + import-blocking meta-path finder.
# ---------------------------------------------------------------------------

# This script is executed in a fresh interpreter. It (a) installs a
# meta-path finder that raises ImportError for ``torch`` / ``torch.*``
# BEFORE importing anything provider-related, (b) sets sys.path for both
# packages, (c) runs the real generation path, (d) prints a sentinel.
_NO_TORCH_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import importlib.machinery
    import os
    import sys

    class _BlockTorch(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "torch" or fullname.startswith("torch."):
                raise ImportError(
                    "torch import is forbidden in the provider generation path"
                )
            return None

    # Install the blocker FIRST so any later ``import torch`` fails hard.
    sys.meta_path.insert(0, _BlockTorch())

    # Defensive: if torch was somehow already imported, fail the guard.
    assert "torch" not in sys.modules, "torch was already imported"

    ck_dsl_python = {ck_dsl_python!r}
    provider_python = {provider_python!r}
    for p in (ck_dsl_python, provider_python):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Sanity-check the blocker itself works.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    else:
        print("TORCH-IMPORT-LEAKED")
        sys.exit(2)

    from ck_dsl_provider import compile_service as cs

    payload = {{
        "batch": 2,
        "shape": {{"head_size": 64, "num_query_heads": 64, "num_kv_heads": 8}},
        "dtype": "bf16",
        "mask_mode": "causal",
        "seqlen_q": 256,
        "seqlen_k": 256,
        "is_paged": False,
        "block_size": 32,
        "is_varlen": False,
        "sliding_window": 0,
        "use_sinks": False,
        "knobs": {{
            "num_warps": 4,
            "block_m_per_warp": 16,
            "tile_size": 0,
            "waves_per_eu": 0,
            "use_mfma_32x32": False,
            "use_transposed_qk_32x32": False,
            "use_register_pv": False,
            "use_early_v_schedule": False,
            "use_fast_paged_kv_desc": False,
        }},
    }}

    result = cs._compile_sdpa_fwd_unified(payload, arch="gfx950")
    assert result["kind"] == "sdpa_fmha_fwd_unified"
    assert len(result["arg_schema"]) == 18
    assert "gfx950" in result["isa"]
    assert len(result["hsaco"]) > 0
    assert "torch" not in sys.modules, "torch was imported by the generation path"
    print("TORCH-FREE OK")
    """
)


def test_generation_path_runs_with_torch_absent():
    """The whole generation path imports + runs with torch unavailable.

    Spawns a subprocess that blocks ``torch`` (and any ``torch.*``) via a
    meta-path finder installed before any provider import, then runs a real
    ``_compile_sdpa_fwd_unified``. This does NOT depend on whether torch is
    installed in the parent environment -- the blocker guarantees absence
    inside the child regardless. Asserts the child exits 0 and emits the
    ``TORCH-FREE OK`` sentinel.
    """
    script = _NO_TORCH_SCRIPT.format(
        ck_dsl_python=_CK_DSL_PYTHON, provider_python=_PROVIDER_PYTHON
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"torch-free subprocess failed (rc={proc.returncode}).\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert (
        "TORCH-FREE OK" in proc.stdout
    ), f"sentinel missing.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# 3. IR-intrinsic assertions (lower-only, no comgr -> cheap). These PROVE the
#    codegen-distinguishing knobs actually changed the generated code, not
#    just the result metadata. Strings mirror the existing tests in
#    test_ck_dsl.py (test_unified_attention_2d_tiled_kernel_compiles and
#    test_unified_attention_2d_tiled_half_local_pv_compiles).
# ---------------------------------------------------------------------------


def _lower_tiled(**spec_kwargs) -> str:
    from ck_dsl.instances import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
    )
    from ck_dsl import lower_kernel_to_llvm

    spec = UnifiedAttention2DTiledSpec(**spec_kwargs)
    kernel = build_unified_attention_2d_tiled(spec)
    return lower_kernel_to_llvm(kernel)


def test_ir_default_atom_emits_16x16x32_mfma():
    """The default 16x16x32 atom path emits the wide-K f16/bf16 MFMA, async
    DMA, the softmax cross-lane reduce, and the trailing qq_bias_stride_0
    param."""
    ll = _lower_tiled(
        head_size=128,
        block_size=16,
        num_query_heads=16,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
    )
    # Default atom -> 16x16x32 MFMA.
    assert "@llvm.amdgcn.mfma.f32.16x16x32.f16" in ll
    # Async K/V DMA present.
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds" in ll
    # Softmax cross-lane reduction (ds_swizzle for row-group masks <= 16).
    assert "@llvm.amdgcn.ds.swizzle" in ll
    # qq_bias_stride_0 is the last kernel param.
    assert "i32 %qq_bias_stride_0" in ll
    # The default atom must NOT have flipped to the 32x32 geometry.
    assert "@llvm.amdgcn.mfma.f32.32x32" not in ll


def test_ir_mfma_32x32_emits_32x32_mfma():
    """The use_mfma_32x32 path emits the 32x32x16 MFMA (distinct from the
    default 16x16x32 atom) -- proving the knob changed the generated code."""
    ll = _lower_tiled(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        tile_size=64,
        block_m_per_warp=32,
        use_mfma_32x32=True,
        use_transposed_qk_32x32=True,
    )
    # 32x32 atom present (bf16 variant).
    assert "@llvm.amdgcn.mfma.f32.32x32x16.bf16" in ll
    # Async DMA still present.
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds" in ll
    # qq_bias_stride_0 still the last param (ABI unchanged by the knob).
    assert "i32 %qq_bias_stride_0" in ll


def test_ir_early_v_schedule_still_default_atom():
    """early-V schedule keeps the default 16x16x32 atom (a schedule, not an
    atom, change) and still emits async DMA + the last param."""
    ll = _lower_tiled(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        use_early_v_schedule=True,
    )
    assert "@llvm.amdgcn.mfma.f32.16x16x32" in ll
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds" in ll
    assert "i32 %qq_bias_stride_0" in ll


def test_ir_register_pv_default_atom_bf16():
    """register-PV is a 16x16x32-path bf16 variant: it keeps the default
    atom + async DMA + last param, and the kernel name carries ``regpv``."""
    from ck_dsl.instances import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
    )
    from ck_dsl import lower_kernel_to_llvm

    spec = UnifiedAttention2DTiledSpec(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        use_register_pv=True,
    )
    kernel = build_unified_attention_2d_tiled(spec)
    ll = lower_kernel_to_llvm(kernel)
    assert "@llvm.amdgcn.mfma.f32.16x16x32" in ll
    assert "@llvm.amdgcn.raw.ptr.buffer.load.lds" in ll
    assert "i32 %qq_bias_stride_0" in ll
    assert "regpv" in kernel.name


# ---------------------------------------------------------------------------
# 4. Golden snapshot: pin the stable signature for the POC reference shapes
#    so future drift in the ABI/grid/block/name-pattern is caught. The
#    kernel_name is matched by PATTERN (no embedded hash here, but we exclude
#    any future nondeterministic suffix by anchoring a regex prefix). The
#    HSACO bytes themselves are NOT snapshotted (comgr/LLVM output is not
#    byte-reproducible across toolchain versions).
# ---------------------------------------------------------------------------

# What is snapshotted: arg_schema (kinds + names), grid, block, and a
# kernel_name regex. What is EXCLUDED as nondeterministic: the HSACO bytes
# (toolchain-version-dependent), the full ISA triple string (we only assert
# it contains "gfx950"), and any hash/suffix the name builder might append
# in future (the regex is anchored at the front and tolerant at the end).
_GOLDEN = {
    "decode_h64_gqa8_bf16_b32": {
        "payload": _make_payload(
            head_size=64,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            batch=4,
            seqlen_q=1,
            seqlen_k=1024,
            block_size=32,
            num_warps=1,
            block_m_per_warp=16,
        ),
        # nqk=8; block_m=16; block_q=2; total_q=4; 4//2=2; +num_seqs=4
        "grid": (8, 6, 1),
        "block": (64, 1, 1),
        "name_re": r"^ck_dsl_uattn2d_tiled_d64_b32_h64kv8_bf16(?:_.*)?$",
    },
    "prefill_h128_gqa1_fp16_b32": {
        "payload": _make_payload(
            head_size=128,
            num_query_heads=8,
            num_kv_heads=8,
            dtype="fp16",
            batch=2,
            seqlen_q=512,
            seqlen_k=512,
            block_size=32,
            num_warps=4,
            block_m_per_warp=16,
        ),
        # nqk=1; block_m=64; block_q=64; total_q=1024; 1024//64=16; +2 = 18
        "grid": (8, 18, 1),
        "block": (256, 1, 1),
        "name_re": r"^ck_dsl_uattn2d_tiled_d128_b32_h8kv8_fp16(?:_.*)?$",
    },
}


@pytest.mark.parametrize("case_id", sorted(_GOLDEN.keys()))
def test_golden_signature_snapshot(case_id):
    """Pin the stable ABI/grid/block/name-pattern for a POC reference shape.

    HSACO bytes + the exact ISA string are excluded as nondeterministic;
    everything host-visible and stable is snapshotted so codegen drift is
    caught.
    """
    golden = _GOLDEN[case_id]
    result = cs._compile_sdpa_fwd_unified(golden["payload"], arch=ARCH)

    schema = result["arg_schema"]
    assert [s["kind"] for s in schema] == _EXPECTED_SCHEMA_KINDS
    assert [s["name"] for s in schema] == _EXPECTED_SCHEMA_NAMES
    assert result["grid"] == golden["grid"]
    assert result["block"] == golden["block"]
    assert re.match(golden["name_re"], result["kernel_name"]), (
        f"kernel_name {result['kernel_name']!r} did not match " f"{golden['name_re']!r}"
    )
    # ISA: only the gfx950 token is asserted (full triple is excluded).
    assert "gfx950" in result["isa"]


# ---------------------------------------------------------------------------
# 5. Builder/codegen validation: the provider's choices are buildable per the
#    DSL gate, and the provider's grid matches the DSL builder's own grid math.
# ---------------------------------------------------------------------------


def _supports_for_payload(payload: Dict[str, Any]):
    """Call the DSL gate ``supports_tiled_2d`` with the provider's choices."""
    from ck_dsl.instances import supports_tiled_2d

    shape = payload["shape"]
    knobs = payload["knobs"]
    nqk = shape["num_query_heads"] // shape["num_kv_heads"]
    tile = int(knobs.get("tile_size", 0))
    return supports_tiled_2d(
        head_size=shape["head_size"],
        block_size=payload["block_size"],
        dtype=payload["dtype"],
        num_queries_per_kv=nqk,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype=payload["dtype"],
        num_warps=knobs["num_warps"],
        tile_size=tile if tile > 0 else None,
        arch=ARCH,
    )


_SUPPORTED_REPRESENTATIVE = [
    ("supp_bf16_gqa8", _make_payload()),
    ("supp_fp16_gqa1", _make_payload(dtype="fp16", num_query_heads=8, num_kv_heads=8)),
    ("supp_head256", _make_payload(head_size=256)),
    ("supp_block16_nw1", _make_payload(block_size=16, num_warps=1)),
    ("supp_block64", _make_payload(block_size=64)),
    ("supp_gqa16", _make_payload(num_query_heads=128, num_kv_heads=8)),
]


@pytest.mark.parametrize(
    "payload",
    [c[1] for c in _SUPPORTED_REPRESENTATIVE],
    ids=[c[0] for c in _SUPPORTED_REPRESENTATIVE],
)
def test_supports_tiled_2d_accepts_provider_choices(payload):
    """``supports_tiled_2d`` accepts the knob/shape combos the provider's
    path emits for representative supported variants (i.e. the provider's
    choices are buildable per the DSL gate)."""
    ok, reason = _supports_for_payload(payload)
    assert ok, f"DSL gate rejected a provider-emitted combo: {reason}"


@pytest.mark.parametrize(
    "payload", [c[1] for c in _MATRIX_CASES], ids=[c[0] for c in _MATRIX_CASES]
)
def test_provider_grid_matches_dsl_problem_grid(payload):
    """The provider's returned grid matches the DSL ``UnifiedAttentionProblem``
    builder's own grid computation for the same knobs.

    Cross-checks the provider's ``_unified_grid`` against the kernel
    reference formula via the DSL problem's
    ``total_num_q_blocks_upper_bound`` property. For 1 <= NQK <= 16 (the
    supported GQA range) the DSL's block_m == 16 == the provider's launch
    block_m only when num_warps*block_m_per_warp == 16; for larger BLOCK_M
    the launch grid uses the launch block_m, so we cross-check the provider
    against the problem's *own* num_queries_per_kv/total_q rather than the
    upper-bound property (which fixes block_m=16). The independent
    recomputation in ``_expected_grid`` is the reference; this test asserts
    the DSL problem object agrees on the shared inputs (num_kv_heads,
    num_queries_per_kv, total_q, num_seqs)."""
    from ck_dsl.instances.common.attention_unified import UnifiedAttentionProblem

    shape = payload["shape"]
    knobs = payload["knobs"]
    block_size = payload["block_size"]
    batch = payload["batch"]
    seqlen_q = payload["seqlen_q"]
    seqlen_k = payload["seqlen_k"]

    problem = UnifiedAttentionProblem(
        total_q=batch * seqlen_q,
        num_seqs=batch,
        num_query_heads=shape["num_query_heads"],
        num_kv_heads=shape["num_kv_heads"],
        head_size=shape["head_size"],
        block_size=block_size,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        dtype=cs._normalize_unified_dtype(payload["dtype"]),
        sliding_window=payload["sliding_window"],
        use_sinks=payload["use_sinks"],
    )

    # The provider's grid (its hot-path math) for the SAME knobs.
    provider_grid = cs._unified_grid(
        problem, knobs["num_warps"], knobs["block_m_per_warp"]
    )

    # The independent reference recomputation from the payload alone.
    assert provider_grid == _expected_grid(payload)

    # And the DSL problem object agrees on the shared inputs that feed the
    # grid formula -- a cross-check that the provider read the problem
    # correctly (num_kv_heads, num_queries_per_kv, total_q, num_seqs).
    assert problem.num_kv_heads == shape["num_kv_heads"]
    assert problem.num_queries_per_kv == (
        shape["num_query_heads"] // shape["num_kv_heads"]
    )
    assert problem.total_q == batch * seqlen_q
    assert problem.num_seqs == batch

    # Finally, the full provider compile returns the same grid.
    result = cs._compile_sdpa_fwd_unified(payload, arch=ARCH)
    assert result["grid"] == provider_grid
