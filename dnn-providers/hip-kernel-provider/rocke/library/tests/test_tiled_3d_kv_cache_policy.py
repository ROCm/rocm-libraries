# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K/V cache policy (``kv_cache_policy``) of the gfx950 and gfx942 3D segment
kernels."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kernels.common import attention_unified as au
from kernels.common.attention_unified import UnifiedAttentionProblem
from kernels.gfx942 import attention_tiled_3d as gfx942_3d
from kernels.gfx950.attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
)


def _problem(**kw) -> UnifiedAttentionProblem:
    base: dict[str, object] = {
        "total_q": 16,
        "num_seqs": 16,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_size": 128,
        "block_size": 16,
        "max_seqlen_q": 1,
        "max_seqlen_k": 8192,
        "dtype": "bf16",
    }
    base.update(kw)
    return UnifiedAttentionProblem(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "arch, kw, expected",
    [
        ("gfx950", {}, "stream"),
        ("gfx950", {"total_q": 64, "max_seqlen_q": 4}, "all"),
        ("gfx950", {"use_fp8": True}, "all"),
        ("gfx942", {}, "stream"),
        ("gfx942", {"total_q": 64, "max_seqlen_q": 4}, "all"),
        ("gfx942", {"use_fp8": True}, "all"),
        ("gfx1250", {}, "all"),
    ],
)
def test_dispatch_policy(arch, kw, expected):
    assert au._kv_cache_policy_3d(_problem(**kw), arch=arch) == expected


def test_fp8_kv_rejects_streaming():
    with pytest.raises(ValueError, match="fp16/bf16 KV storage only"):
        UnifiedAttention3DTiledSpec(
            head_size=128,
            block_size=16,
            num_query_heads=64,
            num_kv_heads=8,
            dtype="bf16",
            use_sinks=False,
            sliding_window=0,
            has_softcap=False,
            num_segments=16,
            kv_storage_dtype="fp8e4m3",
            kv_cache_policy="stream",
        )


@pytest.mark.parametrize("kv_layout", ["paged", "strided"])
def test_native_builder_matches_python(kv_layout):
    from dataclasses import asdict

    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    engine = pytest.importorskip("rocke_engine")
    spec = UnifiedAttention3DTiledSpec(
        head_size=128,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=8,
        num_seqs=3,
        kv_layout=kv_layout,
        use_decode_grid=True,
        kv_cache_policy="stream",
    )
    reduce = UnifiedAttentionReduceTiledSpec(
        head_size=128,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=8,
    )
    expected = _lower_kernel_to_llvm_python(
        build_unified_attention_3d_tiled(spec, arch="gfx950"), arch="gfx950"
    ) + _lower_kernel_to_llvm_python(
        build_unified_attention_reduce_tiled(reduce, arch="gfx950"), arch="gfx950"
    )
    actual = engine.gfx950_attention_tiled_3d_lower_llvm(asdict(spec), "gfx950")
    assert actual == expected


def _objdump() -> str | None:
    for candidate in (
        os.environ.get("LLVM_OBJDUMP"),
        os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm/bin/llvm-objdump"),
        shutil.which("llvm-objdump"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


@pytest.mark.parametrize("policy, bits", [("all", set()), ("stream", {"nt"})])
@pytest.mark.parametrize("tile", [None, 32])
def test_kv_copies_carry_the_policy_bits(policy, bits, tile):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    spec = UnifiedAttention3DTiledSpec(
        head_size=128,
        block_size=16,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=16,
        tile_size_override=tile,
        use_decode_grid=True,
        kv_cache_policy=policy,
    )
    kernel = build_unified_attention_3d_tiled(spec, arch="gfx950")
    hsaco = compile_kernel(kernel, arch="gfx950").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", "--mcpu=gfx950", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    copies = [
        ln.split("//")[0].split()
        for ln in isa.splitlines()
        if "buffer_load_dwordx4 " in ln and " lds" in ln
    ]
    assert copies
    assert all(set(c) & {"sc0", "sc1", "nt"} == bits for c in copies), copies


def _gfx942_spec(**kw) -> gfx942_3d.UnifiedAttention3DTiledSpec:
    base: dict[str, object] = {
        "head_size": 128,
        "block_size": 16,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "dtype": "bf16",
        "use_sinks": False,
        "sliding_window": 0,
        "has_softcap": False,
        "num_segments": 16,
        "use_decode_grid": True,
    }
    base.update(kw)
    return gfx942_3d.UnifiedAttention3DTiledSpec(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", ["global", "nt"])
def test_gfx942_wide_loader_takes_all_or_stream(policy):
    _gfx942_spec(kv_cache_policy=policy)
    with pytest.raises(ValueError, match="wide K/V loader"):
        _gfx942_spec(use_wide_kv_load=True, kv_cache_policy=policy)


@pytest.mark.parametrize(
    "kw",
    [
        {"use_wide_kv_load": True, "kv_cache_policy": "stream"},
        {"kv_cache_policy": "nt"},
        {"kv_layout": "strided", "kv_cache_policy": "stream"},
    ],
)
def test_gfx942_native_builder_matches_python(kw):
    from dataclasses import asdict

    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    engine = pytest.importorskip("rocke_engine")
    spec = _gfx942_spec(num_query_heads=32, num_segments=8, num_seqs=3, **kw)
    # The gfx942 binding lowers the segment kernel only.
    expected = _lower_kernel_to_llvm_python(
        gfx942_3d.build_unified_attention_3d_tiled(spec, arch="gfx942"), arch="gfx942"
    )
    actual = engine.gfx942_attention_tiled_3d_lower_llvm(asdict(spec), "gfx942")
    assert actual == expected


@pytest.mark.parametrize(
    "policy, bits", [("all", set()), ("stream", {"nt"}), ("nt", {"sc0", "nt"})]
)
@pytest.mark.parametrize("wide", [True, False])
def test_gfx942_kv_copies_carry_the_policy_bits(policy, bits, wide):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime import comgr

    if wide and policy == "nt":
        pytest.skip("the wide loader takes 'all' or 'stream'")
    objdump = _objdump()
    if objdump is None:
        pytest.skip("llvm-objdump not found (LLVM_OBJDUMP / ROCM_PATH / PATH)")
    try:
        comgr._resolve_lib()
    except comgr.ComgrError as e:
        pytest.skip(f"COMGR not loadable: {str(e)[:200]}")
    spec = _gfx942_spec(use_wide_kv_load=wide, kv_cache_policy=policy)
    kernel = gfx942_3d.build_unified_attention_3d_tiled(spec, arch="gfx942")
    hsaco = compile_kernel(kernel, arch="gfx942").hsaco
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        isa = subprocess.run(
            [objdump, "-d", "--mcpu=gfx942", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    lines = [ln.split("//")[0].split() for ln in isa.splitlines()]
    policy_bits = {"sc0", "sc1", "nt"}
    if wide:
        # Every 16-byte global load but Q's 4 per lane (16 rows x 128 dims) is
        # a K/V copy.
        loads = [ln for ln in lines if ln and ln[0] == "global_load_dwordx4"]
        flagged = [c for c in loads if set(c) & policy_bits]
        assert len(loads) > 4
        assert len(flagged) == (len(loads) - 4 if bits else 0), loads
        assert all(set(c) & policy_bits == bits for c in flagged), flagged
    else:
        copies = [
            ln for ln in lines if ln and ln[0] == "buffer_load_dword" and "lds" in ln
        ]
        assert copies
        assert all(set(c) & policy_bits == bits for c in copies), copies
