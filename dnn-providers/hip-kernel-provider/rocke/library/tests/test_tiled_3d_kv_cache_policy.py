# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""K/V cache policy of the gfx950 3D split-KV segment kernel.

The dispatcher streams decode K/V (``kv_cache_policy="stream"``, the ``nt``
bit) and leaves every other batch at the default caching. No GPU: the ISA test
uses COMGR + ``llvm-objdump`` and skips when either is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kernels.common import attention_unified as au
from kernels.common.attention_unified import UnifiedAttentionProblem
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
        ("gfx942", {}, "all"),
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
