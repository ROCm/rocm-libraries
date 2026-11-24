# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense prefill paged K/V past the i32 range (``use_i64_kv_addr``): the spec rule
and the per-page buffer resources in the gfx942 and gfx950 IR."""

import importlib
import re
from dataclasses import replace

import pytest
from kernels.common._attention_dense_paged import paged_page_bytes
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.common._spec_checks import I32_KV_BUFFER_BYTES
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

_ARCHS = ("gfx942", "gfx950")
_MODULES = {
    arch: importlib.import_module(f"kernels.{arch}.attention_dense") for arch in _ARCHS
}
_SPEC_CLASSES = {
    "gfx942": _MODULES["gfx942"].Gfx942AttentionDenseSpec,
    "gfx950": _MODULES["gfx950"].Gfx950AttentionDenseSpec,
}


def _paged(arch, **kw):
    fields = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=16,
        num_kv_heads=8,
        head_size=128,
        dtype="fp16",
        sliding_window=256,
        paged=True,
        block_size=16,
        num_kv_blocks=32,
    )
    return _SPEC_CLASSES[arch](**{**fields, **kw})


def _paged_varlen(arch, **kw):
    """Three packed sequences whose maxima end inside a Q and a KV tile."""
    return replace(
        _paged(arch),
        batch=3,
        seqlen_q=1000,
        seqlen_kv=1000,
        varlen=True,
        num_kv_blocks=192,
        **kw,
    )


_BODIES = {"single_seq": _paged, "varlen": _paged_varlen}

# Blocks of one page that fill exactly the i32 K/V buffer range.
_I32_BLOCKS = I32_KV_BUFFER_BYTES // paged_page_bytes(_paged("gfx950"))


@pytest.mark.parametrize("body", sorted(_BODIES))
@pytest.mark.parametrize("arch", _ARCHS)
def test_threshold_is_the_unified_i32_buffer_range(arch, body):
    supports = _MODULES[arch].supports_attention_dense
    paged = _BODIES[body](arch)
    assert _I32_BLOCKS * paged_page_bytes(paged) == I32_KV_BUFFER_BYTES
    at_limit = replace(paged, num_kv_blocks=_I32_BLOCKS)
    assert supports(at_limit) == (True, "")
    with pytest.raises(ValueError, match="use_i64_kv_addr"):
        replace(paged, num_kv_blocks=_I32_BLOCKS + 1)
    past = replace(paged, num_kv_blocks=_I32_BLOCKS + 1, use_i64_kv_addr=True)
    assert supports(past) == (True, "")


@pytest.mark.parametrize("arch", _ARCHS)
def test_flag_is_paged_only_and_may_be_forced_on_a_small_cache(arch):
    paged = _paged(arch)
    forced = replace(paged, use_i64_kv_addr=True)
    assert _MODULES[arch].supports_attention_dense(forced) == (True, "")
    with pytest.raises(ValueError, match="requires paged"):
        replace(paged, paged=False, block_size=0, num_kv_blocks=0, use_i64_kv_addr=True)


@pytest.mark.parametrize("body", sorted(_BODIES))
@pytest.mark.parametrize("arch", _ARCHS)
def test_name_and_cache_key_split_only_on_the_flag(arch, body):
    paged = _BODIES[body](arch)
    i64 = replace(paged, use_i64_kv_addr=True)
    assert "i64kv" in i64.kernel_name()
    assert ("varlen" in i64.kernel_name()) is paged.varlen
    assert "i64kv" not in paged.kernel_name()
    assert i64.kernel_name().replace("_i64kv", "") == paged.kernel_name()
    key32 = attention_dense_cache_key(paged, arch=arch)
    key64 = attention_dense_cache_key(i64, arch=arch)
    assert key32 != key64
    diff = set(key32[2]) ^ set(key64[2])
    assert {name for name, _ in diff} == {"use_i64_kv_addr"}


def _lower(spec, arch):
    return _lower_kernel_to_llvm_python(
        _MODULES[arch].build_attention_dense(spec), arch=arch, llvm_flavor="llvm22"
    )


def _counts(spec, arch):
    ir = _lower(spec, arch)
    return dict(
        resources=len(re.findall(r"@llvm\.amdgcn\.make\.buffer\.rsrc\.p8\.p1\(", ir)),
        copies=ir.count("call void @llvm.amdgcn.raw.ptr.buffer.load.lds"),
        # gfx942 cfvst V: one i32 register load per key row and lane.
        v_reg_loads=ir.count("call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32"),
        page_loads=len(
            re.findall(
                r"getelementptr inbounds i32, ptr addrspace\(1\) %block_tables", ir
            )
        ),
    )


def _page_bytes_terms(defs, value):
    """``(block_size, head_size)`` of a page-size value built as
    ``num_kv_heads * head_size * elem_bytes * block_size`` (fp16, 2 bytes)."""
    rows, block_size = re.fullmatch(
        r"mul nsw i32 (%[\w.]+), (\d+)", defs[value]
    ).groups()
    row_bytes, elem = re.fullmatch(r"mul nsw i32 (%[\w.]+), (\d+)", defs[rows]).groups()
    if row_bytes == "%num_kv_heads":
        # One mul by head_size * elem_bytes.
        return int(block_size), int(elem) // 2
    assert int(elem) == 2
    heads, head_size = re.fullmatch(
        r"mul nsw i32 (%[\w.]+), (\d+)", defs[row_bytes]
    ).groups()
    assert heads == "%num_kv_heads"
    return int(block_size), int(head_size)


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "body,variant",
    [
        ("single_seq", {}),
        ("single_seq", {"head_size": 64}),
        ("single_seq", {"sliding_window": 0}),
        ("varlen", {}),
        ("varlen", {"head_size": 64}),
    ],
    ids=["d128_swa", "d64_swa", "d128_causal", "varlen_d128_swa", "varlen_d64_swa"],
)
def test_one_resource_per_wave_per_kv_tile_load(arch, body, variant):
    """i64 replaces the two whole-cache K/V resources with one per K or V tile
    load, shared by all of the wave's copies (or gfx942 cfvst register loads) of
    that tile (a resource per copy would add one per copy). The varlen body's
    other resources (gfx950 bounds its packed Q with one) stay as they are.
    Counts only; the GPU tests check the addresses."""
    spec = _BODIES[body](arch, **variant)
    i32 = _counts(spec, arch)
    i64 = _counts(replace(spec, use_i64_kv_addr=True), arch)
    rows_per_instr = 128 // spec.head_size
    copies_per_tile = spec.block_n // spec.num_waves // rows_per_instr
    assert i64["copies"] == i32["copies"]
    assert i64["copies"] % copies_per_tile == 0
    tile_loads = i64["copies"] // copies_per_tile
    assert tile_loads > 0
    # cfvst V loads each of the wave's rows of the tile with one register load.
    rows_per_wave = spec.block_n // spec.num_waves
    assert i64["v_reg_loads"] == i32["v_reg_loads"]
    assert i64["v_reg_loads"] % rows_per_wave == 0
    tile_loads += i64["v_reg_loads"] // rows_per_wave
    assert i64["resources"] == i32["resources"] - 2 + tile_loads
    # The page lookup stays one per tile load, as on the i32 path.
    assert i64["page_loads"] == i32["page_loads"] == tile_loads


@pytest.mark.parametrize("arch", _ARCHS)
@pytest.mark.parametrize(
    "body,variant",
    [("single_seq", {}), ("varlen", {}), ("varlen", {"head_size": 64})],
    ids=["d128", "varlen_d128", "varlen_d64"],
)
def test_page_base_clamps_the_block_id_to_the_last_block(arch, body, variant):
    """A corrupt id must not become an arbitrary 64-bit page base: every K/V page
    base is ``zext(ult(id, nb) ? id : nb - 1) * zext(page_bytes)`` of the
    wave-uniform id, with ``nb`` the ``num_kv_blocks`` kernarg and ``page_bytes``
    ``block_size`` rows of the ``num_kv_heads`` kernarg. The unsigned compare also
    clamps negative ids."""
    spec = _BODIES[body](arch, use_i64_kv_addr=True, **variant)
    ir = _lower(spec, arch)
    defs = dict(re.findall(r"^\s*(%[\w.]+) = (.*)$", ir, flags=re.M))
    bases = re.findall(
        r"getelementptr inbounds i8, ptr addrspace\(1\) %[kv]_ptr, i64 (%[\w.]+)", ir
    )
    assert len(bases) == _counts(spec, arch)["page_loads"]
    for base in bases:
        ext, ext_bytes = re.fullmatch(
            r"mul (?:nsw )?i64 (%[\w.]+), (%[\w.]+)", defs[base]
        ).groups()
        (page_bytes,) = re.fullmatch(
            r"zext i32 (%[\w.]+) to i64", defs[ext_bytes]
        ).groups()
        assert _page_bytes_terms(defs, page_bytes) == (spec.block_size, spec.head_size)
        (sel,) = re.fullmatch(r"zext i32 (%[\w.]+) to i64", defs[ext]).groups()
        cond, block_id, last = re.fullmatch(
            r"select i1 (%[\w.]+), i32 (%[\w.]+), i32 (%[\w.]+)", defs[sel]
        ).groups()
        assert defs[last] == "sub nsw i32 %num_kv_blocks, 1"
        assert defs[cond] == f"icmp ult i32 {block_id}, %num_kv_blocks"
        assert defs[block_id].startswith("call i32 @llvm.amdgcn.readfirstlane.i32(")
