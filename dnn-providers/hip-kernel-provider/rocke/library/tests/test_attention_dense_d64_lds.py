# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""D=64 K group-pad invariants for the gfx950 dense flash-attn kernel.

At D=128 one ``async_buffer_load_lds`` fills exactly one row, so K can carry a
per-row pad. At D=64 one instruction fills TWO rows, so the pad has to sit
between DMA row-groups instead (``_LDS_PAD_K64``). These tests lock the two
properties that make that safe:

* the pad is inert at D=128 -- same IR bytes, same LDS -- so the tuned D=128
  schedule cannot drift when the D=64 pad is re-swept;
* at D=64 the pad is live and grows LDS by exactly one pad per row-group.

Plus the alignment precondition: a group pitch that is not 16-byte aligned would
break the ``ds_read_b128`` QK read.

Pure text lowering -- no GPU and no comgr required.
"""

import hashlib
import os
import re

import pytest

import kernels.gfx950.attention_dense as ad
from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

_NBUF = 2  # module-private double-buffer depth, mirrored here for the LDS math
_POOL_RE = re.compile(r"addrspace\(3\)\s+global\s+\[(\d+) x i8\]")


def _spec(head_size, *, persistent=False, block_n=64):
    return AttentionDenseSpec(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=head_size,
        causal=True,
        dtype="bf16",
        block_n=block_n,
        persistent=persistent,
        num_persistent=256,
    )


def _lower(spec):
    from rocke.core.lower_llvm import (
        _lower_kernel_to_llvm_python,
        _resolve_llvm_flavor,
    )

    return _lower_kernel_to_llvm_python(
        build_attention_dense(spec, arch="gfx950"),
        arch="gfx950",
        llvm_flavor=_resolve_llvm_flavor(),
    )


def _sha(spec):
    return hashlib.sha256(_lower(spec).encode("utf-8")).hexdigest()


def _lds_pool_bytes(spec):
    """Size of the unified addrspace(3) smem pool for this spec."""
    m = _POOL_RE.search(_lower(spec))
    assert m, "no addrspace(3) smem pool global found in the lowered IR"
    return int(m.group(1))


def _with_pad(monkeypatch, pad):
    """The builders read ``_LDS_PAD_K64`` as a module global at build time."""
    monkeypatch.setattr(ad, "_LDS_PAD_K64", pad)


def test_k_group_pad_is_16_byte_aligned():
    """A group pitch of ROWS_PER_INSTR*D + pad elements must stay 16-byte
    aligned or the QK ds_read_b128 loses its alignment."""
    assert ad._LDS_PAD_K64 % 8 == 0, (
        f"_LDS_PAD_K64={ad._LDS_PAD_K64} must be a multiple of 8 bf16 elements "
        "(16 bytes) to keep the K group pitch ds_read_b128-aligned"
    )
    assert ad._LDS_PAD_K64 >= 0


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("pad", [0, 8, 32])
def test_pad_is_inert_at_d128(monkeypatch, persistent, pad):
    """D=128 packs one row per DMA instr and keeps its own per-row pad, so the
    D=64 group pad must not perturb its IR at all."""
    spec = _spec(128, persistent=persistent)
    _with_pad(monkeypatch, 0)
    base = _sha(spec)
    _with_pad(monkeypatch, pad)
    assert _sha(spec) == base, (
        f"_LDS_PAD_K64={pad} changed the D=128 IR (persistent={persistent}); "
        "the D=128 path must be byte-identical across D=64 pad changes"
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_pad_is_live_at_d64(monkeypatch, persistent):
    """Sanity counterpart: the pad must actually reach the D=64 codegen."""
    spec = _spec(64, persistent=persistent)
    _with_pad(monkeypatch, 0)
    unpadded = _sha(spec)
    _with_pad(monkeypatch, 16)
    assert (
        _sha(spec) != unpadded
    ), f"_LDS_PAD_K64 had no effect on the D=64 IR (persistent={persistent})"


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("block_n", [32, 64, 128])
@pytest.mark.parametrize("pad", [8, 16, 32])
def test_d64_lds_grows_by_one_pad_per_row_group(monkeypatch, block_n, pad, persistent):
    """One pad per DMA row-group, on the K tile only. rows_per_instr = 128//D = 2
    at D=64, so the group count is block_n // 2 and the growth is
    NBUF * (block_n // 2) * pad * 2 bytes. Checked on both builders -- the
    persistent one is the configuration that ships for long sequences."""
    spec = _spec(64, block_n=block_n, persistent=persistent)
    _with_pad(monkeypatch, 0)
    base = _lds_pool_bytes(spec)
    _with_pad(monkeypatch, pad)
    grown = _lds_pool_bytes(spec)
    rows_per_instr = 128 // 64
    expect = _NBUF * (block_n // rows_per_instr) * pad * 2
    assert grown - base == expect, (
        f"D=64 block_n={block_n} pad={pad} persistent={persistent}: LDS grew "
        f"{grown - base} bytes, expected {expect} (one pad per K row-group, "
        "V untouched)"
    )


@pytest.mark.parametrize("bad", [4, 12, -8])
def test_bad_pad_is_rejected(bad):
    """A pitch that is not 16-byte aligned must fail loudly: smem_load_vN stamps
    `align 16` on the n=8 read unconditionally, so an 8-byte-aligned pitch keeps
    the ds_read_b128 and breaks its alignment contract silently."""
    import importlib

    import pytest as _pytest

    old = os.environ.get("ROCKE_DENSE_KPAD64")
    os.environ["ROCKE_DENSE_KPAD64"] = str(bad)
    try:
        with _pytest.raises(ValueError, match="multiple of 8"):
            importlib.reload(ad)
    finally:
        if old is None:
            os.environ.pop("ROCKE_DENSE_KPAD64", None)
        else:
            os.environ["ROCKE_DENSE_KPAD64"] = old
        importlib.reload(ad)


@pytest.mark.parametrize("pad", [8, 16, 32])
def test_d128_lds_unchanged_by_pad(monkeypatch, pad):
    spec = _spec(128)
    _with_pad(monkeypatch, 0)
    base = _lds_pool_bytes(spec)
    _with_pad(monkeypatch, pad)
    assert (
        _lds_pool_bytes(spec) == base
    ), f"_LDS_PAD_K64={pad} changed the D=128 LDS footprint"
