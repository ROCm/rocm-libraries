# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""D=64 K group-pad invariants for the gfx950 dense flash-attn kernel.

At head_size=128 one ``async_buffer_load_lds`` fills exactly one row, so K can
carry a per-row pad. At 64 one instruction fills TWO rows, so the pad has to sit
between DMA row-groups instead (the ``lds_k_group_pad`` spec field). These tests
lock the properties that make that safe:

* the pad is inert at head_size=128 -- same IR bytes, same LDS -- so the tuned
  128 schedule cannot drift when the 64 pad is re-swept;
* at 64 the pad is live and grows LDS by exactly one pad per row-group;
* a pad that would break ``ds_read_b128`` alignment is rejected by the spec;
* the pad participates in ``kernel_name()``, so two layouts cannot collide on one
  symbol name (or one launcher-cache entry).

Every invariant is checked on both allowed dtypes: they share the packed
group-pad path and are both 2 bytes wide, so bf16 passing does not by itself
establish fp16.

Pure text lowering -- no GPU and no comgr required.
"""

import hashlib
import re

import pytest

from kernels.gfx950.attention_dense import AttentionDenseSpec, build_attention_dense

_NBUF = 2  # module-private double-buffer depth, mirrored here for the LDS math
_POOL_RE = re.compile(r"addrspace\(3\)\s+global\s+\[(\d+) x i8\]")
_DTYPES = ("bf16", "fp16")


def _spec(head_size, *, pad=8, persistent=False, block_n=64, dtype="bf16"):
    return AttentionDenseSpec(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=64,
        num_kv_heads=8,
        head_size=head_size,
        causal=True,
        dtype=dtype,
        block_n=block_n,
        persistent=persistent,
        num_persistent=256,
        lds_k_group_pad=pad,
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


@pytest.mark.parametrize("bad", [4, 12, -8])
def test_misaligned_pad_is_rejected(bad):
    """A group pitch that is not 16-byte aligned must fail loudly: smem_load_vN
    stamps `align 16` on the n=8 read unconditionally, so an 8-byte-aligned pitch
    would keep the ds_read_b128 and break its alignment contract silently."""
    with pytest.raises(ValueError, match="multiple of 8"):
        _spec(64, pad=bad)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("pad", [0, 8, 32])
def test_pad_is_inert_at_d128(persistent, pad, dtype):
    """head_size=128 packs one row per DMA instr and keeps its own per-row pad, so
    the group pad must not perturb its IR at all."""
    base = _sha(_spec(128, pad=0, persistent=persistent, dtype=dtype))
    assert _sha(_spec(128, pad=pad, persistent=persistent, dtype=dtype)) == base, (
        f"lds_k_group_pad={pad} changed the head_size=128 {dtype} IR "
        f"(persistent={persistent}); that path must be byte-identical across "
        "group-pad changes"
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("persistent", [False, True])
def test_pad_is_live_at_d64(persistent, dtype):
    """Sanity counterpart: the pad must actually reach the D=64 codegen."""
    unpadded = _sha(_spec(64, pad=0, persistent=persistent, dtype=dtype))
    assert (
        _sha(_spec(64, pad=16, persistent=persistent, dtype=dtype)) != unpadded
    ), f"lds_k_group_pad had no effect on the D=64 {dtype} IR (persistent={persistent})"


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("block_n", [32, 64, 128])
@pytest.mark.parametrize("pad", [8, 16, 32])
def test_d64_lds_grows_by_one_pad_per_row_group(block_n, pad, persistent, dtype):
    """One pad per DMA row-group, on the K tile only. 128//head_size = 2 at 64, so
    the group count is block_n // 2 and the growth is
    NBUF * (block_n // 2) * pad * 2 bytes. Checked on both builders -- the
    persistent one is the configuration that ships for long sequences. Both dtypes
    are 2 bytes wide, so the expected growth is the same for each."""
    base = _lds_pool_bytes(
        _spec(64, pad=0, block_n=block_n, persistent=persistent, dtype=dtype)
    )
    grown = _lds_pool_bytes(
        _spec(64, pad=pad, block_n=block_n, persistent=persistent, dtype=dtype)
    )
    rows_per_instr = 128 // 64
    expect = _NBUF * (block_n // rows_per_instr) * pad * 2
    assert grown - base == expect, (
        f"D=64 block_n={block_n} pad={pad} persistent={persistent} dtype={dtype}: "
        f"LDS grew {grown - base} bytes, expected {expect} (one pad per K "
        "row-group, V untouched)"
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("pad", [8, 16, 32])
def test_d128_lds_unchanged_by_pad(pad, dtype):
    base = _lds_pool_bytes(_spec(128, pad=0, dtype=dtype))
    assert (
        _lds_pool_bytes(_spec(128, pad=pad, dtype=dtype)) == base
    ), f"lds_k_group_pad={pad} changed the head_size=128 {dtype} LDS footprint"


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pad_is_part_of_the_kernel_identity(dtype):
    """Two D=64 layouts must not share a symbol name -- kernel_name() is also the
    launcher-cache key, so a collision would silently reuse the wrong binary."""
    n0 = _spec(64, pad=0, dtype=dtype).kernel_name()
    n8 = _spec(64, pad=8, dtype=dtype).kernel_name()
    assert n0 != n8, f"pad absent from kernel_name(): {n0}"
    assert "kpad8" in n8 and "kpad0" in n0

    # ... and it must stay out of the head_size=128 name, which is unchanged.
    n128a = _spec(128, pad=0, dtype=dtype).kernel_name()
    n128b = _spec(128, pad=32, dtype=dtype).kernel_name()
    assert n128a == n128b, "group pad leaked into the head_size=128 kernel name"
    assert "kpad" not in n128a
