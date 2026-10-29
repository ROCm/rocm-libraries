# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packed paged varlen inputs and checks shared by the gfx942 and gfx950 dense
numeric tests. Each arch file passes its own runner, spec factory and reference,
so both arches run the same cases against the same poisoned cache."""

import math

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

# Per-sequence lengths of a packed paged varlen batch. "edges": tile-aligned (256,
# 512), page-aligned (48), 1, 17, block_n - 1, block_n + 1 and an empty sequence.
PAGED_VARLEN_EDGES = [256, 1, 17, 63, 65, 0, 512, 48]
_PAD_ID = 0x7FFF_FFFF
_TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def paged_varlen_lengths(name):
    if name == "edges":
        return list(PAGED_VARLEN_EDGES)
    if name == "single":
        return [777]
    gen = torch.Generator().manual_seed(1)
    return torch.randint(1, 1024, (3,), generator=gen).tolist()  # "random3"


def paged_varlen_inputs(lens, block_size, d, dtype, hq=16, hkv=4):
    """Packed Q and per-sequence K/V scattered into a paged cache built to catch
    any read the kernel must not use: a pool twice the needed pages at permuted
    ids, NaN in every unused page (block 0 included, which out-of-sequence lookups
    read), NaN in each last page's slots past the sequence's length, and table
    columns past each sequence's pages set to an invalid id."""
    tdt = _TORCH_DT[dtype]
    total = sum(lens)
    q = torch.randn(total, hq, d, device="cuda", dtype=tdt)
    ks = [torch.randn(n, hkv, d, device="cuda", dtype=tdt) for n in lens]
    vs = [torch.randn(n, hkv, d, device="cuda", dtype=tdt) for n in lens]
    pages = [-(-n // block_size) for n in lens]
    width = max(1, -(-max(lens) // block_size))
    num_kv_blocks = 2 * sum(pages) + 1
    phys = 1 + torch.randperm(num_kv_blocks - 1, device="cuda")[: sum(pages)]
    k_cache = torch.full(
        (num_kv_blocks, block_size, hkv, d), float("nan"), device="cuda", dtype=tdt
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    table = torch.full((len(lens), width), _PAD_ID, device="cuda", dtype=torch.int32)
    first = 0
    for b, n in enumerate(lens):
        ids = phys[first : first + pages[b]]
        first += pages[b]
        table[b, : pages[b]] = ids.to(torch.int32)
        for p in range(pages[b]):
            rows = min(n, (p + 1) * block_size) - p * block_size
            k_cache[ids[p], :rows] = ks[b][p * block_size : p * block_size + rows]
            v_cache[ids[p], :rows] = vs[b][p * block_size : p * block_size + rows]
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    kv_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return q, ks, vs, k_cache, v_cache, table, kv_lens, cu


def check_paged_varlen(
    run, make_spec, reference, tolerance, lens, block_size, d, dtype, window
):
    """One packed launch of ``lens`` against ``reference(q, k, v, scale, window)``
    per sequence ([1, S, H, D] tensors). ``make_spec(lens, block_size, d, dtype,
    num_kv_blocks, window=...)`` builds the arch's varlen spec."""
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(d)
    q, ks, vs, k_cache, v_cache, table, kv_lens, cu = paged_varlen_inputs(
        lens, block_size, d, dtype
    )
    spec = make_spec(lens, block_size, d, dtype, k_cache.shape[0], window=window)
    assert spec.varlen
    # Unwritten rows stay NaN. The rows past the packed output catch a tail
    # store of the last sequence's partial q-block that its guard missed.
    total = q.shape[0]
    out_buf = torch.full(
        (total + 256, *q.shape[1:]), float("nan"), device="cuda", dtype=q.dtype
    )
    out = out_buf[:total]
    run(
        spec=spec,
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        scale=scale,
        block_tables=table,
        kv_lens=kv_lens,
        cu_seqlens_q=cu,
    )
    torch.cuda.synchronize()
    assert torch.isnan(out_buf[total:]).all(), "store past the packed output"
    for b, n in enumerate(lens):
        if n == 0:
            continue
        name = f"{dtype} D{d} pgd{block_size} swa{window} seq{b} len{n}"
        got = out[int(cu[b]) : int(cu[b + 1])]
        assert torch.isfinite(got).all(), f"{name}: NaN (unwritten row or bad read)"
        ref = reference(
            q[None, int(cu[b]) : int(cu[b + 1])],
            ks[b][None],
            vs[b][None],
            scale,
            window,
        )[0]
        max_abs = (ref - got.float()).abs().max().item()
        assert max_abs < tolerance(dtype), f"{name}: max_abs={max_abs:.3e}"


def check_whole_tile_sequences(run, make_spec, single_spec, lens, block_size, d):
    """Each sequence of one packed fp16 launch must equal, bit for bit, its own
    single-sequence launch with ``single_spec(n, num_kv_blocks)``."""
    dtype = "fp16"
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(d)
    q, _, _, k_cache, v_cache, table, kv_lens, cu = paged_varlen_inputs(
        lens, block_size, d, dtype
    )
    nb = k_cache.shape[0]
    out = torch.full_like(q, float("nan"))
    run(
        spec=make_spec(lens, block_size, d, dtype, nb),
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        scale=scale,
        block_tables=table,
        kv_lens=kv_lens,
        cu_seqlens_q=cu,
    )
    for b, n in enumerate(lens):
        q_b = q[int(cu[b]) : int(cu[b + 1])][None].contiguous()
        out_b = torch.empty_like(q_b)
        run(
            spec=single_spec(n, nb),
            q=q_b,
            k=k_cache,
            v=v_cache,
            out=out_b,
            scale=scale,
            block_tables=table[b : b + 1, : -(-n // block_size)].contiguous(),
            kv_lens=kv_lens[b : b + 1].contiguous(),
        )
        torch.cuda.synchronize()
        got = out[int(cu[b]) : int(cu[b + 1])]
        assert torch.equal(got, out_b[0]), f"D{d} seq{b} len{n}: bits differ"
