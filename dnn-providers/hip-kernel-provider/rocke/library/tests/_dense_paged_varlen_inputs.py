# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Packed paged varlen inputs and checks shared by the gfx942 and gfx950 dense
numeric tests. Each arch file passes its own runner, spec factory and reference,
so both arches run the same cases against the same poisoned cache."""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

# Per-sequence lengths of a packed paged varlen batch. "edges": tile-aligned (256,
# 512), page-aligned (48), 1, 17, block_n - 1, block_n + 1 and an empty sequence.
PAGED_VARLEN_EDGES = [256, 1, 17, 63, 65, 0, 512, 48]
# A ragged batch for the i64 page-base checks: a long sequence ending inside a Q
# and a KV tile, a two-page one, an empty one and a tile-aligned one.
PAGED_VARLEN_RAGGED = [1000, 17, 0, 512]
# Top-left paged varlen with q_len != kv_len, as (kv_lens, q_lens): unequal maxima
# both ways, equal maxima over mixed sequences, sequences without keys (kv_len 0),
# and under a window, rows past kv_len + window that see no key.
PAGED_VARLEN_TOP_LEFT_CROSS = {
    "q_lt_kv": ([512, 300, 17, 65], [256, 300, 1, 64]),
    "q_gt_kv": ([256, 17, 0, 100], [512, 64, 33, 100]),
    "equal_maxima": ([512, 256, 0], [256, 512, 40]),
}
_PAD_ID = 0x7FFF_FFFF
_TORCH_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}


def paged_varlen_lengths(name):
    if name == "edges":
        return list(PAGED_VARLEN_EDGES)
    if name == "single":
        return [777]
    gen = torch.Generator().manual_seed(1)
    return torch.randint(1, 1024, (3,), generator=gen).tolist()  # "random3"


def paged_varlen_inputs(lens, block_size, d, dtype, hq=16, hkv=4, q_lens=None):
    """Packed Q and per-sequence K/V scattered into a paged cache built to catch
    any read the kernel must not use: a pool twice the needed pages at permuted
    ids, NaN in every unused page (block 0 included, which out-of-sequence lookups
    read), NaN in each last page's slots past the sequence's length, and table
    columns past each sequence's pages set to an invalid id. ``lens`` are the KV
    lengths; the query lengths are ``q_lens`` (bottom-right), else the same."""
    tdt = _TORCH_DT[dtype]
    q_lens = lens if q_lens is None else q_lens
    total = sum(q_lens)
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
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    kv_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return q, ks, vs, k_cache, v_cache, table, kv_lens, cu


def check_paged_varlen(
    run,
    make_spec,
    reference,
    tolerance,
    lens,
    block_size,
    d,
    dtype,
    window,
    q_lens=None,
    sinks=False,
):
    """One packed launch of ``lens`` against ``reference(q, k, v, scale, window)``
    per sequence ([1, S, H, D] tensors). ``make_spec(lens, block_size, d, dtype,
    num_kv_blocks, window=...)`` builds the arch's varlen spec. ``q_lens``: query
    lengths that differ from the KV lengths ``lens`` (any under top-left,
    ``q_len <= kv_len`` under bottom-right). ``sinks``:
    per-head sink logits go to the launch and to ``reference(..., sinks=...)``."""
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(d)
    q, ks, vs, k_cache, v_cache, table, kv_lens, cu = paged_varlen_inputs(
        lens, block_size, d, dtype, q_lens=q_lens
    )
    q_lens = lens if q_lens is None else q_lens
    spec = make_spec(lens, block_size, d, dtype, k_cache.shape[0], window=window)
    assert spec.varlen
    sink_kw = {}
    if sinks:
        spec = dataclasses.replace(spec, use_sinks=True)
        # Large enough that a dropped sink moves every row.
        sink_kw["sinks"] = (2.0 * torch.randn(q.shape[1], device="cuda")).to(q.dtype)
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
        **sink_kw,
    )
    torch.cuda.synchronize()
    assert torch.isnan(out_buf[total:]).all(), "store past the packed output"
    for b, n in enumerate(lens):
        if q_lens[b] == 0:
            continue
        name = f"{dtype} D{d} pgd{block_size} swa{window} seq{b} len{q_lens[b]}x{n}"
        got = out[int(cu[b]) : int(cu[b + 1])]
        assert torch.isfinite(got).all(), f"{name}: NaN (unwritten row or bad read)"
        ref = reference(
            q[None, int(cu[b]) : int(cu[b + 1])],
            ks[b][None],
            vs[b][None],
            scale,
            window,
            **sink_kw,
        )[0]
        max_abs = (ref - got.float()).abs().max().item()
        assert max_abs < tolerance(dtype), f"{name}: max_abs={max_abs:.3e}"
        keyless = (ref == 0).flatten(1).all(dim=1)
        assert not got[keyless].any(), f"{name}: a row without keys is not zero"
    # For negative controls against other references.
    return q, ks, vs, out, cu, scale


def check_top_left_cross_length(
    run, make_spec, reference, tolerance, case, block_size, d, dtype, window, sinks
):
    """:func:`check_paged_varlen` over a :data:`PAGED_VARLEN_TOP_LEFT_CROSS` case;
    the spec's query maximum comes from the query lengths."""
    kv_lens, q_lens = PAGED_VARLEN_TOP_LEFT_CROSS[case]

    def make_cross_spec(lens, *args, **kw):
        spec = make_spec(lens, *args, **kw)
        return dataclasses.replace(spec, seqlen_q=max(q_lens))

    check_paged_varlen(
        run,
        make_cross_spec,
        reference,
        tolerance,
        kv_lens,
        block_size,
        d,
        dtype,
        window,
        q_lens=q_lens,
        sinks=sinks,
    )


def check_one_kernel_serves_two_caches(
    run, make_spec, reference, tolerance, launcher_cache, cache_key
):
    """Two packed launches whose batch, lengths and cache size all differ run on
    one compiled kernel: the same cache slot holds the same launcher after both,
    and both match the reference. ``cache_key(spec)`` is the arch's
    ``attention_dense_cache_key``."""
    runs = ([256, 17, 65], [1000, 48])  # batch 3, then 2 with a larger cache
    launchers = []
    sizes = []
    for lens in runs:
        # The cache size paged_varlen_inputs allocates.
        sizes.append(2 * sum(-(-n // 16) for n in lens) + 1)
        spec = make_spec(lens, 16, 128, "fp16", sizes[-1], window=0)
        check_paged_varlen(
            run, make_spec, reference, tolerance, lens, 16, 128, "fp16", 0
        )
        launchers.append(launcher_cache[cache_key(spec)])
    assert sizes[0] != sizes[1]
    assert launchers[0] is launchers[1], "a second compile for another cache size"


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


def _run_packed(run, spec, q, k_cache, v_cache, table, kv_lens, cu, **kw):
    out = torch.full_like(q, float("nan"))
    run(
        spec=spec,
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        scale=1.0 / math.sqrt(spec.head_size),
        block_tables=table,
        kv_lens=kv_lens,
        cu_seqlens_q=cu,
        **kw,
    )
    torch.cuda.synchronize()
    return out


def check_i64_matches_i32(
    run, make_spec, lens, block_size, d, dtype, window, q_lens=None
):
    """One packed launch of ``lens`` through the i64 and the i32 page bases on
    the same poisoned cache: a page base or in-page offset off by anything reads
    a NaN page, a NaN slot or wrong keys, so the bits differ. ``q_lens`` as in
    ``check_paged_varlen``."""
    torch.manual_seed(0)
    q, _, _, k_cache, v_cache, table, kv_lens, cu = paged_varlen_inputs(
        lens, block_size, d, dtype, q_lens=q_lens
    )
    spec = make_spec(lens, block_size, d, dtype, k_cache.shape[0], window=window)
    assert spec.varlen and not spec.use_i64_kv_addr
    i32, i64 = (
        _run_packed(
            run,
            dataclasses.replace(spec, use_i64_kv_addr=flag),
            q,
            k_cache,
            v_cache,
            table,
            kv_lens,
            cu,
        )
        for flag in (False, True)
    )
    assert torch.isfinite(i64).all(), "i64: NaN (unwritten row or bad read)"
    assert torch.equal(i32, i64), "i64 != i32 bits"


def check_bad_block_id_reads_the_last_block(run, make_spec, lens, seq, column, bad_id):
    """With ``validate_paged=False`` an out-of-range id in a used ``column`` of
    sequence ``seq`` is clamped to the last block: the output equals that of a
    table naming it. ``bad_id="nb"`` stands for the cache's block count."""
    block_size, d, dtype = 16, 128, "fp16"
    assert column < -(-lens[seq] // block_size), "the column must be read"
    torch.manual_seed(0)
    q, _, _, k_cache, v_cache, table, kv_lens, cu = paged_varlen_inputs(
        lens, block_size, d, dtype
    )
    # Finite unused pages, so that reading the last block gives a finite output
    # that can be compared bit for bit.
    k_cache = torch.where(k_cache.isnan(), torch.randn_like(k_cache), k_cache)
    v_cache = torch.where(v_cache.isnan(), torch.randn_like(v_cache), v_cache)
    num_kv_blocks = k_cache.shape[0]
    spec = dataclasses.replace(
        make_spec(lens, block_size, d, dtype, num_kv_blocks), use_i64_kv_addr=True
    )
    bad, clamped = table.clone(), table.clone()
    bad[seq, column] = num_kv_blocks if bad_id == "nb" else bad_id
    clamped[seq, column] = num_kv_blocks - 1
    args = (q, k_cache, v_cache)
    out_bad = _run_packed(run, spec, *args, bad, kv_lens, cu, validate_paged=False)
    out_clamped = _run_packed(run, spec, *args, clamped, kv_lens, cu)
    assert torch.equal(out_bad, out_clamped)
