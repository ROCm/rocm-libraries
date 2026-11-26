# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 tiled 2D K/V copies read through one buffer resource per page of a
tile, under i32 and i64 K/V addressing alike, and the loop carries the page ids
of the current and next tile instead of reading one per copy.

The structural checks run on the CPU. The numeric checks need a gfx942 GPU and
run without torch, except the LSE case (the runner takes the LSE output as a
torch tensor) and the cache past 2^31 bytes. They pin the selector's 2D spec
(with a few knobs set over it) through an explicit tuning spec, over decode and
causal prefill, fp16 and bf16, page sizes 16 to 64, head sizes 64 to 256,
sliding window, softcap, ALiBi, LSE, the two-phase causal loop, the 32x32 MFMA
paths and the global_load_lds K copies. Overstating ``num_kv_blocks`` makes a
small cache take the i64 path, which the launch guard allows. On i32, page ids
at or past the 32-bit buffer bound (or negative) read as zeros through the
buffer copies; the opt-in global_load_lds K path is not bounded. Ids below that
bound but past the real block count still read past the allocation, as through
the old whole-cache resource.
"""

import re
from dataclasses import dataclass, replace

import numpy as np
import pytest

from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

# The selector's spec for bf16 D128 decode with 256 sequences of 4096 tokens,
# GQA 32/8 and block size 16: each 64-token tile spans four pages.
_DECODE = UnifiedAttention2DTiledSpec(
    head_size=128,
    block_size=16,
    num_query_heads=32,
    num_kv_heads=8,
    dtype="bf16",
    use_sinks=False,
    sliding_window=0,
    has_softcap=False,
    num_seqs=256,
    waves_per_eu=2,
    use_register_pv=True,
    tile_size=64,
)

# raw.ptr.buffer.load.lds(rsrc, lds, size, voffset, soffset, offset, aux)
_DMA = re.compile(
    r"call void @llvm\.amdgcn\.raw\.ptr\.buffer\.load\.lds\("
    r"ptr addrspace\(8\) [^,]+, ptr addrspace\(3\) [^,]+, i32 \d+, i32 [^,]+, "
    r"(i32 [^,]+),"
)


def _ir(spec):
    return _lower_kernel_to_llvm_python(
        build_unified_attention_2d_tiled(spec, arch="gfx942"),
        arch="gfx942",
        llvm_flavor="llvm22",
    )


def _counts(ir):
    return dict(
        resources=len(
            re.findall(r"call ptr addrspace\(8\) @llvm\.amdgcn\.make\.buffer\.rsrc", ir)
        ),
        copies=len(_DMA.findall(ir)),
        page_loads=len(
            re.findall(
                r"getelementptr inbounds i32, ptr addrspace\(1\) %block_tables_ptr", ir
            )
        ),
        carried=len(re.findall(r"%page_cur\d+ = phi", ir)),
    )


def _copy_soffsets(ir):
    """The soffset operand of every K/V copy."""
    return _DMA.findall(ir)


@pytest.mark.parametrize(
    "spec",
    [
        _DECODE,
        # Four warps and two 32-token pages per tile.
        replace(_DECODE, block_size=32, num_warps=4),
        # One 64-token page per tile.
        replace(_DECODE, block_size=64),
        # Head size 256 with one warp: each copy covers half a row.
        replace(_DECODE, head_size=256, tile_size=32),
        # Head size 192 with two warps: a copy is neither whole rows nor a part
        # of one row, so each copy builds its whole offset within the page.
        replace(_DECODE, head_size=192, tile_size=32, num_warps=2),
    ],
    ids=["nw1_4pages", "nw4_2pages", "nw1_1page", "d256_part_row", "d192_unsplit"],
)
def test_copies_share_one_resource_per_page(spec):
    """Counts only: the numeric tests check the addressing. Builds the spec
    directly, not through the selector, so it runs without a cache of any size."""
    i32 = _counts(_ir(spec))
    i64 = _counts(_ir(replace(spec, use_i64_kv_addr=True)))
    # i32 and i64 addressing build the same K/V loader.
    assert i32 == i64
    pages = spec.n_blocks_per_tile
    # Each lane copies one dword (two halves) per copy.
    copies_per_tile = spec.tile_size * spec.head_size // (64 * spec.num_warps * 2)
    tile_loads = i32["copies"] // copies_per_tile
    assert tile_loads > 0 and i32["copies"] % copies_per_tile == 0
    # The two whole-cache resources, then one per page of each tile load, where
    # a per-copy resource would add one per copy.
    assert i32["resources"] == 2 + pages * tile_loads
    # Page ids are read for the first two tiles before the loop and for the tile
    # after next in each iteration; no copy reads its own.
    assert i32["page_loads"] == 3 * pages
    assert i32["carried"] == pages


def test_part_row_copies_put_their_offset_in_soffset():
    """Head size 256 with one warp: a copy is half a row, and its offset within
    the page goes to soffset; whole-row copies keep soffset zero."""
    part_row = _copy_soffsets(_ir(replace(_DECODE, head_size=256, tile_size=32)))
    whole_row = _copy_soffsets(_ir(_DECODE))
    assert any(s != "i32 0" for s in part_row)
    assert all(s == "i32 0" for s in whole_row)


@pytest.fixture(scope="module")
def gfx942_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from rocke.runtime.hip_module import get_device_arch

    if get_device_arch() != "gfx942":
        pytest.skip("requires a gfx942 GPU")
    return torch


_BLOCK_SIZE = 16


def _run_decode(torch, q, k, v, block_table, lengths, num_kv_blocks, i64):
    """Launch bf16 decode on the 2D route and return the output; the spec must
    take the K/V page copies with the expected addressing."""
    from kernels.common.attention_unified import (
        UnifiedAttentionProblem,
        _tiled_spec_from_problem,
        run_unified_attention_torch,
    )

    batch, num_query_heads, head_size = q.shape
    problem = UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=num_query_heads,
        num_kv_heads=k.shape[2],
        head_size=head_size,
        block_size=_BLOCK_SIZE,
        max_seqlen_q=1,
        max_seqlen_k=max(lengths),
        dtype="bf16",
        num_kv_blocks=num_kv_blocks,
        # One CTA target makes the 2D grid count as filling the device.
        target_ctas=1,
    )
    assert problem.select_path("gfx942") == "2d"
    spec = _tiled_spec_from_problem(problem, "gfx942")
    assert spec.use_i64_kv_addr == i64 and spec.n_blocks_per_tile > 1
    out = torch.empty_like(q)
    run_unified_attention_torch(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        softmax_scale=head_size**-0.5,
        block_table=block_table,
        softcap=0.0,
        backend="tiled",
    )
    return out


def _assert_matches_reference(torch, out, q, k, v, block_table, lengths):
    num_kv_heads, head_size = k.shape[2], k.shape[3]
    group = q.shape[1] // num_kv_heads
    for b, length in enumerate(lengths):
        pages = block_table[b, : -(-length // _BLOCK_SIZE)].long()
        keys = k[pages].reshape(-1, num_kv_heads, head_size)[:length].float()
        values = v[pages].reshape(-1, num_kv_heads, head_size)[:length].float()
        qb = q[b].view(num_kv_heads, group, head_size).float()
        scores = torch.einsum("hgd,thd->hgt", qb, keys) * head_size**-0.5
        ref = torch.einsum("hgt,thd->hgd", scores.softmax(-1), values).reshape(
            -1, head_size
        )
        # On long sequences the output's spread is about 0.05 (max |o| about
        # 0.2), so a looser absolute bound could hide one misaddressed page.
        # rtol covers the bf16 rounding of the output, which grows with |o| on
        # short sequences (half an ulp is at most |o| / 256).
        torch.testing.assert_close(out[b].float(), ref, atol=2e-3, rtol=2**-8)


def test_i64_decode_reads_past_2gib(gfx942_torch):
    """A cache just over 2 GiB per tensor whose pages straddle the 2^31-byte
    offset, so the i64 page bases are exercised with real addresses."""
    torch = gfx942_torch
    head_size = 128
    batch, seqlen, num_query_heads, num_kv_heads = 4, 1024, 32, 8
    block_bytes = _BLOCK_SIZE * num_kv_heads * head_size * 2
    first_past = 2**31 // block_bytes
    num_blocks = first_past + 128
    pages = seqlen // _BLOCK_SIZE
    free, _ = torch.cuda.mem_get_info()
    if free < 2 * num_blocks * block_bytes + (1 << 30):
        pytest.skip("needs about 5 GiB of free device memory")
    torch.manual_seed(2)
    k = torch.empty(
        num_blocks,
        _BLOCK_SIZE,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.empty_like(k)
    # Only the pages the sequences use are filled: half below 2^31 bytes, half
    # above.
    used = torch.arange(num_blocks - batch * pages, num_blocks, device="cuda")
    assert int(used[0]) < first_past <= int(used[-1])
    k[used] = torch.randn_like(k[used])
    v[used] = torch.randn_like(v[used])
    q = torch.randn(
        batch, num_query_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    block_table = (
        used[torch.randperm(batch * pages, device="cuda")]
        .view(batch, pages)
        .to(torch.int32)
    )
    lengths = [seqlen] * batch
    out = _run_decode(torch, q, k, v, block_table, lengths, num_blocks, True)
    _assert_matches_reference(torch, out, q, k, v, block_table, lengths)


# ---------------------------------------------------------------------------
# Numerics, without torch: rocke.runtime device buffers and a numpy reference,
# so the GPU lanes (which have no torch) run them. Each case pins the kernel
# spec it launches; test_numeric_cases_take_page_copies checks on the CPU that
# every one of them reads K/V through per-page copies with carried page ids.
# ---------------------------------------------------------------------------

_DECODE_LENS = ([1] * 8, [1000, 999, 977, 961, 640, 513, 17, 1000])
# Causal prefill with prior context, a decode row and a context-free row.
_PREFILL_LENS = ([128, 77, 1, 200], [384, 300, 513, 200])


@dataclass(frozen=True)
class _Case:
    dtype: str = "bf16"
    head_size: int = 128
    block_size: int = 16
    num_query_heads: int = 32
    num_kv_heads: int = 8
    lens: tuple = _DECODE_LENS
    sliding_window: int = 0
    softcap: float = 0.0
    alibi: bool = False
    lse: bool = False
    # Spec fields set over the selector's choice.
    overrides: tuple = ()


_CASES = {
    "decode_d128": _Case(),
    "decode_d128_64x4": _Case(num_query_heads=64, num_kv_heads=4),
    "decode_d256": _Case(head_size=256),
    "decode_d192_part_copies": _Case(
        head_size=192, overrides=(("tile_size", 32), ("num_warps", 2))
    ),
    "decode_d128_bs32_fp16": _Case(dtype="fp16", block_size=32),
    "decode_d64_bs32": _Case(head_size=64, block_size=32),
    "decode_global_load_lds_k": _Case(overrides=(("use_global_load_lds_k", True),)),
    "prefill_d128": _Case(lens=_PREFILL_LENS),
    "prefill_d64_fp16": _Case(dtype="fp16", head_size=64, lens=_PREFILL_LENS),
    "prefill_d128_bs64": _Case(block_size=64, lens=_PREFILL_LENS),
    "prefill_phase_split": _Case(
        lens=_PREFILL_LENS, overrides=(("use_causal_mask_phase_split", True),)
    ),
    "prefill_sliding_window": _Case(lens=_PREFILL_LENS, sliding_window=100),
    "prefill_softcap": _Case(lens=_PREFILL_LENS, softcap=30.0),
    "prefill_alibi": _Case(lens=_PREFILL_LENS, alibi=True),
    "prefill_lse": _Case(lens=_PREFILL_LENS, lse=True),
    "prefill_mfma32": _Case(
        lens=_PREFILL_LENS,
        overrides=(
            ("use_mfma_32x32x8", True),
            ("block_m_per_warp", 32),
            ("num_warps", 1),
            ("use_register_pv", False),
        ),
    ),
    "prefill_transposed32": _Case(
        lens=_PREFILL_LENS,
        overrides=(
            ("use_mfma_32x32x8", True),
            ("use_transposed_qk_32x32", True),
            ("block_m_per_warp", 32),
            ("num_warps", 1),
        ),
    ),
}

_ADDRESSING = pytest.mark.parametrize("i64", [False, True], ids=["i32", "i64"])
_CASE_IDS = pytest.mark.parametrize("case_id", list(_CASES))


def _num_blocks(case):
    """Every block-table entry gets its own page, plus spares."""
    kv_lens = case.lens[1]
    return len(kv_lens) * max(-(-k // case.block_size) for k in kv_lens) + 16


def _problem(case, i64):
    from kernels.common.attention_unified import UnifiedAttentionProblem

    q_lens, kv_lens = case.lens
    return UnifiedAttentionProblem(
        total_q=sum(q_lens),
        num_seqs=len(q_lens),
        num_query_heads=case.num_query_heads,
        num_kv_heads=case.num_kv_heads,
        head_size=case.head_size,
        block_size=case.block_size,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        dtype=case.dtype,
        sliding_window=case.sliding_window,
        softcap=case.softcap,
        use_alibi=case.alibi,
        emit_lse=case.lse,
        # Overstating num_kv_blocks makes this small cache take the i64 path,
        # which the launch guard allows.
        num_kv_blocks=(1 << 22) if i64 else _num_blocks(case),
        # One CTA target makes the 2D grid count as filling the device.
        target_ctas=1,
    )


def _tuning_spec(case, i64):
    """The selector's tiled 2D spec for ``case`` with its overrides, pinned."""
    from dispatch.attention.common import AttentionTuningSpec
    from kernels.common.attention_unified import _tiled_spec_from_problem

    problem = _problem(case, i64)
    assert problem.select_path("gfx942") == "2d"
    spec = _tiled_spec_from_problem(
        problem, "gfx942", overrides=dict(case.overrides) or None
    )
    assert spec.use_i64_kv_addr == i64
    return problem, AttentionTuningSpec(
        path="2d",
        arch="gfx942",
        builder_kind="tiled",
        compile_backend="llvm",
        candidate_name="kv_pages_test",
        tuning_id="kv_pages_test",
        kernel_spec=spec,
        num_kv_blocks=problem.num_kv_blocks,
    )


@_ADDRESSING
@_CASE_IDS
def test_numeric_cases_take_page_copies(case_id, i64):
    """Every numeric case launches a kernel whose K/V copies read per-page
    resources and whose loop carries page ids, so none passes vacuously."""
    _, tuning = _tuning_spec(_CASES[case_id], i64)
    counts = _counts(_ir(tuning.kernel_spec))
    assert counts["carried"] == tuning.kernel_spec.n_blocks_per_tile
    assert counts["resources"] > 2


@pytest.fixture(scope="module")
def rt():
    from rocke.runtime.hip_module import Runtime, get_device_arch

    try:
        arch = get_device_arch()
    except Exception as exc:  # no HIP runtime or no device
        pytest.skip(f"requires a gfx942 GPU ({exc})")
    if arch != "gfx942":
        pytest.skip(f"requires a gfx942 GPU, found {arch}")
    return Runtime()


def _encode(x, dtype):
    """float32 values as the kernel's fp16 or bf16 (raw uint16) host array."""
    from rocke.numeric.bf16 import f32_to_bf16

    x = np.asarray(x, dtype=np.float32)
    return x.astype(np.float16) if dtype == "fp16" else f32_to_bf16(x)


def _decode(x, dtype):
    from rocke.numeric.bf16 import bf16_to_f32

    return x.astype(np.float32) if dtype == "fp16" else bf16_to_f32(x)


def _rounded(rng, shape, dtype):
    return _decode(_encode(rng.standard_normal(shape), dtype), dtype)


def _alibi_slopes(num_heads):
    return np.array(
        [2.0 ** (-8.0 * (h + 1) / num_heads) for h in range(num_heads)],
        dtype=np.float32,
    )


def _reference(case, q, k_cache, v_cache, block_table, scale):
    """float64 O [total_q, Hq, D] and natural-log LSE [total_q, Hq]:
    bottom-right causal, with the case's window, softcap and ALiBi."""
    q_lens, kv_lens = case.lens
    hq, hkv = case.num_query_heads, case.num_kv_heads
    outs, lses, start = [], [], 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        pages = block_table[b, : -(-kv_len // case.block_size)]
        keys = k_cache[pages].reshape(-1, hkv, case.head_size)[:kv_len]
        values = v_cache[pages].reshape(-1, hkv, case.head_size)[:kv_len]
        keys = np.repeat(keys.transpose(1, 0, 2), hq // hkv, axis=0).astype(np.float64)
        values = np.repeat(values.transpose(1, 0, 2), hq // hkv, axis=0)
        qs = q[start : start + q_len].transpose(1, 0, 2).astype(np.float64)
        s = qs @ keys.transpose(0, 2, 1) * scale
        if case.softcap:
            s = case.softcap * np.tanh(s / case.softcap)
        context = kv_len - q_len
        key = np.arange(kv_len)[None, :]
        if case.alibi:
            s += _alibi_slopes(hq)[:, None, None] * (key - context)
        diag = np.arange(q_len)[:, None] + context
        visible = key <= diag
        if case.sliding_window:
            visible &= diag - key < case.sliding_window
        s = np.where(visible, s, -np.inf)
        m = s.max(-1, keepdims=True)
        e = np.exp(s - m)
        total = e.sum(-1, keepdims=True)
        outs.append(((e / total) @ values).transpose(1, 0, 2))
        lses.append((m + np.log(total))[..., 0].T)
        start += q_len
    return np.concatenate(outs), np.concatenate(lses)


_BAD_IDS = (-1, 0x7FFFFFFF)


def _run_case(rt, case, i64, *, table_pad=0, bad_ids=False):
    """Launch ``case`` and check O (and LSE) against the reference.

    ``table_pad`` extra block-table rows past the last sequence hold an
    unmapped page id: the loop reads page ids up to two tiles ahead, so a
    short last sequence reads into them unless the read is masked.

    ``bad_ids`` puts out-of-range page ids inside two sequences and in padding
    columns past every row; the kernel must read those pages as zeros."""
    from kernels.common.attention_unified import run_unified_attention_torch

    from ._attention_device_array import DeviceArray

    problem, tuning = _tuning_spec(case, i64)
    q_lens, kv_lens = case.lens
    batch, hq, hkv, d = (
        len(q_lens),
        case.num_query_heads,
        case.num_kv_heads,
        case.head_size,
    )
    num_blocks = _num_blocks(case)
    rng = np.random.default_rng(0)
    q = _rounded(rng, (sum(q_lens), hq, d), case.dtype)
    k_cache = _rounded(rng, (num_blocks, case.block_size, hkv, d), case.dtype)
    v_cache = _rounded(rng, (num_blocks, case.block_size, hkv, d), case.dtype)
    width = max(-(-k // case.block_size) for k in kv_lens)
    pad_cols = 3 if bad_ids else 0
    table = np.full((batch + table_pad, width + pad_cols), 0x7FFFFFFF, np.int32)
    table[:batch, :width] = rng.permutation(num_blocks)[: batch * width].reshape(
        batch, width
    )
    ref_table, ref_k, ref_v = table, k_cache, v_cache
    if bad_ids:
        table[:batch, width::2] = _BAD_IDS[0]
        table[0, 5], table[1, 10] = _BAD_IDS
        # The reference reads every bad id as an extra all-zero page.
        zero = np.zeros((1, *k_cache.shape[1:]), np.float32)
        ref_k, ref_v = np.concatenate([k_cache, zero]), np.concatenate([v_cache, zero])
        ref_table = np.where(np.isin(table, _BAD_IDS), num_blocks, table)
    scale = d**-0.5

    # Every byte 0xFF is NaN in fp16 and bf16: a row the kernel skips fails.
    out_host = np.full(q.shape, 0xFFFF, dtype=np.uint16)
    if case.dtype == "fp16":
        out_host = out_host.view(np.float16)
    torch = None
    if case.lse:
        # The runner takes an LSE output only as a torch tensor, and then
        # every other buffer as one too.
        torch = pytest.importorskip("torch")

    def dev(host, rows=None):
        if torch is None:
            return DeviceArray(
                rt, host, shape=None if rows is None else (rows, *host.shape[1:])
            )
        t = torch.from_numpy(
            np.ascontiguousarray(host).view(
                np.int16 if host.dtype == np.uint16 else host.dtype
            )
        ).cuda()
        t = t.view(torch.bfloat16) if host.dtype == np.uint16 else t
        return t if rows is None else t[:rows]

    def read(buf):
        if torch is None:
            return buf.read()
        if buf.dtype == torch.bfloat16:
            return buf.view(torch.int16).cpu().numpy().view(np.uint16)
        return buf.cpu().numpy()

    out = dev(out_host)
    lse = (
        torch.full((sum(q_lens), hq, 1), float("nan"), device="cuda")
        if case.lse
        else None
    )
    run_unified_attention_torch(
        problem=problem,
        q=dev(_encode(q, case.dtype)),
        k=dev(_encode(k_cache, case.dtype)),
        v=dev(_encode(v_cache, case.dtype)),
        out=out,
        cu_seqlens_q=dev(np.cumsum([0, *q_lens], dtype=np.int32)),
        seqused_k=dev(np.array(kv_lens, dtype=np.int32)),
        softmax_scale=scale,
        block_table=dev(table, rows=batch),
        softcap=case.softcap,
        alibi_slopes=dev(_alibi_slopes(hq)) if case.alibi else None,
        backend="tiled",
        tuning_spec=tuning,
        lse=lse,
    )
    o_ref, lse_ref = _reference(case, q, ref_k, ref_v, ref_table, scale)
    got = _decode(read(out), case.dtype)
    # A misread page moves O by about its share of the row's softmax mass times
    # |V| (about 0.02 for one page of a 1000-token row), so the bound stays
    # under that; rtol covers the output rounding (half an ulp is |o| / 256 in
    # bf16) and P rounded to the K/V type before PV.
    atol, rtol = (4e-3, 2**-7) if case.dtype == "bf16" else (1e-3, 2**-9)
    np.testing.assert_allclose(got, o_ref, atol=atol, rtol=rtol)
    if case.lse:
        np.testing.assert_allclose(read(lse)[..., 0], lse_ref, atol=1e-3, rtol=0)


@pytest.mark.gpu
@_ADDRESSING
@_CASE_IDS
def test_matches_reference(rt, case_id, i64):
    _run_case(rt, _CASES[case_id], i64)


@pytest.mark.gpu
@_ADDRESSING
def test_block_table_tail_is_masked(rt, i64):
    """Ragged decode whose last tile runs past the end of the block table."""
    case = _CASES["decode_d128"]
    # The last 64-token tile covers pages 60..63; page 63 is past every row.
    assert -(-max(case.lens[1]) // case.block_size) % 4 != 0
    _run_case(rt, case, i64, table_pad=2)


@pytest.mark.gpu
@pytest.mark.parametrize("case_id", ["decode_d128", "prefill_d128"])
def test_i32_out_of_range_page_ids_read_zeros(rt, case_id):
    """i32 addressing: a page id at or past the 32-bit buffer bound (or
    negative), inside a sequence or in padding past it, reads as zeros through
    the buffer copies, as it did through the whole-cache resource, instead of
    faulting. The cases use buffer K copies: the opt-in global_load_lds K path
    builds its address from the page id and is not bounded. i64 has no bound to
    check against."""
    _run_case(rt, _CASES[case_id], False, bad_ids=True)
