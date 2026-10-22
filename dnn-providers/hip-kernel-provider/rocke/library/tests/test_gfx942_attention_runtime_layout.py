# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 tiled-2D unified attention in ``runtime_layout`` mode.

``runtime_layout=True`` makes one compiled object serve any head count, sliding
window, DENSE or RAGGED query layout, Q/O strides and paged-KV layout (vLLM's
interleaved ``[nb, Hkv, bs, 2D]`` view, NHD, HND) through 16 appended kernargs.

The CPU tests check that every catalog spec passes ``supports_tiled_2d_spec``
and emits, and that incompatible specs are rejected. The GPU test launches the
catalog kernels against an fp32 numpy gather-then-dense reference: shuffled page
tables with garbage past each sequence's used blocks, empty sequences, padded
DENSE rows, RAGGED offsets with gaps, fused-QKV Q strides, padded O strides, a
sliding window, and grids smaller than the q-block count (grid-stride loop).
Rows of sequences without visible keys must be exactly 0 and every O element the
kernel must not write (gaps, padded rows, stride padding, guard regions) must
keep its sentinel.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import zlib

import numpy as np
import pytest

from kernels.gfx942.attention_tiled_2d import (
    RUNTIME_LAYOUT_EXTRA_ARGS,
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    runtime_layout_catalog_spec,
    supports_tiled_2d_spec,
)

_CATALOG = list(
    itertools.product(
        ("fp16", "bf16"), (64, 128, 256), (16, 32, 64), (1, 2, 4, 5, 6, 7, 8, 16)
    )
)


# ---------------------------------------------------------------------------
# CPU: catalog predicate + emit, rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,head_size,block_size,qpk", _CATALOG)
def test_catalog_spec_supported_and_emits(dtype, head_size, block_size, qpk):
    spec = runtime_layout_catalog_spec(
        dtype=dtype, head_size=head_size, block_size=block_size, num_queries_per_kv=qpk
    )
    ok, why = supports_tiled_2d_spec(spec, arch="gfx942")
    assert ok, why
    kernel = build_unified_attention_2d_tiled(spec, arch="gfx942")
    names = [p.name.lstrip("%") for p in kernel.params]
    assert names[18:] == [n for n, _ in RUNTIME_LAYOUT_EXTRA_ARGS]
    assert kernel.name.endswith("_rtl")


def _rtl_spec(**overrides):
    base = dict(
        head_size=128,
        block_size=32,
        num_query_heads=4,
        num_kv_heads=1,
        dtype="bf16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_warps=4,
        runtime_layout=True,
    )
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "overrides,needle",
    [
        (dict(use_sinks=True), "use_sinks"),
        (dict(has_softcap=True), "has_softcap"),
        (dict(use_alibi=True), "use_alibi"),
        (dict(use_qq_bias=True), "use_qq_bias"),
        (dict(sliding_window=128), "sliding_window"),
        (dict(use_fast_paged_kv_desc=True), "use_fast_paged_kv_desc"),
        (dict(use_q_major_grid=True), "use_q_major_grid"),
        (dict(block_m_per_warp=32), "block_m_per_warp"),
        (dict(tile_size=24), "tile_size"),
        (dict(tile_size=64), "tile_size"),
        (dict(num_query_heads=5, num_kv_heads=2), "multiple"),
    ],
)
def test_incompatible_spec_rejected(overrides, needle):
    with pytest.raises(ValueError, match=needle):
        UnifiedAttention2DTiledSpec(**_rtl_spec(**overrides))


def test_lds_budget_rejected_with_reason():
    spec = UnifiedAttention2DTiledSpec(**_rtl_spec(head_size=256, block_size=64))
    ok, why = supports_tiled_2d_spec(spec, arch="gfx942")
    assert not ok and "LDS" in why


def test_runtime_layout_off_keeps_name():
    plain = UnifiedAttention2DTiledSpec(**_rtl_spec(runtime_layout=False))
    assert not plain.kernel_name().endswith("rtl")
    rtl = dataclasses.replace(plain, runtime_layout=True)
    assert rtl.kernel_name() == plain.kernel_name() + "_rtl"


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _gpu_ready() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx942"
    except Exception:  # noqa: BLE001
        return False


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)

_NO_WINDOW = 0x3FFFFFFF
_HKV = 2
_GUARD = 1024  # elements of sentinel before and after the O view
_SENTINEL = -777.0
_GPU_KERNELS = [
    (dt, hd, bs, qpk)
    for dt in ("bf16", "fp16")
    for hd in (64, 128, 256)
    for bs in (16, 64)
    for qpk in (1, 4, 5, 8)
] + [
    ("bf16", 128, 32, 4),
    ("bf16", 128, 32, 8),
    ("fp16", 64, 32, 1),
    ("fp16", 64, 32, 5),
    ("bf16", 256, 32, 8),
]

_LAUNCHERS = {}


def _launcher(spec):
    from kernels.common.attention_unified import _attn_signature
    from rocke import compile_kernel
    from rocke.helpers.spec import SignatureBuilder
    from rocke.runtime import KernelLauncher

    key = spec.kernel_name()
    if key not in _LAUNCHERS:
        artifact = compile_kernel(
            build_unified_attention_2d_tiled(spec, arch="gfx942"),
            arch="gfx942",
            capture_ir_text=False,
        )
        sb = SignatureBuilder().extend(
            _attn_signature(
                spec.dtype, include_bt_stride=True, include_qq_bias_stride=True
            )
        )
        for name, ty in RUNTIME_LAYOUT_EXTRA_ARGS:
            if ty == "ptr<i32>":
                sb.ptr(name, "i32")
            else:
                sb.scalar(name, ty)
        _LAUNCHERS[key] = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=sb.build(),
            cache_key=("rtl_test", key),
        )
    return _LAUNCHERS[key]


# Scenario fields: mode (dense/ragged), seq_lens, q_lens (per-seq query rows),
# L (dense rows per seq), has_lq, gaps (ragged trailing gap per seq), kv layout,
# fused_q (Q is the Q slice of a fused QKV row), o_pad (O token-stride padding),
# window (None = off), grid_y (None = full grid).
def _scenarios(bs):
    return [
        dict(
            name="dense_L1_empty_seqs",
            mode="dense",
            L=1,
            has_lq=0,
            seq_lens=[0, 1, 17, 3 * bs + 5, 2 * bs, 70],
            q_lens=None,
            kv="nhd",
            fused_q=False,
            o_pad=0,
            window=None,
            grid_y=None,
        ),
        dict(
            name="dense_L3_padded_rows_vllm_window",
            mode="dense",
            L=3,
            has_lq=1,
            seq_lens=[40, 2, 0, 0, 3, bs + 1],
            q_lens=[3, 1, 0, 2, 3, 2],
            kv="vllm",
            fused_q=True,
            o_pad=8,
            window=5,
            grid_y=None,
        ),
        dict(
            name="dense_L3_gridstride_hnd",
            mode="dense",
            L=3,
            has_lq=0,
            seq_lens=[3, 50, 2 * bs + 7, 9],
            q_lens=None,
            kv="hnd",
            fused_q=False,
            o_pad=8,
            window=None,
            grid_y=1,
        ),
        dict(
            name="ragged_mixed_gaps_vllm",
            mode="ragged",
            has_lq=1,
            seq_lens=[33, 37 + 2 * bs + 3, 1, 0, 26, 64],
            q_lens=[1, 37, 1, 0, 19, 1],
            gaps=[2, 0, 5, 1, 3, 8],
            kv="vllm",
            fused_q=True,
            o_pad=8,
            window=None,
            grid_y=None,
        ),
        dict(
            name="ragged_offsets_window_hnd",
            mode="ragged",
            has_lq=0,
            seq_lens=[4, 90, 60, 1],
            q_lens=[4, 1, 50, 1],
            gaps=None,
            kv="hnd",
            fused_q=False,
            o_pad=0,
            window=16,
            grid_y=None,
        ),
        dict(
            name="ragged_gridstride_nhd",
            mode="ragged",
            has_lq=0,
            seq_lens=[70, 100, 2 * bs + 1],
            q_lens=[70, 33, 1],
            gaps=None,
            kv="nhd",
            fused_q=True,
            o_pad=8,
            window=None,
            grid_y=2,
        ),
    ]


def _reference(q_rows, k_logical, v_logical, table, seq_lens, q_lens, qpk, bs, window):
    """fp32 gather-then-dense attention. Returns {(seq, s): [Hq, D]}."""
    hq = q_rows[0].shape[1] if q_rows else 0
    out = {}
    for b, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        if ql == 0:
            continue
        blocks = [table[b, t // bs] for t in range(sl)]
        k = (
            np.stack([k_logical[blk, t % bs] for t, blk in enumerate(blocks)])
            if sl
            else None
        )
        v = (
            np.stack([v_logical[blk, t % bs] for t, blk in enumerate(blocks)])
            if sl
            else None
        )
        ctx = sl - ql
        for s in range(ql):
            d = ctx + s
            keys = np.arange(sl)
            vis = (keys <= d) & (d - keys < window)
            res = np.zeros((hq, q_rows[b].shape[2]), np.float32)
            if vis.any():
                for h in range(hq):
                    kv_h = h // qpk
                    sc = (k[vis, kv_h] @ q_rows[b][s, h]) / math.sqrt(
                        q_rows[b].shape[2]
                    )
                    p = np.exp(sc - sc.max())
                    res[h] = (p @ v[vis, kv_h]) / p.sum()
            out[(b, s)] = res
    return out


def _run_case(spec, sc, rng):
    import torch

    tdt = torch.bfloat16 if spec.dtype == "bf16" else torch.float16
    hd, bs, qpk = spec.head_size, spec.block_size, spec.num_queries_per_kv
    hq = qpk * _HKV
    nseq = len(sc["seq_lens"])
    seq_lens = list(sc["seq_lens"])
    bq = spec.block_q
    dense = sc["mode"] == "dense"

    # ---- query layout ----
    if dense:
        L = sc["L"]
        q_lens = list(sc["q_lens"]) if sc["has_lq"] else [L] * nseq
        q_lens_kernel = [min(L, x) for x in q_lens]
        tok_base = [b * L for b in range(nseq)]
        total_tok = nseq * L
    else:
        q_lens = list(sc["q_lens"])
        gaps = sc["gaps"] or [0] * nseq
        off = [0]
        for ql, g in zip(q_lens, gaps):
            off.append(off[-1] + ql + g)
        q_lens_kernel = q_lens
        tok_base = off[:-1]
        total_tok = off[-1]
    q_row = (hq + 2 * _HKV) * hd if sc["fused_q"] else hq * hd
    q_alloc = rng.standard_normal((max(total_tok, 1), q_row)).astype(np.float32)
    q_dev = torch.from_numpy(q_alloc).to(tdt).cuda()
    q_np = q_dev.float().cpu().numpy()
    q_rows = [
        q_np[tok_base[b] : tok_base[b] + q_lens_kernel[b], : hq * hd].reshape(
            -1, hq, hd
        )
        for b in range(nseq)
    ]

    # ---- paged KV ----
    used = [-(-sl // bs) for sl in seq_lens]
    nb = sum(used) + 3
    perm = rng.permutation(nb)
    max_blocks = max(used + [1])
    table = np.full((nseq, max_blocks + 1), 0x7FFFFFFF, np.int32)
    nxt = 0
    for b in range(nseq):
        for i in range(used[b]):
            table[b, i] = perm[nxt]
            nxt += 1
    kv_layout = sc["kv"]
    if kv_layout == "vllm":
        kv = (
            torch.from_numpy(
                rng.standard_normal((nb, _HKV, bs, 2 * hd)).astype(np.float32)
            )
            .to(tdt)
            .cuda()
        )
        kv_np = kv.float().cpu().numpy()
        k_logical = kv_np[..., :hd].transpose(0, 2, 1, 3)
        v_logical = kv_np[..., hd:].transpose(0, 2, 1, 3)
        k_ptr = kv.data_ptr()
        v_ptr = kv.data_ptr() + hd * kv.element_size()
        strides = (_HKV * bs * 2 * hd, 2 * hd, bs * 2 * hd)
        keep = (kv,)
    else:
        shape = (nb, bs, _HKV, hd) if kv_layout == "nhd" else (nb, _HKV, bs, hd)
        k = (
            torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
            .to(tdt)
            .cuda()
        )
        v = (
            torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
            .to(tdt)
            .cuda()
        )
        k_np, v_np = k.float().cpu().numpy(), v.float().cpu().numpy()
        if kv_layout == "nhd":
            k_logical, v_logical = k_np, v_np
            strides = (bs * _HKV * hd, _HKV * hd, hd)
        else:
            k_logical, v_logical = k_np.transpose(0, 2, 1, 3), v_np.transpose(
                0, 2, 1, 3
            )
            strides = (_HKV * bs * hd, hd, bs * hd)
        k_ptr, v_ptr = k.data_ptr(), v.data_ptr()
        keep = (k, v)
    kv_block_stride, kv_tok_stride, kv_head_stride = strides
    span_bytes = ((_HKV - 1) * kv_head_stride + (bs - 1) * kv_tok_stride + hd) * 2

    # ---- output with sentinel guards ----
    o_tok = hq * hd + sc["o_pad"]
    o_elems = max(total_tok, 1) * o_tok
    o_all = torch.full((_GUARD + o_elems + _GUARD,), _SENTINEL, dtype=tdt).cuda()
    o_ptr = o_all.data_ptr() + _GUARD * o_all.element_size()

    seq_lens_dev = torch.tensor(seq_lens, dtype=torch.int32).cuda()
    table_dev = torch.from_numpy(table).cuda()
    lq_dev = torch.tensor(
        q_lens_kernel if not dense else q_lens, dtype=torch.int32
    ).cuda()
    off_dev = torch.tensor(off, dtype=torch.int32).cuda() if not dense else seq_lens_dev

    if dense:
        bps = -(-sc["L"] // bq)
        full_grid = nseq * bps
        q_seq_stride, o_seq_stride = sc["L"] * q_row, sc["L"] * o_tok
    else:
        full_grid = off[-1] // bq + nseq
        q_seq_stride, o_seq_stride = q_row, o_tok
    grid_y = sc["grid_y"] or full_grid
    window = sc["window"]
    values = dict(
        output_ptr=o_ptr,
        query_ptr=q_dev.data_ptr(),
        key_cache_ptr=k_ptr,
        value_cache_ptr=v_ptr,
        sink_ptr=0,
        block_tables_ptr=table_dev.data_ptr(),
        seq_lens_ptr=seq_lens_dev.data_ptr(),
        alibi_slopes_ptr=0,
        qq_bias_ptr=0,
        query_start_len_ptr=off_dev.data_ptr(),
        scale=1.0 / math.sqrt(hd),
        k_scale=1.0,
        v_scale=1.0,
        out_scale=1.0,
        softcap=0.0,
        num_seqs=nseq,
        block_table_stride=table.shape[1],
        qq_bias_stride_0=0,
        seq_lens_q_ptr=(lq_dev if sc["has_lq"] else seq_lens_dev).data_ptr(),
        has_seq_lens_q=sc["has_lq"],
        q_uniform_len=sc["L"] if dense else 0,
        num_query_heads=hq,
        sliding_window=window if window is not None else _NO_WINDOW,
        q_seq_stride=q_seq_stride,
        q_tok_stride=q_row,
        q_head_stride=hd,
        o_seq_stride=o_seq_stride,
        o_tok_stride=o_tok,
        o_head_stride=hd,
        kv_block_stride=kv_block_stride,
        kv_tok_stride=kv_tok_stride,
        kv_head_stride=kv_head_stride,
        kv_block_span_bytes=span_bytes,
        q_block_grid=grid_y,
    )
    from rocke.runtime import LaunchConfig

    _launcher(spec)(
        values,
        config=LaunchConfig(grid=(_HKV, grid_y, 1), block=(spec.num_warps * 64, 1, 1)),
    )
    torch.cuda.synchronize()
    del keep

    got = o_all.float().cpu().numpy()
    raw = o_all.cpu()
    ref = _reference(
        q_rows,
        k_logical,
        v_logical,
        table,
        seq_lens,
        q_lens_kernel,
        qpk,
        bs,
        window if window is not None else _NO_WINDOW,
    )
    written = np.zeros(got.shape, bool)
    for (b, s), res in ref.items():
        base = _GUARD + (tok_base[b] + s) * o_tok
        for h in range(hq):
            sl = slice(base + h * hd, base + (h + 1) * hd)
            written[sl] = True
            out = got[sl]
            assert np.isfinite(
                out
            ).all(), f"{sc['name']}: non-finite seq {b} row {s} head {h}"
            if not np.any(res[h]):
                assert not np.any(
                    out
                ), f"{sc['name']}: seq {b} row {s} head {h} not zero"
            np.testing.assert_allclose(
                out,
                res[h],
                atol=2e-2,
                rtol=2e-2,
                err_msg=f"{sc['name']}: seq {b} row {s} head {h}",
            )
    sentinel = torch.tensor(_SENTINEL, dtype=tdt)
    untouched = (raw == sentinel).numpy()
    bad = np.flatnonzero(~written & ~untouched)
    assert bad.size == 0, (
        f"{sc['name']}: {bad.size} elements outside the written rows changed "
        f"(first at {bad[0] - _GUARD} relative to the O view)"
    )


@pytest.mark.gpu
@requires_gfx942_gpu
@pytest.mark.parametrize("dtype,head_size,block_size,qpk", _GPU_KERNELS)
def test_runtime_layout_numerics(dtype, head_size, block_size, qpk):
    spec = runtime_layout_catalog_spec(
        dtype=dtype, head_size=head_size, block_size=block_size, num_queries_per_kv=qpk
    )
    rng = np.random.default_rng(
        zlib.crc32(f"{dtype}:{head_size}:{block_size}:{qpk}".encode())
    )
    for sc in _scenarios(block_size):
        _run_case(spec, sc, rng)
