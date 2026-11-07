# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 tiled-2D unified attention in ``runtime_layout`` mode.

``runtime_layout=True`` makes one compiled object serve any head count, sliding
window, DENSE or RAGGED query layout, Q/O strides and paged-KV layout (vLLM's
interleaved ``[nb, Hkv, bs, 2D]`` view, NHD, HND): the head counts and window
are the body's runtime shape kernargs, the rest 15 appended kernargs.

The CPU tests check that every catalog spec passes ``supports_tiled_2d_spec``
and emits, and that the Python and native (C) checks reject incompatible specs.
The GPU test runs through the HIP runtime alone (no torch) and launches the
catalog kernels against an fp32 numpy gather-then-dense reference: shuffled page
tables with garbage past each sequence's used blocks, empty sequences, padded
DENSE rows, RAGGED offsets with gaps, fused-QKV Q strides, padded O strides, a
sliding window, grids smaller than the q-block count (grid-stride loop), and a
separate V block table that places V in other blocks than K.
Rows of sequences without visible keys, and padded DENSE rows past seq_lens_q,
must be exactly 0, and every O element the kernel must not write (RAGGED gap
rows past seq_lens_q, stride padding, guard regions) must keep its sentinel.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
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
    assert names[18:21] == list(UnifiedAttention2DTiledSpec.RUNTIME_KERNARG_FIELDS)
    assert names[21:] == [n for n, _ in RUNTIME_LAYOUT_EXTRA_ARGS]
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


_REJECTED = [
    (dict(has_softcap=True), "has_softcap"),
    (dict(use_alibi=True), "use_alibi"),
    (dict(use_qq_bias=True), "use_qq_bias"),
    (dict(sliding_window=128), "sliding_window"),
    (dict(use_fast_paged_kv_desc=True), "use_fast_paged_kv_desc"),
    (dict(use_q_major_grid=True), "use_q_major_grid"),
    (dict(block_m_per_warp=32), "block_m_per_warp"),
    (dict(tile_size=24), "tile_size"),
    (dict(tile_size=64), "tile_size"),
    (dict(num_query_heads=5, num_kv_heads=2), "divisible"),
    (dict(emit_lse=True), "emit_lse"),
]


@pytest.mark.parametrize("overrides,needle", _REJECTED)
def test_incompatible_spec_rejected(overrides, needle):
    with pytest.raises(ValueError, match=needle):
        UnifiedAttention2DTiledSpec(**_rtl_spec(**overrides))


@pytest.mark.parametrize("overrides,needle", _REJECTED)
def test_native_builder_rejects_incompatible_spec(overrides, needle):
    """The C twin of the reject check refuses the same specs, with the same cause."""
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError, match=needle):
        engine.gfx942_attention_tiled_2d_lower_llvm(_rtl_spec(**overrides), "gfx942")


def test_4warp_gqa_builder_refuses_runtime_layout():
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

    spec = UnifiedAttention2DTiledSpec(**_rtl_spec())
    with pytest.raises(NotImplementedError, match="runtime_layout"):
        build_gfx942_4warp_gqa(spec)


def test_sinks_accepted_with_runtime_layout():
    spec = UnifiedAttention2DTiledSpec(**_rtl_spec(use_sinks=True))
    ok, why = supports_tiled_2d_spec(spec, arch="gfx942")
    assert ok, why
    catalog = runtime_layout_catalog_spec(
        dtype="bf16", head_size=128, block_size=32, num_queries_per_kv=4, use_sinks=True
    )
    assert catalog.use_sinks
    assert "sinks" in catalog.kernel_name()
    assert catalog.kernel_name().endswith("_rtl")
    kernel = build_unified_attention_2d_tiled(catalog, arch="gfx942")
    names = [p.name.lstrip("%") for p in kernel.params]
    assert "sink_ptr" in names
    assert names[21:] == [n for n, _ in RUNTIME_LAYOUT_EXTRA_ARGS]


# D256, 64-token tile, 4 warps: 2*64*256*2 + 64*264*2 + 64*72*2 + 64*256*2
# = 141312 B, over gfx942's 65536 B.
_OVER_LDS = dict(head_size=256, block_size=64)
_OVER_LDS_NEEDLE = r"estimated LDS 141312 B .* exceeds the gfx942 65536 B LDS budget"


def test_lds_budget_rejected_with_reason():
    spec = UnifiedAttention2DTiledSpec(**_rtl_spec(**_OVER_LDS))
    ok, why = supports_tiled_2d_spec(spec, arch="gfx942")
    assert not ok
    assert re.search(_OVER_LDS_NEEDLE, why), why


def test_builder_rejects_lds_over_budget():
    spec = UnifiedAttention2DTiledSpec(**_rtl_spec(**_OVER_LDS))
    with pytest.raises(ValueError, match=_OVER_LDS_NEEDLE):
        build_unified_attention_2d_tiled(spec, arch="gfx942")


def test_native_builder_rejects_lds_over_budget():
    """The C builder charges the same LDS bytes against the same capacity."""
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError, match=_OVER_LDS_NEEDLE):
        engine.gfx942_attention_tiled_2d_lower_llvm(_rtl_spec(**_OVER_LDS), "gfx942")


def test_runtime_layout_off_keeps_name():
    plain = UnifiedAttention2DTiledSpec(**_rtl_spec(runtime_layout=False))
    assert not plain.kernel_name().endswith("rtl")
    rtl = dataclasses.replace(plain, runtime_layout=True)
    assert rtl.kernel_name() == plain.kernel_name() + "_rtl"


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _gpu_ready() -> bool:
    # Through the HIP runtime only, no torch: TheRock's test environment has none.
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch(0) == "gfx942"
    except Exception:  # noqa: BLE001 - no runtime / no device
        return False


requires_gfx942_gpu = pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx942 GPU")

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
    from kernels.common.attention_unified import _tiled_2d_signature
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
        sb = SignatureBuilder().extend(_tiled_2d_signature("gfx942", spec.dtype))
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
# L (dense rows per seq), has_lq, gaps (ragged trailing gap per seq), offsets
# (ragged off[0..B] as given: non-decreasing, off[B] the end; needs has_lq),
# kv layout, fused_q (Q is the Q slice of a fused QKV row), o_pad (O
# token-stride padding), window (None = off), grid_y (None = full grid).
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
        # K and V page through two tables: V's blocks are another permutation.
        dict(
            name="dense_L1_separate_v_table_vllm",
            mode="dense",
            L=1,
            has_lq=0,
            seq_lens=[0, 1, 2 * bs + 3, 90, bs],
            q_lens=None,
            kv="vllm",
            fused_q=False,
            o_pad=0,
            window=None,
            grid_y=None,
            v_table=True,
        ),
        dict(
            name="ragged_separate_v_table_window_hnd",
            mode="ragged",
            has_lq=1,
            seq_lens=[50, 3 * bs + 2, 1, 0],
            q_lens=[50, 4, 1, 0],
            gaps=[1, 0, 2, 0],
            kv="hnd",
            fused_q=True,
            o_pad=8,
            window=24,
            grid_y=None,
            v_table=True,
        ),
        # 0 < seq_len < q_len: the leading rows see no key (zeros), the rest see the
        # keys up to their own bottom-right diagonal.
        dict(
            name="dense_L5_keyless_leading_rows",
            mode="dense",
            L=5,
            has_lq=1,
            seq_lens=[2, 1, 40, 4],
            q_lens=[5, 5, 3, 5],
            kv="vllm",
            fused_q=True,
            o_pad=0,
            window=None,
            grid_y=None,
        ),
        dict(
            name="ragged_keyless_leading_rows_window",
            mode="ragged",
            has_lq=1,
            seq_lens=[2, 30, 1],
            q_lens=[5, 1, 4],
            gaps=[1, 0, 2],
            kv="nhd",
            fused_q=False,
            o_pad=8,
            window=16,
            grid_y=None,
        ),
        # One decode row whose origin lies past the first q-blocks: those blocks
        # find no sequence and write nothing (rows before off[0] keep the
        # sentinel).
        dict(
            name="ragged_shifted_first_origin",
            mode="ragged",
            has_lq=1,
            seq_lens=[20],
            q_lens=[1],
            offsets=[64, 65],
            kv="nhd",
            fused_q=False,
            o_pad=0,
            window=None,
            grid_y=None,
        ),
    ]


def _reference(
    q_rows,
    k_logical,
    v_logical,
    table,
    seq_lens,
    q_lens,
    qpk,
    bs,
    window,
    sinks=None,
    v_table=None,
):
    """fp32 gather-then-dense attention. Returns {(seq, s): [Hq, D]}.

    ``sinks`` ([Hq] logits, not scaled) join the softmax max and denominator;
    rows with no visible key stay zero. V pages through ``v_table`` when given,
    else through ``table``.
    """
    hq = q_rows[0].shape[1] if q_rows else 0
    v_table = table if v_table is None else v_table
    out = {}
    for b, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        if ql == 0:
            continue
        k = (
            np.stack([k_logical[table[b, t // bs], t % bs] for t in range(sl)])
            if sl
            else None
        )
        v = (
            np.stack([v_logical[v_table[b, t // bs], t % bs] for t in range(sl)])
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
                    m = sc.max() if sinks is None else max(sc.max(), sinks[h])
                    p = np.exp(sc - m)
                    denom = p.sum() + (0.0 if sinks is None else np.exp(sinks[h] - m))
                    res[h] = (p @ v[vis, kv_h]) / denom
            out[(b, s)] = res
    return out


def _encode(a, dtype):
    """fp32 -> raw 16-bit storage of ``dtype``."""
    from rocke.numeric.bf16 import f32_to_bf16

    if dtype == "bf16":
        return f32_to_bf16(np.ascontiguousarray(a, dtype=np.float32))
    return a.astype(np.float16).view(np.uint16)


def _decode(a, dtype):
    from rocke.numeric.bf16 import bf16_to_f32

    if dtype == "bf16":
        return bf16_to_f32(a)
    return a.view(np.float16).astype(np.float32)


def _run_case(spec, sc, rng, sink_f32=False):
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig

    dt = spec.dtype
    rt = Runtime()
    keep = []

    def upload(a):
        a = np.ascontiguousarray(a)
        m = DeviceMem(max(a.nbytes, 4))
        if a.nbytes:
            rt.memcpy_h2d(m.ptr(), as_u8_buffer(a), a.nbytes)
        keep.append(m)
        return m.ptr()

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
        if sc.get("offsets") is not None:
            assert sc["has_lq"], "explicit offsets need seq_lens_q"
            off = list(sc["offsets"])
            rows = q_lens
        else:
            gaps = sc["gaps"] or [0] * nseq
            off = [0]
            for ql, g in zip(q_lens, gaps):
                off.append(off[-1] + ql + g)
            rows = [off[b + 1] - off[b] for b in range(nseq)]
        q_lens_kernel = q_lens
        tok_base = off[:nseq]
        total_tok = max(t + r for t, r in zip(tok_base, rows))
    q_row = (hq + 2 * _HKV) * hd if sc["fused_q"] else hq * hd
    q_alloc = rng.standard_normal((max(total_tok, 1), q_row)).astype(np.float32)
    q_host = _encode(q_alloc, dt)
    q_ptr = upload(q_host)
    q_np = _decode(q_host, dt)
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
    v_table = table
    if sc.get("v_table"):
        # V's blocks: another permutation, so V lives in other blocks than K.
        v_perm = rng.permutation(nb)
        v_table = np.full_like(table, 0x7FFFFFFF)
        nxt = 0
        for b in range(nseq):
            for i in range(used[b]):
                v_table[b, i] = v_perm[nxt]
                nxt += 1
        assert not np.array_equal(v_table, table)
    kv_layout = sc["kv"]
    if kv_layout == "vllm":
        kv = _encode(rng.standard_normal((nb, _HKV, bs, 2 * hd)), dt)
        kv_np = _decode(kv, dt)
        k_logical = kv_np[..., :hd].transpose(0, 2, 1, 3)
        v_logical = kv_np[..., hd:].transpose(0, 2, 1, 3)
        k_ptr = upload(kv)
        v_ptr = k_ptr + hd * 2
        strides = (_HKV * bs * 2 * hd, 2 * hd, bs * 2 * hd)
    else:
        shape = (nb, bs, _HKV, hd) if kv_layout == "nhd" else (nb, _HKV, bs, hd)
        k = _encode(rng.standard_normal(shape), dt)
        v = _encode(rng.standard_normal(shape), dt)
        k_np, v_np = _decode(k, dt), _decode(v, dt)
        if kv_layout == "nhd":
            k_logical, v_logical = k_np, v_np
            strides = (bs * _HKV * hd, _HKV * hd, hd)
        else:
            k_logical, v_logical = k_np.transpose(0, 2, 1, 3), v_np.transpose(
                0, 2, 1, 3
            )
            strides = (_HKV * bs * hd, hd, bs * hd)
        k_ptr, v_ptr = upload(k), upload(v)
    kv_block_stride, kv_tok_stride, kv_head_stride = strides
    span_bytes = ((_HKV - 1) * kv_head_stride + (bs - 1) * kv_tok_stride + hd) * 2

    # ---- sinks (one per query head; fp32 when sink_f32, else I/O dtype) ----
    sinks_np = None
    sink_ptr = 0
    if spec.use_sinks:
        # Spread from "negligible" to "dominates the row" so a head that loses
        # (or reads another head's) sink fails the comparison.
        sinks_host = rng.uniform(-4.0, 6.0, hq).astype(np.float32)
        if sink_f32:
            sinks_np = sinks_host
            sink_ptr = upload(sinks_host)
        else:
            sinks_raw = _encode(sinks_host, dt)
            sinks_np = _decode(sinks_raw, dt)
            sink_ptr = upload(sinks_raw)

    # ---- output with sentinel guards ----
    o_tok = hq * hd + sc["o_pad"]
    o_elems = max(total_tok, 1) * o_tok
    sentinel = _encode(np.array([_SENTINEL], np.float32), dt)[0]
    o_host = np.full(_GUARD + o_elems + _GUARD, sentinel, np.uint16)
    o_base = upload(o_host)
    o_ptr = o_base + _GUARD * 2

    seq_lens_dev = upload(np.array(seq_lens, np.int32))
    table_dev = upload(table)
    v_table_dev = upload(v_table) if v_table is not table else table_dev
    lq_dev = upload(np.array(q_lens_kernel if not dense else q_lens, np.int32))
    off_dev = upload(np.array(off, np.int32)) if not dense else seq_lens_dev

    if dense:
        full_grid = nseq * -(-sc["L"] // bq)
    else:
        full_grid = off[-1] // bq + nseq
    if dense:
        q_seq_stride, o_seq_stride = sc["L"] * q_row, sc["L"] * o_tok
    else:
        q_seq_stride, o_seq_stride = q_row, o_tok
    grid_y = sc["grid_y"] or full_grid
    window = sc["window"]
    values = dict(
        output_ptr=o_ptr,
        query_ptr=q_ptr,
        key_cache_ptr=k_ptr,
        value_cache_ptr=v_ptr,
        sink_ptr=sink_ptr,
        block_tables_ptr=table_dev,
        seq_lens_ptr=seq_lens_dev,
        alibi_slopes_ptr=0,
        qq_bias_ptr=0,
        query_start_len_ptr=off_dev,
        scale=1.0 / math.sqrt(hd),
        k_scale=1.0,
        v_scale=1.0,
        out_scale=1.0,
        softcap=0.0,
        num_seqs=nseq,
        block_table_stride=table.shape[1],
        qq_bias_stride_0=0,
        seq_lens_q_ptr=lq_dev if sc["has_lq"] else seq_lens_dev,
        block_tables_v_ptr=v_table_dev,
        has_seq_lens_q=sc["has_lq"],
        q_uniform_len=sc["L"] if dense else 0,
        num_query_heads=hq,
        num_kv_heads=_HKV,
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
        sink_is_f32=int(sink_f32),
    )
    _launcher(spec)(
        values,
        config=LaunchConfig(
            grid=(_HKV, grid_y, 1), block=(spec.num_warps * 64, 1, 1), fence=True
        ),
    )
    rt.memcpy_d2h(as_u8_buffer(o_host), o_base, o_host.nbytes)
    del keep
    got = _decode(o_host, dt)
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
        sinks_np,
        v_table,
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
            # The kernel rounds P and O to bf16/fp16 (8 or 11 mantissa bits); the
            # reference stays in fp32. 2e-2 covers that rounding, not a looser kernel.
            np.testing.assert_allclose(
                out,
                res[h],
                atol=2e-2,
                rtol=2e-2,
                err_msg=f"{sc['name']}: seq {b} row {s} head {h}",
            )
    # A dense Q's rows past seq_lens_q are padding: they store zeros. A ragged
    # Q's gap rows (past seq_lens_q, up to the next offset) belong to no
    # sequence: the kernel must not write them, so they keep the sentinel.
    for b in range(nseq if dense else 0):
        for s in range(q_lens_kernel[b], sc["L"]):
            base = _GUARD + (tok_base[b] + s) * o_tok
            sl = slice(base, base + hq * hd)
            written[sl] = True
            assert not np.any(
                got[sl]
            ), f"{sc['name']}: padding row {s} of seq {b} not zero"
    untouched = o_host == sentinel
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


# Sink objects: _HKV=2 KV heads, so query heads of KV head 1 (h >= qpk) need
# the runtime query-head bound and mapping to find their sink.
_GPU_SINK_KERNELS = [
    ("bf16", 64, 16, 1),
    ("bf16", 128, 32, 4),
    ("fp16", 128, 64, 5),
    ("bf16", 256, 16, 8),
    ("fp16", 64, 32, 8),
]


@pytest.mark.gpu
@requires_gfx942_gpu
@pytest.mark.parametrize("sink_f32", (False, True), ids=("sink_qdtype", "sink_f32"))
@pytest.mark.parametrize("dtype,head_size,block_size,qpk", _GPU_SINK_KERNELS)
def test_runtime_layout_sinks_numerics(dtype, head_size, block_size, qpk, sink_f32):
    spec = runtime_layout_catalog_spec(
        dtype=dtype,
        head_size=head_size,
        block_size=block_size,
        num_queries_per_kv=qpk,
        use_sinks=True,
    )
    rng = np.random.default_rng(
        zlib.crc32(f"sinks:{dtype}:{head_size}:{block_size}:{qpk}".encode())
    )
    for sc in _scenarios(block_size):
        _run_case(spec, sc, rng, sink_f32=sink_f32)
