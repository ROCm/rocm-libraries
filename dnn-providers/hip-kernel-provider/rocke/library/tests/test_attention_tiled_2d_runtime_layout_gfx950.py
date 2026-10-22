# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the gfx950 tiled-2D unified attention in runtime-layout mode.

``runtime_layout=True`` takes head counts, the sliding window, every Q/O/KV
stride and a dense-or-ragged Q mapping as kernel arguments. Each catalog kernel
runs a set of scenarios against a numpy gather-then-dense reference:

- vLLM's interleaved KV view (one ``[nb, Hkv, bs, 2D]`` allocation, K is
  ``[..., :D]``, V is ``[..., D:]``) and plain NHD caches;
- shuffled page tables with ``0x7fffffff`` past each sequence's used blocks and
  NaN in every cache slot that is not a live token (so any load of a padded
  token or a stale table entry shows up as a NaN or a fault);
- dense Q (L=1, L=3 with padded rows) and ragged Q (mixed decode and chunked
  prefill rows, offset gaps with ``seq_lens_q``, exact offsets without);
- sliding window on and off, Q token strides of a fused QKV split, padded O
  token strides, and grids smaller than the q-block count (grid-stride loop).

Outputs must be finite and within tolerance; rows that see no key must be
exactly zero; every O element the kernel must not write (padding, gaps, guard
regions) must keep its sentinel. The device test is skipped unless the visible
device is gfx950; the catalog check needs no device.
"""

from __future__ import annotations

import itertools
import math
import zlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

_ARCH = None
try:
    from rocke.runtime.hip_module import get_device_arch

    _ARCH = get_device_arch(0)
except Exception:  # noqa: BLE001 - no runtime / no device
    _ARCH = None

_NEEDS_GFX950 = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs gfx950; device is {_ARCH}"
)

NO_WINDOW = 0x3FFFFFFF
GARBAGE_BLOCK = 0x7FFFFFFF
SENTINEL = 0xDEAD  # raw 16-bit pattern of every O element the kernel must not write
GUARD = 64  # elements of sentinel before and after the O view
NUM_KV_HEADS = 2
ATOL = 2e-2
RTOL = 2e-2


@dataclass(frozen=True)
class Scenario:
    name: str
    seq_lens: Tuple[int, ...]
    # DENSE when ``q_uniform_len > 0``: every sequence owns ``q_uniform_len`` Q
    # rows; ``q_lens`` (if given) is ``seq_lens_q``. RAGGED otherwise: ``q_lens``
    # are the query lengths and ``gaps`` the unused tokens after each sequence.
    q_uniform_len: int = 0
    q_lens: Optional[Tuple[int, ...]] = None
    gaps: Optional[Tuple[int, ...]] = None
    has_seq_lens_q: bool = False
    window: int = NO_WINDOW
    kv_layout: str = "vllm"  # "vllm" interleaved or "nhd"
    fused_qkv: bool = True  # Q token stride = (Hq + 2*Hkv) * D
    grid_y_div: int = 1  # launch grid.y = ceil(full grid.y / grid_y_div)


SCENARIOS = (
    # Decode batch; seq_len 0 rows still exist (no seq_lens_q) and must be 0.
    Scenario("dense_l1", seq_lens=(37, 0, 1, 130, 64), q_uniform_len=1),
    # Speculative-style L=3 with padded rows (seq_lens_q < L, one empty).
    Scenario(
        "dense_l3_padded",
        seq_lens=(50, 3, 0, 77),
        q_uniform_len=3,
        q_lens=(3, 2, 0, 1),
        has_seq_lens_q=True,
        window=21,
        kv_layout="nhd",
    ),
    Scenario(
        "dense_l3_grid_stride",
        seq_lens=(9, 40, 3, 100, 18, 66),
        q_uniform_len=3,
        kv_layout="nhd",
        fused_qkv=False,
        grid_y_div=3,
    ),
    # Mixed decode + chunked prefill, offset gaps, an empty sequence.
    Scenario(
        "ragged_mixed_gap",
        seq_lens=(33, 17, 1, 0, 140, 69),
        q_lens=(1, 17, 1, 0, 40, 5),
        gaps=(0, 3, 0, 2, 7, 2),
        has_seq_lens_q=True,
    ),
    Scenario(
        "ragged_exact_window",
        seq_lens=(6, 90, 50),
        q_lens=(6, 1, 23),
        window=16,
        kv_layout="nhd",
    ),
    Scenario(
        "ragged_grid_stride_window",
        seq_lens=(33, 17, 1, 140, 69),
        q_lens=(1, 17, 1, 40, 5),
        gaps=(0, 3, 0, 7, 2),
        has_seq_lens_q=True,
        window=29,
        grid_y_div=4,
    ),
)


def _kernel_cases():
    """(dtype, head_size, block_size, qpk) kernels exercised on the device."""
    cases = list(
        itertools.product(("bf16", "fp16"), (64, 128, 256), (16, 64), (1, 4, 5, 8))
    )
    cases += [(dt, hd, 32, 4) for dt in ("bf16", "fp16") for hd in (64, 128, 256)]
    return cases


_LAUNCHERS: dict = {}


def _launcher(dtype: str, head_size: int, block_size: int, qpk: int):
    key = (dtype, head_size, block_size, qpk)
    if key not in _LAUNCHERS:
        from kernels.gfx950.attention_tiled_2d import (
            build_unified_attention_2d_tiled,
            runtime_layout_catalog_spec,
            supports_tiled_2d_spec,
        )
        from rocke.helpers.compile import compile_kernel
        from rocke.helpers.spec import SignatureBuilder
        from rocke.runtime.launcher import KernelLauncher

        spec = runtime_layout_catalog_spec(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_queries_per_kv=qpk,
        )
        ok, why = supports_tiled_2d_spec(spec, arch="gfx950")
        assert ok, why
        art = compile_kernel(build_unified_attention_2d_tiled(spec), arch="gfx950")
        io = "f16" if dtype == "fp16" else "bf16"
        sb = SignatureBuilder()
        for name in ("output_ptr", "query_ptr", "key_cache_ptr", "value_cache_ptr"):
            sb.ptr(name, io)
        sb.ptr("sink_ptr", io)
        sb.ptr("block_tables_ptr", "i32")
        sb.ptr("seq_lens_ptr", "i32")
        sb.ptr("alibi_slopes_ptr", "f32")
        sb.ptr("qq_bias_ptr", "f32")
        sb.ptr("query_start_len_ptr", "i32")
        for name in ("scale", "k_scale", "v_scale", "out_scale", "softcap"):
            sb.scalar(name, "f32")
        for name in ("num_seqs", "block_table_stride", "qq_bias_stride_0"):
            sb.scalar(name, "i32")
        sb.ptr("seq_lens_q_ptr", "i32")
        for name in (
            "has_seq_lens_q",
            "q_uniform_len",
            "num_query_heads",
            "sliding_window",
            "q_seq_stride",
            "q_tok_stride",
            "q_head_stride",
            "o_seq_stride",
            "o_tok_stride",
            "o_head_stride",
            "kv_block_stride",
            "kv_tok_stride",
            "kv_head_stride",
            "kv_block_span_bytes",
            "q_block_grid",
        ):
            sb.scalar(name, "i32")
        launcher = KernelLauncher(
            hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sb.build()
        )
        _LAUNCHERS[key] = (spec, launcher)
    return _LAUNCHERS[key]


def _encode(a: np.ndarray, dtype: str) -> np.ndarray:
    """fp32 -> raw 16-bit storage of ``dtype``."""
    from rocke.numeric.bf16 import f32_to_bf16

    if dtype == "bf16":
        return f32_to_bf16(a.astype(np.float32))
    return a.astype(np.float16).view(np.uint16)


def _decode(a: np.ndarray, dtype: str) -> np.ndarray:
    from rocke.numeric.bf16 import bf16_to_f32

    if dtype == "bf16":
        return bf16_to_f32(a)
    return a.view(np.float16).astype(np.float32)


def _reference(
    q: np.ndarray,
    k: Sequence[np.ndarray],
    v: Sequence[np.ndarray],
    seq_len: int,
    q_len: int,
    qpk: int,
    window: int,
    scale: float,
) -> np.ndarray:
    """Bottom-right causal (+ window) attention of one sequence, fp32.

    ``q`` is ``[q_len, Hq, D]``; ``k``/``v`` are the gathered ``[seq_len, Hkv, D]``
    live tokens. Rows with no visible key are zero.
    """
    hq, d = q.shape[1], q.shape[2]
    out = np.zeros((q_len, hq, d), dtype=np.float32)
    if q_len == 0:
        return out
    diag = (seq_len - q_len) + np.arange(q_len)[:, None]
    key = np.arange(seq_len)[None, :]
    visible = (key <= diag) & (diag - key < window)
    for h in range(hq):
        kvh = h // qpk
        s = (q[:, h, :] @ k[:, kvh, :].T) * scale
        s = np.where(visible, s, -np.inf)
        row_max = np.max(s, axis=1, initial=-np.inf, where=visible)
        has_key = np.isfinite(row_max)
        p = np.where(visible, np.exp(s - np.where(has_key, row_max, 0.0)[:, None]), 0.0)
        denom = p.sum(axis=1)
        o = p @ v[:, kvh, :]
        out[:, h, :] = np.where(
            has_key[:, None], o / np.where(has_key, denom, 1.0)[:, None], 0.0
        )
    return out


def _run(dtype: str, head_size: int, block_size: int, qpk: int, sc: Scenario):
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import DeviceMem, LaunchConfig

    spec, launcher = _launcher(dtype, head_size, block_size, qpk)
    rng = np.random.default_rng(
        zlib.crc32(f"{dtype}/{head_size}/{block_size}/{qpk}/{sc.name}".encode())
    )
    D, BS = head_size, block_size
    hkv = NUM_KV_HEADS
    hq = hkv * qpk
    B = len(sc.seq_lens)
    seq_lens = np.array(sc.seq_lens, dtype=np.int32)
    dense = sc.q_uniform_len > 0
    if dense:
        L = sc.q_uniform_len
        slq = np.array(sc.q_lens if sc.q_lens else (L,) * B, dtype=np.int32)
        q_lens = np.minimum(slq, L) if sc.has_seq_lens_q else np.full(B, L, np.int32)
        rows_per_seq = np.full(B, L)
        starts = np.arange(B) * L
        offsets = None
    else:
        q_lens = np.array(sc.q_lens, dtype=np.int32)
        gaps = np.array(sc.gaps if sc.gaps else (0,) * B, dtype=np.int32)
        offsets = np.zeros(B + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(q_lens + gaps)
        slq = q_lens
        rows_per_seq = q_lens + gaps
        starts = offsets[:-1]
    total_rows = int(rows_per_seq.sum())

    # ---- Q (optionally a fused-QKV row) ----
    q_tok = (hq + 2 * hkv) * D if sc.fused_qkv else hq * D
    q_host_f = rng.standard_normal((max(total_rows, 1), q_tok)).astype(np.float32)
    q_host = _encode(q_host_f, dtype)
    q_view = _decode(q_host, dtype)[:, : hq * D].reshape(-1, hq, D)

    # ---- O with padded token stride and sentinel guards ----
    o_tok = hq * D + 8
    o_host = np.full(GUARD + max(total_rows, 1) * o_tok + GUARD, SENTINEL, np.uint16)

    # ---- paged KV cache: NaN everywhere except live tokens ----
    nblk = [(int(s) + BS - 1) // BS for s in seq_lens]
    num_blocks = sum(nblk) + 3
    perm = rng.permutation(num_blocks).astype(np.int32)
    bt_width = max(nblk) + 2
    block_tables = np.full((B, bt_width), GARBAGE_BLOCK, dtype=np.int32)
    nan_bits = _encode(np.array([np.nan], np.float32), dtype)[0]
    if sc.kv_layout == "vllm":
        cache = np.full((num_blocks, hkv, BS, 2 * D), nan_bits, np.uint16)
        kv_block_stride, kv_head_stride, kv_tok_stride = (
            hkv * BS * 2 * D,
            BS * 2 * D,
            2 * D,
        )
    else:
        k_cache = np.full((num_blocks, BS, hkv, D), nan_bits, np.uint16)
        v_cache = np.full((num_blocks, BS, hkv, D), nan_bits, np.uint16)
        kv_block_stride, kv_tok_stride, kv_head_stride = BS * hkv * D, hkv * D, D
    kv_span_bytes = ((hkv - 1) * kv_head_stride + (BS - 1) * kv_tok_stride + D) * 2
    gathered = []
    used = 0
    for b in range(B):
        n = nblk[b]
        blocks = perm[used : used + n]
        used += n
        block_tables[b, :n] = blocks
        sl = int(seq_lens[b])
        kf = _encode(rng.standard_normal((sl, hkv, D)).astype(np.float32), dtype)
        vf = _encode(rng.standard_normal((sl, hkv, D)).astype(np.float32), dtype)
        for t in range(sl):
            blk, tt = blocks[t // BS], t % BS
            if sc.kv_layout == "vllm":
                cache[blk, :, tt, :D] = kf[t]
                cache[blk, :, tt, D:] = vf[t]
            else:
                k_cache[blk, tt] = kf[t]
                v_cache[blk, tt] = vf[t]
        gathered.append((_decode(kf, dtype), _decode(vf, dtype)))

    # ---- launch ----
    bq = spec.block_q
    if dense:
        full_grid_y = B * ((sc.q_uniform_len + bq - 1) // bq)
    else:
        full_grid_y = int(offsets[-1]) // bq + B
    grid_y = max(1, -(-full_grid_y // sc.grid_y_div))
    scale = 1.0 / math.sqrt(D)

    rt = Runtime()
    bufs = []

    def upload(a: np.ndarray) -> DeviceMem:
        a = np.ascontiguousarray(a)
        m = DeviceMem(max(a.nbytes, 4))
        if a.nbytes:
            rt.memcpy_h2d(m.ptr(), as_u8_buffer(a), a.nbytes)
        bufs.append(m)
        return m

    q_dev = upload(q_host)
    o_dev = upload(o_host)
    seq_lens_dev = upload(seq_lens)
    bt_dev = upload(block_tables)
    slq_dev = upload(slq) if sc.has_seq_lens_q else seq_lens_dev
    # DENSE: a one-entry offsets buffer, so only index 0 is addressable.
    cu_dev = upload(offsets if offsets is not None else np.zeros(1, np.int32))
    if sc.kv_layout == "vllm":
        cache_dev = upload(cache)
        k_ptr, v_ptr = cache_dev.ptr(), cache_dev.ptr() + D * 2
    else:
        k_ptr, v_ptr = upload(k_cache).ptr(), upload(v_cache).ptr()
    values = {
        "output_ptr": o_dev.ptr() + GUARD * 2,
        "query_ptr": q_dev,
        "key_cache_ptr": k_ptr,
        "value_cache_ptr": v_ptr,
        "sink_ptr": 0,
        "block_tables_ptr": bt_dev,
        "seq_lens_ptr": seq_lens_dev,
        "alibi_slopes_ptr": 0,
        "qq_bias_ptr": 0,
        "query_start_len_ptr": cu_dev,
        "scale": scale,
        "k_scale": 1.0,
        "v_scale": 1.0,
        "out_scale": 1.0,
        "softcap": 0.0,
        "num_seqs": B,
        "block_table_stride": bt_width,
        "qq_bias_stride_0": 0,
        "seq_lens_q_ptr": slq_dev,
        "has_seq_lens_q": int(sc.has_seq_lens_q),
        "q_uniform_len": sc.q_uniform_len,
        "num_query_heads": hq,
        "sliding_window": sc.window,
        "q_seq_stride": sc.q_uniform_len * q_tok if dense else q_tok,
        "q_tok_stride": q_tok,
        "q_head_stride": D,
        "o_seq_stride": sc.q_uniform_len * o_tok if dense else o_tok,
        "o_tok_stride": o_tok,
        "o_head_stride": D,
        "kv_block_stride": kv_block_stride,
        "kv_tok_stride": kv_tok_stride,
        "kv_head_stride": kv_head_stride,
        "kv_block_span_bytes": kv_span_bytes,
        "q_block_grid": grid_y,
    }
    launcher(
        values,
        config=LaunchConfig(
            grid=(hkv, grid_y, 1), block=(64 * spec.num_warps, 1, 1), fence=True
        ),
    )
    rt.memcpy_d2h(as_u8_buffer(o_host), o_dev.ptr(), o_host.nbytes)

    # ---- check ----
    written = np.zeros(o_host.shape, dtype=bool)
    for b in range(B):
        ql = int(q_lens[b])
        if ql == 0:
            continue
        rows = int(starts[b]) + np.arange(ql)
        kf, vf = gathered[b]
        ref = _reference(
            q_view[rows], kf, vf, int(seq_lens[b]), ql, qpk, sc.window, scale
        )
        idx = (
            GUARD
            + rows[:, None, None] * o_tok
            + np.arange(hq)[None, :, None] * D
            + np.arange(D)[None, None, :]
        )
        written[idx] = True
        got = _decode(o_host[idx], dtype)
        where = f"{sc.name} seq {b} (seq_len {seq_lens[b]}, q_len {ql})"
        assert np.isfinite(got).all(), f"non-finite output in {where}"
        no_key = ~np.any(ref != 0.0, axis=2)
        assert (
            got[no_key] == 0.0
        ).all(), f"rows with no visible key not zero in {where}"
        err = np.abs(got - ref) - (ATOL + RTOL * np.abs(ref))
        assert (
            err.max() <= 0.0
        ), f"{where}: max_abs {np.abs(got - ref).max():.3e} exceeds tolerance"
    untouched = o_host[~written]
    assert (untouched == SENTINEL).all(), (
        f"{sc.name}: {(untouched != SENTINEL).sum()} O elements outside the live "
        "rows were written (padding, gap or guard)"
    )
    del bufs


@pytest.mark.gpu
@_NEEDS_GFX950
@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.parametrize(
    "dtype,head_size,block_size,qpk",
    _kernel_cases(),
    ids=lambda v: str(v),
)
def test_runtime_layout_matches_reference(dtype, head_size, block_size, qpk, sc):
    _run(dtype, head_size, block_size, qpk, sc)


def test_runtime_layout_catalog_supported_and_named():
    """Every catalog combination passes the support predicate and carries ``rtl``."""
    from kernels.gfx950.attention_tiled_2d import (
        runtime_layout_catalog_spec,
        supports_tiled_2d_spec,
    )

    for dtype, hd, bs, qpk in itertools.product(
        ("fp16", "bf16"), (64, 128, 256), (16, 32, 64), (1, 2, 4, 5, 6, 7, 8, 16)
    ):
        spec = runtime_layout_catalog_spec(
            dtype=dtype, head_size=hd, block_size=bs, num_queries_per_kv=qpk
        )
        ok, why = supports_tiled_2d_spec(spec, arch="gfx950")
        assert ok, (spec, why)
        assert spec.kernel_name().endswith("_rtl")
