# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA prefill acceptance against the numpy oracle: scores, then output.

Two verification modes over the same ragged paged-KV problem generator:

``--mode scores`` (milestone 2, the original behaviour of this file)
    Runs :func:`kernels.mla.mla_prefill_gfx942.build_mla_prefill_score_probe`
    and compares the emitted score tile against
    :func:`builders.mla.ref_mla_attn.ref_mla_prefill_scores` called with
    ``causal=False``. No softmax, no PV -- the probe writes raw
    ``scale * q . k`` and nothing else, so that is all this checks.

``--mode full`` (milestone 3)
    Runs the full forward kernel and compares ``(out, softmax_lse)`` against
    :func:`builders.mla.ref_mla_attn.ref_mla_prefill` called with
    ``causal=True``. The mask is *not* optional here: the oracle's contract is
    that a fully masked query row (bottom-right causal with ``S_q > S_k``)
    yields ``out == 0`` and ``lse == -inf``, and those are asserted exactly
    rather than folded into a residual, which would either pass vacuously or
    go NaN.

Both modes drive the same scenario table (``--scenario``), which crosses four
shape families with two head counts. See :data:`SCENARIOS`.

Run::

    source .sandbox-env
    "$PY" -m builders.mla.verify_score_probe_gfx942                   # scores, all scenarios
    "$PY" -m builders.mla.verify_score_probe_gfx942 --mode full
    "$PY" -m builders.mla.verify_score_probe_gfx942 --scenario page_tail_h64
    "$PY" -m builders.mla.verify_score_probe_gfx942 --scenario adhoc --q-lens 5 7 --k-lens 20 19
    "$PY" -m builders.mla.verify_score_probe_gfx942 --mode full --dry-run   # geometry only, no GPU

Tolerance (fixed before the first run, not fitted to the result): the kernel
rounds two things to bf16 that the oracle keeps in fp32 -- the projected query
and the expanded ``K_nope = c_kv @ W_UK`` -- so a mismatch of a few bf16 ulps is
expected and is not a defect. bf16 carries 8 total mantissa bits, i.e. a relative
epsilon of ``2**-9 == 2.0e-3``; a 192-term dot product with sign cancellation
amplifies that against the tile peak. ``--tol 2e-2`` on
``max|got - ref| / max|ref|`` is ~10 ulps of headroom over that.

To separate "kernel is wrong" from "bf16 is coarse" the run also reports a
second residual against a numpy model that rounds Q and the expanded K to bf16
exactly as the kernel does. That one should land near fp32 noise; it is a
diagnostic, not the acceptance criterion.

In ``--mode full`` the same ``--tol`` bounds ``out``; ``out`` additionally lands
in bf16 storage, so the run reports a second residual against the bf16-rounded
oracle as the analogous diagnostic. ``lse`` is fp32 on both sides and is bounded
by ``--tol`` relative to ``max|lse_ref|`` over the live rows only.

.. warning::

   The full-forward kernel entry point is being written concurrently with this
   driver. Everything it needs is resolved by name at call time (see
   :func:`_resolve_fwd_entry`) so that ``--mode scores`` keeps working while
   ``build_mla_prefill_fwd`` does not yet exist, and the assumed argument order
   is spelled out in :func:`mla_prefill_fwd_args` -- one place to reconcile if
   the kernel lands with a different ABI.

Layering note: the ``make_problem`` helper below deliberately duplicates the one
in ``library/tests/test_mla_reference.py`` rather than importing it. ``builders``
sits below ``tests``, and importing upward is what
``library/tests/test_library_layering.py`` forbids.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import struct
import sys
from typing import NamedTuple, Sequence

import numpy as np

from builders.mla.ref_mla_attn import (
    DEFAULT_ROPE_LAYOUT,
    MlaGeometry,
    _expand_latent_kv,
    _project_query,
    bottom_right_causal_mask,
    gather_paged_kv,
    ref_mla_prefill,
    ref_mla_prefill_scores,
)
from kernels.mla import mla_prefill_gfx942 as _mla_mod
from kernels.mla.mla_prefill_gfx942 import (
    MlaPrefillSpec,
    build_mla_prefill_score_probe,
    mla_prefill_block,
    mla_prefill_score_probe_grid,
    supports_mla_prefill,
)
from rocke.helpers import compile_kernel
from rocke.runtime.hip_module import Runtime, get_device_arch

# --------------------------------------------------------------------------
# Scenario table
# --------------------------------------------------------------------------


class Scenario(NamedTuple):
    """One named (shape family, head count) case.

    ``shape`` is the family name; the full case name is ``f"{shape}_h{heads}"``,
    which is what ``--scenario`` takes.
    """

    shape: str
    heads: int
    q_lens: tuple
    k_lens: tuple
    why: str


# Four shape families. ``page_block_size`` is 16 (== block_k, enforced by
# MlaPrefillSpec.__post_init__), so "a page tail" means ``S_k % 16 != 0``.
_SHAPES = (
    (
        "square",
        (32, 32),
        (32, 32),
        "S_q == S_k, both page-aligned: the no-special-case baseline",
    ),
    (
        "sq_lt_sk",
        (8, 16),
        (32, 48),
        "S_q < S_k (chunked prefill), still page-aligned so the tail is not "
        "also under test",
    ),
    (
        "page_tail",
        (12, 20),
        (20, 37),
        "S_k % 16 != 0 (20 -> tail 4, 37 -> tail 5): the last page is partial "
        "and its garbage rows must not reach the accumulator",
    ),
    (
        "mixed_varlen",
        (5, 33, 24, 16),
        (20, 48, 19, 7),
        "4 sequences, every (S_q, S_k) pair distinct; seq0/seq2/seq3 are page "
        "tails; seq2 and seq3 have S_q > S_k, which is what produces fully "
        "masked query rows for the degenerate-row assert",
    ),
)

# H_q = 64 is a Kimi-K2 shaped model, 128 a DeepSeek-V3 shaped one.
_HEAD_COUNTS = (64, 128)

SCENARIOS: dict = {
    f"{shape}_h{heads}": Scenario(shape, heads, tuple(q), tuple(k), why)
    for (shape, q, k, why) in _SHAPES
    for heads in _HEAD_COUNTS
}


# --------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------


def _bf16_bits(a: np.ndarray) -> np.ndarray:
    """fp32 -> the raw ``uint16`` bf16 payload, round-to-nearest-even.

    Done by hand rather than via ``ml_dtypes``, which is not in the shared venv;
    bf16 is just fp32 truncated to its top 16 bits, so the only real work is the
    tie-break. Inputs here are finite by construction, so NaN payload
    preservation is not a concern.
    """
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16)).astype(np.uint16)


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round-trip through bf16 and back to fp32.

    Inputs are generated pre-rounded so that the kernel and the oracle read the
    *same* numbers; the only precision gap left is the one the kernel introduces
    internally, which is what the tolerance is actually budgeting for.
    """
    return (_bf16_bits(a).astype(np.uint32) << 16).view(np.float32)


def _bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    """Raw ``uint16`` bf16 payload -> fp32. The decode half of :func:`_bf16`.

    Used on the way *back* from the device, where ``out`` is stored as bf16 and
    numpy has no native bf16 to read it as.
    """
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


# bf16 quiet NaN. Seeded into the device ``out`` buffer before launch for the
# same reason ``scores`` is seeded with fp32 NaN: a slot the kernel never writes
# must fail loudly instead of reading back as a plausible zero -- and zero is
# exactly the value a fully masked row is *supposed* to have, so an unseeded
# buffer would make the degenerate-row assert pass vacuously.
_BF16_NAN_BITS = np.uint16(0x7FC0)


def _rope_tables(d_rope: int, max_pos: int, base: float = 10000.0):
    half = d_rope // 2
    inv = 1.0 / (base ** (np.arange(half, dtype=np.float64) / half))
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


class Problem(NamedTuple):
    geom: MlaGeometry
    q_latent: np.ndarray
    c_kv: np.ndarray
    k_rope: np.ndarray
    w_uq: np.ndarray
    w_uk: np.ndarray
    cu_seqlens_q: np.ndarray
    cu_seqlens_k: np.ndarray
    block_table: np.ndarray
    positions: np.ndarray
    cos_table: np.ndarray
    sin_table: np.ndarray
    scale: float

    def oracle_kwargs(self) -> dict:
        return {
            "q_latent": self.q_latent,
            "c_kv": self.c_kv,
            "k_rope": self.k_rope,
            "w_uq": self.w_uq,
            "w_uk": self.w_uk,
            "cu_seqlens_q": self.cu_seqlens_q,
            "cu_seqlens_k": self.cu_seqlens_k,
            "block_table": self.block_table,
            "positions": self.positions,
            "scale": self.scale,
            "cos_table": self.cos_table,
            "sin_table": self.sin_table,
        }


def make_problem(
    spec: MlaPrefillSpec,
    *,
    q_lens: Sequence[int] = (5, 7),
    k_lens: Sequence[int] = (20, 19),
    r_q: int = 1536,
    seed: int = 0,
) -> Problem:
    """A ragged paged-KV problem at the spec's real geometry.

    ``k_lens`` are deliberately not multiples of ``page_block_size`` so the
    final page of each sequence is partly garbage -- that is the case the
    kernel's store predicate has to drop.
    """
    if len(q_lens) != len(k_lens):
        raise ValueError(f"q_lens {q_lens} and k_lens {k_lens} must be the same length")

    rng = np.random.default_rng(seed)
    geom = MlaGeometry(
        num_heads=spec.num_heads,
        d_nope=spec.d_nope,
        d_rope=spec.d_rope,
        d_v=spec.d_v,
        r_kv=spec.r_kv,
        r_q=r_q,
    )
    page = spec.page_block_size
    batch = len(q_lens)
    total_q = int(sum(q_lens))

    max_blocks = max((k + page - 1) // page for k in k_lens)
    num_blocks = batch * max_blocks
    block_table = (
        rng.permutation(num_blocks).reshape(batch, max_blocks).astype(np.int32)
    )

    def rand(*shape):
        # 0.3 keeps the 192-term dot away from bf16's exponent extremes without
        # shrinking it to a regime where cancellation dominates.
        return _bf16(rng.standard_normal(shape).astype(np.float32) * 0.3)

    q_latent = rand(total_q, r_q)
    c_kv = rand(num_blocks, page, geom.r_kv)
    k_rope = rand(num_blocks, page, geom.d_rope)
    w_uq = rand(geom.num_heads, r_q, geom.head_dim_qk)
    w_uk = rand(geom.num_heads, geom.r_kv, geom.d_nope + geom.d_v)

    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    cu_k = np.concatenate([[0], np.cumsum(k_lens)]).astype(np.int32)

    positions = np.concatenate(
        [np.arange(max(k - q, 0), max(k - q, 0) + q) for q, k in zip(q_lens, k_lens)]
    ).astype(np.int32)
    cos_table, sin_table = _rope_tables(geom.d_rope, int(positions.max()) + 1)

    return Problem(
        geom=geom,
        q_latent=q_latent,
        c_kv=c_kv,
        k_rope=k_rope,
        w_uq=w_uq,
        w_uk=w_uk,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        block_table=block_table,
        positions=positions,
        cos_table=cos_table,
        sin_table=sin_table,
        scale=1.0 / math.sqrt(geom.head_dim_qk),
    )


def projected_query(p: Problem) -> np.ndarray:
    """``[total_q, H, head_dim_qk]`` fp32, laid out exactly as ``q_ptr`` wants.

    The probe takes the query *after* the W_UQ projection and RoPE, so the
    driver runs the oracle's own projection rather than a second copy of it.
    """
    q_nope, q_rope = _project_query(
        p.q_latent,
        p.w_uq,
        p.positions,
        p.cos_table,
        p.sin_table,
        p.geom,
        DEFAULT_ROPE_LAYOUT,
    )
    return np.concatenate([q_nope, q_rope], axis=-1).astype(np.float32)


def bf16_model(p: Problem, q: np.ndarray) -> list:
    """Diagnostic reference: the oracle's math with the kernel's bf16 roundings.

    Rounds the projected query and the expanded ``K_nope`` to bf16, which is
    where the two implementations legitimately part ways. Residual against this
    should be fp32 noise; anything larger is a real kernel defect that the
    coarser tolerance against the fp32 oracle might have let through.
    """
    out = []
    gathered = gather_paged_kv(p.c_kv, p.k_rope, p.block_table, p.cu_seqlens_k)
    q_bf = _bf16(q)
    for i, (latent, rope) in enumerate(gathered):
        lo, hi = int(p.cu_seqlens_q[i]), int(p.cu_seqlens_q[i + 1])
        k_nope, _ = _expand_latent_kv(latent, p.w_uk, p.geom)
        k_full = np.concatenate(
            [
                _bf16(k_nope),
                np.broadcast_to(
                    _bf16(rope)[:, None, :], k_nope.shape[:2] + (rope.shape[-1],)
                ),
            ],
            axis=-1,
        )
        out.append(
            p.scale * np.einsum("thd,shd->ths", q_bf[lo:hi], k_full, optimize=True)
        )
    return out


# --------------------------------------------------------------------------
# Device run
# --------------------------------------------------------------------------


def _u8(a: np.ndarray) -> ctypes.Array:
    a = np.ascontiguousarray(a)
    return (ctypes.c_uint8 * max(1, int(a.nbytes))).from_buffer_copy(
        a if a.nbytes else np.zeros(1, np.uint8)
    )


def _upload(rt: Runtime, items) -> dict:
    """``[(name, host_array)]`` -> ``{name: device_pointer}``, copied up."""
    bufs = {}
    for name, arr in items:
        arr = np.ascontiguousarray(arr)
        dev = rt.alloc(max(1, int(arr.nbytes)))
        rt.memcpy_h2d(dev, _u8(arr), int(arr.nbytes))
        bufs[name] = dev
    return bufs


def _download(rt: Runtime, dev, like: np.ndarray) -> np.ndarray:
    """Read ``dev`` back into a fresh array shaped and typed like ``like``."""
    nbytes = int(like.nbytes)
    raw = (ctypes.c_uint8 * nbytes)()
    rt.memcpy_d2h(raw, dev, nbytes)
    return np.frombuffer(bytes(raw), dtype=like.dtype).reshape(like.shape).copy()


def _launch_geometry(spec: MlaPrefillSpec, p: Problem) -> dict:
    """The packed-varlen counts both launches need, derived from the problem."""
    q_lens = np.diff(p.cu_seqlens_q.astype(np.int64))
    k_lens = np.diff(p.cu_seqlens_k.astype(np.int64))
    max_k = int(k_lens.max()) if k_lens.size else 0
    return {
        "total_q": int(p.cu_seqlens_q[-1]),
        "q_lens": q_lens,
        "k_lens": k_lens,
        "max_k": max_k,
        "num_k_tiles": (max_k + spec.block_k - 1) // spec.block_k,
        # NOT ``sum(ceil(S_q / Bq))``. Both kernels invert the shared
        # ``binary_search_seq_idx`` invariant ``cu_q[i] // Bq + i <= target``,
        # so sequence ``i`` owns blocks starting at ``cu_q[i] // Bq + i``. That
        # numbering leaves a gap whenever ``cu_q[i]`` is a multiple of ``Bq``
        # (e.g. q_lens=[32, 32], Bq=16: seq 1 starts at block 3, not 2), and a
        # sum-of-ceilings grid then never launches the tail blocks -- they come
        # back as unwritten NaN/0x7FC0 slots. ``total_q // Bq + n_seqs`` is the
        # matching upper bound; the extra blocks early-exit on ``qb_start >= S_q``.
        "num_q_blocks": int(p.cu_seqlens_q[-1]) // spec.block_q + int(len(q_lens)),
        "n_seqs": int(len(q_lens)),
        "bt_stride": int(p.block_table.shape[1]),
    }


def _load(spec: MlaPrefillSpec, kernel, *, arch: str):
    art = compile_kernel(kernel, arch=arch)
    rt = Runtime()
    blob = getattr(art, "hsaco", None) or art.hsaco_bytes
    return rt, rt.load_module(blob).get_function(art.kernel_name)


def _shared_inputs(p: Problem, q: np.ndarray) -> tuple:
    """The seven read-only buffers both entry points take, in ABI order."""
    return (
        ("q", _bf16_bits(q)),
        ("c_kv", _bf16_bits(p.c_kv)),
        ("k_rope", _bf16_bits(p.k_rope)),
        ("w_uk", _bf16_bits(p.w_uk)),
        ("cu_q", p.cu_seqlens_q.astype(np.int32)),
        ("cu_k", p.cu_seqlens_k.astype(np.int32)),
        ("bt", p.block_table.astype(np.int32)),
    )


def run_on_device(spec: MlaPrefillSpec, p: Problem, q: np.ndarray, *, arch: str):
    """Launch the probe and return the raw ``[total_q, H, max_k]`` fp32 buffer."""
    rt, fn = _load(spec, build_mla_prefill_score_probe(spec, arch=arch), arch=arch)
    g = _launch_geometry(spec, p)

    # The store predicate skips every out-of-range (query, key) pair, so those
    # slots keep whatever the buffer was seeded with. Seed with a value the
    # kernel can never produce, so a missing write is a loud failure rather than
    # a plausible zero.
    host_scores = np.full(
        (g["total_q"], spec.num_heads, g["max_k"]), np.nan, dtype=np.float32
    )

    bufs = _upload(rt, (("scores", host_scores),) + _shared_inputs(p, q))
    packed = struct.pack(
        "<" + "Q" * 8 + "f" + "i" * 3,
        bufs["scores"],
        bufs["q"],
        bufs["c_kv"],
        bufs["k_rope"],
        bufs["w_uk"],
        bufs["cu_q"],
        bufs["cu_k"],
        bufs["bt"],
        float(p.scale),
        g["n_seqs"],
        g["bt_stride"],
        g["max_k"],
    )
    grid = mla_prefill_score_probe_grid(
        spec, num_q_blocks=g["num_q_blocks"], num_k_tiles=g["num_k_tiles"]
    )
    rt.launch(fn, grid, mla_prefill_block(spec), packed)
    rt.sync()

    got = _download(rt, bufs["scores"], host_scores)
    for dev in bufs.values():
        rt.free(dev)
    return got


# --------------------------------------------------------------------------
# Full forward: kernel entry resolution and launch
# --------------------------------------------------------------------------

#: Name of the full-forward builder expected in ``kernels.mla.mla_prefill_gfx942``.
FWD_BUILDER = "build_mla_prefill_fwd"
#: Optional grid helper; :func:`_fwd_grid` falls back if it is absent.
FWD_GRID = "mla_prefill_fwd_grid"


def _resolve_fwd_entry():
    """Look up the full-forward builder by name, at call time.

    Deliberately not a module-level import: this driver's ``--mode scores`` path
    must keep running while the forward kernel is still being written, and an
    unresolvable name should read as "that milestone has not landed" rather than
    as an ImportError on the whole module.
    """
    build = getattr(_mla_mod, FWD_BUILDER, None)
    if build is None:
        raise NotImplementedError(
            f"kernels.mla.mla_prefill_gfx942.{FWD_BUILDER} does not exist yet, so "
            "--mode full has nothing to launch. Use --mode scores, or --dry-run to "
            "exercise the scenario table and the oracle without a device."
        )
    return build


def _fwd_grid(spec: MlaPrefillSpec, g: dict) -> tuple:
    """Grid for the full forward.

    Prefers a ``mla_prefill_fwd_grid`` exported by the kernel module. The
    fallback assumes the forward owns its whole key range (it must, to carry a
    running softmax), so unlike the score probe there is no k dimension in the
    grid -- one workgroup per (q block, head).
    """
    grid_fn = getattr(_mla_mod, FWD_GRID, None)
    if grid_fn is not None:
        return grid_fn(spec, num_q_blocks=g["num_q_blocks"])
    return (g["num_q_blocks"], spec.num_heads, 1)


def mla_prefill_fwd_args(bufs: dict, p: Problem, g: dict) -> bytes:
    """Pack the assumed full-forward argument block.

    ASSUMED ABI -- the single place to reconcile if the kernel lands different.
    It is the score probe's signature with ``scores_ptr`` replaced by
    ``out_ptr`` and a new ``lse_ptr`` inserted immediately after it, then the
    same four scalars::

        out_ptr   bf16 [total_q, H_q, d_v]
        lse_ptr   f32  [total_q, H_q]
        q_ptr, c_kv_ptr, k_rope_ptr, w_uk_ptr, cu_q_ptr, cu_k_ptr, bt_ptr
        scale f32, num_seqs i32, block_table_stride i32, max_k i32
    """
    return struct.pack(
        "<" + "Q" * 9 + "f" + "i" * 3,
        bufs["out"],
        bufs["lse"],
        bufs["q"],
        bufs["c_kv"],
        bufs["k_rope"],
        bufs["w_uk"],
        bufs["cu_q"],
        bufs["cu_k"],
        bufs["bt"],
        float(p.scale),
        g["n_seqs"],
        g["bt_stride"],
        g["max_k"],
    )


def run_full_on_device(spec: MlaPrefillSpec, p: Problem, q: np.ndarray, *, arch: str):
    """Launch the full forward and return ``(out fp32, lse fp32)``.

    ``out`` comes back decoded from its bf16 storage into fp32 so the caller can
    compare it against the fp32 oracle without a second dtype convention.
    """
    build = _resolve_fwd_entry()
    rt, fn = _load(spec, build(spec, arch=arch), arch=arch)
    g = _launch_geometry(spec, p)

    # Both outputs are NaN-seeded for the score probe's reason, but it matters
    # more here: a fully masked row's correct answer is 0 / -inf, so a buffer
    # left at its allocation value could satisfy the degenerate-row assert
    # without the kernel having written anything at all.
    host_out_bits = np.full(
        (g["total_q"], spec.num_heads, spec.d_v), _BF16_NAN_BITS, dtype=np.uint16
    )
    host_lse = np.full((g["total_q"], spec.num_heads), np.nan, dtype=np.float32)

    bufs = _upload(
        rt, (("out", host_out_bits), ("lse", host_lse)) + _shared_inputs(p, q)
    )
    rt.launch(
        fn,
        _fwd_grid(spec, g),
        mla_prefill_block(spec),
        mla_prefill_fwd_args(bufs, p, g),
    )
    rt.sync()

    out = _bf16_from_bits(_download(rt, bufs["out"], host_out_bits))
    lse = _download(rt, bufs["lse"], host_lse)
    for dev in bufs.values():
        rt.free(dev)
    return out, lse


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _residual(got: np.ndarray, ref: np.ndarray) -> tuple:
    peak = float(np.max(np.abs(ref))) if ref.size else 0.0
    denom = peak if peak > 0 else 1.0
    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
    return err, err / denom, peak


def verify(spec: MlaPrefillSpec, p: Problem, *, arch: str, tol: float) -> bool:
    q = projected_query(p)
    got = run_on_device(spec, p, q, arch=arch)
    ref = ref_mla_prefill_scores(**p.oracle_kwargs(), causal=False, geometry=p.geom)
    model = bf16_model(p, q)

    ok = True
    for i in range(len(ref)):
        lo, hi = int(p.cu_seqlens_q[i]), int(p.cu_seqlens_q[i + 1])
        s_k = int(p.cu_seqlens_k[i + 1] - p.cu_seqlens_k[i])
        # got is [token, head, k]; the oracle returns [S_q, H, S_k] per sequence.
        tile = got[lo:hi, :, :s_k]

        if not np.all(np.isfinite(tile)):
            n = int(np.count_nonzero(~np.isfinite(tile)))
            print(f"  seq {i}: FAIL -- {n}/{tile.size} in-range slots never written")
            ok = False
            continue

        abs_o, rel_o, peak = _residual(tile, ref[i])
        _, rel_m, _ = _residual(tile, model[i])
        verdict = "ok" if rel_o <= tol else "FAIL"
        if rel_o > tol:
            ok = False
        print(
            f"  seq {i}: S_q={hi - lo} S_k={s_k} peak={peak:.4f} "
            f"max_abs={abs_o:.3e} rel_vs_oracle={rel_o:.3e} "
            f"rel_vs_bf16_model={rel_m:.3e}  {verdict}"
        )
    return ok


def _check_degenerate_rows(
    got_out: np.ndarray, got_lse: np.ndarray, dead: np.ndarray, label: str
) -> bool:
    """Assert the oracle's fully-masked-row contract exactly.

    ``dead`` is the ``[S_q, H]`` boolean of rows the bottom-right causal mask
    leaves with no visible key. For those the contract is ``out == 0`` and
    ``lse == -inf`` -- checked as exact equality and :func:`np.isneginf`, not as
    a relative residual, which against a zero/-inf reference would either pass
    for any small garbage or evaluate to NaN and be silently dropped by the
    comparison.
    """
    if not dead.any():
        return True
    ok = True
    bad_lse = int(np.count_nonzero(~np.isneginf(got_lse[dead])))
    if bad_lse:
        sample = got_lse[dead][~np.isneginf(got_lse[dead])][:4]
        print(
            f"  {label}: FAIL -- {bad_lse}/{int(dead.sum())} fully masked rows have "
            f"lse != -inf, e.g. {list(map(float, sample))}"
        )
        ok = False
    bad_out = int(np.count_nonzero(got_out[dead] != 0.0))
    if bad_out:
        vals = got_out[dead][got_out[dead] != 0.0][:4]
        print(
            f"  {label}: FAIL -- {bad_out}/{got_out[dead].size} out elements on "
            f"fully masked rows are nonzero, e.g. {list(map(float, vals))}"
        )
        ok = False
    return ok


def verify_full(spec: MlaPrefillSpec, p: Problem, *, arch: str, tol: float) -> bool:
    """Compare ``(out, lse)`` against :func:`ref_mla_prefill` with ``causal=True``.

    The mask is on here, unlike the score mode, so this is the first check that
    covers the causal geometry and the fully masked rows it produces.
    """
    q = projected_query(p)
    got_out, got_lse = run_full_on_device(spec, p, q, arch=arch)
    ref_out, ref_lse = ref_mla_prefill(
        **p.oracle_kwargs(), causal=True, geometry=p.geom
    )

    ok = True
    for i in range(len(p.cu_seqlens_q) - 1):
        lo, hi = int(p.cu_seqlens_q[i]), int(p.cu_seqlens_q[i + 1])
        s_q, s_k = hi - lo, int(p.cu_seqlens_k[i + 1] - p.cu_seqlens_k[i])
        label = f"seq {i}"

        # Derive `dead` from the mask rather than from ref_lse, so a broken
        # oracle cannot make the degenerate check vacuous.
        dead_row = ~bottom_right_causal_mask(s_q, s_k).any(axis=-1)  # [S_q]
        dead = np.broadcast_to(dead_row[:, None], (s_q, spec.num_heads))
        alive = ~dead

        o_got, o_ref = got_out[lo:hi], ref_out[lo:hi]
        l_got, l_ref = got_lse[lo:hi], ref_lse[lo:hi]

        if not _check_degenerate_rows(o_got, l_got, dead, label):
            ok = False

        if not alive.any():
            print(f"  {label}: S_q={s_q} S_k={s_k} every row fully masked  ok")
            continue

        live_out_got, live_out_ref = o_got[alive], o_ref[alive]
        if not np.all(np.isfinite(live_out_got)) or not np.all(
            np.isfinite(l_got[alive])
        ):
            n = int(np.count_nonzero(~np.isfinite(live_out_got))) + int(
                np.count_nonzero(~np.isfinite(l_got[alive]))
            )
            print(f"  {label}: FAIL -- {n} live out/lse slots never written")
            ok = False
            continue

        abs_o, rel_o, peak = _residual(live_out_got, live_out_ref)
        _, rel_bf, _ = _residual(live_out_got, _bf16(live_out_ref))
        abs_l, rel_l, peak_l = _residual(l_got[alive], l_ref[alive])
        verdict = "ok" if (rel_o <= tol and rel_l <= tol) else "FAIL"
        if verdict == "FAIL":
            ok = False
        print(
            f"  {label}: S_q={s_q} S_k={s_k} dead_rows={int(dead_row.sum())} "
            f"out[peak={peak:.4f} max_abs={abs_o:.3e} rel={rel_o:.3e} "
            f"rel_vs_bf16_ref={rel_bf:.3e}] "
            f"lse[peak={peak_l:.4f} max_abs={abs_l:.3e} rel={rel_l:.3e}]  {verdict}"
        )
    return ok


# --------------------------------------------------------------------------
# Scenario driving
# --------------------------------------------------------------------------


def describe_problem(spec: MlaPrefillSpec, p: Problem) -> str:
    """One multi-line geometry dump, used by ``--dry-run``.

    Reports exactly the properties the scenario names claim -- per-sequence
    ``S_q``/``S_k``, the page-tail remainder, the pages each sequence needs, the
    block-table shape, and how many query rows the causal mask leaves dead.
    """
    page = spec.page_block_size
    rows = []
    for i in range(len(p.cu_seqlens_q) - 1):
        s_q = int(p.cu_seqlens_q[i + 1] - p.cu_seqlens_q[i])
        s_k = int(p.cu_seqlens_k[i + 1] - p.cu_seqlens_k[i])
        tail = s_k % page
        dead = int((~bottom_right_causal_mask(s_q, s_k).any(axis=-1)).sum())
        rows.append(
            f"    seq{i}: S_q={s_q:>3} S_k={s_k:>3} "
            f"pages={(s_k + page - 1) // page} tail={tail if tail else 0}"
            f"{' (PARTIAL last page)' if tail else ''} "
            f"{'S_q>S_k' if s_q > s_k else ('S_q<S_k' if s_q < s_k else 'S_q==S_k')} "
            f"dead_rows={dead}"
        )
    g = _launch_geometry(spec, p)
    rows.append(
        f"    block_table={tuple(p.block_table.shape)} total_q={g['total_q']} "
        f"max_k={g['max_k']} num_q_blocks={g['num_q_blocks']} "
        f"num_k_tiles={g['num_k_tiles']}"
    )
    return "\n".join(rows)


def run_case(
    name: str,
    heads: int,
    q_lens,
    k_lens,
    *,
    arch: str,
    mode: str,
    seed: int,
    tol: float,
    dry_run: bool,
) -> bool:
    """One (shape, heads) case end to end. Returns True on pass."""
    spec = MlaPrefillSpec(num_heads=heads)
    # --dry-run reports problem geometry, which is arch-independent, so it stays
    # runnable on a host with no device.
    if not dry_run:
        ok, reason = supports_mla_prefill(spec, arch=arch)
        if not ok:
            print(f"[{name}] unsupported: {reason}")
            return False

    p = make_problem(spec, q_lens=q_lens, k_lens=k_lens, seed=seed)
    print(
        f"[{name}] heads={heads} q_lens={list(q_lens)} k_lens={list(k_lens)} "
        f"mode={mode} scale={p.scale:.6f} tol={tol:g}"
    )
    if dry_run:
        print(describe_problem(spec, p))
        return True
    if mode == "scores":
        passed = verify(spec, p, arch=arch, tol=tol)
    else:
        passed = verify_full(spec, p, arch=arch, tol=tol)
    print(f"[{name}] {'PASS' if passed else 'FAIL'}")
    return passed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--mode",
        choices=("scores", "full"),
        default="scores",
        help="scores: raw QK tile vs ref_mla_prefill_scores(causal=False). "
        "full: (out, lse) vs ref_mla_prefill(causal=True). Default: scores.",
    )
    ap.add_argument(
        "--scenario",
        default="all",
        choices=("all", "adhoc") + tuple(SCENARIOS),
        help="'all' (default) runs the whole table in this one process; "
        "'adhoc' uses --heads/--q-lens/--k-lens instead.",
    )
    ap.add_argument("--heads", type=int, default=4, help="--scenario adhoc only")
    ap.add_argument(
        "--q-lens", type=int, nargs="+", default=[5, 7], help="--scenario adhoc only"
    )
    ap.add_argument(
        "--k-lens", type=int, nargs="+", default=[20, 19], help="--scenario adhoc only"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-2, help="relative-to-peak bound")
    ap.add_argument("--arch", default=None, help="default: the local device")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print each case's geometry and stop; no compile, no device",
    )
    ap.add_argument(
        "--list-scenarios", action="store_true", help="print the table and exit"
    )
    args = ap.parse_args(argv)

    if args.list_scenarios:
        for name, sc in SCENARIOS.items():
            print(
                f"{name:<20} heads={sc.heads:<4} q_lens={list(sc.q_lens)} "
                f"k_lens={list(sc.k_lens)}\n    {sc.why}"
            )
        return 0

    arch = args.arch or get_device_arch() or ""
    if not args.dry_run:
        # Arch admission does not depend on the scenario's head count, so one
        # check up front keeps "this box cannot run MLA prefill" a skip (2)
        # rather than N case failures (1).
        ok, reason = supports_mla_prefill(MlaPrefillSpec(num_heads=64), arch=arch)
        if not ok:
            print(f"unsupported: {reason}")
            return 2

    if args.scenario == "adhoc":
        cases = [("adhoc", args.heads, tuple(args.q_lens), tuple(args.k_lens))]
    elif args.scenario == "all":
        cases = [(n, s.heads, s.q_lens, s.k_lens) for n, s in SCENARIOS.items()]
    else:
        s = SCENARIOS[args.scenario]
        cases = [(args.scenario, s.heads, s.q_lens, s.k_lens)]

    print(f"arch={arch} mode={args.mode} scenario={args.scenario} cases={len(cases)}")
    results = []
    for name, heads, q_lens, k_lens in cases:
        results.append(
            (
                name,
                run_case(
                    name,
                    heads,
                    q_lens,
                    k_lens,
                    arch=arch,
                    mode=args.mode,
                    seed=args.seed,
                    tol=args.tol,
                    dry_run=args.dry_run,
                ),
            )
        )

    failed = [n for n, passed in results if not passed]
    print(
        f"SUMMARY mode={args.mode} {len(results) - len(failed)}/{len(results)} passed"
        + (f"; failed: {', '.join(failed)}" if failed else "")
    )
    print("PASS" if not failed else "FAIL")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
