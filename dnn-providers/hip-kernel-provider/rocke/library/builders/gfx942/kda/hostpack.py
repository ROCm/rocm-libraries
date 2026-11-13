# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Torch-free host pack and token-serial oracle for gfx942 fused KDA.

The torch builders in ``kda_chunk_fused.py`` stay the local GPU experiment
path. This module is the layout the external ``rocke.run_manifest`` runner
(and ``benchmark.remote_test``) bind: numpy only, bf16 stored as uint16
bits because stock numpy has no bfloat16 dtype.

Packed layout matches the fused kernel contract:

* ``q/k/g``: ``[BH * parts * NC, C * DK]``
* ``beta``: ``[BH * parts * NC, C]``
* ``v/o``: ``[BH * parts * NC, C * head_v]``
* ``h0/ht``: ``[BH * parts, head_v, DK]`` fp32 (state stored transposed)

``parts = logical_head_v // head_v``. Q/K/g/beta are duplicated across V
partitions; V is split. ``--shape`` is ``B,H,T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import numpy as np
except Exception as e:  # pragma: no cover - environment dependent
    raise RuntimeError("kda hostpack requires numpy") from e

# Match torch.nn.functional.normalize(..., eps=1e-12).
_L2_EPS = 1e-12


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round IEEE f32 to bf16 bits (round-to-nearest-even)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return ((u + rounding_bias) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(u: np.ndarray) -> np.ndarray:
    """Interpret bf16 bits as f32 (zero-extend the mantissa)."""
    u = np.ascontiguousarray(u, dtype=np.uint16)
    return (u.astype(np.uint32) << np.uint32(16)).view(np.float32)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, _L2_EPS)


def make_inputs(
    B: int,
    H: int,
    T: int,
    DK: int,
    DV: int,
    *,
    gate_low: float = -0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense ``[B,H,T,*]`` inputs. ``q/k/v`` are bf16 bits; ``g/beta`` are f32."""
    rng = np.random.default_rng(seed)
    q = _l2_normalize(rng.standard_normal((B, H, T, DK), dtype=np.float32))
    k = _l2_normalize(rng.standard_normal((B, H, T, DK), dtype=np.float32))
    v = rng.standard_normal((B, H, T, DV), dtype=np.float32) * np.float32(0.2)
    g = np.float32(gate_low) * rng.random((B, H, T, DK), dtype=np.float32)
    beta = rng.random((B, H, T), dtype=np.float32)
    return (
        f32_to_bf16_bits(q),
        f32_to_bf16_bits(k),
        f32_to_bf16_bits(v),
        np.ascontiguousarray(g, dtype=np.float32),
        np.ascontiguousarray(beta, dtype=np.float32),
    )


@dataclass
class PackedFused:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    g: np.ndarray
    beta: np.ndarray
    bh: int
    nc: int
    parts: int
    chunk: int
    head_k: int
    head_v: int
    logical_head_v: int


def pack_v_partitions(
    chunk: int,
    head_v: int,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
) -> PackedFused:
    """Pack dense ``[B,H,T,*]`` inputs into independent gfx942 V partitions."""
    B, H, T, DK = q.shape
    DV = int(v.shape[-1])
    if T % chunk:
        raise ValueError(f"T ({T}) must be divisible by chunk ({chunk})")
    if DV % head_v:
        raise ValueError(
            f"logical head_v ({DV}) must be divisible by partition "
            f"head_v ({head_v})"
        )
    BH, NC = B * H, T // chunk
    parts = DV // head_v

    # ``broadcast_to`` returns a read-only view, and ``ascontiguousarray`` does
    # not copy one whose layout is already contiguous -- which is exactly the
    # single-partition case (logical head_v == 64), where the broadcast adds a
    # length-1 axis and changes nothing. The pack would then hand back a
    # read-only array, and every consumer here uploads through
    # ``ctypes.from_buffer``, which requires a writable buffer. Copy explicitly
    # so the result does not depend on how many partitions there happen to be.
    def materialize(view: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        return np.array(view, copy=True).reshape(*shape)

    def duplicate(x: np.ndarray, tail: int) -> np.ndarray:
        x = np.ascontiguousarray(x).reshape(BH, NC, chunk, tail)
        x = np.broadcast_to(x[:, None, ...], (BH, parts, NC, chunk, tail))
        return materialize(x, (BH * parts * NC, chunk * tail))

    qf = duplicate(q, DK)
    kf = duplicate(k, DK)
    gf = duplicate(g, DK)
    bf = np.ascontiguousarray(beta).reshape(BH, NC, chunk)
    bf = np.broadcast_to(bf[:, None, ...], (BH, parts, NC, chunk))
    bf = materialize(bf, (BH * parts * NC, chunk))
    vf = np.ascontiguousarray(v).reshape(BH, NC, chunk, parts, head_v)
    # Same ownership guarantee as the duplicated fields: at one partition the
    # transpose is a no-op and would otherwise alias the caller's ``v``.
    vf = materialize(vf.transpose(0, 3, 1, 2, 4), (BH * parts * NC, chunk * head_v))
    return PackedFused(
        q=qf,
        k=kf,
        v=vf,
        g=gf,
        beta=bf,
        bh=BH,
        nc=NC,
        parts=parts,
        chunk=chunk,
        head_k=DK,
        head_v=head_v,
        logical_head_v=DV,
    )


def pack_initial_state(head_v: int, h0: np.ndarray, parts: int) -> np.ndarray:
    """``[B,H,DK,DV]`` fp32 -> ``[BH * parts, head_v, DK]`` (S stored as Sᵀ)."""
    B, H, DK, DV = h0.shape
    BH = B * H
    packed = np.ascontiguousarray(h0).reshape(BH, DK, parts, head_v)
    packed = np.ascontiguousarray(packed.transpose(0, 2, 3, 1))
    return packed.reshape(BH * parts, head_v, DK)


def unpack_outputs(
    *,
    chunk: int,
    head_k: int,
    head_v: int,
    o: np.ndarray,
    ht: np.ndarray,
    B: int,
    H: int,
    T: int,
    DV: int,
    NC: int,
    parts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Packed ``o`` / ``ht`` back to dense ``[B,H,T,DV]`` and ``[B,H,DK,DV]``."""
    BH = B * H
    out = np.ascontiguousarray(o).reshape(BH, parts, NC, chunk, head_v)
    out = np.ascontiguousarray(out.transpose(0, 2, 3, 1, 4)).reshape(B, H, T, DV)
    final = np.ascontiguousarray(ht).reshape(BH, parts, head_v, head_k)
    final = np.ascontiguousarray(final.transpose(0, 3, 1, 2)).reshape(B, H, head_k, DV)
    return out, final


def ref_token_serial(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    scale: float,
    h0: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Token-serial float64 oracle for the gated delta rule.

    Per token, with ``S`` the ``DK x DV`` state::

        S <- Diag(exp(g_t)) S
        u <- beta_t (v_t - k_t^T S)
        S <- S + k_t u^T
        o_t = scale * q_t^T S
    """
    B, H, T, DK = q.shape
    DV = int(v.shape[-1])
    qd = q.astype(np.float64, copy=False)
    kd = k.astype(np.float64, copy=False)
    vd = v.astype(np.float64, copy=False)
    gd = g.astype(np.float64, copy=False)
    bd = beta.astype(np.float64, copy=False)
    o = np.zeros((B, H, T, DV), dtype=np.float64)
    if h0 is None:
        S = np.zeros((B, H, DK, DV), dtype=np.float64)
    else:
        S = np.array(h0, dtype=np.float64, copy=True)
    for t in range(T):
        S = S * np.exp(gd[:, :, t, :])[..., None]
        kt = kd[:, :, t, :]
        kv = np.einsum("bhd,bhde->bhe", kt, S)
        u = bd[:, :, t, None] * (vd[:, :, t, :] - kv)
        S = S + kt[..., None] * u[:, :, None, :]
        o[:, :, t, :] = scale * np.einsum("bhd,bhde->bhe", qd[:, :, t, :], S)
    return o, S
