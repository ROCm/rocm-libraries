# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host pack for gfx942 fused / split KDA: dense ``[B,H,T,*]`` <-> V partitions.

gfx942 cannot keep a full ``DV=128`` state mirror plus the C16 tile builder
under its 64 KiB LDS ceiling, so each workgroup owns ``head_v`` value channels
of one logical head. The builders in this package, the external
``rocke.run_manifest`` runner (``manifest.py``) and the gfx942 benchmark all
pack through here. numpy only; bf16 travels as ``uint16`` bits.

Packed layout matches the fused kernel contract:

* ``q/k/g``: ``[BH * parts * NC, C * DK]``
* ``beta``: ``[BH * parts * NC, C]``
* ``v/o``: ``[BH * parts * NC, C * head_v]``
* ``h0/ht``: ``[BH * parts, head_v, DK]`` fp32 (state stored transposed)

``parts = logical_head_v // head_v``. Q/K/g/beta are duplicated across V
partitions; V is split. Inputs and the float64 oracle come from
:mod:`builders.common.kda_host` and :mod:`rocke.numeric.gated_delta`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    def materialize(view: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Packed ``o`` / ``ht`` back to dense ``[B,H,T,DV]`` and ``[B,H,DK,DV]``."""
    BH = B * H
    out = np.ascontiguousarray(o).reshape(BH, parts, NC, chunk, head_v)
    out = np.ascontiguousarray(out.transpose(0, 2, 3, 1, 4)).reshape(B, H, T, DV)
    final = np.ascontiguousarray(ht).reshape(BH, parts, head_v, head_k)
    final = np.ascontiguousarray(final.transpose(0, 3, 1, 2)).reshape(B, H, head_k, DV)
    return out, final
