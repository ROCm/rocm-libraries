# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``run_manifest`` adapter for gfx942 fused KDA (numpy pack + oracle).

Importing this module registers kind ``kda_chunk_fused_bf16``. The fused
example writes ``runner_module`` into the manifest so
``python -m rocke.run_manifest`` can import this file on a cluster node
that has ``library/`` on ``PYTHONPATH`` — no platform edit, no torch.

``--shape`` is ``B,H,T``. Chunk, ``DK``, partition ``DV``, and logical
``DV`` live on the manifest because the remote CLI only forwards three
ints.

The torch builder GPU tests in ``kda_chunk_fused.py`` are a separate
experimentation lane and are not used here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rocke.helpers.manifest import make_simple_op_manifest
from rocke.runtime.host_buffers import as_u8_buffer, nbytes, require_numpy
from rocke.runtime.hip_module import Runtime
from rocke.runtime.packing import pack_args
from rocke.run_manifest import register_manifest_runner

from .hostpack import (
    PackedFused,
    bf16_bits_to_f32,
    make_inputs,
    pack_initial_state,
    pack_v_partitions,
    ref_token_serial,
    unpack_outputs,
)

KIND = "kda_chunk_fused_bf16"
RUNNER_MODULE = "builders.gfx942.kda.manifest"

_DEFAULT_SHAPE = (2, 4, 256)
_DEFAULT_TOL = 3e-2


def _with_size_bytes(signature: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for arg in signature:
        item = dict(arg)
        ty = str(item["type"])
        if ty.startswith("ptr"):
            item.setdefault("size_bytes", 8)
        elif ty in ("i32", "f32"):
            item.setdefault("size_bytes", 4)
        else:
            item.setdefault("size_bytes", 8)
        out.append(item)
    return out


def make_kda_fused_manifest(
    *,
    artifact,
    spec,
    args_signature: Sequence[Mapping[str, Any]],
    logical_head_v: int,
    default_shape: Sequence[int] = _DEFAULT_SHAPE,
    warmup_iters: int = 5,
    timed_iters: int = 100,
    notes: str = "",
) -> Dict[str, Any]:
    """Emit a v1 manifest the remote ``run_manifest`` path can execute."""
    B, H, T = (int(x) for x in default_shape)
    note = notes or (
        "gfx942 fused KDA prefill. --shape is B,H,T; pack/check live in "
        f"{RUNNER_MODULE}."
    )
    return make_simple_op_manifest(
        artifact=artifact,
        kind=KIND,
        op="kda_chunk_fused",
        dtype=str(getattr(spec, "dtype", "bf16")),
        threads_per_block=int(spec.tile.block_size),
        default_shape=(B, H, T),
        args_signature=_with_size_bytes(args_signature),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        notes=note,
        extra={
            "runner_module": RUNNER_MODULE,
            "chunk": int(spec.tile.chunk),
            "head_k": int(spec.head_k),
            "head_v": int(spec.head_v),
            "logical_head_v": int(logical_head_v),
            "has_initial_state": bool(spec.has_initial_state),
            "scale": float(spec.head_k) ** -0.5,
            "gate_low": -0.5,
            "verify_tol": _DEFAULT_TOL,
        },
    )


def _bht(manifest: dict, shape: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if shape is None or shape == (0, 0, 0):
        ds = manifest.get("default_shape", list(_DEFAULT_SHAPE))
        if len(ds) != 3:
            raise ValueError(f"{KIND} default_shape must be [B, H, T], got {ds!r}")
        return int(ds[0]), int(ds[1]), int(ds[2])
    return int(shape[0]), int(shape[1]), int(shape[2])


def run_kda_fused_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Problem builder: ``(make_args, grid, block, flop, bytes_xfer, check)``."""
    np = require_numpy()
    B, H, T = _bht(manifest, shape)
    chunk = int(manifest["chunk"])
    head_k = int(manifest["head_k"])
    head_v = int(manifest["head_v"])
    logical_head_v = int(manifest["logical_head_v"])
    has_h0 = bool(manifest.get("has_initial_state"))
    scale = float(manifest.get("scale", head_k**-0.5))
    gate_low = float(manifest.get("gate_low", -0.5))
    tol = float(manifest.get("verify_tol", _DEFAULT_TOL))
    signature = list(manifest["args_signature"])
    threads = int(manifest["threads_per_block"])

    q, k, v, g, beta = make_inputs(B, H, T, head_k, logical_head_v, gate_low=gate_low)
    packed: PackedFused = pack_v_partitions(chunk, head_v, q, k, v, g, beta)
    nt = packed.bh * packed.parts * packed.nc
    o = np.zeros((nt, packed.chunk * packed.head_v), dtype=np.uint16)
    ht = np.zeros(
        (packed.bh * packed.parts, packed.head_v, packed.head_k),
        dtype=np.float32,
    )
    h0_dense = None
    h0_packed = None
    if has_h0:
        rng = np.random.default_rng(7)
        h0_dense = np.float32(0.1) * rng.standard_normal(
            (B, H, head_k, logical_head_v), dtype=np.float32
        )
        h0_packed = pack_initial_state(head_v, h0_dense, packed.parts)

    grid = (packed.bh * packed.parts, 1, 1)
    block = (threads, 1, 1)
    flop = 4.0 * B * H * T * head_k * logical_head_v
    bytes_xfer = float(
        packed.q.nbytes
        + packed.k.nbytes
        + packed.v.nbytes
        + packed.g.nbytes
        + packed.beta.nbytes
        + o.nbytes
        + ht.nbytes
        + (0 if h0_packed is None else h0_packed.nbytes)
    )

    q_f = bf16_bits_to_f32(q)
    k_f = bf16_bits_to_f32(k)
    v_f = bf16_bits_to_f32(v)

    def make_args(rt: Runtime):
        q_dev = rt.alloc(nbytes(packed.q))
        k_dev = rt.alloc(nbytes(packed.k))
        g_dev = rt.alloc(nbytes(packed.g))
        beta_dev = rt.alloc(nbytes(packed.beta))
        v_dev = rt.alloc(nbytes(packed.v))
        o_dev = rt.alloc(nbytes(o))
        ht_dev = rt.alloc(nbytes(ht))
        rt.memcpy_h2d(q_dev, as_u8_buffer(packed.q), nbytes(packed.q))
        rt.memcpy_h2d(k_dev, as_u8_buffer(packed.k), nbytes(packed.k))
        rt.memcpy_h2d(g_dev, as_u8_buffer(packed.g), nbytes(packed.g))
        rt.memcpy_h2d(beta_dev, as_u8_buffer(packed.beta), nbytes(packed.beta))
        rt.memcpy_h2d(v_dev, as_u8_buffer(packed.v), nbytes(packed.v))
        rt.memset(o_dev, 0, nbytes(o))
        rt.memset(ht_dev, 0, nbytes(ht))
        if h0_packed is None:
            h0_dev = ht_dev
            ptrs = (q_dev, k_dev, g_dev, beta_dev, v_dev, o_dev, ht_dev)
        else:
            h0_dev = rt.alloc(nbytes(h0_packed))
            rt.memcpy_h2d(h0_dev, as_u8_buffer(h0_packed), nbytes(h0_packed))
            ptrs = (q_dev, k_dev, g_dev, beta_dev, v_dev, o_dev, h0_dev, ht_dev)
        args = pack_args(
            signature,
            {
                "q_ptr": q_dev,
                "k_ptr": k_dev,
                "g_ptr": g_dev,
                "beta_ptr": beta_dev,
                "v_ptr": v_dev,
                "o_ptr": o_dev,
                "h0_ptr": h0_dev,
                "ht_ptr": ht_dev,
                "scale": scale,
                "nc": packed.nc,
            },
        )
        return args, ptrs

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, int(o.size)
        o_dev = ptrs[5]
        ht_dev = ptrs[-1]
        rt.memcpy_d2h(as_u8_buffer(o), o_dev, nbytes(o))
        rt.memcpy_d2h(as_u8_buffer(ht), ht_dev, nbytes(ht))
        o_got_bits, ht_got = unpack_outputs(
            chunk=packed.chunk,
            head_k=packed.head_k,
            head_v=packed.head_v,
            o=o,
            ht=ht,
            B=B,
            H=H,
            T=T,
            DV=logical_head_v,
            NC=packed.nc,
            parts=packed.parts,
        )
        o_got = bf16_bits_to_f32(o_got_bits).astype(np.float64)
        o_ref, s_ref = ref_token_serial(q_f, k_f, v_f, g, beta, scale, h0=h0_dense)
        # Same criterion as the torch builder: max_abs / ref_absmax vs verify_tol.
        max_abs = 0.0
        bad_total = 0
        total = 0
        for got, ref in ((o_got, o_ref), (ht_got.astype(np.float64), s_ref)):
            err = np.abs(got - ref)
            d = float(err.max()) if err.size else 0.0
            den = max(float(np.abs(ref).max()) if ref.size else 0.0, 1e-30)
            max_abs = max(max_abs, d)
            bad_total += int(np.count_nonzero(err > tol * den))
            total += int(got.size)
        return max_abs, bad_total, total

    return make_args, grid, block, flop, bytes_xfer, check


register_manifest_runner(KIND, run_kda_fused_manifest_problem)
