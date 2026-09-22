# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``run_manifest`` adapter for the gfx942 lightning indexer (numpy pack + oracle).

Importing this module registers kind ``lightning_indexer_bf16``. The manifest
carries ``runner_module`` so ``python -m rocke.run_manifest`` can import this file
on a cluster node that has ``library/`` on ``PYTHONPATH`` -- no platform edit, no
torch. This is the numpy GPU lane: it allocates device buffers, launches the
kernel, copies the scores back, and grades them against the CPU oracle in
``hostpack.ref_indexer_scores``.

``--shape`` is ``seqlen_q, seqlen_k, n_index_heads`` (the three ints the remote
CLI forwards); ``index_head_dim`` and ``q_pos_base`` live on the manifest.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from rocke.helpers.manifest import make_simple_op_manifest
from rocke.runtime.host_buffers import as_u8_buffer, nbytes, require_numpy
from rocke.runtime.hip_module import Runtime
from rocke.runtime.packing import pack_args
from rocke.run_manifest import register_manifest_runner

from .hostpack import NEG_INF_SCORE, f32_to_bf16_bits, make_inputs, ref_indexer_scores

KIND = "lightning_indexer_bf16"
RUNNER_MODULE = "builders.gfx942.dsa.manifest"

# seqlen_q, seqlen_k, n_index_heads. seqlen_k > 2048 is where sparsity would
# matter, but the indexer scores every key regardless, so a small shape is a
# fine smoke default here.
_DEFAULT_SHAPE = (8, 64, 4)
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


def make_lightning_indexer_manifest(
    *,
    artifact,
    spec,
    args_signature: Sequence[Mapping[str, Any]],
    q_pos_base: int = 0,
    default_shape: Sequence[int] = _DEFAULT_SHAPE,
    warmup_iters: int = 5,
    timed_iters: int = 100,
    notes: str = "",
) -> Dict[str, Any]:
    """Emit a v1 manifest the remote ``run_manifest`` path can execute."""
    Q, Sk, HI = (int(x) for x in default_shape)
    note = notes or (
        "gfx942 lightning indexer. --shape is seqlen_q,seqlen_k,n_index_heads; "
        f"pack/check live in {RUNNER_MODULE}."
    )
    return make_simple_op_manifest(
        artifact=artifact,
        kind=KIND,
        op="lightning_indexer",
        dtype=str(getattr(spec, "dtype", "bf16")),
        threads_per_block=int(spec.tile.block_size),
        default_shape=(Q, Sk, HI),
        args_signature=_with_size_bytes(args_signature),
        warmup_iters=warmup_iters,
        timed_iters=timed_iters,
        notes=note,
        extra={
            "runner_module": RUNNER_MODULE,
            "index_head_dim": int(spec.index_head_dim),
            "q_pos_base": int(q_pos_base),
            "verify_tol": _DEFAULT_TOL,
        },
    )


def _shape(manifest: dict, shape: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if shape is None or shape == (0, 0, 0):
        ds = manifest.get("default_shape", list(_DEFAULT_SHAPE))
        if len(ds) != 3:
            raise ValueError(f"{KIND} default_shape must be [Q, Sk, HI], got {ds!r}")
        return int(ds[0]), int(ds[1]), int(ds[2])
    return int(shape[0]), int(shape[1]), int(shape[2])


def run_lightning_indexer_manifest_problem(
    manifest: dict, shape: Optional[Tuple[int, int, int]], verify: bool
) -> tuple:
    """Problem builder: ``(make_args, grid, block, flop, bytes_xfer, check)``."""
    np = require_numpy()
    Q, Sk, HI = _shape(manifest, shape)
    D_I = int(manifest["index_head_dim"])
    q_pos_base = int(manifest.get("q_pos_base", 0))
    tol = float(manifest.get("verify_tol", _DEFAULT_TOL))
    signature = list(manifest["args_signature"])
    threads = int(manifest["threads_per_block"])

    ins = make_inputs(Q, HI, D_I, Sk, seed=0)
    index_q_bits = f32_to_bf16_bits(ins["index_q"])
    index_k_bits = f32_to_bf16_bits(ins["index_k"])
    w = ins["w"].astype(np.float32)
    scores = np.zeros((Q, Sk), dtype=np.float32)
    ref = ref_indexer_scores(ins["index_q"], ins["index_k"], w, q_pos_base)

    grid = (Q, 1, 1)
    block = (threads, 1, 1)
    # Two flops per (q, key, head, dim): one multiply, one add in the dot.
    flop = 2.0 * Q * Sk * HI * D_I
    bytes_xfer = float(
        index_q_bits.nbytes + index_k_bits.nbytes + w.nbytes + scores.nbytes
    )

    def make_args(rt: Runtime):
        q_dev = rt.alloc(nbytes(index_q_bits))
        k_dev = rt.alloc(nbytes(index_k_bits))
        w_dev = rt.alloc(nbytes(w))
        s_dev = rt.alloc(nbytes(scores))
        rt.memcpy_h2d(q_dev, as_u8_buffer(index_q_bits), nbytes(index_q_bits))
        rt.memcpy_h2d(k_dev, as_u8_buffer(index_k_bits), nbytes(index_k_bits))
        rt.memcpy_h2d(w_dev, as_u8_buffer(w), nbytes(w))
        rt.memset(s_dev, 0, nbytes(scores))
        ptrs = (q_dev, k_dev, w_dev, s_dev)
        args = pack_args(
            signature,
            {
                "index_q": q_dev,
                "index_k": k_dev,
                "w": w_dev,
                "scores": s_dev,
                "q_pos_base": q_pos_base,
            },
        )
        return args, ptrs

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, int(scores.size)
        s_dev = ptrs[3]
        rt.memcpy_d2h(as_u8_buffer(scores), s_dev, nbytes(scores))
        got = scores.astype(np.float64)
        # Only grade the causal-valid region: masked entries are the sentinel on
        # both sides and would swamp the relative error otherwise.
        valid = ref > (0.5 * NEG_INF_SCORE)
        err = np.abs(got - ref)[valid]
        d = float(err.max()) if err.size else 0.0
        den = max(float(np.abs(ref[valid]).max()) if valid.any() else 0.0, 1e-30)
        bad = int(np.count_nonzero(err > tol * den))
        return d, bad, int(valid.sum())

    return make_args, grid, block, flop, bytes_xfer, check


register_manifest_runner(KIND, run_lightning_indexer_manifest_problem)
