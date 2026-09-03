# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pieces of the KDA family: the request and the gates every candidate
re-uses.

Arch-neutral by construction even though the only arch today is gfx942: the
arch module imports this one and nothing here imports it, which keeps the
registry assembly in ``__init__`` free of import-order dependence.

SCOPE -- what this dispatcher decides
-------------------------------------
Which of the two chunkwise prefill paths runs: the **fused** kernel (one
workgroup per (batch, head, V partition) walks that head's chunks, keeping the
six per-chunk tiles in LDS), or the **split** path (a tile-builder kernel that
is one workgroup per chunk, then a state-scan kernel over the materialized
tiles). Both compute the same recurrence; they trade HBM traffic against
parallelism. See :mod:`kernels.gfx942.kda_chunkwise`.

The request carries the *logical* value width ``head_v`` (128 for a Kimi Delta
Attention head). The partition into ``KDA_PARTITION_HEAD_V``-channel workgroups
is an implementation fact of gfx942's LDS budget, so the candidates apply it
when they build the spec and when they report their grid -- exactly as
``builders/gfx942/kda`` does on the host.

DEFERRED -- the fused/split crossover
-------------------------------------
The kernel docstrings state the trade in the right direction (split wins once
there is enough batch/head parallelism to fill the device, fused wins for
prefill at batch scale) but there is no measured crossover, so there is no
honest threshold to encode. The split candidates are therefore **opt-in** and
fused is the default, rather than a heuristic that would look measured and is
not. Naming ``algorithm="chunk_prep"`` / ``"chunk_scan"`` selects the split
halves.

Also deferred: ``bind``. The launch-side pack for these kernels lives in
``builders/gfx942/kda/hostpack.py`` and is already reachable through the
manifest runner, which is a different seam; wiring it as a ``ProblemBinding``
is a separate change, so the registry does not set ``require_binding``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

from kernels.gfx942.kda_chunkwise import (
    # Re-exported, never redeclared. The kernel owns what it covers; dispatch's
    # job is to state that coverage as a Capability. Copying the numbers would
    # drift in the direction that fails silently -- the prefilter rejecting a
    # shape the kernel had since learned to run.
    KDA_CHUNK_SIZES,
    KDA_DTYPES,
    KDA_PARTITION_HEAD_V,
)
from rocke.core.arch import ArchTarget
from rocke.dispatch.core import KernelCandidate, OperatorRequest

FAMILY = "kda_chunkwise"
KDA_ABI_VERSION = "rocke-kda-chunkwise/v1"


@dataclass(frozen=True)
class KdaRequest(OperatorRequest):
    """Normalized chunkwise Kimi Delta Attention prefill request.

    ``head_v`` is the logical value width of one attention head, not the
    per-workgroup partition. ``seqlen`` is the padded per-sequence length; this
    family has no varlen path, so a ragged batch must be padded by the caller.
    """

    batch: int
    num_heads: int
    seqlen: int
    arch: str
    head_k: int = 128
    head_v: int = 128
    chunk_size: int = 16
    op: str = "kda"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    has_initial_state: bool = False
    store_final_state: bool = True

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        return d

    def dims(self) -> dict[str, int]:
        return {
            "batch": int(self.batch),
            "num_heads": int(self.num_heads),
            "seqlen": int(self.seqlen),
            "head_k": int(self.head_k),
            "head_v": int(self.head_v),
            "chunk_size": int(self.chunk_size),
            "num_chunks": self.num_chunks,
        }

    def features(self) -> frozenset[str]:
        active = set()
        if bool(self.has_initial_state):
            active.add("initial_state")
        if bool(self.store_final_state):
            active.add("final_state")
        return frozenset(active)

    @property
    def num_chunks(self) -> int:
        """Chunks per sequence. Zero when ``seqlen`` does not tile exactly."""
        chunk = int(self.chunk_size)
        if chunk <= 0 or int(self.seqlen) % chunk:
            return 0
        return int(self.seqlen) // chunk

    @property
    def v_partitions(self) -> int:
        """Workgroups per logical head, from gfx942's value-channel partition."""
        return int(self.head_v) // KDA_PARTITION_HEAD_V

    @property
    def workgroups(self) -> int:
        """Independent (batch, head, V partition) scan streams."""
        return int(self.batch) * int(self.num_heads) * self.v_partitions


KDA_DIM_VOCABULARY = (
    "batch",
    "num_heads",
    "seqlen",
    "head_k",
    "head_v",
    "chunk_size",
    "num_chunks",
)

KDA_FEATURES = frozenset({"initial_state", "final_state"})


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, KdaRequest):
        return [f"expected KdaRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "kda":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("batch", "num_heads", "seqlen", "head_k", "head_v", "chunk_size"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


def _selector_matches(req: KdaRequest, candidate: KernelCandidate) -> Tuple[bool, str]:
    algorithm = req.algorithm.strip().lower()
    spec_id = req.spec_id.strip().lower()
    if algorithm not in ("auto", candidate.algorithm):
        return False, f"request algorithm {req.algorithm!r} != {candidate.algorithm!r}"
    if spec_id not in ("auto", candidate.spec_id):
        return False, f"request spec_id {req.spec_id!r} != {candidate.spec_id!r}"
    return True, "ok"


__all__ = [
    "FAMILY",
    "KDA_ABI_VERSION",
    "KDA_CHUNK_SIZES",
    "KDA_DIM_VOCABULARY",
    "KDA_DTYPES",
    "KDA_FEATURES",
    "KDA_PARTITION_HEAD_V",
    "KdaRequest",
    "_request_errors",
    "_selector_matches",
]
