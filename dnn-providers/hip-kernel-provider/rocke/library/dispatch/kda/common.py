# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pieces of the KDA family: the request and the gates every candidate
re-uses.

Arch-neutral by construction: each arch module imports this one and nothing
here imports an arch module, which keeps the
registry assembly in ``__init__`` free of import-order dependence.

SCOPE -- what this dispatcher decides
-------------------------------------
Which of the two chunkwise prefill paths runs: the **fused** kernel (one
workgroup per recurrence stream walks that stream's chunks, keeping the six
per-chunk tiles in LDS), or the **split** path (a tile-builder kernel that is
one workgroup per chunk, then a state-scan kernel over the materialized tiles).
Both compute the same recurrence and trade HBM traffic against parallelism.

The request carries the *logical* value width ``head_v``. gfx942 partitions a
logical head into ``KDA_PARTITION_HEAD_V``-channel workgroups to fit its LDS
budget; gfx950 keeps the whole value width in one recurrence stream and may
apply the scan spec's own ``value_splits`` in its grid helper.

DEFERRED -- the fused/split crossover
-------------------------------------
The available measurements do not establish a stable cross-architecture
threshold, so the split candidates are **opt-in** and fused is the default.
Naming ``algorithm="chunk_prep"`` / ``"chunk_scan"`` selects the split halves.

Also deferred: ``bind``. gfx942 launches through its manifest host pack while
gfx950 launches through ``KernelLauncher`` with torch tensors. The split path
also needs two launches and a tile workspace, which is not represented by one
:class:`~rocke.dispatch.core.ProblemBinding`.
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
# v2 gives raw gfx950 beta distinct BF16/FP32 pointer ABIs plus three strides.
KDA_ABI_VERSION = "rocke-kda-chunkwise/v2"


@dataclass(frozen=True)
class KdaRequest(OperatorRequest):
    """Normalized chunkwise Kimi Delta Attention prefill request.

    ``head_v`` is the logical value width of one attention head, not the
    per-workgroup partition used on gfx942. ``seqlen`` is the padded
    per-sequence length; this family has no varlen path, so a ragged batch must
    be padded by the caller. An omitted ``chunk_size`` selects the tuned
    architecture default: 16 on gfx942 and 32 on gfx950.

    The raw/token-major fields describe the framework-facing gfx950 split
    pipeline. They default off so existing prepared, chunk-packed requests keep
    their byte-identical specs.
    """

    batch: int
    num_heads: int
    seqlen: int
    arch: str
    head_k: int = 128
    head_v: int = 128
    chunk_size: int | None = None
    op: str = "kda"
    dtype: str = "bf16"
    algorithm: str = "auto"
    spec_id: str = "auto"
    has_initial_state: bool = False
    store_final_state: bool = True
    raw_inputs: bool = False
    fp32_beta_dtype: bool = False
    token_major_io: bool = False
    fuse_qk_l2norm: bool = False
    fuse_gate: bool = False
    fuse_beta_sigmoid: bool = False
    has_dt_bias: bool = False
    lower_bound: float = -5.0
    # None lets the architecture dispatcher choose its measured geometry.
    # An integer preserves the explicit override used by tuning/debug tools.
    value_splits: int | None = None

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        d["chunk_size"] = self.effective_chunk_size
        return d

    def dims(self) -> dict[str, int]:
        return {
            "batch": int(self.batch),
            "num_heads": int(self.num_heads),
            "seqlen": int(self.seqlen),
            "head_k": int(self.head_k),
            "head_v": int(self.head_v),
            "chunk_size": self.effective_chunk_size,
            "num_chunks": self.num_chunks,
            "value_splits": (
                0 if self.value_splits is None else int(self.value_splits)
            ),
        }

    def features(self) -> frozenset[str]:
        active = set()
        if bool(self.has_initial_state):
            active.add("initial_state")
        if bool(self.store_final_state):
            active.add("final_state")
        return frozenset(active)

    @property
    def effective_chunk_size(self) -> int:
        """Requested chunk, or the architecture's tuned default."""
        if self.chunk_size is not None:
            return int(self.chunk_size)
        return 16 if self.arch == "gfx942" else 32

    @property
    def num_chunks(self) -> int:
        """Chunks per sequence. Zero when ``seqlen`` does not tile exactly."""
        chunk = self.effective_chunk_size
        if chunk <= 0 or int(self.seqlen) % chunk:
            return 0
        return int(self.seqlen) // chunk

    @property
    def v_partitions(self) -> int:
        """Workgroups per logical head before any spec-local value splits."""
        if self.arch == "gfx942":
            return int(self.head_v) // KDA_PARTITION_HEAD_V
        return 1

    @property
    def workgroups(self) -> int:
        """Independent recurrence streams for the selected architecture."""
        return int(self.batch) * int(self.num_heads) * self.v_partitions


KDA_DIM_VOCABULARY = (
    "batch",
    "num_heads",
    "seqlen",
    "head_k",
    "head_v",
    "chunk_size",
    "num_chunks",
    "value_splits",
)

KDA_FEATURES = frozenset({"initial_state", "final_state"})


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, KdaRequest):
        return [f"expected KdaRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "kda":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("batch", "num_heads", "seqlen", "head_k", "head_v"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if req.effective_chunk_size <= 0:
        errors.append("chunk_size must be positive")
    if req.value_splits is not None and int(req.value_splits) not in (1, 2, 4, 8):
        errors.append("value_splits must be one of (1, 2, 4, 8)")
    if req.fp32_beta_dtype and not req.raw_inputs:
        errors.append("fp32_beta_dtype requires raw_inputs")
    if req.has_dt_bias and not req.fuse_gate:
        errors.append("has_dt_bias requires fuse_gate")
    fused_preprocessing = req.fuse_qk_l2norm or req.fuse_gate or req.fuse_beta_sigmoid
    if fused_preprocessing and not req.raw_inputs:
        errors.append("fused preprocessing requires raw_inputs")
    if req.raw_inputs and not fused_preprocessing:
        errors.append("raw_inputs requires fused preprocessing")
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
