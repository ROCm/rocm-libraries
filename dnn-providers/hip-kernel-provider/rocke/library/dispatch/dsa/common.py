# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared pieces of the DSA lightning-indexer family: the request and the gates
every candidate re-uses.

Arch-neutral by construction: each arch module imports this one and nothing here
imports an arch module, so the registry assembly in ``__init__`` is free of
import-order dependence.

SCOPE -- what this dispatcher decides
-------------------------------------
Which lightning-indexer kernel serves a scoring request. The indexer scores
every allowed key (the causal set) and emits the score matrix that top-k
consumes; it does not select. The dense-fallback routing (Sk <= k routes to
dense MLA instead of the sparse path) is an outer decision above this registry,
per the design doc -- it is not a gate here, because the indexer's job does not
change with Sk. The DSA op string is admitted only by this family's
``_request_errors``, so DSA and standard-attention candidate sets stay mutually
exclusive by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from kernels.common.lightning_indexer import (
    INDEXER_DTYPES,
    IndexerSpec,
    IndexerTileSpec,
    build_lightning_indexer,
    is_valid_spec,
    lightning_indexer_grid,
    lightning_indexer_signature,
)
from rocke.core.arch import ArchTarget
from rocke.dispatch.core import (
    Capability,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
    selector_matches,
)

FAMILY = "lightning_indexer"
DSA_INDEXER_ABI_VERSION = "rocke-lightning-indexer/v1"

# The indexer scores every key it is given, so the request geometry is the
# indexer's own dimensions plus the two sequence lengths that size the grid and
# the score matrix. q_pos_base is a launch argument (the causal offset), not a
# selection dimension, so it is not carried here.
DSA_INDEXER_DIM_VOCABULARY = (
    "seqlen_q",
    "seqlen_k",
    "n_index_heads",
    "index_head_dim",
)

DSA_INDEXER_FEATURES = frozenset()


@dataclass(frozen=True)
class IndexerRequest(OperatorRequest):
    """Normalized lightning-indexer forward request.

    ``seqlen_q`` is the number of query tokens (batch axis in decode, packed
    query count in prefill); ``seqlen_k`` is the key length Sk. ``n_index_heads``
    (H_I) and ``index_head_dim`` (D_I) are the indexer geometry, distinct from
    the MLA attention geometry.
    """

    seqlen_q: int
    seqlen_k: int
    n_index_heads: int
    arch: str
    index_head_dim: int = 128
    op: str = "lightning_indexer"
    dtype: str = "bf16"
    block_size: int = 256
    algorithm: str = "auto"
    spec_id: str = "auto"

    def normalized(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.lower()
        return d

    def dims(self) -> dict[str, int]:
        return {
            "seqlen_q": int(self.seqlen_q),
            "seqlen_k": int(self.seqlen_k),
            "n_index_heads": int(self.n_index_heads),
            "index_head_dim": int(self.index_head_dim),
        }

    def features(self) -> frozenset[str]:
        return frozenset()


def _request_errors(req: OperatorRequest) -> list[str]:
    if not isinstance(req, IndexerRequest):
        return [f"expected IndexerRequest, got {type(req).__name__}"]
    errors: list[str] = []
    if req.op != "lightning_indexer":
        errors.append(f"unsupported op {req.op!r}")
    for field in ("seqlen_q", "seqlen_k", "n_index_heads", "index_head_dim"):
        if int(getattr(req, field)) <= 0:
            errors.append(f"{field} must be positive")
    if int(req.block_size) <= 0:
        errors.append("block_size must be positive")
    try:
        ArchTarget.from_gfx(req.arch)
    except KeyError as e:
        errors.append(str(e))
    return errors


# Shared pin-selector: identical across families, re-exported under the family's
# own name.
_selector_matches = selector_matches


# ---- Shared candidate factory (arch-parametric) ---------------------------
# The kernel is arch-neutral, so gfx942 and gfx950 candidates differ only in the
# capability's arch tuple and the candidate names; everything else is shared here.

# Declared coverage a request can be filtered on without building a spec. The
# validator's spec-computed gates (block_size/wave cover) stay residual.
_SHARED_SHAPES = (
    ShapeRange("seqlen_q", min=1),
    ShapeRange("seqlen_k", min=1),
    ShapeRange("n_index_heads", min=1),
    ShapeRange("index_head_dim", min=1),
)
# The MFMA candidate needs 16-aligned tiles, so its capability prefilters
# unaligned requests out -- they fall to the scalar candidate.
_MFMA_SHAPES = (
    ShapeRange("seqlen_q", min=16, multiple_of=16),
    ShapeRange("seqlen_k", min=16, multiple_of=16),
    ShapeRange("n_index_heads", min=1),
    ShapeRange("index_head_dim", min=16, multiple_of=16),
)


def _spec(req: OperatorRequest, body: str) -> IndexerSpec:
    assert isinstance(req, IndexerRequest)
    # The MFMA body pins one wave64; the scalar body honors the request block.
    block = 64 if body == "mfma" else int(req.block_size)
    return IndexerSpec(
        n_index_heads=int(req.n_index_heads),
        index_head_dim=int(req.index_head_dim),
        seqlen_q=int(req.seqlen_q),
        seqlen_k=int(req.seqlen_k),
        dtype=req.dtype.lower(),
        body=body,
        tile=IndexerTileSpec(block_size=block),
    )


def _grid(spec: IndexerSpec, req: OperatorRequest):
    # The grid is fully determined by the spec; the request is accepted for
    # signature parity with the family's grid callback.
    return lightning_indexer_grid(spec)


def make_candidate(
    *, arches, name, algorithm, spec_id, priority, body, shapes
) -> KernelCandidate:
    """Build one arch-scoped indexer candidate (mfma or scalar)."""

    def support(req: OperatorRequest):
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, IndexerRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        return is_valid_spec(_spec(req, body), arch=req.arch)

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _spec(req, body)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=DSA_INDEXER_ABI_VERSION,
        priority=priority,
        capability=Capability(
            arches=tuple(arches),
            dtypes=INDEXER_DTYPES,
            shapes=shapes,
            relations=(),
            supports_features=frozenset(),
        ),
        _supports=support,
        select_spec=select,
        build=build_lightning_indexer,
        grid=_grid,
        block=lambda spec: (spec.tile.block_size, 1, 1),
        signature=lightning_indexer_signature,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


__all__ = [
    "FAMILY",
    "DSA_INDEXER_ABI_VERSION",
    "DSA_INDEXER_DIM_VOCABULARY",
    "DSA_INDEXER_FEATURES",
    "INDEXER_DTYPES",
    "IndexerRequest",
    "_MFMA_SHAPES",
    "_SHARED_SHAPES",
    "_request_errors",
    "_selector_matches",
    "make_candidate",
]
