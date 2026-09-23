# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 lightning-indexer candidate (CDNA3, bf16, scalar-reduction v1).

``kernels/gfx942/lightning_indexer.py`` carries the spec, validator, builder,
grid and signature; this module is the registration that makes it selectable
through the DSA registry. One candidate, because the indexer is a single emitted
kernel with one grid and ABI. The MFMA score body and the fp8 phase will add
their own candidates later.
"""

from __future__ import annotations

from typing import Tuple

from kernels.gfx942.lightning_indexer import (
    INDEXER_DTYPES,
    IndexerSpec,
    IndexerTileSpec,
    build_lightning_indexer,
    is_valid_spec,
    lightning_indexer_grid,
    lightning_indexer_signature,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    FAMILY,
    DSA_INDEXER_ABI_VERSION,
    IndexerRequest,
    _request_errors,
    _selector_matches,
)

# Declared coverage: the data gates a request can be filtered on without building
# a spec. Everything the validator computes from the spec (the LDS budget, the
# block_size/wave cover) stays in the residual predicate below.
_SHARED_SHAPES = (
    ShapeRange("seqlen_q", min=1),
    ShapeRange("seqlen_k", min=1),
    ShapeRange("n_index_heads", min=1),
    ShapeRange("index_head_dim", min=1),
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


# The MFMA candidate needs 16-aligned tiles (validator-gated), so the capability
# prefilters unaligned requests out — they fall to the scalar candidate below.
_MFMA_SHAPES = (
    ShapeRange("seqlen_q", min=16, multiple_of=16),
    ShapeRange("seqlen_k", min=16, multiple_of=16),
    ShapeRange("n_index_heads", min=1),
    ShapeRange("index_head_dim", min=16, multiple_of=16),
)


def _make_candidate(*, name, algorithm, spec_id, priority, body, shapes) -> KernelCandidate:
    def support(req: OperatorRequest) -> Tuple[bool, str]:
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
            arches=("gfx942",),
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


def register(registry: CandidateRegistry) -> None:
    # MFMA is the default (lower priority number wins) for 16-aligned shapes; the
    # scalar body is the fallback for unaligned shapes and the correctness oracle.
    registry.register(
        _make_candidate(
            name="lightning_indexer_gfx942_mfma",
            algorithm="mfma_v1",
            spec_id="gfx942_mfma_v1",
            priority=5,
            body="mfma",
            shapes=_MFMA_SHAPES,
        )
    )
    registry.register(
        _make_candidate(
            name="lightning_indexer_gfx942",
            algorithm="scalar_v1",
            spec_id="gfx942_scalar_v1",
            priority=10,
            body="scalar",
            shapes=_SHARED_SHAPES,
        )
    )
