# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 lightning-indexer candidates (CDNA3, bf16).

Registration only. The arch-neutral kernel lives in
``kernels/common/lightning_indexer.py`` and the shared candidate factory in
``.common``; this module just registers the gfx942-scoped candidates. Two
candidates: mfma_v1 (preferred for 16-aligned shapes) and scalar_v1 (the fallback
for unaligned shapes and the correctness oracle).
"""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry

from .common import _MFMA_SHAPES, _SHARED_SHAPES, make_candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(
        make_candidate(
            arches=("gfx942",),
            name="lightning_indexer_gfx942_mfma",
            algorithm="mfma_v1",
            spec_id="gfx942_mfma_v1",
            priority=5,
            body="mfma",
            shapes=_MFMA_SHAPES,
        )
    )
    registry.register(
        make_candidate(
            arches=("gfx942",),
            name="lightning_indexer_gfx942",
            algorithm="scalar_v1",
            spec_id="gfx942_scalar_v1",
            priority=10,
            body="scalar",
            shapes=_SHARED_SHAPES,
        )
    )
