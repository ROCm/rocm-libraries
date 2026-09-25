# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 lightning-indexer candidates (CDNA4, bf16).

The kernel is arch-neutral (the 16x16x16 bf16 atom and the register-only MFMA
body work on gfx950 too, and neither body uses LDS), so this module is
registration only -- same shared factory as gfx942, just the gfx950 arch tuple
and candidate names. Two candidates: mfma_v1 (preferred for 16-aligned shapes)
and scalar_v1 (the fallback for unaligned shapes and the correctness oracle).
"""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry

from .common import _MFMA_SHAPES, _SHARED_SHAPES, make_candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(
        make_candidate(
            arches=("gfx950",),
            name="lightning_indexer_gfx950_mfma",
            algorithm="mfma_v1",
            spec_id="gfx950_mfma_v1",
            priority=5,
            body="mfma",
            shapes=_MFMA_SHAPES,
        )
    )
    registry.register(
        make_candidate(
            arches=("gfx950",),
            name="lightning_indexer_gfx950",
            algorithm="scalar_v1",
            spec_id="gfx950_scalar_v1",
            priority=10,
            body="scalar",
            shapes=_SHARED_SHAPES,
        )
    )
