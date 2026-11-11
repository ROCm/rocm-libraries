# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 MLA candidates.

One candidate: the weight-absorbed decode kernel, for ``max_seqlen_q == 1``
(absorbed query in, normalized latent + LSE out; ``dispatch.mla.decode``).
There is no gfx950 prefill kernel yet, so a prefill-shaped request is refused
on gfx950 with every candidate's reason.
"""

from __future__ import annotations

from kernels.mla.mla_decode_gfx950 import (
    MlaDecodeGfx950Spec,
    build_mla_decode_gfx950,
)
from rocke.dispatch.core import CandidateRegistry

from .decode import make_mla_decode_candidate

# About 256 VGPRs per wave hold two workgroups per CU, as on gfx942; gfx950's
# larger LDS does not raise that while VGPRs bind first.
_DECODE_WGS_PER_CU = 2
# CU count assumed when neither the request nor the live device supplies one.
# Too high only splits further than needed.
_DECODE_FALLBACK_NUM_CUS = 256


def register(registry: CandidateRegistry) -> None:
    registry.register(
        make_mla_decode_candidate(
            spec_cls=MlaDecodeGfx950Spec,
            build_fn=build_mla_decode_gfx950,
            wgs_per_cu=_DECODE_WGS_PER_CU,
            fallback_num_cus=_DECODE_FALLBACK_NUM_CUS,
        )
    )


__all__ = ["register"]
