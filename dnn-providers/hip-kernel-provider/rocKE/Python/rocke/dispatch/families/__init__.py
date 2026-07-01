# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Operator-family scaffolds for the CK DSL dispatcher.

The GEMM family (``rocke.dispatch.gemm``) is fully implemented as the worked
reference: two cases (fp16 / bf16 RCR) on top of the operator-agnostic
``core.py`` contracts (``OperatorRequest`` / ``KernelCandidate`` /
``CandidateRegistry`` / ``DispatchResult``) plus a generic config predicate
(``gemm_config_supported``) and the arch-family gate (``arch_family_supported``).

This package holds **documented scaffolds** for the remaining families so the
extension pattern is obvious and uniform. Each scaffold:

* defines a frozen ``OperatorRequest`` subclass with the family's normalized
  fields (so request hashing / cache identity already work),
* declares a ``CandidateRegistry`` for the family,
* documents exactly which existing pieces a full implementation reuses
  (the instance builders under ``rocke.instances`` and the per-family
  ``is_valid_spec`` validators), and
* exposes a ``dispatch_<family>`` entry point that raises
  ``NotImplementedError`` with a precise TODO until candidates are registered.

A scaffold is "filled in" by registering ``KernelCandidate`` objects on its
registry exactly the way ``gemm/fp16_rcr.py`` does:

    1. write per-candidate ``_spec_*`` factories (one per tuned tile/algorithm),
    2. write a ``support`` predicate = family request errors
       + ``arch_family_supported`` + a family config predicate
       (generalize ``gemm_config_supported``) + a runtime-shape check,
    3. register the candidates with (priority, arch_family),
    4. point ``dispatch_<family>`` at ``registry.select(req)``.
"""

from __future__ import annotations

from .attention import ATTENTION_REGISTRY, AttentionRequest, dispatch_attention
from .conv import CONV_REGISTRY, ConvRequest, dispatch_conv
from .moe import MOE_REGISTRY, MoeRequest, dispatch_moe
from .norm import NORM_REGISTRY, NormRequest, dispatch_norm

__all__ = [
    "ATTENTION_REGISTRY",
    "AttentionRequest",
    "dispatch_attention",
    "CONV_REGISTRY",
    "ConvRequest",
    "dispatch_conv",
    "MOE_REGISTRY",
    "MoeRequest",
    "dispatch_moe",
    "NORM_REGISTRY",
    "NormRequest",
    "dispatch_norm",
]
