# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Operator-family scaffolds for the rocKE dispatcher.

The GEMM family (``rocke.dispatch.gemm``) is fully implemented as the worked
reference: two cases (fp16 / bf16 RCR) on top of the operator-agnostic
``core.py`` contracts (``OperatorRequest`` / ``KernelCandidate`` /
``CandidateRegistry`` / ``DispatchResult``) plus a generic config predicate
(``gemm_config_supported``) and a declared ``Capability`` per candidate.

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
    2. declare a ``Capability`` per candidate: the explicit ``arches`` it serves,
       plus the dtypes / layouts / shape bounds it is built for. This is
       mandatory -- ``register()`` rejects a candidate without one, because a
       candidate that declares nothing is invisible to ``for_arch`` and
       ``coverage()``,
    3. write a ``support`` predicate for what is left: family request errors, a
       family config predicate (generalize ``gemm_config_supported``), and any
       runtime-shape check the capability cannot express as data,
    4. register the candidates, and call ``candidate.admits(req)`` anywhere the
       full verdict is wanted -- the residual predicate is stored as
       ``_supports`` precisely so that calling it alone reads as a mistake,
    5. point ``dispatch_<family>`` at ``registry.select(req)``.

Each family also passes a ``dim_vocabulary`` to its registry, which turns a
misspelled dimension name in a ``ShapeRange`` into an import-time error instead
of a constraint that silently never matches.
"""

from __future__ import annotations

from .moe import MOE_REGISTRY, MoeRequest, dispatch_moe
from .norm import NORM_REGISTRY, NormRequest, dispatch_norm

__all__ = [
    "MOE_REGISTRY",
    "MoeRequest",
    "dispatch_moe",
    "NORM_REGISTRY",
    "NormRequest",
    "dispatch_norm",
]
