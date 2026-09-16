# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GEMM family pipeline/capability policy.

``GemmPipelinePolicy`` is the family-side validity surface: it answers "is this
spec legal on this target?" by composing :class:`rocke.core.arch.ArchTarget`
hardware predicates (legal MMA atoms, LDS capacity, max threads/block) with
GEMM-specific geometry rules. It is arch-agnostic code that *reads* arch facts —
no LLVM intrinsic names, no per-gfx tables.

For the first milestone this delegates to the (now arch-aware) ``is_valid_spec``
in ``instances/common/gemm_universal.py`` so there is a single source of truth
for the rules while the policy abstraction is introduced. The body of
``is_valid_spec`` migrates into this class as the policy surface widens (see
``dsl_docs/architecture/multi_arch_data_layout.md`` → "Migration").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Tuple

from ...core.arch import ArchTarget


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str = "ok"

    @staticmethod
    def accept() -> "ValidationResult":
        return ValidationResult(True, "ok")

    @staticmethod
    def reject(reason: str) -> "ValidationResult":
        return ValidationResult(False, reason)

    def __bool__(self) -> bool:  # so callers can `if policy.validate(...):`
        return self.ok


class GemmPipelinePolicy:
    """Validity policy for the Universal GEMM family."""

    family: ClassVar[str] = "gemm"

    def valid_warp_tiles(
        self, target: ArchTarget, dtypes: Tuple[str, str, str]
    ) -> List[Tuple[int, int, int]]:
        """The legal MFMA warp-tile shapes for ``dtypes=(a,b,c)`` on ``target``,
        sourced from the target's MMA catalog (so a new arch picks up its atoms
        with no edit here)."""
        a, b, c = dtypes
        return sorted(
            op.shape for op in target.mma.enumerate(a_dtype=a, b_dtype=b, c_dtype=c)
        )

    def validate(self, target: ArchTarget, spec) -> ValidationResult:
        """Return a :class:`ValidationResult` for ``spec`` on ``target``.

        Composes the arch-aware GEMM rules (single source of truth in
        ``is_valid_spec``) keyed by ``target.gfx``.
        """
        from .gemm_universal import is_valid_spec

        ok, reason = is_valid_spec(spec, arch=target.gfx)
        return ValidationResult(ok, reason)
