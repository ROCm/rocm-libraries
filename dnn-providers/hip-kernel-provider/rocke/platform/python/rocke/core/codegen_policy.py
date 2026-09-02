# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Typed per-kernel policy for backend code generation.

Semantic kernel operations remain in :class:`KernelDef`; this module owns the
small set of validated backend choices that affect the compiled artifact without
becoming an unrestricted compiler-option interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .ir import KernelDef

SCHEDULER_STRATEGY_ATTR = "scheduler_strategy"


class SchedulerStrategy(str, Enum):
    """AMDGPU machine-scheduler strategies supported as stable kernel policy."""

    MAX_ILP = "max-ilp"
    MAX_MEMORY_CLAUSE = "max-memory-clause"
    ITERATIVE_ILP = "iterative-ilp"
    ITERATIVE_MINREG = "iterative-minreg"
    ITERATIVE_MAXOCC = "iterative-maxocc"


_SCHEDULER_STRATEGY_VALUES = frozenset(strategy.value for strategy in SchedulerStrategy)


def normalize_scheduler_strategy(value: object) -> str | None:
    """Return the canonical scheduler string or reject an unsupported value."""

    if value is None:
        return None
    if isinstance(value, SchedulerStrategy):
        return value.value
    if isinstance(value, str) and value in _SCHEDULER_STRATEGY_VALUES:
        return value
    choices = ", ".join(sorted(_SCHEDULER_STRATEGY_VALUES))
    raise ValueError(
        f"unsupported scheduler_strategy {value!r}; expected one of: {choices}, or None"
    )


@dataclass(frozen=True)
class CodegenPolicy:
    """Validated code-generation choices belonging to one compiled kernel."""

    scheduler_strategy: str | SchedulerStrategy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scheduler_strategy",
            normalize_scheduler_strategy(self.scheduler_strategy),
        )


def codegen_policy_from_attrs(attrs: Mapping[str, Any]) -> CodegenPolicy:
    """Construct and validate policy from a kernel attribute mapping."""

    return CodegenPolicy(scheduler_strategy=attrs.get(SCHEDULER_STRATEGY_ATTR))


def codegen_policy_for_kernel(kernel: KernelDef) -> CodegenPolicy:
    """Return the validated code-generation policy carried by ``kernel``."""

    return codegen_policy_from_attrs(kernel.attrs)


def apply_codegen_policy(kernel: KernelDef, policy: CodegenPolicy) -> None:
    """Apply validated policy to ``kernel`` before serialization and lowering."""

    if not isinstance(policy, CodegenPolicy):
        raise TypeError(f"policy must be CodegenPolicy, got {type(policy).__name__}")
    if policy.scheduler_strategy is None:
        kernel.attrs.pop(SCHEDULER_STRATEGY_ATTR, None)
    else:
        kernel.attrs[SCHEDULER_STRATEGY_ATTR] = policy.scheduler_strategy


__all__ = [
    "CodegenPolicy",
    "SCHEDULER_STRATEGY_ATTR",
    "SchedulerStrategy",
    "apply_codegen_policy",
    "codegen_policy_for_kernel",
    "codegen_policy_from_attrs",
    "normalize_scheduler_strategy",
]
