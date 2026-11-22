# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: find the floating-point relaxations rocKE's numerics policy
forbids in lowered LLVM IR.

rocKE emits no fast-math flags, no ``!fpmath`` accuracy metadata and no
floating-point mode function attributes (``dsl_docs/numerics/README.md``,
"Shared rules"), so the compiler may not assume NaN or infinity away, fuse a
multiply and an add, relax a division or flush subnormals on its own.
:func:`fp_policy_violations` lists every line of a module that breaks that
rule, each prefixed with the kind of relaxation it carries.
"""

from __future__ import annotations

import re

__all__ = ["FAST_MATH_FLAGS", "FP_MODE_ATTRIBUTES", "fp_policy_violations"]

FAST_MATH_FLAGS = ("fast", "contract", "afn", "reassoc", "nnan", "ninf", "nsz", "arcp")
FP_MODE_ATTRIBUTES = (
    "denormal-fp-math",
    "unsafe-fp-math",
    "no-nans-fp-math",
    "no-infs-fp-math",
    "approx-func-fp-math",
    "no-signed-zeros-fp-math",
)
# The words between the opcode of a value-producing instruction that may carry
# fast-math flags and its type: where LLVM prints those flags.
_FLAGS_AFTER_OPCODE = re.compile(r"= (?:f\w+|call|select|phi) ((?:\w+ )*)")


def fp_policy_violations(llvm_ir: str) -> list[str]:
    """Every line of ``llvm_ir`` that carries a fast-math flag, ``!fpmath``
    metadata or a floating-point mode attribute, as ``"<kind>: <line>"``."""
    violations: list[str] = []
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        flags = _FLAGS_AFTER_OPCODE.search(stripped)
        if flags and set(flags.group(1).split()) & set(FAST_MATH_FLAGS):
            violations.append(f"fast-math flags: {stripped}")
        if "!fpmath" in stripped:
            violations.append(f"!fpmath metadata: {stripped}")
        if stripped.startswith("attributes #"):
            for attribute in FP_MODE_ATTRIBUTES:
                if f'"{attribute}' in stripped:
                    violations.append(f"FP mode attribute {attribute}: {stripped}")
    return violations
