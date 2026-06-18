################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Shared exception type and helpers for input-YAML type validation.

Every in-place TensileLite validator that owns a section of an input YAML
(GlobalParameters, ProblemType, BenchmarkCommon/ForkParameters,
InternalSupportParams, LibraryLogic) raises ``ConfigTypeError`` with a
structured message produced by :func:`formatMismatch`. The shared
location guarantees identical message shape across validators and lets
tests (and the worker-process re-raise in ``BenchmarkProblems.py``)
catch the exception by type.

The module is deliberately small: an exception class, one formatter, an
optional line-number recovery helper that re-parses the file only on
the error path, and a staged-landing kill switch. No dispatcher,
no registry — each validator owns its own section.

Strict will be the only behaviour (mismatches raise ``ConfigTypeError``)
once ``_STRICT_GATE_ENABLED`` is removed in commit 2; until then the gate
ships off by default and every validator early-returns to warn-only.
"""

import os
from typing import Iterable, Optional, Type


# Staged-landing kill switch: when False, every input-yaml validator
# returns immediately without raising. The next commit flips this to
# True and then removes the boolean entirely so no traces remain.
_STRICT_GATE_ENABLED = False


class ConfigTypeError(Exception):
    """Raised when an input YAML value has the wrong Python type.

    Distinct from generic ``Exception`` so tests can catch it precisely
    and workers can re-raise it past the broad ``except Exception``
    swallow in ``BenchmarkProblems._generate_single_solution``.
    """
    pass


def _typeNames(expectedTypes: Iterable[Type]) -> str:
    """Render a set/iterable of types as ``\"int\"`` / ``\"int or str\"``."""
    names = sorted({t.__name__ for t in expectedTypes})
    return " or ".join(names) if names else "<unspecified>"


def _readLineNo(srcFile: str, keyLeaf: str) -> Optional[int]:
    """Recover a line number for ``keyLeaf`` in ``srcFile`` via yaml.compose.

    Best-effort: returns the first matching line or ``None`` on any
    failure (file missing, yaml parse error, key not found). Paid only
    on the error path because PyYAML's default loader discards position
    info.
    """
    if not srcFile or not os.path.isfile(srcFile):
        return None
    try:
        import yaml
        with open(srcFile, "r") as f:
            node = yaml.compose(f)
    except Exception:
        return None

    # Walk the composed node tree looking for the first scalar key whose
    # value matches keyLeaf. This is a best-effort hint; multiple matches
    # all return the first occurrence.
    from yaml.nodes import MappingNode, SequenceNode, ScalarNode

    def _walk(n):
        if isinstance(n, MappingNode):
            for k, v in n.value:
                if isinstance(k, ScalarNode) and k.value == keyLeaf:
                    return k.start_mark.line + 1
                hit = _walk(v)
                if hit is not None:
                    return hit
        elif isinstance(n, SequenceNode):
            for child in n.value:
                hit = _walk(child)
                if hit is not None:
                    return hit
        return None

    try:
        return _walk(node)
    except Exception:
        return None


def formatMismatch(
    srcFile: str,
    keyPath: str,
    value,
    expectedTypes,
    *,
    lineNo: Optional[int] = None,
) -> str:
    """Render a single type-mismatch record as a one-line message.

    The shape is:

        <srcFile>:<lineNo>: <keyPath> = <repr(value)> (<actualType>);
        expected <expectedTypes>

    Either ``<srcFile>`` or ``<lineNo>`` (or both) may be omitted if not
    available. ``repr(value)`` is used so ``True`` / ``1`` / ``\"1\"``
    are unambiguous in the output. Callers may pass ``lineNo`` directly
    (cheap) or let the formatter recover it from ``srcFile`` via
    :func:`_readLineNo` (only paid on the error path).
    """
    if lineNo is None and srcFile:
        # Use the last keypath segment as the leaf key to search for.
        leaf = keyPath.rsplit(".", 1)[-1].split("[", 1)[0]
        lineNo = _readLineNo(srcFile, leaf)

    prefix_parts = []
    if srcFile:
        prefix_parts.append(srcFile)
    if lineNo is not None:
        prefix_parts.append(str(lineNo))
    prefix = ":".join(prefix_parts)
    if prefix:
        prefix += ": "

    actualType = type(value).__name__
    expectedStr = _typeNames(expectedTypes)
    return (
        f"{prefix}{keyPath} = {value!r} ({actualType}); expected {expectedStr}"
    )
