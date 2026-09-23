# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Evaluate the static GEMM sizes/strides in Tensile's runtime predicates.

Candidate selection can reject a known size mismatch before compilation. The
predicate tree owns the constraints and their tuning-dependent values; this
module evaluates the subset available without device buffers or runtime state.
The host runtime must still evaluate the complete predicate tree before launch.
"""

from Tensile.Contractions import ProblemPredicate, ProblemType


def problemSizeRejection(solution, sizes, strides, tensorSizes=None):
    """Return the first known size/stride mismatch, or ``None``.

    ``sizes`` is [M, N, batch, K]. ``strides`` and optional ``tensorSizes`` map
    A/B/C/D to their physical three-dimensional layouts. A/B are not accessed
    when K=0, and no tensor is accessed when M=0 or N=0. The request does not
    carry beta, so C bounds are conservatively checked as if beta were nonzero.
    This is a partial static check, not a replacement for runtime support checks.
    """
    problemType = ProblemType.FromOriginalState(solution["ProblemType"])
    predicate = ProblemPredicate.FromOriginalState(solution, problemType)
    m, n, batch, k = sizes
    types = dict(zip("ABCD", (problemType.aType, problemType.bType,
                              problemType.cType, problemType.dType)))
    physical = {
        "A": [k, m, batch] if problemType.transA else [m, k, batch],
        "B": [n, k, batch] if problemType.transB else [k, n, batch],
        "C": [m, n, batch], "D": [m, n, batch],
    }
    physical.update({key: value for key, value in (tensorSizes or {}).items()
                     if value is not None})

    def withinBuffer(tensor, tile, shift=0):
        offset = strides[tensor][1] * min(tile, physical[tensor][1]) + shift
        return offset * types[tensor].numBytes() < 2**32

    def check(node):
        tag, value = node.tag, node.value
        if tag == "And":
            for child in value:
                reason = check(child)
                if reason:
                    return reason
            return None
        if tag in ("Free0SizeMultiple", "Free1SizeMultiple", "BoundSizeMultiple"):
            extent = {"Free0SizeMultiple": m, "Free1SizeMultiple": n,
                      "BoundSizeMultiple": k}[tag]
            valid = extent % value == 0
        elif tag == "BatchSizeEqual":
            valid = batch == value
        elif tag == "BatchSizeMultiple":
            valid = batch % value == 0
        elif tag in ("SizeEqual", "SizeMultiple", "SizeGreaterThan", "SizeLessThan"):
            extent = sizes[node.index]
            valid = {"SizeEqual": lambda: extent == value,
                     "SizeMultiple": lambda: extent % value == 0,
                     "SizeGreaterThan": lambda: extent > value,
                     "SizeLessThan": lambda: extent < value}[tag]()
        elif m == 0 or n == 0:
            return None
        elif tag in ("LeadingFree0SizesGreaterOrEqual", "LeadingFree1SizesGreaterOrEqual"):
            valid = k == 0 or (m if tag == "LeadingFree0SizesGreaterOrEqual" else n) >= value
        elif tag == "BufferLoadOffsetLimitCheck":
            valid = k == 0 or (withinBuffer("A", value["DUorMT0"], value["ShiftPtrElemA"])
                               and withinBuffer("B", value["DUorMT1"], value["ShiftPtrElemB"]))
        elif tag == "BufferLoadOffsetLimitCheck_Beta":
            valid = withinBuffer("C", value)
        elif tag == "BufferStoreOffsetLimitCheck":
            valid = withinBuffer("D", value)
        elif tag == "WorkgroupNumberCheck":
            tileM, tileN, splitK = value
            valid = splitK == -1 or ((m + tileM - 1) // tileM
                                     * ((n + tileN - 1) // tileN)
                                     * max(splitK, 1) * batch <= 2**24)
        elif tag == "GlobalSplitUCheckMinK":
            minimum, splitK = value
            valid = splitK == -1 or k >= minimum * (splitK if splitK > 1 else 0)
        elif tag == "CDStridesEqual":
            valid = strides["C"] == strides["D"]
        else:
            # Scalar values, pointers, workspace, and hardware-dependent checks
            # are evaluated by the existing runtime predicates before execution.
            return None
        return None if valid else f"Problem violates {tag}={value}"

    return check(predicate)
