# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Evaluates the problem predicates emitted for a skinny MXFP8 solution.

This mirrors the C++ semantics in ContractionProblemPredicates.hpp for the predicate tags the
skinny kernels emit. An unknown tag raises so that no predicate is skipped silently.
"""

from dataclasses import dataclass

from tensilelite.Contractions import ProblemPredicate

A_B_BYTES = 1
C_D_BYTES = 2


@dataclass(frozen=True)
class Tensor:
    sizes: tuple
    strides: tuple
    element_bytes: int

    def total_allocated_bytes(self):
        last = sum((size - 1) * stride for size, stride in zip(self.sizes, self.strides))
        return (last + 1) * self.element_bytes


@dataclass(frozen=True)
class Problem:
    sizes: tuple  # [m, n, batch, k]
    a: Tensor
    b: Tensor
    c: Tensor
    d: Tensor


def make_problem(m, n, batch, k, ldd, ldc, lda, ldb):
    """TN problem from an ExactLogic key: A is [k, m, batch], B [k, n, batch], C/D [m, n, batch]."""
    return Problem(
        sizes=(m, n, batch, k),
        a=Tensor((k, m, batch), (1, lda, lda * m), A_B_BYTES),
        b=Tensor((k, n, batch), (1, ldb, ldb * n), A_B_BYTES),
        c=Tensor((m, n, batch), (1, ldc, ldc * n), C_D_BYTES),
        d=Tensor((m, n, batch), (1, ldd, ldd * n), C_D_BYTES),
    )


def flatten(predicate):
    if predicate.tag == "And":
        for child in predicate.value:
            yield from flatten(child)
    else:
        yield predicate


def emitted_predicates(state):
    """The problem predicates the library builder derives from a solution's Assert keys."""
    result = []
    for pair in state.items():
        predicate = ProblemPredicate.FromOriginalKeyPair(pair)
        if predicate is not None:
            result.extend(flatten(predicate))
    return result


def _stride_multiple(stride, index, value):
    # A zero multiple or a broadcast (zero) stride is rejected, as in the C++ predicate.
    if value == 0 or index >= len(stride.strides) or stride.strides[index] == 0:
        return False
    return stride.strides[index] % value == 0


def evaluate(predicate, problem):
    tag, index, value = predicate.tag, predicate.index, predicate.value
    if tag == "SizeEqual":
        return problem.sizes[index] == value
    if tag == "SizeGreaterThan":
        return problem.sizes[index] > value
    if tag == "SizeLessThan":
        return problem.sizes[index] < value
    if tag == "SizeMultiple":
        return value != 0 and problem.sizes[index] % value == 0
    if tag == "BoundSizeMultiple":
        # Index -1 is the last summation index, which is K.
        return problem.sizes[3] % value == 0
    tensors = {"A": problem.a, "B": problem.b, "C": problem.c, "D": problem.d}
    if tag.startswith("Stride") and tag.endswith("Equal"):
        return tensors[tag[len("Stride")]].strides[index] == value
    if tag in ("StrideAMultiple", "StrideBMultiple"):
        return _stride_multiple(tensors[tag[len("Stride")]], index, value)
    if tag == "TensorBytesLessThan":
        return all(t.total_allocated_bytes() < value for t in tensors.values())
    raise AssertionError(f"predicate tag {tag!r} has no evaluator")


def failing(predicates, problem):
    """The emitted predicates the problem violates."""
    return [p for p in predicates if not evaluate(p, problem)]


def accepts(predicates, problem):
    return not failing(predicates, problem)
