# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Deriving `tflops` and `gbs` from what an operation declares and a run measured.

RFC 0019.13 §8.3 makes both metrics derived rather than collected: a producer supplies a time,
and the quantities that turn a time into a rate come from the operation's `.opmeta.json`.
`flops` and `elements` are declared there, closed-form over ``$q.*`` and free of anything the
runtime measures -- ``2*M*N*K`` is the same count however a matmul is tiled -- so a corpus from
any source can be given the metrics rather than asked for them.

Separate from the reader and the writer so it can be tested without either: everything here is
arithmetic over a row and a declaration, which is what makes the failure it guards visible. A
wrong FLOP formula does not raise, it produces a plausible throughput, and every model trained on
it ranks kernels by a number that is wrong by a constant factor nobody sees.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["DTYPE_BYTES", "evaluate", "dtype_width", "derive_metrics", "UnsupportedExpression"]


class UnsupportedExpression(Exception):
    """An opmeta declaration used an operator this evaluator does not implement.

    Raised rather than skipped. A declaration this cannot read is a metric it would otherwise
    emit as null, which reads downstream as "not measured" -- indistinguishable from a genuine
    failure, and wrong in the direction that hides the problem.
    """


#: Width in bytes of every dtype the seven operation declarations admit.
#:
#: Sub-byte types are absent deliberately rather than rounded to one: none of the operations
#: declare them today, and a rounded width would make `gbs` quietly wrong for the first one that
#: does. Adding a dtype here is the deliberate step that should accompany declaring it.
DTYPE_BYTES: Mapping[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "fp8_e4m3_fnuz": 1,
    "fp8_e5m2_fnuz": 1,
}


def evaluate(expression: Any, query: Mapping[str, Any]) -> float:
    """Evaluates an opmeta `flops`/`elements` declaration against one row's ``q.*`` values.

    The JsonLogic subset those declarations use: ``* + - /`` and ``ceil_div``, over literals and
    ``$q.<name>`` references. Deliberately not the full evaluator -- these are closed-form
    arithmetic, and an expression here reaching for a comparison or a variable outside ``$q.``
    is a declaration that has outgrown what a corpus row can answer.
    """
    if isinstance(expression, str):
        if not expression.startswith("$q."):
            raise UnsupportedExpression(
                f"only $q.* references are resolvable here, got {expression!r}"
            )
        name = expression[3:]
        if name not in query:
            raise UnsupportedExpression(f"row has no column q.{name}")
        return float(query[name])

    if isinstance(expression, (int, float)):
        return float(expression)

    if not isinstance(expression, dict) or len(expression) != 1:
        raise UnsupportedExpression(f"expected a single-operator object, got {expression!r}")

    (operator, operands), = expression.items()
    values = [evaluate(operand, query) for operand in operands]

    if operator == "*":
        product = 1.0
        for value in values:
            product *= value
        return product
    if operator == "+":
        return math.fsum(values)
    if operator == "-":
        return values[0] - values[1]
    if operator == "/":
        if values[1] == 0:
            raise UnsupportedExpression("division by zero in a declaration")
        return values[0] / values[1]
    if operator == "ceil_div":
        if values[1] == 0:
            raise UnsupportedExpression("ceil_div by zero in a declaration")
        return float(math.ceil(values[0] / values[1]))

    raise UnsupportedExpression(f"unsupported operator {operator!r}")


def dtype_width(dtype: str) -> int:
    """Bytes per element, by the name the operation declaration uses."""
    try:
        return DTYPE_BYTES[dtype]
    except KeyError:
        raise UnsupportedExpression(
            f"no byte width known for dtype {dtype!r}; add it to DTYPE_BYTES alongside "
            "declaring it in the operation's .opmeta.json"
        ) from None


def derive_metrics(
    query: Mapping[str, Any],
    time_ms: float | None,
    opmeta: Mapping[str, Any],
) -> dict[str, float | None]:
    """`tflops` and `gbs` for one row, or nulls where there is nothing to derive from.

    Null, not zero, in every absent case. A zero throughput sorts *last* and so is merely wrong;
    a zero *time* would divide into an infinity that outranks every real measurement. Keeping
    the absent case null throughout means no arithmetic here can manufacture a winner.

    `tflops` is null for an operation that declares no `flops`. That is not an omission: a FLOP
    count for layernorm is a convention rather than a fact, and those operations are memory-bound
    anyway, so `gbs` is what ranks them.
    """
    absent: dict[str, float | None] = {"tflops": None, "gbs": None}

    # No measurement, or one that cannot yield a rate. A time of zero is impossible rather than
    # fast, so it is treated as no measurement rather than divided by.
    if time_ms is None or not math.isfinite(time_ms) or time_ms <= 0.0:
        return absent

    seconds = time_ms / 1000.0
    derived: dict[str, float | None] = dict(absent)

    if "flops" in opmeta:
        derived["tflops"] = evaluate(opmeta["flops"], query) / seconds / 1e12

    if "elements" in opmeta:
        elements = evaluate(opmeta["elements"], query)
        width = dtype_width(str(query["dtype"]))
        derived["gbs"] = elements * width / seconds / 1e9

    return derived
