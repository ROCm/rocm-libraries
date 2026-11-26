# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: evaluate a flat list of scalar integer IR ops on Python ints, with LLVM's poison rules.

For host tests of integer helpers whose exact bits matter (Philox, counter and
index math): the result is what the GPU computes, and an op whose ``nsw`` /
``nuw`` flag the lowering emits but whose exact result leaves its range raises
:class:`IntPoison` instead of returning the wrapped value. ``arith.select``
and f32 constants pass values through, so a helper that masks floats can be
checked for which operand it picks; ``arith.bitop3_b32`` is evaluated by its
truth table.
"""

from __future__ import annotations

from typing import Iterable, Mapping

__all__ = ["IntPoison", "evaluate_int_ops"]


class IntPoison(AssertionError):
    """An op with a wrap flag overflowed: LLVM would produce poison."""


_WIDTH = {"i1": 1, "i32": 32, "i64": 64}
_ARITH = {
    "arith.add": lambda x, y: x + y,
    "arith.sub": lambda x, y: x - y,
    "arith.mul": lambda x, y: x * y,
}


def _signed(v: int, bits: int) -> int:
    return v - (1 << bits) if v >= (1 << (bits - 1)) else v


def evaluate_int_ops(builder, inputs: Mapping[str, object], outputs: Iterable) -> list:
    """Run ``builder.kernel.body.ops`` (no regions) and return the values of ``outputs``.

    ``inputs`` maps parameter names (with or without the leading ``%``) to ints,
    taken mod ``2**bits`` of the parameter's type (so a negative i32 is its u32
    word), or floats. Integers come back unsigned.
    """
    widths = {
        "%" + p.name.lstrip("%"): _WIDTH.get(p.type.name) for p in builder.kernel.params
    }
    env = {}
    for name, v in inputs.items():
        key = name if name.startswith("%") else "%" + name
        if key not in widths:
            raise KeyError(f"{name}: no such parameter")
        bits = widths[key]
        env[key] = v & ((1 << bits) - 1) if isinstance(v, int) and bits else v

    def val(v):
        return env[v.name]

    for op in builder.kernel.body.ops:
        if op.name == "func.return":
            continue
        if op.regions:
            raise NotImplementedError(f"{op.name}: ops with regions")
        res = op.results[0]
        a = val(op.operands[0]) if op.operands else None
        c = val(op.operands[1]) if len(op.operands) > 1 else None
        if op.name == "arith.constant":
            out = op.attrs["value"]
            if res.type.name in _WIDTH:
                out = int(out) & ((1 << _WIDTH[res.type.name]) - 1)
            env[res.name] = out
            continue
        if op.name == "arith.select":
            env[res.name] = c if a else val(op.operands[2])
            continue
        bits = _WIDTH[res.type.name]
        mask = (1 << bits) - 1
        if op.name in ("arith.add", "arith.sub", "arith.mul"):
            apply = _ARITH[op.name]
            exact_u = apply(a, c)
            if op.attrs.get("wrap") == "nuw":
                if not 0 <= exact_u <= mask:
                    raise IntPoison(f"{op.name} nuw overflow: {a}, {c}")
            else:
                sa, sc = _signed(a, bits), _signed(c, bits)
                exact_s = apply(sa, sc)
                if not -(1 << (bits - 1)) <= exact_s < (1 << (bits - 1)):
                    raise IntPoison(f"{op.name} nsw overflow: {sa}, {sc}")
            out = exact_u & mask
        elif op.name in ("arith.div", "arith.mod"):
            # sdiv / srem: truncating; a zero divisor or INT_MIN / -1 is
            # undefined behaviour.
            sa, sc = _signed(a, bits), _signed(c, bits)
            if sc == 0 or (sa == -(1 << (bits - 1)) and sc == -1):
                raise IntPoison(f"{op.name} undefined: {sa} / {sc}")
            quot = abs(sa) // abs(sc) * (1 if (sa < 0) == (sc < 0) else -1)
            rem = sa - quot * sc
            out = (quot if op.name == "arith.div" else rem) & mask
        elif op.name == "arith.xor":
            out = a ^ c
        elif op.name == "arith.and":
            out = a & c
        elif op.name == "arith.shl":
            out = (a << c) & mask
        elif op.name == "arith.lshr":
            out = a >> c
        elif op.name == "arith.bitop3_b32":
            t, d = op.attrs["table"], val(op.operands[2])
            # Bit i of the result is table bit (a_i << 2 | b_i << 1 | d_i).
            out = 0
            for i in range(32):
                idx = ((a >> i) & 1) << 2 | ((c >> i) & 1) << 1 | ((d >> i) & 1)
                out |= ((t >> idx) & 1) << i
        elif op.name == "arith.umul_hi_i32":
            out = (a * c) >> 32
        elif op.name == "arith.zext":
            out = a
        elif op.name == "arith.trunc":
            out = a & mask
        elif op.name == "arith.cmp":
            ib = _WIDTH[op.operands[0].type.name]
            pred = op.attrs["pred"]
            sa, sc = _signed(a, ib), _signed(c, ib)
            out = int(
                {
                    "lt": sa < sc,
                    "le": sa <= sc,
                    "gt": sa > sc,
                    "ge": sa >= sc,
                    "eq": a == c,
                    "ne": a != c,
                    "ult": a < c,
                }[pred]
            )
        else:
            raise NotImplementedError(op.name)
        env[res.name] = out
    return [env[v.name] for v in outputs]
