---
name: i1
aliases: [i1, bool, boolean]
family: predicate
bits: [1]
variants:
  - {arch: all, encoding: "LLVM i1; HIP bool"}
status: on develop
related: [platform/python/rocke/core/ir.py, platform/python/rocke/core/lower_llvm.py]
---

# i1 (predicates)

rocKE has no separate `bool` type in its IR; comparisons produce `i1`, and
`select` consumes it. This file covers how predicates meet NaN.

The SDPA dispatch dtype `bool` (hipDNN `BOOLEAN`) is different: it names a
mask tensor stored one byte per element, which a kernel loads as bytes, not
an `i1` value
(`Dtype` in [library/dispatch/sdpa/common.py](../../../library/dispatch/sdpa/common.py)).

## Float comparisons

`fcmp(pred, a, b)` accepts `olt`, `ole`, `ogt`, `oge`, `oeq`, `one`, `ord`
and `uno` ([core/ir.py](../../python/rocke/core/ir.py)). The ordered
predicates are false when either operand is NaN; `uno` is true when either
is NaN, `ord` when neither is. There are no unordered relational predicates
(`ult`, `ugt` ...), so "x is not greater than y, or NaN" is spelled
`lor(fcmp("ole", x, y), fcmp("uno", x, y))`.

## Negation

`lnot` lowers to `xor i1 %x, true` in LLVM IR
(`_op_arith_not` in [core/lower_llvm.py](../../python/rocke/core/lower_llvm.py)), so
`lnot(fcmp("oeq", l, 0))` is true for a NaN `l`. Attention's "row has mass"
test wants exactly that (a NaN row keeps NaN instead of being treated as
empty), and builds it as `fcmp("one", l, 0) OR fcmp("uno", l, 0)` because of
the HIP lowering below
(`lse_row_has_mass` in [_lse_store.py](../../../library/kernels/common/_lse_store.py)).

The HIP debugging lowering emits `bool r = ~x;` for `lnot`
(`_op_arith_not` in [core/lower_hip.py](../../python/rocke/core/lower_hip.py)); `~` on a
`bool` promotes to `int`, so the result is true for both inputs. Avoid `lnot`
on `i1` in code that is also lowered to HIP; use the complementary predicate.

## Policy

Predicates follow LLVM's `fcmp` semantics: ordered predicates are false on
NaN. Code that must treat NaN as a hit says so with `uno`.

## Tests

No dedicated test; the "row has mass" predicate is exercised by the
attention LSE tests listed in [datatype_f32.md](./datatype_f32.md#tests).

## How to decide

- **A test that must treat NaN as "true"**: OR the ordered predicate with
  `uno`.
- **Selecting on NaN**: `select(fcmp("uno", x, x), x, other)` keeps a NaN
  through a clamp or a max ([README](./README.md#nan-and-infinity-in-arithmetic)).
