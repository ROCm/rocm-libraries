# Resisting functions / lines — `Tensile/Common/DataType.py`

**Nothing resisted.** The suite reaches **100.00% line and 100.00% branch**
(161 stmts, 0 miss, 24 branches, 0 partial). This file records the items that
*looked* like they might resist (defensive raises, an asserted dead branch,
generated comparison methods) and why each was in fact reachable or out of
scope — so a future reader does not mistake the clean result for missing rigour.
New file in the per-target dir per the add-only rule.

## Nominally-defensive paths that turned out reachable

| Code | Line(s) | Why it could have resisted | How it was reached |
|---|---|---|---|
| `__init__` else → `raise RuntimeError` | 281-282 | Requires an input that is none of enum/int/str/DataType. | `test_init_invalid_type_raises` passes `float`/`None`/`tuple`/`list`. |
| `DataType.lookup[value.lower()]` KeyError | 278 | The str path normally gets a valid key. | `test_init_unknown_str_key_raises` passes a non-existent name; the dict access raises `KeyError`. |
| `toDevice` else → `assert 0` | 298 | Only hit for a non-`"HIP"` language. | `test_to_device_non_hip_asserts` parametrizes `HLSL`/`OCL`/`""`/`"hip"` (case-sensitive). |
| `__eq__` / `__lt__` `NotImplemented` arms | 464-465, 470-471 | Need a non-`DataType` right-hand operand. | `test_eq` compares against str/int/None; `test_lt_vs_non_datatype_raises` asserts the resulting `TypeError`. |
| `_populateLookupTable` index-mismatch raise | 484-485 | The real table is self-consistent (index == enum value), so this never fires at import. | `test_populate_lookup_table_index_mismatch_raises` feeds a 1-row synthetic list with `Double` (value 1) at index 0. |
| `_populateLookupTable` duplicate-key raise | 489 | The real table has no duplicate `char`/`enum` keys. | `test_populate_lookup_table_duplicate_key_raises` feeds two rows sharing `char='S'` at indices 0/1. |
| `zeroString` `vectorWidth > 1` branch | 308-309 | Width suffix only emitted for vw > 1. | `test_zero_string` parametrizes vw 1 (no suffix) and 2/4 (suffix). |

## Logically-dead but line-covered (no Miss)

| Code | Line | Note |
|---|---|---|
| `isNone` → `return self.value == None` | 438-439 | `self.value` is always an `int` row index (the constructor never stores `None` — a `None` input raises in `__init__`), so the equality is always `False`. The **line executes** (covered, returns `False`); only the `True` *outcome* is unreachable, and `==` is an expression, not a counted branch — so this costs neither a Miss nor a partial branch. Pinned in the predicate matrix (`isNone` row, all `False`). |

## Out of scope for line coverage (not part of this file)

| Item | Why not counted |
|---|---|
| `__le__` / `__gt__` / `__ge__` / `__ne__` | Synthesized by `@functools.total_ordering` / Python from `__lt__`+`__eq__`; their code lives in the stdlib, not in `DataType.py`, so they do not appear in the module's statement count. Still exercised behaviourally (`test_lt_ordering` uses `<=` and `>`, `test_eq` uses `!=`) to confirm the derived operators agree with the snapshotted `__lt__`/`__eq__`. |
| `_is8bitFloat` `@lru_cache` wrapper | The cache wrapper line (29) runs at import; the function body (30-40) is covered via the `is8bitFloat` predicate row of the matrix. |

## Determinism notes

- Every output is a pure function of the dtype + arguments — no paths,
  timestamps, versions, or env coupling — so **no normalisation was needed**.
- The dtype set and the `is*` predicate roster are both **discovered from the
  module** (table iteration + `inspect.getmembers`) and pinned by their own
  snapshots, so a future table/predicate change surfaces as a snapshot diff
  rather than silently shifting coverage.
- `__hash__` is pinned structurally (equal objects hash equal; hash matches
  `getAttributes()`; distinct values differ) rather than by snapshotting the
  raw integer, which—though stable for small-int tuples in CPython—is opaque.
