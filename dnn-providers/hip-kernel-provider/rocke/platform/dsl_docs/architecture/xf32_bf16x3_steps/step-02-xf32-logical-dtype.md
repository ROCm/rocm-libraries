# Step 2 — register `xf32` as a logical dtype

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4, step 2).
Ticket AICK-2208.

## Contents

- [Why](#why)
- [The plan named the wrong file](#the-plan-named-the-wrong-file)
- [The rejection guard needed no new code](#the-rejection-guard-needed-no-new-code)
- [What changed](#what-changed)
- [Design notes](#design-notes)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

`xf32` names a *matrix-selection tag*: "fp32 operands, reduced-precision matrix
path". It selects either a native XF32 MMA or the BF16x3 emulation. It is not a
storage format — operands stay fp32 in memory and in registers — so the name has
to be recognised by dtype normalization and the arch catalog while staying out of
everything that implies a representation.

## The plan named the wrong file

Plan §3.1 said to mirror the registration in `cpp/core/ir/core_types.cpp`. That
file is the **IR scalar type system** — the interned singletons (`f32`, `bf16`, …)
that have an LLVM representation. Registering `xf32` there would assert exactly
the thing that is false about it.

The real mirror of [`core/dtypes.py`](../../../python/rocke/core/dtypes.py) is
[`cpp/core/dtypes.cpp`](../../../cpp/core/dtypes.cpp), the dtype alias table.
`fp6e2m3` is the precedent already in the tree: a logical dtype that appears in
the alias table and the arch catalog, and deliberately nowhere in the IR types.
`xf32` takes the same shape. Plan §3.1 has been corrected.

## The rejection guard needed no new code

The plan asked for a guard making `xf32` rejected by the load/store builders.
Both engines already reject it, by construction and without knowing it exists:

- `helpers/io.py::io_ir_type` raises `ValueError` for anything but f16/fp16/bf16.
- `rocke_scalar_by_name` returns `NULL` for any name not in its explicit whitelist.

So the correct work was not to add a check but to **pin the existing behaviour**,
so that a later step cannot make `xf32` loadable without a test going red.

## What changed

| File | Change |
|---|---|
| [`python/rocke/core/dtypes.py`](../../../python/rocke/core/dtypes.py) | `xf32` → `xf32`, `tf32` → `xf32` |
| [`cpp/core/dtypes.cpp`](../../../cpp/core/dtypes.cpp) | same two rows, mirrored |
| [`tests/core/test_xf32_logical_dtype.py`](../../../tests/core/test_xf32_logical_dtype.py) | new — the boundary tests |
| [`tests/core/test_dtypes.py`](../../../tests/core/test_dtypes.py), [`.cpp`](../../../tests/core/test_dtypes.cpp) | two normalization cases each, both engines |

## Design notes

**`tf32` is accepted as an alias.** The ticket is titled in terms of TF32 and CK
uses `tf32` as its tag name, so that is the spelling people will reach for. It
canonicalises to `xf32`, following the existing `fp8` → `fp8e4m3` pattern: accept
the familiar spelling, store one key. Note the emulation is *more* accurate than
real TF32 — the name is inherited vocabulary, not a precision claim.

**The new tests assert a boundary, not a feature.** Four things are pinned: both
spellings normalize; no IR scalar type carries the name; `io_ir_type` rejects it;
and **no catalog atom claims `xf32` yet**. That last one means ordinary
`(fp32, fp32, fp32)` selection is provably unchanged by this commit — it is plan
test 7, pinned early and cheaply, while the answer is still trivially true.

## Verification

- Byte identity, llvm20 and llvm22: 69 GREEN, 1 DRIFT — the same pre-existing
  `gfx942_attention_tiled_3d` drift documented in
  [step 1](step-01-mma-result-dtype.md). No new drift.
- `tests/core` pytest: 366 passed, 22 skipped.
- ctest `rocke_dtypes` (which now carries the new cases) and the arch/MMA targets: pass.
- `ruff check` (no `--fix` — emitter code): clean on both touched files.

Two `test_debug_info.py` failures seen during a full `run_all.py` run were stale
`__pycache__`: bytecode compiled against one source path and reused under another.
Clearing the caches and re-running gives 55 passed. Unrelated to this step, but
worth knowing when running the suite under a different mount path than it was last
byte-compiled at.

## Follow-ups left open

- Nothing selects `xf32` yet; the catalog rows land in steps 3–4.
- The pre-existing `gfx942_attention_tiled_3d` drift is still open and unowned.
