# Step 1 — separate the MMA result dtype (D) from the accumulator (C)

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4, step 1).
Ticket AICK-2208.

## Contents

- [Why](#why)
- [What the step turned out to be](#what-the-step-turned-out-to-be)
- [What changed](#what-changed)
- [Design notes](#design-notes)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

An MMA atom has an accumulator input (C) and a result (D). The catalog schema
conflated the two because every atom it ships produces the dtype it accumulates.
The BF16x3 emulation needs them stated independently: the emulated atom consumes
bf16 operands and threads a single fp32 accumulator across three MMAs, so a row
must be able to say what it *produces* without that being implied by what it
*accumulates* — even in the common case where the two coincide.

This is a prerequisite: steps 3 and 4 cannot add emulated catalog rows until the
schema can express them.

## What the step turned out to be

The plan sized this as a schema change plus a mechanical edit of every catalog
row. It is not. All 33 shipped `op_id`s have D == C, so no existing row needs a
new value — only the *ability* to spell one is missing.

That reframing is what makes the change small: the work is adding an optional
key and its two accessors, not touching 68 initializer rows.

## What changed

| File | Change |
|---|---|
| [`python/rocke/core/arch/target.py`](../../../python/rocke/core/arch/target.py) | `MmaOp.d_dtype` field, `MmaOp.result_dtype` property, optional `"d"` key in `_build_mma_op`, `_op_id_d_dtype()` lookup |
| [`cpp/include/rocke/arch_target.h`](../../../cpp/include/rocke/arch_target.h) | trailing `d_dtype` struct member + two declarations |
| [`cpp/core/arch/query.cpp`](../../../cpp/core/arch/query.cpp) | `rocke_arch_mma_op_d_dtype`, `rocke_arch_mma_op_id_d_dtype` |
| [`cpp/core/arch/data.cpp`](../../../cpp/core/arch/data.cpp) | comment only — records that the field is zero-init |
| [`tests/core/test_arch_mma_ssot.py`](../../../tests/core/test_arch_mma_ssot.py) | `TestOpIdDDtype`, 4 tests |

Absent `d` means "same as `c`". Read the value through `MmaOp.result_dtype` /
`rocke_arch_mma_op_d_dtype`, never off the raw field — the raw field is `None`/`NULL`
on every shipped atom and answers the wrong question.

## Design notes

**Zero churn on the C catalog.** The new struct member is appended *last*, after
the scale fields, rather than placed next to `c_dtype`. C aggregate initialization
zero-fills members the initializer list does not reach, so all 68 positional
catalog rows keep their existing meaning and pick up `d_dtype = NULL` for free.
Placing it beside `c_dtype` would have shifted every positional initializer.

This is safe under `-Werror` because the build already sets
`-Wno-missing-field-initializers`; the same mechanism is what the trailing scale
fields rely on.

**Field order in Python is forced, not chosen.** A defaulted dataclass field may
not precede a required one, so `d_dtype` sits after `op_id` rather than next to
`c_dtype`. The two orderings therefore differ between the engines, which is fine —
the C side is positional and the Python side is by keyword.

**The `op_id` lookup keeps the catalog contract.** `_op_id_d_dtype` mirrors
`_op_id_c_dtype`: first-hit-wins across arches, and raises on cross-arch drift
rather than silently preferring one arch's answer.

Nothing on an emission path reads `d_dtype` yet. This step is metadata only.

## Verification

- Byte identity, llvm20 and llvm22: 69 GREEN, 1 DRIFT.
- The one DRIFT (`gfx942_attention_tiled_3d`, config 3) reproduces **identically**
  on a pristine tree at the parent commit with none of these files present. It is
  pre-existing and not caused by this change. It remains open and unowned here.
- `tests/core` pytest: pass. `ruff check` (no `--fix` — emitter code): no new findings.

## Follow-ups left open

- The pre-existing `gfx942_attention_tiled_3d` drift needs its own triage; this
  step neither caused nor fixed it.
- Plan §3.1 still describes CK's `tf32_t` as a pure selection tag. That is
  incomplete: `ck_tile::tfloat32_t` is a real 32-bit storage type that masks the
  low fraction bits, and is a different thing from the gfx950 atoms tagged `tf32_t`
  that actually run bf16x3. Worth correcting in the plan.
