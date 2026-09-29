# Step 4 — the two gfx950 XF32 catalog rows

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4, step 4).
Ticket AICK-2208.

## Contents

- [Why](#why)
- [What changed](#what-changed)
- [Divergence from the plan: no `"d"` key](#divergence-from-the-plan-no-d-key)
- [The C mirror cannot derive](#the-c-mirror-cannot-derive)
- [Two "not yet" tests became real invariants](#two-not-yet-tests-became-real-invariants)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

[Step 3](step-03-emulation-descriptor.md) added the vocabulary for an emulated
atom but left the catalog with none. This step adds the two gfx950 rows the
ticket asks for. After it, `xf32` is *selectable* — the lowering still cannot
expand it, which is steps 5 and 6.

## What changed

Two rows appended to the gfx950 MMA list in
[`arch_specs.json`](../../../python/rocke/core/arch/data/arch_specs.json):

| op_id | shape | physical atom |
|---|---|---|
| `xf32_bf16x3_f32_16x16x32` | 16×16×32 | `mfma_f32_16x16x32_bf16` |
| `xf32_bf16x3_f32_32x32x16` | 32×32×16 | `mfma_f32_32x32x16_bf16` |

Both `a`/`b` = `xf32`, `c` = `fp32`, `recipe` = `bf16x3`. Mirrored in
[`cpp/core/arch/data.cpp`](../../../cpp/core/arch/data.cpp).

New tests:
[`test_xf32_catalog_rows.py`](../../../tests/core/test_xf32_catalog_rows.py) and
`test_emulated_rows_mirror_physical` in
[`test_mma_catalog_lookup.cpp`](../../../tests/core/test_mma_catalog_lookup.cpp).

## Divergence from the plan: no `"d"` key

Plan §3.2's row snippet spells `"d": "fp32"`. These rows omit it.

D and C coincide for this atom — it accumulates fp32 and produces fp32 — so the
key carries no information. It does carry a cost: `d_dtype` unset means
`MmaOp.result_dtype` falls back to the accumulator, and
`test_shipped_atoms_produce_what_they_accumulate` in
[`test_arch_mma_ssot.py`](../../../tests/core/test_arch_mma_ssot.py) pins exactly
that across the whole catalog. Spelling a redundant `"d"` would have broken a
live invariant in exchange for nothing, and created a second place for the two to
disagree.

Step 1's `d_dtype` field therefore still has no shipped user. That was already
its expected shape: it exists for a *native* XF32 atom (AICK-1714), where D and C
genuinely can differ.

## The C mirror cannot derive

Python derives an emulated row's frag lengths, wave size and layout maps from
`physical_op_id` at catalog-build time, which is what makes drift
unrepresentable there. `data.cpp` is a static table with no build step, so its
two rows *restate* that metadata — `8, 8, 4, 64, &lm_mfma_16x16x32_{a,b,c}` and
`8, 8, 16, 64, &lm_mfma_32x32x16_{a,b,c}`.

That reintroduces exactly the drift the derivation was meant to prevent, so it is
pinned on both sides:

- Python — `test_fragment_metadata_equals_the_physical_atom` compares the
  emulated `MmaOp` field-by-field against the bf16 `MmaOp`.
- C — `test_emulated_rows_mirror_physical` walks every non-native row on every
  arch, finds its physical atom by `op_id`, and compares frag lengths, wave size
  and **layout pointer identity** (the same `lm_*` object, not a copy).

Both also refuse an expansion chain: the physical atom must itself be native, so
one hop always reaches real hardware. The comment on `rocke_mma_op_t` in
[`arch_target.h`](../../../cpp/include/rocke/arch_target.h) was corrected in this
change — as written in step 3 it claimed the C side copied the metadata at
catalog-build time, which a static table cannot do.

## Two "not yet" tests became real invariants

Steps 2 and 3 left behind two tests asserting that nothing claimed `xf32` and
that every atom was native. Step 4 is the change that makes both false, so they
were replaced rather than deleted — each with the permanent invariant it was
standing in for:

- `test_no_shipped_catalog_atom_claims_it_yet` →
  `test_xf32_appears_only_as_an_operand_dtype` (xf32 may be A or B; C and D
  describe values that really exist in registers, and none of them is xf32) plus
  `test_xf32_operands_are_confined_to_gfx950_emulated_rows` (blast radius: only
  gfx950, only emulated, never a mixed-operand row).
- `test_every_shipped_atom_is_native` →
  `test_emulation_is_confined_to_the_xf32_rows`.

`test_enumerate_is_unchanged_when_nothing_is_emulated` kept passing unmodified
once real emulated rows existed — the emulated rows are reachable only through an
xf32 query and there is no native xf32 atom to shadow them — so only its premise
(now stale) was rewritten.

## Verification

- [`tests/core/test_xf32_catalog_rows.py`](../../../tests/core/test_xf32_catalog_rows.py)
  — 9 tests. Rows present with the right shape; recipe and physical atom; frag
  metadata equal to the physical atom; physical atom native; fp32/unscaled/no-D;
  xf32 queries reach both rows; `select_largest_k` picks 16×16×32; no other arch
  offers xf32; and — the point of the exercise — gfx950 fp32 and bf16 enumeration
  return exactly the same op_id sets as before.
- `tests/core` — 391 passed, 22 skipped, 315 subtests. The only tests that moved
  are the two "not yet" pins above.
- C build clean; `ctest` 12/12, including the new mirror check.
- Byte identity at llvm20 **and** llvm22: 69 GREEN / 1 DRIFT, the DRIFT being the
  `gfx942_attention_tiled_3d` cfg[3] divergence proven pre-existing in
  [step 1](step-01-mma-result-dtype.md). No new drift — expected, since no
  emitted kernel selects xf32 yet.

## Follow-ups left open

- gfx942 has no XF32 row. Native gfx942 XF32 is AICK-1714's; this ticket only
  supplies the preference rule that will pick it over emulation when it lands.
- The bare-op_id fragment lookups (`_mma_c_frag_len` in Python,
  `rocke_arch_mma_c_frag_len` in C) do not know the emulated op_ids. That is
  currently harmless — `IRBuilder.mma` prefers the `MmaOp`'s own `c_frag_len`
  whenever an atom object is passed, which is how the xf32 atoms are reached —
  but a caller that passes the bare op_id string would get "unknown MMA op_id".
  Worth revisiting in step 5 when the lowering's actual call shape is settled.
