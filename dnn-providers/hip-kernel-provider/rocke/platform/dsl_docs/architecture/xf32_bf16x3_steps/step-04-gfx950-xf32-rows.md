# Step 4 — the two gfx950 emulated TF32 catalog rows

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4, step 4).

## Contents

- [Why](#why)
- [What changed](#what-changed)
- [The C mirror cannot derive](#the-c-mirror-cannot-derive)
- [Two "not yet" tests became real invariants](#two-not-yet-tests-became-real-invariants)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

[Step 3](step-03-emulation-descriptor.md) added the vocabulary for an emulated
atom but left the catalog with none. This step adds the two gfx950 rows this plan
calls for. After it, a TF32 atom is *selectable* on gfx950 — the lowering still
cannot expand it, which is steps 5 and 6.

## What changed

Two rows appended to the gfx950 MMA list in
[`arch_specs.json`](../../../python/rocke/core/arch/data/arch_specs.json):

| op_id | shape | physical atom |
|---|---|---|
| `mfma_f32_16x16x32_xf32` | 16×16×32 | `mfma_f32_16x16x32_bf16` |
| `mfma_f32_32x32x16_xf32` | 32×32×16 | `mfma_f32_32x32x16_bf16` |

Both `a`/`b` = `tf32`, `c` = `fp32`, `recipe` = `bf16x3`. (This step first landed the
rows as `xf32_bf16x3_f32_*` over an `xf32` selection tag; [step 11](step-11-tf32-operand-abi.md)
moved them onto the base's `tf32` type and these names.) Mirrored in
[`cpp/core/arch/data.cpp`](../../../cpp/core/arch/data.cpp).

New tests:
[`test_xf32_catalog_rows.py`](../../../tests/core/test_xf32_catalog_rows.py) and
`test_emulated_rows_mirror_physical` in
[`test_mma_catalog_lookup.cpp`](../../../tests/core/test_mma_catalog_lookup.cpp).

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

Earlier steps left behind two tests asserting that nothing claimed the new dtype and
that every atom was native. This step is the change that makes both false, so they
were replaced rather than deleted — each with the permanent invariant it was standing
in for. After the move onto the base's TF32 type they read:

- `test_tf32_atoms_are_native_on_gfx942_emulated_on_gfx950_and_absent_elsewhere`
  (blast radius: which arch offers which TF32 atoms, and no other) in
  `test_xf32_catalog_rows.py`;
- `test_emulation_is_confined_to_the_tf32_rows` in `test_mma_emulation_schema.py`
  (every atom that is not TF32 is still native).

`test_enumerate_is_unchanged_for_every_shipped_query` kept passing unmodified once
real emulated rows existed — the emulated rows are reachable only through a TF32
query, and no arch lists a native and an emulated atom for one `(m, n)` for the
native-preferred rule to arbitrate — so only its premise was rewritten.

## Verification

- [`tests/core/test_xf32_catalog_rows.py`](../../../tests/core/test_xf32_catalog_rows.py)
  — rows present with the right shape; recipe and physical atom; frag metadata equal
  to the physical atom; physical atom native; fp32 accumulator and unscaled; TF32
  queries reach both rows; `select_largest_k` picks 16×16×32; which arch offers which
  TF32 atoms; and — the point of the exercise — gfx950 fp32 and bf16 enumeration
  return exactly the same op_id sets as before.
- The C twin, `test_emulated_rows_mirror_physical`, runs under `ctest` with the rest of
  the engine's catalog tests.
- Byte identity at llvm20 **and** llvm22 was 69 GREEN / 1 DRIFT when this step landed,
  the DRIFT being the `gfx942_attention_tiled_3d` cfg[3] divergence, proven
  pre-existing (see [step 10](step-10-byte-identity.md)). No new drift — expected,
  since no emitted kernel selected an emulated atom yet.

## Follow-ups left open

- gfx942 is offered no emulated row. Its native TF32 atoms came with the base, and the
  K-packed bf16 atoms these rows expand into do not exist there (plan §3.2).
- **Closed after review.** The bare-op_id fragment lookups (`_frag_info` in
  Python, `rocke_arch_mma_c_frag_len` in C) originally did not know the emulated
  op_ids, so the bare-string form of `IRBuilder.mma` and the C builder's
  `rocke_b_mma` — which has no other form — both failed with "unknown MMA
  op_id". Both lookups now resolve a logical op_id through its physical atom,
  one hop, via `target._op_id_physical_op_id` /
  `rocke_arch_mma_op_id_physical_op_id`. The frag table itself still holds
  physical atoms only, so the borrowed fragment cannot drift from the one it
  borrows. Pinned by `tests/core/test_mma_frag_tables.py`,
  `tests/core/mma_frag_ssot.cpp` and `tests/core/test_mma_catalog_lookup.cpp`.
