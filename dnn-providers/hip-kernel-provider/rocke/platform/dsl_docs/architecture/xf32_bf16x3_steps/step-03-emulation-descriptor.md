# Step 3 — the `emulation` descriptor and the native-preferred rule

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4, step 3).

## Contents

- [Why](#why)
- [What changed](#what-changed)
- [Design notes](#design-notes)
  - [Fragment metadata is derived, never restated](#fragment-metadata-is-derived-never-restated)
  - [One predicate, four C entry points](#one-predicate-four-c-entry-points)
  - [The all-native fast path](#the-all-native-fast-path)
  - [Zero-init keeps the catalog rows untouched](#zero-init-keeps-the-catalog-rows-untouched)
- [Verification](#verification)
- [Follow-ups left open](#follow-ups-left-open)

## Why

An emulated atom is a catalog row with no hardware encoding behind it: the
lowering expands it into repeated issues of a *different*, real atom. The catalog
schema had no way to say that. This step adds the vocabulary — and the selection
rule that keeps an emulated atom from ever displacing hardware that can do the
job directly.

No catalog row uses it yet. The rows land in
[step 4](step-04-gfx950-xf32-rows.md); step 3 is schema only.

## What changed

Python — [`core/arch/target.py`](../../../python/rocke/core/arch/target.py):

- `MmaEmulationRecipe` — a `str` enum; `BF16X3` is its only member.
- `MmaEmulation` — frozen dataclass of `(recipe, physical_op_id)`, coercing the
  recipe to the enum and rejecting an empty `physical_op_id` in `__post_init__`.
- `MmaOp.emulation: Optional[MmaEmulation] = None`, plus an `is_native` property
  for the common "does the hardware do this" read.
- `_build_mma_emulation(spec)` — load-time validation: unknown keys, missing
  keys, unknown recipe, and a `physical_op_id` absent from `_MMA_FRAGMENT_INFO`
  are all `ValueError` at catalog construction.
- `_prefer_native(ops)` — drops an emulated atom when a native atom of the same
  `(m, n, k)` is in the same result list. `MmaCatalog.enumerate` returns through it.
- Both new names are re-exported from
  [`core/arch/__init__.py`](../../../python/rocke/core/arch/__init__.py).

C++ — mirrored in the same change:

- [`include/rocke/arch_target.h`](../../../cpp/include/rocke/arch_target.h):
  `rocke_mma_emulation_recipe_t` (with `ROCKE_MMA_EMULATION_NONE = 0`), two
  trailing `rocke_mma_op_t` members, and the two accessor declarations.
- [`cpp/core/arch/query.cpp`](../../../cpp/core/arch/query.cpp):
  `rocke_arch_mma_op_is_native`, `rocke_arch_mma_op_physical_op_id`,
  `rocke_ati_op_shadowed_by_native`, and `rocke_ati_op_selectable`.

## Design notes

### Fragment metadata is derived, never restated

An emulated row carries no fragment lengths, wave size, or layout maps of its
own. `_build_mma_op` looks up `_MMA_FRAGMENT_INFO` under
`emulation.physical_op_id` when an emulation descriptor is present, and under
`op_id` otherwise. Duplicating a `_MMA_FRAGMENT_INFO` row for the logical atom
would have worked too — and would have created a pair of rows free to drift
apart. Deriving makes the drift unrepresentable.

That is also why an unregistered `physical_op_id` is a hard load error: without
the check the row would silently inherit zero-length fragments.

### One predicate, four C entry points

Python funnels `has_shape`, `select_largest_k`, and `op_for_shape` through
`enumerate`, so filtering inside `enumerate` covers all of them. C has four
independent loops over `cat->ops`, each previously calling
`rocke_ati_op_matches` directly. All four now call `rocke_ati_op_selectable`
(`matches && !shadowed_by_native`), so no entry point can forget the rule.

### The all-native fast path

`_prefer_native` returns its argument list *identically* (`assertIs`-tested) when
every op is native, and the C `shadowed_by_native` returns `false` immediately
for a native op without scanning. Every query that predates XF32 therefore takes
a path that cannot have changed behaviour — which is what makes the "selection is
unchanged" claim checkable rather than asserted.

### Zero-init keeps the catalog rows untouched

The two new struct members are appended last, so C aggregate zero-init gives the
68 positional rows in [`cpp/core/arch/data.cpp`](../../../cpp/core/arch/data.cpp)
`ROCKE_MMA_EMULATION_NONE` and a null `physical_op_id` for free — no row edits.
The comment at the head of that table now records the constraint, including the
one that bites in step 4: C++ forbids mixing designated and positional
initializers in a single braced-init-list, so a row that needs to set an
emulation field must spell out every field positionally.

## Verification

- [`tests/core/test_mma_emulation_schema.py`](../../../tests/core/test_mma_emulation_schema.py)
  — 14 tests, 102 subtests, GREEN. Covers descriptor parsing and typing,
  inherited fragment metadata, all five load-time rejections, the four
  `_prefer_native` behaviours, and two whole-catalog guards: every shipped atom
  is native, and `enumerate` output is equal to the same query with the
  comparator patched out.
- `tests/core` — 380 passed, 22 skipped.
- C build clean; `ctest` 12/12 passed.
- Byte identity at llvm20 **and** llvm22: 69 GREEN / 1 DRIFT, the one DRIFT being
  the `gfx942_attention_tiled_3d` cfg[3] divergence, proven pre-existing (see
  [step 10](step-10-byte-identity.md)). No new drift at either flavor.

## Follow-ups left open

- `MmaEmulationRecipe` has one member. A second recipe (say an fp8-based split)
  would want the expansion cost — issue count — to become a catalog field rather
  than knowledge baked into the lowering's `switch`.
- `_prefer_native` compares on `(m, n, k)` only. If a native and an emulated atom
  ever differ in scale-block granularity at the same shape, that comparison needs
  to widen.
- The same full-shape key means a native atom of a *different* K does not shadow
  an emulated one, and `select_largest_k` then prefers the larger K. No arch lists
  both for one `(m, n)`, a test pins that, and the rule has to be re-keyed on
  `(m, n)` before one does ([step 11](step-11-tf32-operand-abi.md)).

## Later changes

Step 11 adds three load-time checks to this schema: the recipe fixes the operand
dtype, the physical atom must be a native row of the *same* arch, and an emulated
TF32 atom's operand fragments are readable from the catalog by op_id. The result-
dtype field the first draft of this step relied on was removed.
