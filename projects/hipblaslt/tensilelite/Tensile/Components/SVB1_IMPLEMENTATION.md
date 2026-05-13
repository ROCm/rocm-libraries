# svb1: Unify build_idmap and build_id_to_category_per_iter

Implementation memo for bead `rocm-libraries-svb1`.

## Scope (post-4up4 / post-hdem)

Per the svb1 bead comment 2026-05-13: the original correctness-drift urgency
is gone — identity no longer consumes category (see hdem `1bf40be09b7`). The
remaining work is a **good-hygiene refactor**:

1. Shared private helper for the categorization rules both factories share.
2. Conformance tests asserting both factories produce identical category
   strings for leaves they share.
3. Sparse-MX gap fix: kwargs added to `build_idmap` so future sparse-MX CMS
   work has a categorization slot ready.

## Shared helper surface

`_shared_category_assignments` (in `Tensile/Components/ScheduleCapture.py`)
is a generator that yields `(category_string, module)` pairs for the
categories BOTH public factories emit. It is the SINGLE source of truth for:

- Category-name string literals: `"GRIncA"`, `"GRIncB"`, `"GRA"`, `"GRB"`.
- Per-iter category-name templates: `f"LRA{subiter}"`, `f"LRB{subiter}"`,
  `f"PackA{subiter}"`, `f"PackB{subiter}"`.
- Ordering of category emission within a subiter.

Each public face still owns its own input-shape resolution and writes
the helper's output into its native target shape:

- `build_idmap` (returns `{category: module}`) calls the helper twice:
  once for the non-per-iter slots (GRIncA/B, GRA/B) and once per `u`
  in `range(num_loop_iter)` for the per-iter slots.
- `build_id_to_category_per_iter` (returns `{id(item): category}`) calls
  the helper for each resolved per-side-per-subiter module, walking
  `flatitems()` to populate the id-keyed map. It uses `tag_module`
  (override semantics) for GRIncA/B/LR/Pack and `tag_module_setdefault`
  (preserve-existing semantics) for GRA/B so a leaf that appears in
  both inc and load inputs retains the more-specific GRIncA/B tag.

## Categories each public factory retains for itself

`build_idmap`-only (CMS schedule pieces; no NLL/NGL counterpart):

- `LWA`, `LWB` (local writes per side)
- `LRSA`, `LRSB`, `LWSA`, `LWSB` (local-read/write swap modules)
- `LCC` (loop counter code)
- `SYNC`, `SNOP` (CMS-arbitrary wait/snop slots)

`build_id_to_category_per_iter`-only:

- Generic `"GR"` fallback (for globalReadCode leaves not already tagged
  by the more-specific GRIncA/B/GRA/B helper output).
- Generic `"LW"` fallback (for localWriteCode leaves).
- `LRMXSA{subiter}`, `LRMXSB{subiter}`, `LRMetadata{subiter}`
  (sparse-MX local-read sub-modules — historically only this factory
  exposed them; svb1 deliverable 3 below adds the matching kwargs to
  `build_idmap` so the schema slot exists symmetrically).

## Sparse-MX gap fix

`build_idmap` gained three optional kwargs (default `None`):

- `LRCodeMXSA`: per-iter list of modules → `LRMXSA{u}` categories.
- `LRCodeMXSB`: per-iter list of modules → `LRMXSB{u}` categories.
- `LRCodeMetadata`: per-iter list of modules → `LRMetadata{u}` categories.

Category-string format (`f"LRMXSA{u}"` etc.) deliberately matches the
template (`"LRMXSA{}".format(subiter)`) `build_id_to_category_per_iter`
already used for `LocalReadDoMXSA_I{iui}` sub-modules, so the two
factories agree on naming when both have populated inputs.

Existing callers (KernelWriter `_loopBody`, CustomSchedule.dispatch) are
unaffected — none pass the new kwargs, and they default to unpopulated.

## Tests added (regression guards)

In `Tensile/Tests/unit/test_idmap_helper.py`:

1. `test_factories_agree_on_shared_categories_subiter0`
2. `test_factories_agree_on_shared_categories_subiter2`
3. `test_build_idmap_accepts_sparse_mx_kwargs` (Cycle 2 RED — TypeError)
4. `test_build_idmap_sparse_mx_kwargs_default_empty`
5. `test_factories_agree_on_sparse_mx_categories_subiter0`
6. `test_factories_agree_on_sparse_mx_categories_subiter3`

## TDD cycle notes

- **Cycle 1 RED unexpectedly passed.** The conformance test for shared
  categories did not fail before the helper extraction — both factories
  already aligned by convention. Surfaced and reframed: the test is a
  regression guard, not a RED→GREEN driver. The shared-helper extraction
  proceeded as a refactor with the test continuing to pass.
- **Cycle 2 was a clean RED→GREEN.** `TypeError: build_idmap() got an
  unexpected keyword argument 'LRCodeMXSA'` → kwargs added.
- **Cycle 3 RED unexpectedly passed.** Sparse-MX category strings already
  matched because Cycle 2 GREEN deliberately reused the same format-string
  pattern as `build_id_to_category_per_iter`. Test kept as a regression
  guard.

Both surprises were surfaced rather than silently massaged. The refactor
is principled (single source of truth for the shared schema) and the
guards mechanically prevent the drift the bead worried about.

## File:line landmarks

- Shared helper: `Tensile/Components/ScheduleCapture.py:998` (`_shared_category_assignments`).
- `build_idmap`: `Tensile/Components/ScheduleCapture.py:1045` (was `:998` pre-svb1).
- `build_id_to_category_per_iter`: `Tensile/Components/ScheduleCapture.py:1178` (was `:1107` pre-svb1).
- Tests: `Tensile/Tests/unit/test_idmap_helper.py` (595 lines, was 270 pre-svb1).
