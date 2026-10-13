# Step 7 — nothing that predates XF32 moved

Part of [XF32 via BF16x3 MMA emulation](../xf32_bf16x3_emulation_plan.md) (§4,
step 7).

## Contents

- [What is actually at risk](#what-is-actually-at-risk)
- [Why a differential test, not a table](#why-a-differential-test-not-a-table)
- [What is pinned](#what-is-pinned)
- [The C engine twin](#the-c-engine-twin)
- [Negative controls](#negative-controls)
- [Verification](#verification)

## What is actually at risk

Steps 3 and 4 did two things to the catalog: added rows, and added a preference
rule (`_prefer_native`) that runs on *every* query, not just xf32 ones. Every
GEMM, attention and convolution instance in the tree picks its atom out of this
catalog, and several take `select_largest_k` or the first `enumerate` match — so
a perturbed answer is not a failed lookup, it is a silently different kernel
that still compiles and still runs.

Two distinct failure modes follow, and the step guards both:

1. **Leak** — an emulated row reachable from a query that predates it.
2. **Perturbation** — the comparator dropping or reordering a *native* atom.

The second is the subtler one. A leak needs a dtype-filter mistake; a
perturbation needs only the shape test in `_prefer_native` to be slightly wrong,
and it shows up on arches that have no emulated rows at all.

## Why a differential test, not a table

The obvious test writes down the expected `op_id` set per query. That rots: the
next native atom added to any arch makes it fail for a reason that has nothing
to do with XF32, and whoever fixes it updates the table without re-deriving
whether the change was legitimate.

So the test is differential instead. Per arch it builds a *stripped* catalog —
the live rows minus the emulated ones, which reconstructs the catalog exactly as
it stood before step 4 — and asks both catalogs the same question. Adding a
native atom updates both sides at once and the test stays green; a leak or a
perturbation updates only one and it goes red. The queries are generated from
the shipped atoms rather than listed, so coverage tracks the catalog too.

## What is pinned

[`tests/core/test_xf32_selection_regression.py`](../../../tests/core/test_xf32_selection_regression.py),
over **every** known arch and **every** pre-existing atom:

- all four resolving entry points — `enumerate`, `has_shape`,
  `select_largest_k`, `op_for_shape` — answer identically on both catalogs, for
  the atom's full operand contract;
- the same for the scale-free spelling instances actually use, which is the
  widest net a caller can cast;
- `ArchTarget.supports_dtype_combo`, the arch-level predicate that gates whole
  instances;
- `by_op_id` still resolves each pre-existing atom to the *same object*;
- no non-xf32 query returns an emulated atom, stated directly rather than left
  as a consequence of the differential;
- an fp32 query never reaches an xf32 atom — the one confusion the emulation
  invites. XF32 is opt-in; asking for fp32 gets the native fp32 MFMA or nothing,
  never a silently lower-precision emulation;
- the comparator is inert on an all-native catalog, which is *why* no arch
  without emulated rows could have moved.

A raised ambiguity counts as an answer rather than being swallowed:
`select_largest_k` and `op_for_shape` refuse an ambiguous query instead of
picking arbitrarily, and a new row turning a formerly unique query ambiguous is
exactly the regression being hunted.

## The C engine twin

The Python catalog is built from `arch_specs.json`; the C one is a hand-written
static table, so the two can drift independently and a Python-only guard leaves
the engine that actually ships in the provider ungated. Two tests were added to
[`tests/core/test_mma_catalog_lookup.cpp`](../../../tests/core/test_mma_catalog_lookup.cpp)
(ctest `rocke_mma_catalog_lookup`).

There is no stripped catalog on the C side — the table is static. It is not
needed: every pre-existing row is native, so "the catalog as it stood before"
is just the native subset, and the expected answer is computed in the test
rather than tabulated. `enumerate` is checked on **count and order**, because
callers that take the first match would otherwise move silently.

Coverage is every native row on every arch — the whole catalog less the emulated
rows.

## Negative controls

A regression test that cannot fail is decoration, so both failure modes were
induced and observed to go red:

- **Perturbation** — `_prefer_native` wrapped to reverse its output: 3 failures,
  on gfx90a, gfx942 and gfx950, from the inertness check. The differential pairs
  stay green here by construction, since a global patch moves both catalogs
  together — which is precisely why the inertness check is a separate assertion
  and not folded into the differential.
- **Leak** — the emulated rows relabelled `a_dtype="fp32"`: 9 failures across
  four of the test methods, all on gfx950.

## Verification

- `tests/core/test_xf32_selection_regression.py` is green, and `ctest` runs its C
  twin as `rocke_mma_catalog_lookup`, which reports how many pre-existing native atoms
  it found unmoved.
- No emission changed in this step, so byte-identity is untouched by
  construction; the gate result from
  [steps 5+6](step-05-bf16x3-lowering.md) still stands. The current gate results are
  in [step 10](step-10-byte-identity.md) and [step 11](step-11-tf32-operand-abi.md).
