# `hipdnn_validate_descriptors` mutation fixtures

Each directory is a complete, standalone generic-kernel-ingestor descriptor bundle
(KMD + UHD + UED + UMD + UDD + KDP with two inline kernels), modeled on the shipped
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv_fwd/`
example. `valid/` is the unmutated baseline; every malformed directory below differs by
exactly one deliberate defect and must make `hipdnn_validate_descriptors <dir>` exit
non-zero. The role fixtures at the end instead pin which role-bound models the validator
admits, and must agree with the runtime admission for that role.

## `valid/`

The baseline bundle: one engine (`hipkernel:ValidateFixture`), one pack targeting
`gfx942`, two inline kernels with distinct `(block_size, dtype)` metadata tuples.
Expected: exit 0.

## `bad_arch/`

The pack's `arch` list is `["GFX942"]` (uppercase) instead of `["gfx942"]`.

Expected failure: `requireArchList` rejects it at load time — `DescriptorLoader.hpp`'s
`isPlausibleArchBaseId()` requires everything after the `gfx` prefix to be lowercase, so
`GFX942` is not a plausible base id and the whole KDP fails to parse.

**Deliberately not `gfx94`.** `isPlausibleArchBaseId` is a shape check, not an existence
check: it accepts `gfx` followed by any run of `[a-z0-9_-]`, so `gfx94` parses as
well-formed and loads clean. The comment above `requireArchList()` implies otherwise;
`isPlausibleArchBaseId()` is authoritative.

## `dangling_uuid/`

The UED's `metadata` field names a UUID (`9341b3cb-3540-44f6-9066-f3695a3b6a2d`) that no
KMD in the bundle defines (the real KMD keeps its original id,
`46d64d06-18eb-483d-9bb4-94472d32b78d`).

Expected failure: `DescriptorLoader.hpp`'s `resolveDescriptorSets()` looks up the
engine's metadata schema by id and drops the whole engine when it is not found.

## `duplicate_tuple/`

The pack's second inline kernel carries the same completed metadata tuple as the first
(`block_size: 64, dtype: FLOAT`), and neither narrows its own `arch` (both inherit the
pack's `["gfx942"]`), so they occupy one overlapping-arch group.

Expected failure: `KernelIngestorStateManager.hpp`'s `validateAndIndexPacks()`, run inside
`loadValidatedDescriptorSets`'s throwaway `makeStateManager` probe, throws on a
metadata-tuple collision within one overlapping-arch group (`archOverlaps`); the loader
catches it and drops the whole engine.

## `undeclared_knob/`

The UED's `knobs` list names `tile_count`, a field the KMD's `fields` array does not
declare (the KMD only declares `block_size` and `dtype`).

Expected failure: `findUndeclaredKnob` rejects the engine during
`DescriptorLoader.hpp`'s `resolveDescriptorSets()`.

Note: a *declared-but-non-int* knob cannot be a fixture here — `GenericEngine.hpp`'s
`findUndeclaredKnob()` checks name membership only. The non-int-knob drop happens later,
in `GenericPlanBuilder::getCustomKnobs` at plan-build time against a real graph and
device, which this standalone binary cannot reach.

## Role fixtures

`valid/` plus one change to a role-bound UHD. The engine loads in every case; the verdict
is the bound model's `model_checks` entry and the exit status. A kernel-scoped role is
admitted through `UhdKernelHeuristic::tryCreate`; `predict_engine` through the L1 guards
GenericEngine evaluates it with (`uhd::prediction_detail::validateBinding` and `model`).

- `l2_static_order/`: the `sort_kernel_catalog` UHD is `static_order`. Expected: exit 0
  -- declared order is a legal kernel ranking. `model_checks`: `sort_kernel_catalog`
  succeeds.
- `l2_dynamic_features/`: the `sort_kernel_catalog` UHD is a `native` scorer whose
  signature reads `$kernel.block_size` and `$graph.batch`, with the `features_hash` and
  `trained_against` a feature-consuming model needs. Expected: exit 0 without
  `--feature-samples` -- the model loads and is admitted, and extraction, which needs a
  recorded `$graph.batch`, is skipped with a warning (`feature_rows_checked` 0,
  `feature_extraction_skipped` set). `model_checks`: `sort_kernel_catalog` succeeds. With
  a sample covering the engine, extraction runs for each candidate kernel, and a sample
  leaving `$graph.batch` unbound fails the run.
- `l1_static_order/`: `predict_engine` binds a calibrated `time` UHD whose adapter is
  `static_order`. Expected: non-zero exit -- an L1 estimate needs a `tree_data`, `native`
  or `custom_library` model, so the runtime refuses it. `model_checks`:
  `sort_kernel_catalog` succeeds, `predict_engine` fails.
- `l1_native/`: `predict_engine` binds a calibrated `time` UHD with a signature-less
  `native` scorer. Expected: exit 0 -- the symbol resolves through the UHD scorer
  registry the L1 path uses, not the kernel comparator registry. `model_checks`:
  `sort_kernel_catalog` and `predict_engine` both succeed.

`tests/test_round_trip.py` and the `hipdnn_validate_descriptors_<fixture>` CTest cases
(`tools/CMakeLists.txt`) assert these exit statuses and `model_checks` verdicts.
