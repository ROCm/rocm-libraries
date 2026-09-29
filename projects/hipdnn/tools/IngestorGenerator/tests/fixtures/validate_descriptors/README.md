# `hipdnn_validate_descriptors` mutation fixtures

Each directory is a complete, standalone generic-kernel-ingestor descriptor bundle
(KMD + UHD + UED + UMD + UDD + KDP with two inline kernels), modeled on the shipped
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv_fwd/`
example. `valid/` is the unmutated baseline; every other directory differs by exactly one
deliberate defect and must make `hipdnn_validate_descriptors <dir>` exit non-zero.

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
