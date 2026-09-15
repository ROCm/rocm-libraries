# `hipdnn_validate_descriptors` mutation fixtures

Each directory is a complete, standalone generic-kernel-ingestor descriptor bundle
(KMD + UHD + UED + UMD + UDD + KDP with two inline kernels), modeled on the shipped
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/conv_fwd/`
example. `valid/` is the unmutated baseline; every other directory differs from it by
exactly one deliberate defect and is expected to make
`hipdnn_validate_descriptors <dir>` exit non-zero.

## `valid/`

The baseline bundle: one engine (`hipkernel:ValidateFixture`), one pack targeting
`gfx942`, two inline kernels with distinct `(block_size, dtype)` metadata tuples.
Expected: exit 0.

## `bad_arch/`

The pack's `arch` list is `["GFX942"]` (uppercase) instead of `["gfx942"]`.

Expected failure: `requireArchList` rejects it at load time — `isPlausibleArchBaseId`
requires everything after the `gfx` prefix to be lowercase
(`DescriptorLoader.hpp:634-643`), so `GFX942` is not a plausible base id and the whole
KDP fails to parse.

**Deliberately not `gfx94`.** `isPlausibleArchBaseId` is a shape check, not an
existence check: it accepts `gfx` followed by any run of `[a-z0-9_-]`, so `gfx94`
parses as a well-formed (if fictional) base id and loads clean —
`DescriptorLoader.hpp:634-643`. The comment at `DescriptorLoader.hpp:666` (`archSupports
is a case-sensitive exact compare, so "", " gfx942", or "gfx94" would otherwise
silently disable the pack...`) wrongly implies `gfx94` is among the values this check
catches; the comment is stale and the code at `:634-643` is authoritative. Fixing that
comment is a candidate follow-up, out of scope for this fixture set.

## `dangling_uuid/`

The UED's `metadata` field names a UUID (`9341b3cb-3540-44f6-9066-f3695a3b6a2d`) that no
KMD in the bundle defines (the real KMD keeps its original id,
`46d64d06-18eb-483d-9bb4-94472d32b78d`).

Expected failure: `resolveDescriptorSets` looks up the engine's metadata schema by id
and drops the whole engine when it is not found (`DescriptorLoader.hpp:1660-1668`).

## `duplicate_tuple/`

The pack's second inline kernel has been given the same completed metadata tuple as the
first (`block_size: 64, dtype: FLOAT`), and neither kernel narrows its own `arch` (both
inherit the pack's `["gfx942"]`), so the two occupy the same overlapping-arch group.

Expected failure: `validateAndIndexPacks`, run inside `loadValidatedDescriptorSets`'s
throwaway `makeStateManager` probe, throws on a metadata-tuple collision within one
overlapping-arch group (`KernelIngestorStateManager.hpp:232` ff., using `archOverlaps`);
the loader catches the exception and drops the whole engine.

## `undeclared_knob/`

The UED's `knobs` list names `tile_count`, a field the KMD's `fields` array does not
declare (the KMD only declares `block_size` and `dtype`).

Expected failure: `findUndeclaredKnob` rejects the engine during `resolveDescriptorSets`
(`DescriptorLoader.hpp:1669-1678`).

Note: a *declared-but-non-int* knob is not usable as a fixture here — `findUndeclaredKnob`
(`GenericEngine.hpp:36-51`) only checks name membership, never type. The non-int-knob
drop happens later, in `GenericPlanBuilder::getCustomKnobs` at plan-build time against a
real graph and device, which this standalone binary cannot reach.
