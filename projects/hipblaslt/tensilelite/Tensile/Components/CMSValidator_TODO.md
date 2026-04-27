# CMS Validator — Remaining Test Failures

After wiring `derive_pack_must_start_after` as the primary `must_start_after` path
(commit `23d8627a22`), 6 tests fail because mock idMap registers don't model VW>1
swap pack interleaving or multi-group pack interactions.

The infrastructure for transitioning to real idMaps exists (`generate_real_idmap`,
`kernel_to_solution_config`, `subset_id_map`, `isa_infrastructure` fixture) but
the per-test transition has unresolved complexity around mapping arbitrary test
kernel configs to valid CMS-registered Solution configs.

## Failing tests

### Category A: VW>1 swap pack interleaving (4 tests)

These tests use `VectorWidthA=2 or 4` which triggers `VSwapB32` swap packs that
transpose registers across the T0/X0 register arrays. The mock register layout
in `_make_mock_swap_packs` doesn't model this interleaving, so
`derive_pack_must_start_after` finds incorrect LR dependencies for CVT0 packs
that should depend on swap-touched registers.

- `test_ValidatePack.py::TestValidatePackTF32MultipleGroups::test_failing_two_groups_fully_interleaved`
- `test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4SwapPacks::test_swap_depends_on_specific_lrs_vw4[DU64]`
- `test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4SwapPacks::test_swap_depends_on_specific_lrs_vw4[DU32_ForceUnroll]`
- `test_ValidatePack.py::TestValidatePackTF32MFMA4x4x4SwapPacks::test_swap_depends_on_specific_lrs_vw4[DU32_NoForceUnroll]`

**Fix paths:**
1. **Transition these tests to real idMaps** (preferred). Each test class needs
   a class-scoped real idMap (via `_get_real_idmap` lazy helper using the
   already-implemented `generate_real_idmap`). Tests subset the real idMap to
   match their custom optSchedule via `subset_id_map`.
2. **Implement VW>1 mock register interleaving** — re-implements the kernel
   writer's `dsReadConvTable` and T/X array logic in the mocks. Brittle.

### Category B: TF32 128x128x64 with VW=4 (2 tests)

- `test_CustomSchedule.py::TestCustomScheduleTF32::test_schedule_128x128x64[False-False-True-1-4]`
- `test_CustomSchedule.py::TestCustomScheduleTF32::test_schedule_128x128x64[False-False-False-1-4]`

These call the real `_get_schedule_128x128x64_TF32` which produces a real CMS
schedule, but the test calls `isValid` with mock idMap. The mock can't model
the VW=4 swap interleaving used by this 128x128x64 schedule.

Sample failure: `Code path 0: PackA0 @ idx=8 issued too early, must be issued
after idx=9 (because of LRA0 issued @ idx=4)`.

**Fix path:** Transition `_make_context` in `test_CustomSchedule.py` to use
`generate_real_idmap` instead of `make_mock_id_map`.

## Transition complexity (why this is unresolved)

The straightforward transition path was attempted (see git history) but
revealed:

1. **Test kernel configs don't directly produce valid Solutions.** The base
   kernel from `create_base_kernel()` has `MIWaveTileA=2, MIWaveTileB=2,
   GlobalReadVectorWidthA=0, MacroTile0=0` — a minimal config designed for
   mock-based validation, not for kernel writer use. `kernel_to_solution_config`
   needs to derive sensible defaults that produce a CMS-registered kernel.

2. **Layout mismatch.** Most CMS schedules are registered for TN layout, but
   the base kernel defaults to NN. Forcing TN works for `test_ValidatePack`
   tests (they don't depend on the layout) but `test_CustomSchedule` tests
   verify specific layout behaviors.

3. **Many TF32 datatype tests use FP32 DataType + UseF32XEmulation.** The
   base test kernel for "TF32" tests has FP32 DataType with TF32 emulation
   flags. `kernel_to_solution_config` needs to preserve this combination.

4. **Test schedule subsetting needs to handle SYNC/SNOP carefully.** The real
   idMap's SYNC/SNOP entries are the production schedule's SWaitCnt objects
   with specific `dscnt` values. Tests use their own `syncCode` with
   different `dscnt`. `subset_id_map` already replaces SYNC/SNOP from the
   test's syncCode/snopCode arguments — but tests calling `isValid` directly
   (not via `validate()`) need to pass these explicitly.

5. **Test classes vary in setup pattern.** Most use `setup_method` (auto-called
   by pytest), but some use parametrized `_config` fixtures that re-trigger
   setup. The lazy `_get_real_idmap` approach handles both, but the
   `cms_validation_base.py` changes need to be carefully designed to not break
   the `_inject_isa` fixture ordering.

## What was implemented (and committed)

- `kernel_to_solution_config` helper in `cms_test_utils.py`
- `subset_id_map` helper in `cms_test_utils.py`
- `_frozen_config_key` helper in `cms_test_utils.py`
- `isa_infrastructure` fixture probes both gfx950 and gfx1151
- `generate_real_idmap` in `cms_test_utils.py`
- `writer._last_id_map` side-channel in `customMainLoopSchedule`
- `derive_pack_must_start_after` wired as primary path (RAW + WAR)
- `_hook_up_middle_16_pairs` extracted for TF32 24-group pair constraint
- 21 register-tracing unit tests in `test_register_tracing.py`
- `TestValidatePackTF32MFMA4x4x4SwapPacks` converted from `setUp` → `setup_method`

## What remains for stage 09 completion

1. **Resolve the 6 failing tests** (this TODO).
2. **Delete dead positional code** once tests pass:
   - `_hook_up_packs_bf16`
   - `_hook_up_packs_f32` (keep `_hook_up_middle_16_pairs`)
   - `_hook_up_packs_f32_mfma`
   - Hardcoded `pack_dependencies` dicts
3. **Fix Known Issues 4 & 5** before deleting `set_lr_needed_by_for_VMFMA`
   and `_set_pack_needed_by`:
   - `set_lr_needed_by_from_mfma_operands` doesn't propagate through pack
     chains (LR→Pack→MFMA)
   - `set_pack_needed_by_from_mfma_operands` doesn't trace through
     intermediate packs (CVT0→MFMAPack→CVT1→MFMA)
