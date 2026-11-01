# Lane: GPU architecture handling (`ARCH`)

Find every place the change decides by GPU architecture, and check that the decision is right on every arch it reaches and
exercised on the arch it targets. Trigger: a changed gfx name, device-name test, arch list, wave-size or instruction assumption,
or an arch-keyed data file.

## Where an arch decision lives

Grep the diff and the tree for each token the change adds or removes (`git grep -n 'gfx9\|gfx1[012]\|"gfx'` scoped to the areas below).

- Solver gates: `IsApplicable` device checks (`ctx.GetStream().GetDeviceName()` compared with `==` or `StartsWith(name, "gfx90")`);
  shared predicates `IsXdlopsSupport` (`src/include/miopen/solver/implicitgemm_util.hpp`), `static_ck::IsComposableKernelSupportedHardware`
  (`static_ck_common.hpp`), `mlir::IsMlirSupportedHardware` (`mlir_common.hpp`), `GfxHasMissing*Intrinsics`, `IsTF32Supported`, `IsGfx9ConsumerApu`
  (`target_properties.hpp`), the fusion gate in `src/include/miopen/fusion/utils.hpp`, and the arch checks in `ai_heuristics.cpp`.
- Target properties: `src/target_properties.cpp` (`GetDeviceNameFromMap`, `InitDbId`, xnack and sramecc parsing; overrides
  `MIOPEN_DEVICE_ARCH`, `MIOPEN_DEBUG_ENFORCE_DEVICE`), `Handle::GetDbBasename` (hard-coded gfx11/gfx12 list and CU-count suffix),
  `GetWavefrontWidth`, `GetMaxComputeUnits`, `GetMaxHardwareComputeUnits`.
- Kernels: asm `.amdgcn.gfx_generation_number` / `_minor` ladders in `src/kernels/*.s` and `*.inc`, `__GFX8__`/`__GFX9__` macros
  (`miopen_warp_size.hpp`), CK headers under `src/kernels/static_composable_kernel/`, MLIR and HIP build options.
- Build lists: `ALL_GPU_DATABASES`, `KDB_BZ2_FILES` and `install_gpu_databases` in `CMakeLists.txt`; `_CK_SUPPORTED_ARCHS`
  (`src/ck_impl/CMakeLists.txt`); `MIOPEN_CK_ENABLE_TF32_TARGETS` and `MIOPEN_CK_USE_OCP_FP8_TARGETS` (`projects/miopen/cmake/MiopenCkCompileDefs.cmake`);
  `src/hipconv/CMakeLists.txt`; `GPU_TARGETS` in CI and the `MIOPEN_TEST_GFX*` options in `test/CMakeLists.txt`.
- Data: perf/find DB and model file names `<gfx><CU>` (`gfx90878.db.txt.bz2`), `<arch>.tn.model`, `<arch>_<solver>_metadata.tn.model`, `*.kdb.bz2.dvc`.
- Tests and CI: `Gpu` enum, `GetAllKnownDevices` (`test/gtest/gtest_common.{hpp,cpp}`), `IsTestSupportedByDevice`, `exclude_gpu` and `ex_gpu_*`
  labels in `test/gtest/test_categories.yaml`, `.github/workflows/component-ci-miopen.yml`, the `therock-*ci*.yml` family inputs.
- Docs: `docs/install/install.rst`, `docs/reference/data_type_support.rst` and `docs/data/reference/precision-support.yaml`.

## Checks

1. **New arch support:** list every entry above that names a sibling arch (grep the nearest existing arch, e.g. `gfx950` or `gfx942`) and
   mark which the change updated. A name present in one list and absent from another that the same code path reads (solver gate says
   yes, DB install list or `GetAllKnownDevices` says no) is `IMPORTANT`, `BLOCKING` when the arch then selects a solver that cannot run.
2. **Scope:** a change meant for one arch must not alter others. Widening `==` to `StartsWith`, adding a prefix that also matches
   `gfx1030`/`gfx103x`, `gfx115x`, or reordering an `if` chain changes arches the author did not test. Enumerate the arches each new predicate matches.
3. **Wave size:** wave64 on gfx8/gfx9, wave32 on gfx10+ (`MIOPEN_WARP_SIZE`, `GetWavefrontWidth`). Flag hard-coded 64, `% 64` tiling,
   LDS or reduction sizing that ignores it, and wave-size features reaching the heuristics (`lgbm_pick.cpp`).
4. **Instructions and features:** MFMA/XDLops (gfx908, gfx90a, gfx942, gfx950), WMMA (gfx11/gfx12), dot2 and packed instructions,
   buffer atomics (`support_amd_buffer_atomic_fadd`), fp8 flavor (fnuz vs OCP), TF32, bf16 intrinsics. Each use needs a gate that
   excludes arches lacking it, and the fallback arches must still get a working solver or a clean `IsApplicable == false`.
5. **Feature suffixes:** `:xnack`, `:sramecc`; `target.isXnackEnabled()`; `dbId` suffixes `_sramecc`/`_nosramecc`/`_xnack`. A new DB or model
   for a part with these suffixes must follow `InitDbId`, and gfx906/gfx908 legacy naming must stay unchanged.
6. **Data keyed by arch:** a new DB file name equals `GetDbBasename()` for that part (CU count, hex above 64 CUs, gfx11/gfx12 list). A model
   or DB added for an arch needs its loader entry and install lists.
7. **Executed coverage** ([../policy.md](../policy.md) *Review rules*: green is not coverage): `component-ci-miopen.yml` builds for one
   target with no GPU test; GPU tests run only in TheRock families. For each arch the change names, find a run that executed the
   affected tests on it (job log shows the cases, not a skip or an `ex_gpu_*` exclusion). The `CPU_UnitTestConvSolverDevApplicability*`
   tests run `IsApplicable` against a `MockHandle` for every `GetAllKnownDevices` entry, on any host: they check that the gate agrees
   with the test's declared `Gpu` mask, never a kernel result.

## Without hardware

Decide by reading. If an arch the change names has no executed run, file `IMPORTANT`, Kind `test-gap` (Kind `process` when no CI lane
covers the arch at all), and ask the author for: the arch and CU count, the exact `MIOpenDriver` or gtest filter run, the log excerpt of the executed cases, and
whether `MIOPEN_DEVICE_ARCH`-based no-GPU checks were used (they prove only applicability). The finding stays `UNVERIFIED`; file an
Approval request (command, arch, shapes, duration, where it runs) for the device run. Never run GPU work from this lane.

## Handoffs

`CORR:` applicability logic beyond the gate. `TESTB:` arch-specific test missing. `BUILD:` CMake arch lists, packaging and DB install. `DOCS:` support tables. `SIBLING:` the same gate in direction or backend counterparts. `PERF:` per-arch speed claims.
