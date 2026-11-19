# Tier 0 and tier 1: data changes

These need no engine code. `apply` writes the objects, CSVs, pin and `SOURCE.md`; the build regenerates the kernel registry from the CSVs and repacks the `.kpack` archives.

All paths below are relative to `dnn-providers/hip-kernel-provider/src/`.

## After apply

1. **Reconfigure the build.** `engines/asm_sdpa_engine/CMakeLists.txt` lists the CSVs and objects its codegen and packing steps depend on with `file(GLOB_RECURSE)`, which runs only at configure time. `codegen.py` and `pack.py` glob again when they run, but without a reconfigure an added or removed file does not trigger them.
2. **Check tests that name objects.** `tests/engines/asm_sdpa_engine/TestAsmKpackLoading.cpp` extracts objects from the built archives by archive key, such as `fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co`. `TestAsmKpackArchive.cpp` and `TestSdpaModuleCache.cpp` name such keys too. For every `object_removed` finding, search the tests for the object's name and update them.
3. **Check tests that copy tile sizes.** `TestSdpaFwdPlanGridMath.cpp` hard-codes the tile sizes it tests the grid with; `TestSdpaFwdPlanBuilder.cpp` reads `ts_qo` from the registry. A `row_changed` on `ts_qo` changes the launch grid, the `s_Ts` argument and the Ts overflow gate in `isApplicable`; update hard-coded expected values from the new CSV, never from the old test. The forward engine does not read `ts_kv`.
4. **Know the coverage test.** `TestSdpaFwdPlanBuilder.IsApplicableAvailableKernels` builds a graph for every registry row of the device's arch and requires the builder to accept it. So on a GPU of that arch, a kept row the engine cannot serve fails the unit tests. It builds fp8 rows with FNUZ inputs on every arch, so it cannot tell whether a row's fp8 encoding matches the gate. That test runs only on a matching GPU; without one, check the gates by reading them.

## Per finding

- **`object_changed`**: AITER rebuilt the kernel. Its name, row and arguments are unchanged, but its numerics and performance may not be. Treat it as unvalidated until the forward GPU tests pass on that arch ([validation.md](validation.md)).
- **`row_added`**: the forward GPU integration tests (`IntegrationGpuSdpaFwdBf16` in `integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp`) build their cases from the generated registry and check each one numerically, so a new row is tested automatically on a GPU of that arch. Batch-mode rows are checked against the CPU reference, and group-mode rows against the CPU ragged reference, with the fixed per-batch lengths in `K_GROUP_SEQ_LENS_Q` and `K_GROUP_SEQ_LENS_KV`. Check whether a cross-provider bundle should now claim the row ([validation.md](validation.md)).
- **`object_removed`**: the row is gone from the CSV, so the registry no longer offers it and the engine declines those graphs. Remove any `.support.json` claim under `dnn-providers/integration-tests/integration-test-bundles/` that relied on it, for that arch.
- **`local_differs_from_baseline`**: the tree held bytes that were not those of the recorded commit, so the old provenance was wrong. The update replaces them with the pinned bytes. Mention it in the PR.
- **`row_changed` on tile sizes only**: the grid math, the `s_Ts` argument and the Ts overflow gate read `ts_qo` from the registry, so no code changes. Update tests that hard-code the old values.
