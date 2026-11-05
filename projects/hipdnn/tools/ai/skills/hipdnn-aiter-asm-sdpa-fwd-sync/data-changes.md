# Tier 0 and tier 1: data changes

These need no engine code. `apply` writes the objects, CSVs, pin and `SOURCE.md`; the build regenerates the kernel registry from the CSVs and repacks the `.kpack` archives.

All paths below are relative to `dnn-providers/hip-kernel-provider/src/`.

## After apply

1. **Reconfigure the build.** `engines/asm_sdpa_engine/CMakeLists.txt` finds CSVs and objects with `file(GLOB_RECURSE)`, which runs only at configure time. Without a reconfigure, an added or removed file is not tracked.
2. **Check tests that name objects.** `tests/engines/asm_sdpa_engine/TestAsmKpackLoading.cpp` and `TestAsmKpackArchive.cpp` extract objects by their archive key, such as `fmha_v3_fwd/MI300/fwd_hd128_bf16_rtne.co`. For every `object_removed` finding, search the tests for the object's name and update them.
3. **Check tests that copy CSV values.** `TestSdpaFwdPlanGridMath.cpp` and `TestSdpaFwdPlanBuilder.cpp` use tile sizes and kernel names from the CSVs. A `row_changed` on `ts_qo` or `ts_kv` changes the launch grid; update the expected values from the new CSV, never from the old test.
4. **Know the coverage test.** `TestSdpaFwdPlanBuilder.IsApplicableAvailableKernels` builds a graph for every registry row of the device's arch and requires the builder to accept it. So on a GPU of that arch, a kept row the engine cannot serve fails the unit tests. That test runs only on a matching GPU; without one, check the gates by reading them.

## Per finding

- **`object_changed`**: AITER rebuilt the kernel. Its name, row and arguments are unchanged, but its numerics and performance may not be. Treat it as unvalidated until the forward GPU tests pass on that arch ([validation.md](validation.md)).
- **`row_added`**: the forward GPU integration test (`integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp`) builds its cases from the generated registry, so the new row is tested automatically on a GPU of that arch. Check whether a cross-provider bundle should now claim it ([validation.md](validation.md)).
- **`object_removed`**: the row is gone from the CSV, so the registry no longer offers it and the engine declines those graphs. Remove any `.support.json` claim under `dnn-providers/integration-tests/integration-test-bundles/` that relied on it, for that arch.
- **`local_differs_from_baseline`**: the tree held bytes that were not those of the recorded commit, so the old provenance was wrong. The update replaces them with the pinned bytes. Mention it in the PR.
- **`row_changed` on tile sizes only**: the grid math reads tile sizes from the registry, so no code changes. Update tests that copied the old values.
