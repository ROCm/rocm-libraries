# Phase A — flyDSL kernel end-to-end THROUGH hipDNN (gfx1151): DONE

**Result:** a flyDSL-authored kernel, compiled at build time to a raw HSACO, now
executes correctly through hipDNN's real dispatch-handler contract on this gfx1151
laptop. The gtest passes:

```
[ RUN      ] TestFlydslRawDispatch.LaunchesAFlydslHsacoThroughTheEscapeHatch
[       OK ] TestFlydslRawDispatch.LaunchesAFlydslHsacoThroughTheEscapeHatch (390 ms)
[  PASSED  ] 1 test.
```

It computes `3 + 4 = 7` on device by loading `vadd_flydsl.hsaco` (flyDSL `@kernel`
`vadd`, symbol `vadd_0`) and launching it — but the launch goes through the **actual
hipDNN `IKernelDispatchHandler<Handle>` interface**, driven by the ingestor engine's
own pointwise test fixture (real `MatchContext` / `BoundTokens` / `KernelDefinition` /
uid→pointer `findDeviceBuffer`). This is the "flyDSL running from hipDNN" milestone,
minimal but genuine.

This was done GO-FAST / throwaway. None of it is release-shaped. See "What's real vs.
what's stubbed" below before drawing conclusions.

---

## What proves what

- **Prior standalone proof** (`run_hsaco.cpp`, `run_rmsnorm.cpp`): raw
  `hipModuleLoadData` + `hipModuleGetFunction` + `hipModuleLaunchKernel` on a
  build-time flyDSL HSACO works on gfx1151. That established the *mechanic*.
- **New this session** (`TestFlydslRawDispatchHandler.cpp`): those same three HIP calls
  now sit **inside** a custom `IKernelDispatchHandler<Handle>` — `prepare()` loads the
  module, `launch()` resolves device buffers by uid and launches — and the object is
  invoked exactly the way hipDNN's kernel-ingestor engine invokes a native dispatch
  handler. So the escape hatch is proven *in situ*, not just standalone.

## The key architectural finding (why this is even possible)

`buildIngestorKernelCode(...)` (the switch that throws "no kernel source adapter" on
`HSACO_FILE` / `ROCKE_BUILDER`) is called **only from inside a handler's own
`prepare()`** — never eagerly by the engine. So a UDD author's handler can ignore
source-kind resolution entirely and do its own `hipModuleLoadData`. The interface even
documents itself as the **"Native escape hatch for a UDD"**
(`IKernelDispatchHandler.hpp`). We did not need to fill the `HSACO_FILE` source-kind
stub, add a descriptor JSON, or produce a kpack to get a flyDSL kernel running through
the dispatch path.

---

## Files added/changed on this branch

- **NEW** `dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/TestFlydslRawDispatchHandler.cpp`
  - Local `FlydslRawDispatchHandler : IKernelDispatchHandler<Handle>` +
    `PreparedFlydsl : PreparedDispatch` (owns the loaded `hipModule_t`, unloads in dtor).
  - `prepare()`: read `vadd_flydsl.hsaco` from disk → `hipModuleLoadData` →
    `hipModuleGetFunction("vadd_0")`.
  - `launch()`: `findDeviceBuffer` on `INPUT_A_UID/INPUT_B_UID/OUTPUT_UID` → pack
    `{void* out; void* a; void* b;}` (kernarg layout out@0/a@8/b@16) →
    `hipModuleLaunchKernel(grid 1,1,1 / block 1,1,1)`.
  - Reuses the pointwise fixture (`GraphFixture`, `buildPointwiseGraph`, `matchesGraph`,
    `makeKernel`, uid constants) so the handler is exercised through the same surface as
    the built-in pointwise handler.
  - Guarded by `#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR`.
- **CHANGED** `dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/CMakeLists.txt`
  - Added the new file to `target_sources(hip_kernel_provider_tests ...)` (one line).
- **HSACO artifact** (pre-existing, unchanged): `flydsl_poc_scratch/vadd_flydsl.hsaco`
  (referenced by **absolute path** in the test — throwaway shortcut, see caveats).

No production source (engine, handler registry, source-kind switch, descriptor loader)
was modified. The test is additive and ingestor-gated.

---

## How to reproduce (exact commands that worked)

Toolchain: system cmake 3.28.3 + Ninja, ROCm SDK at `~/rocm-sdk/current`, gfx1151, 32 cores.

1. **One-time host dep** (kpack's Python needs these; system python3 is PEP-668 managed):
   ```
   /usr/bin/python3 -m pip install --user --break-system-packages \
     --trusted-host pypi.org --trusted-host files.pythonhosted.org \
     "zstandard>=0.20.0" msgpack
   ```
2. **Configure** (note: `ROCM_PATH` must be OUT of the env and passed via `-D`, or the
   toolchain warns/misdetects; kpack python dir points at the local rocm-systems tree):
   ```
   env -u ROCM_PATH cmake --preset hip-kernel-provider -G Ninja \
     -DROCM_PATH=/home/brpepers/rocm-sdk/current \
     -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
     -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=/home/brpepers/rocm-systems/shared/kpack/python
   ```
3. **Build** the test binary:
   ```
   env -u ROCM_PATH cmake --build build --target hip_kernel_provider_tests -j 32
   ```
4. **Run**:
   ```
   ./build/bin/hip_kernel_provider_tests --gtest_filter='TestFlydslRawDispatch.*'
   ```

Cold configure ~a few seconds after deps resolved; cold build ~4–5 min (228 objects);
incremental rebuild of just this file + link ~seconds.

---

## Gotchas hit (and the fix), for the morning

1. **kpack has "no skip path" at configure.** With `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`
   the descriptor-packaging subdir requires `rocm_kpack`. Even though our escape-hatch
   test never touches descriptors/kpack, **configure won't complete without it.** Fixed
   by pointing at the local tree: `-DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=~/rocm-systems/shared/kpack/python`
   (this path already exists on the box). Alternative offered by the build:
   `-DHIPKERNELPROVIDER_KPACK_ALLOW_FETCH=ON` (avoided — corp TLS).
2. **kpack import failed for the build's interpreter.** It resolved the tree but
   `/usr/bin/python3` lacked `zstandard`/`msgpack`. Fixed by the pip install in step 1.
   (Could instead `-DPython3_EXECUTABLE=<a python that already has them>`.)
3. **`ROCM_PATH` in environment fights the toolchain.** `cmake/toolchains/rocm-clang.cmake`
   explicitly warns and can misdetect. Use `env -u ROCM_PATH ... -DROCM_PATH=...`.
4. **clang-tidy is `-warnings-as-errors` in this tree.** First build of the test failed
   ONLY on tidy nits (constexpr identifier case must be UPPER_CASE; every parameter must
   be named or `/*commented*/`; no C-style arrays → `std::array<void*,5>` for the launch
   config; locals `const` where possible). Everything else compiled first try. Fixed all
   and the incremental rebuild was clean.

---

## What's real vs. what's stubbed (READ before over-claiming)

REAL:
- flyDSL → HSACO at build time, zero Python/flyDSL at runtime.
- Execution through the genuine `IKernelDispatchHandler<Handle>` `prepare`/`launch`
  contract, with real fixture-built `MatchContext`/`BoundTokens` and uid→pointer
  resolution via the SDK's `findDeviceBuffer`.
- Correct numerical result on device (gfx1151).

STUBBED / SHORTCUT (throwaway):
- The handler is **instantiated directly in the test**, not registered in the
  `DispatchRegistry` and selected by a graph match + heuristic. A full "hipDNN picks it"
  path would register a pack (matcher + score + this dispatch symbol) so the engine
  resolves it by name the way `TestPointwiseAddDispatch...DispatchStaysResolvableAcrossContainerLifetimes`
  does. Mechanically identical launch; the missing part is *selection*, not *execution*.
- HSACO path is a **hardcoded absolute path**, and the kernel is a 1-element `vadd`
  standing in for the pointwise-add fixture — not RMSNorm/attention.
- We reused the pointwise ADD graph/uids purely as a convenient carrier. No new
  descriptor JSON or kpack was produced (deliberately — kpack ruled out of scope).

---

## Suggested next steps (for review, not yet done)

1. **Selection, not just execution.** Register a tiny flyDSL pack (matcher + heuristic +
   dispatch symbol) so the ingestor engine *chooses* the flyDSL handler by graph match,
   closing the last gap to "hipDNN picks flyDSL." This is the honest completion of
   Phase A and is small — clone the pointwise pack registration.
2. **Best-of-breed selection (the actual overarching goal).** Register flyDSL alongside
   rocKE and AITER-ASM handlers for the *same* op and let the heuristic/score pick the
   winner per shape. That's the real "pick best-of-breed among flyDSL/rocKE/AITER"
   demonstration; this Phase A only proves flyDSL can *sit in the ring*.
3. **Phase B (gfx950):** author an attention kernel in the low-level `flir` API (flyDSL
   ships none), compile build-time to HSACO, and drop it into a dispatch handler like
   this one to race vs rocKE and AITER ASM.

## Decisions I made autonomously (flagged for correction)
- Used the **local kpack python tree** instead of `ALLOW_FETCH` (avoids network/TLS).
- Installed `zstandard`/`msgpack` into the **user site** with `--break-system-packages`
  (throwaway box; reversible via `pip uninstall`). If you'd rather isolate, point
  `-DPython3_EXECUTABLE` at a venv instead.
- Kept the handler **unregistered/direct-instantiated** to prove execution fast; did NOT
  build the registry/selection path (item 1 above) — that felt like the right stopping
  point for "keep moving but don't over-build overnight."
