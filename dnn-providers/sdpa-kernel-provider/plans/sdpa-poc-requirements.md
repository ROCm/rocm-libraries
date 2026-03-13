# SDPA Kernel Provider — POC Requirements

## Objective

Prove that AITER ASM Flash Attention v3 forward kernels can be extracted, adapted without an AITER dependency, and launched through hipDNN's plugin pipeline to produce correct SDPA forward-pass attention output on MI300X.

## Scope

| Dimension | POC Boundary |
|-----------|-------------|
| Platform | gfx942 (MI300X) only |
| Data type | BF16 only |
| Direction | Forward pass only |
| Kernel variant | hd128, non-causal, no dropout, no paged attention, no ALiBi, no variable-length sequences, batch mode (not group mode) |
| Starting point | Existing plugin skeleton (handle, container, context, settings, entry point, build system, test infra) |
| Branch model | Feature branch off `rocm-libraries` |

## Constraints

- **No AITER dependency.** The plugin must not `find_package(aiter)` or link against any AITER library. All necessary code is copied and adapted into the provider.
- **Plugin model.** The kernels are exposed through hipDNN's standard plugin SDK API — not through a standalone executable or test harness.
- **Existing skeleton.** Implementation builds on the merged skeleton; no restructuring of the container/handle/context/settings types.

---

## Functional Requirements

### FR-1: Engine and Plan Builder Registration

The plugin must register at least one engine with one plan builder so that hipDNN can discover and dispatch SDPA forward work to it.

| Detail | Specification |
|--------|---------------|
| Engine class | `SdpaKernelEngine` implementing `IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>` |
| Engine ID | Registered via `HIPDNN_REGISTER_ENGINE` in `SdpaKernelContainer` |
| Plan builder | `AsmSdpaFwdPlanBuilder` added to the engine via `engine->addPlanBuilder()`; engine registered via `engineManager.addEngine()` |
| Discovery | `hipdnnEnginePluginGetAllEngineIds()` returns at least one ID |
| Existing tests | The two `TestSdpaKernelContainer` tests that assert zero engines (`CopyEngineIdsReturnsZeroEngines`, `CopyEngineIdsWithBufferReturnsZero`) must be updated to assert one engine |

### FR-2: Graph Pattern Matching

`AsmSdpaFwdPlanBuilder::isApplicable()` must correctly accept graphs matching the POC configuration and reject all others.

**Accept when all of the following hold:**
- Single-node graph with `NodeAttributes::SdpaAttributes`
- Q/K/V tensors are BF16
- Q tensor is rank-4 with `dims[3] == 128` (head dimension)
- `causal_mask == false` and `causal_mask_bottom_right == false` (these fields are deprecated; also check that no `left_bound` / `right_bound` are set, which is the non-deprecated equivalent)
- `dropout_probability` is null or 0.0
- No ALiBi mask (`alibi_mask == false`)
- No padding mask (`padding_mask == false`)
- No variable-length sequences (`seq_len_q` tensor not set — check `!has_value()` on the optional tensor UID, not `== 0`)
- No attention mask / bias tensor (`attn_mask` tensor not set)
- No paged attention (`page_table_k` / `page_table_v` tensors not set)
- Running on gfx942 (requires `hipGetDeviceProperties` device query — this is a device-level check, not a graph attribute)

**Reject otherwise**, returning zero applicable engine IDs.

### FR-3: ASM Kernel Loading and Dispatch

The forward pass requires a single kernel launch. The plugin must load a pre-compiled `.co` binary and launch it.

| Detail | Specification |
|--------|---------------|
| Kernel | Forward attention |
| `.co` file | `fwd_hd128_bf16_rtne.co` (from AITER `hsa/gfx942/fmha_v3_fwd/MI300/`) |
| Load mechanism | `hipModuleLoad` + `hipModuleGetFunction` |
| Launch mechanism | `hipModuleLaunchKernel` with `HIP_LAUNCH_PARAM_BUFFER_POINTER` |
| Module lifecycle | Load on plan build, unload on plan/context destruction |
| Arguments | Populated `fmha_fwd_v3_args` struct with Q/K/V input pointers, O/LSE output pointers, strides, scale, GQA ratio, and control flags |

### FR-4: Correct Computation

The GPU attention output (O) must match the CPU reference within BF16 tolerance for at least the three test configurations below. Verification uses hipDNN's `IntegrationGraphVerificationHarness` — the harness automatically runs the graph on CPU via `CpuReferenceGraphExecutor` and compares against GPU results using registered tolerances.

| Config | B | H_q | H_kv | S_q | S_kv | D | Description |
|--------|---|-----|------|-----|------|---|-------------|
| 1 | 1 | 1 | 1 | 256 | 256 | 128 | Small MHA |
| 2 | 2 | 8 | 8 | 512 | 512 | 128 | Medium MHA |
| 3 | 1 | 8 | 2 | 256 | 256 | 128 | GQA (ratio 4) |

**Tolerance:** `atol = 1e-2`, `rtol = 1e-2`. BF16 has ~7 mantissa bits; the tiled forward kernel accumulates in float but converts to BF16 for output, introducing rounding divergence from the naive CPU implementation.

**Validation:** O tensor — GPU vs CPU reference within tolerance, verified via `registerValidator()` + `verifyGraph()`.

### FR-5: Workspace Size Reporting

The forward plan must report the correct workspace size. The forward pass is a single kernel that reads Q/K/V and writes O and Stats directly — no intermediate buffers are needed.

| Detail | Specification |
|--------|---------------|
| Total workspace | 0 bytes (no intermediate buffers required) |
| Pre-build query | `IPlanBuilder::getMaxWorkspaceSize()` returns 0 |
| Post-build query | `IPlan::getWorkspaceSize()` returns 0 |

### FR-6: Clean Rejection

The plugin must return zero applicable engines (not crash, not throw) for graphs that do not match the POC configuration.

| Rejection case | Expected behavior |
|----------------|-------------------|
| Non-SDPA graph (e.g., BatchNorm) | Zero applicable engines |
| Backward SDPA graph (`SdpaBackwardAttributes`) | Zero applicable engines |
| Forward SDPA with FP16 tensors | Zero applicable engines |
| Forward SDPA with `causal_mask == true` | Zero applicable engines |
| Forward SDPA with `hd != 128` | Zero applicable engines |
| Forward SDPA on non-gfx942 hardware | Zero applicable engines |
| Forward SDPA with dropout | Zero applicable engines |
| Forward SDPA with attention mask / bias tensor | Zero applicable engines |
| Forward SDPA with paged attention | Zero applicable engines |

---

## Engineering Requirements

### ER-1: No AITER Dependency

The build must not reference AITER as an external dependency.

| Check | How to verify |
|-------|---------------|
| No `find_package(aiter)` | Grep CMake files |
| No AITER include paths | Grep `target_include_directories` and `#include` directives |
| No AITER link targets | Grep `target_link_libraries` |
| Self-contained code | All adapted code lives under `src/asm/` with hipDNN plugin logging, `int32_t` replacing `ck_tile::index_t`, direct HIP calls replacing CK tile wrappers |

### ER-2: hipDNN Plugin SDK Conformance

The plugin must implement the full engine plugin API contract.

| API Function | Implementation |
|-------------|----------------|
| `hipdnnEnginePluginGetAllEngineIds` | Returns registered engine ID(s) |
| `hipdnnEnginePluginGetApplicableEngineIds` | Delegates to `isApplicable()` |
| `hipdnnEnginePluginGetEngineDetails` | Serializes `EngineDetails` FlatBuffer |
| `hipdnnEnginePluginGetWorkspaceSize` | Delegates to `getMaxWorkspaceSize()` |
| `hipdnnEnginePluginCreateExecutionContext` | Calls `buildPlan()`, stores plan in context |
| `hipdnnEnginePluginGetWorkspaceSizeFromExecutionContext` | Delegates to `plan.getWorkspaceSize()` |
| `hipdnnEnginePluginExecuteOpGraph` | Calls `plan.execute()` |
| `hipdnnEnginePluginDestroy*` | Proper cleanup, no leaks |

Already handled by the skeleton via `EnginePluginImpl.inl` — the new engine/plan/plan-builder types must conform to `IEngine`, `IPlanBuilder`, and `IPlan` interfaces.

### ER-3: Build System Integration

New source files integrate into the existing CMake targets without restructuring.

| Item | Requirement |
|------|-------------|
| New sources | Added to `sdpa_kernel_plugin_impl` OBJECT target in `src/CMakeLists.txt` |
| `.co` binary | Forward `.co` file installed via `install(DIRECTORY asm_kernels/ ...)`. Note: no `.co` installation infrastructure exists in any provider today — this CMake support must be created from scratch. |
| `.co` path | Compile definition `AITER_ASM_DIR` set to install prefix; runtime override via `HIPDNN_AITER_ASM_DIR` env var |
| Unit tests | New test file added to existing `sdpa_kernel_plugin_tests` target (in `src/tests/`) |
| Integration tests | New test file added to existing `sdpa_kernel_plugin_integration_tests` target (in `src/integration_tests/`) |

### ER-4: Code Quality

Code must pass the project's existing quality gates.

| Gate | Tool | Standard |
|------|------|----------|
| Formatting | `clang-format` | Project `.clang-format` (via `ninja check_format`) |
| Static analysis | `clang-tidy` | Project `.clang-tidy` (via `ninja tidy`) |
| Compiler warnings | clang | `-Werror -Wconversion -Wsign-conversion` |
| Explicit casts | Code review | No implicit narrowing; use `static_cast<>` for all type conversions |

### ER-5: Kernel Arg Struct Verification

The forward kernel arg struct must be verified at compile time to catch layout drift.

| Struct | Check |
|--------|-------|
| `fmha_fwd_v3_args` (forward kernel) | `static_assert` on `sizeof` matching the AITER-defined size |

The struct must use `__attribute__((packed))` with SGPR-aligned padding matching the GPU kernel ABI.

**Note on provenance:** This is an AITER-specific per-kernel ABI struct name. CK's forward attention code may define a separate unified struct used for CPU-side launching — it provides semantic reference for field names and purposes, but not the binary layout. The actual GPU kernel ABI struct must be reverse-engineered from AITER source (`mha_fwd.h` / `mha_fwd.cu`), using CK only as a cross-reference.

### ER-6: AITER Provenance Documentation

All code and binaries copied from AITER must be traceable to their source.

| Item | Required documentation |
|------|----------------------|
| AITER commit hash | The exact commit from which files were copied |
| Source file paths | AITER repo paths for each copied file |
| Adaptations | Summary of what was changed (e.g., "replaced `ck_tile::index_t` with `int32_t`") |
| Location | Comment block at the top of each adapted file, plus a summary in the provider's README |

---

## Research Deliverables

These are analysis outputs (documents or sections in a design doc) that inform the post-POC roadmap. They do not require code implementation.

### RD-1: AITER Kernel Selection Analysis

Document how AITER selects which attention kernel to dispatch for a given problem configuration.

| Topic | Expected content |
|-------|-----------------|
| Dispatch logic | How does AITER's `mha_fwd.cu` / Python layer choose between forward kernel variants? |
| Decision tree | Which parameters drive selection (dtype, head dim, causal, group mode, platform, rounding mode)? |
| CSV metadata | How does the codegen CSV (`fmha_fwd.csv`) define the available variants? |
| Kernel count | How many forward `.co` variants exist for each platform (MI300 vs MI308)? |
| MI300 vs MI308 | How does AITER select between MI300 and MI308 subdirectories? What distinguishes them (CU count tuning)? |

### RD-2: CK and ASM Kernel Relationship

Document the relationship between CK (Composable Kernel) tile-based kernels and AITER's hand-written ASM kernels.

| Topic | Expected content |
|-------|-----------------|
| When CK is used | Which attention configurations use CK tile kernels vs. ASM kernels? |
| Forward coverage | Does CK have forward attention kernels? What is their coverage vs. ASM? |
| Fallback behavior | Does AITER fall back to CK when no ASM kernel matches? |
| Performance delta | Qualitative comparison (ASM is faster for specific configs; CK provides broader coverage) |
| Dependency implications | What does using CK kernels mean for build time, binary size, and dependencies? |

### RD-3: Post-POC Roadmap Input

Provide the data needed to create an incremental plan for expanding beyond the POC.

| Topic | Expected content |
|-------|-----------------|
| Priority variants | Which additional forward kernel variants to add next (causal? hd192? FP8? gfx950? group mode?) |
| Backward pass | Assessment of AITER's backward ASM kernels (3-kernel pipeline: odo, main bwd, dq_convert) and effort to integrate |
| Build time impact | Estimated impact of adding CK tile kernels as a fallback |
| Coverage gaps | What forward SDPA configurations would remain uncovered after adding ASM kernels? |
| Maintainability | Risks of maintaining copied ASM binaries vs. building from AITER source |
| MI308 support | Effort to support MI308 kernel variants alongside MI300 |

---

## Dependencies and Assumptions

| Dependency | Type | Notes |
|------------|------|-------|
| MI300X (gfx942) hardware | Test infrastructure | Required for integration tests; unit tests run on any platform |
| AITER repository access | One-time | Needed to extract `.co` binary, kernel arg struct, and reference the source files |
| Plugin SDK version | API stability | Assumes current `EnginePluginImpl.inl` and interface versions |
| Existing plugin skeleton | Starting point | Handle, container, context, settings, entry point, build system are complete |
| ROCm toolchain | Build dependency | HIP runtime and ROCm clang compiler |
| AITER forward kernel ABI | Reverse engineering | The forward arg struct (`fmha_fwd_v3_args`) must be reverse-engineered from AITER source (`mha_fwd.h` / `mha_fwd.cu`) |

---

## Out of Scope

The following are explicitly excluded from this POC:

| Item | Rationale |
|------|-----------|
| Backward pass | Multi-kernel pipeline (odo + main bwd + dq_convert); significantly more complex; deferred to post-POC |
| FP16 / FP8 data types | Additional kernel variants and arg handling; deferred to post-POC |
| Head dimensions other than 128 | Requires additional `.co` binaries (e.g., hd192x128 variants) |
| Causal masking | Requires different `.co` variants (`fwd_hd128_bf16_causal_rtne.co`) and mask-aware arg setup |
| Dropout | Requires seed/offset tensors, dropout mask, and additional kernel arg fields |
| Attention mask / bias tensors | Requires bias tensor handling and additional kernel args |
| Paged attention | Requires page table tensors and different kernel family |
| ALiBi masking | Requires bias tensor handling |
| Variable-length sequences | Requires group-mode `.co` variants and sequence-length tensors |
| Group mode | Requires group-mode `.co` variants (`fwd_hd128_bf16_rtne_group.co`) |
| Non-gfx942 platforms | Requires platform-specific `.co` binaries |
| MI308 kernel variants | Requires MI300 vs MI308 runtime selection logic |
| Performance benchmarking | POC validates correctness only; performance is a post-POC concern |
| Multi-kernel selection logic | Only one kernel variant; no dispatch logic needed |
| Production error handling | POC uses basic error checking; robust error reporting is post-POC |
| Backward-compatible API surface | No public API commitments from the POC |

---

## Acceptance Criteria

| ID | Criterion | Verification Method |
|----|-----------|-------------------|
| FR-1 | Plugin reports at least one engine ID | `hipdnnEnginePluginGetAllEngineIds()` returns count >= 1; updated unit tests pass |
| FR-2 | `isApplicable()` returns true for BF16 hd128 non-causal forward SDPA on gfx942 | Unit tests with matching `SdpaAttributes` graph configurations |
| FR-3 | Forward kernel loads and launches without HIP errors | Integration test completes without `hipModule*` failures |
| FR-4 | GPU O matches CPU reference (atol=1e-2, rtol=1e-2) for all 3 configs | `IntegrationGpuSdpaKernelFwdBfp16` parameterized test suite passes via `verifyGraph()` |
| FR-5 | Workspace size equals 0 | Unit test asserts `getWorkspaceSize()` returns 0 |
| FR-6 | Non-matching graphs return zero applicable engines | Unit tests with backward SDPA, non-SDPA, FP16, causal, non-gfx942 graphs |
| ER-1 | No AITER references in CMake or includes | Grep-based check; CI build without AITER installed |
| ER-2 | Full plugin lifecycle works end-to-end | Integration test exercises create -> set stream -> get engines -> build -> execute -> destroy |
| ER-3 | `ninja` builds without errors; `.co` file is installed | Build succeeds; `.co` file present at install prefix |
| ER-4 | `ninja check_format`, `ninja tidy` pass; zero `-Werror` warnings | CI quality gates |
| ER-5 | `static_assert` on forward kernel arg struct size compiles | Build succeeds |
| ER-6 | Each adapted file has AITER provenance comment | Code review |
| RD-1 | AITER kernel selection analysis document exists | Document review |
| RD-2 | CK/ASM relationship document exists | Document review |
| RD-3 | Post-POC roadmap input document exists | Document review |

---

## Reference Documents

| Document | Path |
|----------|------|
| POC task breakdown | `dnn-providers/sdpa-kernel-provider/plans/sdpa-poc-tasks.md` |
| hipDNN forward frontend | `projects/hipdnn/frontend/include/hipdnn_frontend/node/SdpaFpropNode.hpp` |
| Forward attributes | `projects/hipdnn/frontend/include/hipdnn_frontend/attributes/SdpaAttributes.hpp` |
| Forward FlatBuffer schema | `projects/hipdnn/data_sdk/schemas/sdpa_attributes.fbs` |
| Integration test harness pattern | `dnn-providers/miopen-provider/integration_tests/IntegrationGraphVerificationHarness.hpp` |
| AITER forward arg struct | `aiter/csrc/include/mha_fwd.h` |
| AITER forward `.co` kernels | `aiter/hsa/gfx942/fmha_v3_fwd/MI300/` |
| AITER forward CSV metadata | `aiter/hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv` |
