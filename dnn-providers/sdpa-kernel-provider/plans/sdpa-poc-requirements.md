# SDPA Kernel Provider — POC Requirements

## Objective

Prove that AITER ASM Flash Attention v3 backward kernels can be extracted, adapted without an AITER dependency, and launched through hipDNN's plugin pipeline to produce correct SDPA backward-pass gradient results on MI300X.

Backward is the priority because IREE/Fusilli will not have backward attention until end of summer — this is the gap the SDPA kernel provider must fill first.

## Scope

| Dimension | POC Boundary |
|-----------|-------------|
| Platform | gfx942 (MI300X) only |
| Data type | BF16 only |
| Direction | Backward pass only |
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

The plugin must register at least one engine with one plan builder so that hipDNN can discover and dispatch SDPA backward work to it.

| Detail | Specification |
|--------|---------------|
| Engine class | `SdpaKernelEngine` implementing `IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>` |
| Engine ID | Registered via `HIPDNN_REGISTER_ENGINE` in `SdpaKernelContainer` |
| Plan builder | `AsmSdpaBwdPlanBuilder` added to the engine via `engine->addPlanBuilder()`; engine registered via `engineManager.addEngine()` |
| Discovery | `hipdnnEnginePluginGetAllEngineIds()` returns at least one ID |
| Existing tests | The two `TestSdpaKernelContainer` tests that assert zero engines (`CopyEngineIdsReturnsZeroEngines`, `CopyEngineIdsWithBufferReturnsZero`) must be updated to assert one engine |

### FR-2: Graph Pattern Matching

`AsmSdpaBwdPlanBuilder::isApplicable()` must correctly accept graphs matching the POC configuration and reject all others.

**Accept when all of the following hold:**
- Single-node graph with `NodeAttributes::SdpaBackwardAttributes`
- Q/K/V/O/dO tensors are BF16
- Stats tensor (LSE from forward pass) is FLOAT
- Q tensor is rank-4 with `dims[3] == 128` (head dimension)
- `causal_mask == false` and `causal_mask_bottom_right == false` (these fields are deprecated; also check that no `left_bound` / `right_bound` are set, which is the non-deprecated equivalent)
- `dropout_probability` is null or 0.0
- No ALiBi mask (`alibi_mask == false`)
- No padding mask (`padding_mask == false`)
- No variable-length sequences (`seq_len_q` tensor not set — check `!has_value()` on the optional tensor UID, not `== 0`)
- No bias / dBias tensors
- Running on gfx942 (requires `hipGetDeviceProperties` device query — this is a device-level check, not a graph attribute)

**Reject otherwise**, returning zero applicable engine IDs.

### FR-3: ASM Kernel Loading and Multi-Kernel Dispatch

The backward pass requires a sequence of kernel launches. The plugin must load pre-compiled `.co` binaries and orchestrate them in the correct order.

**Kernel sequence:**

| Step | Kernel | `.co` file | Purpose |
|------|--------|-----------|---------|
| 1 | O * dO precompute | `bwd_hd128_odo_bf16.co` | Compute `D = rowsum(O * dO)` intermediate (`[B, H, S_q]`, float) |
| 2 | Main backward | `bwd_hd128_bf16_a32_rtne.co` | Compute dK, dV, and dQ accumulator (float) |
| 3 | dQ type conversion | `bwd_hd128_dq_convert_bf16_rtne.co` | Convert dQ from float accumulator to BF16 output |

All `.co` files are copied from AITER `hsa/gfx942/fmha_v3_bwd/`.

| Detail | Specification |
|--------|---------------|
| Load mechanism | `hipModuleLoad` + `hipModuleGetFunction` per kernel |
| Launch mechanism | `hipModuleLaunchKernel` with `HIP_LAUNCH_PARAM_BUFFER_POINTER` |
| Execution order | Steps 1 → 2 → 3, serialized on the same HIP stream |
| Module lifecycle | Load all three on plan build, unload on plan/context destruction |

### FR-4: CPU Backward Reference

A CPU reference implementation for SDPA backward must be created to serve as the test oracle. `CpuFpReferenceSdpa` currently only has `forward()` — a `backward()` method does not exist.

| Detail | Specification |
|--------|---------------|
| Location | `CpuFpReferenceSdpa::backward()` in the test SDK, or a local reference within the provider's test code |
| Algorithm | Naive SDPA backward: recompute `S = softmax(Q * K^T * scale)`, compute `D = rowsum(dO * O)` (numerically preferred form, matches GPU kernel), then `dV = S^T * dO`, `dS = dO * V^T`, `dP = S * (dS - D)` (softmax backward), `dQ = dP * K * scale`, `dK = dP^T * Q * scale` |
| Inputs | Q, K, V, O, dO, Stats (LSE from forward), attn_scale |
| Outputs | dQ, dK, dV |
| Data types | Templated on input type (BF16), compute in float |
| GQA support | Must handle `num_heads_q != num_heads_kv` (GQA ratio) by broadcasting K/V heads and reducing dK/dV gradients |
| Precision | Not required to be fast — correctness only |

### FR-5: Correct Computation

The GPU gradient outputs (dQ, dK, dV) must match the CPU backward reference within BF16 tolerance for at least the three test configurations below.

| Config | B | H_q | H_kv | S_q | S_kv | D | Description |
|--------|---|-----|------|-----|------|---|-------------|
| 1 | 1 | 1 | 1 | 256 | 256 | 128 | Small MHA |
| 2 | 2 | 8 | 8 | 512 | 512 | 128 | Medium MHA |
| 3 | 1 | 8 | 2 | 256 | 256 | 128 | GQA (ratio 4) |

**Tolerance:** `atol = 1e-2`, `rtol = 1e-2`. BF16 has ~7 mantissa bits; the tiled backward kernel accumulates in float but converts to BF16 for output, introducing rounding divergence from the naive CPU implementation.

**Validation per output:**
- dQ: GPU vs CPU reference within tolerance
- dK: GPU vs CPU reference within tolerance
- dV: GPU vs CPU reference within tolerance

**Forward-pass inputs:** The test must first run a forward pass to produce the O and Stats (LSE) tensors that the backward kernel requires as input. Note: the existing `CpuFpReferenceSdpa::forward()` outputs O but does **not** output Stats (LSE). The test setup must either: (a) augment the CPU forward reference to optionally return LSE, or (b) provide a standalone LSE computation utility that computes `LSE_i = log(sum_j(exp(P_ij)))` per query row from Q, K, and scale.

### FR-6: Workspace Size Reporting

The backward plan must report the correct workspace size for the intermediate buffers needed by the multi-kernel sequence.

| Buffer | Size formula | Purpose |
|--------|-------------|---------|
| D (O*dO intermediate) | `B * H_q * S_q * sizeof(float)` | Output of odo kernel, input to main backward kernel |
| dQ accumulator | `B * H_q * S_q * D * sizeof(float)` | Float accumulator for dQ, input to dq_convert kernel |

| Detail | Specification |
|--------|---------------|
| Total workspace | Sum of D buffer and dQ accumulator sizes |
| Pre-build query | `IPlanBuilder::getMaxWorkspaceSize()` returns the total based on graph tensor dimensions |
| Post-build query | `IPlan::getWorkspaceSize()` returns the same value from the built plan |
| Layout | Plugin manages sub-allocation within the workspace (e.g., D at offset 0, dQ_acc at offset `sizeof_D_buffer`) |

### FR-7: Clean Rejection

The plugin must return zero applicable engines (not crash, not throw) for graphs that do not match the POC configuration.

| Rejection case | Expected behavior |
|----------------|-------------------|
| Non-SDPA graph (e.g., BatchNorm) | Zero applicable engines |
| Forward SDPA graph (`SdpaAttributes`) | Zero applicable engines |
| Backward SDPA with FP16 tensors | Zero applicable engines |
| Backward SDPA with `causal_mask == true` | Zero applicable engines |
| Backward SDPA with `hd != 128` | Zero applicable engines |
| Backward SDPA on non-gfx942 hardware | Zero applicable engines |
| Backward SDPA with dropout | Zero applicable engines |
| Backward SDPA with bias/dBias tensors | Zero applicable engines |

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
| `.co` binaries | All three backward `.co` files installed via `install(DIRECTORY asm_kernels/ ...)`. Note: no `.co` installation infrastructure exists in any provider today — this CMake support must be created from scratch. |
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

Each backward kernel arg struct must be verified at compile time to catch layout drift.

| Struct | Check |
|--------|-------|
| `fmha_bwd_v3_args` (main backward) | `static_assert` on `sizeof` matching the AITER-defined size |
| `fmha_bwd_odo_args` (O*dO precompute) | `static_assert` on `sizeof` matching the AITER-defined size |
| `fmha_bwd_dq_convert_args` (dQ conversion) | `static_assert` on `sizeof` matching the AITER-defined size |

All structs must use `__attribute__((packed))` with SGPR-aligned padding matching the GPU kernel ABI.

**Note on provenance:** These are AITER-specific per-kernel ABI struct names. CK's `fmha_bwd.hpp` defines a single unified `fmha_bwd_args` struct used for CPU-side launching — it provides semantic reference for field names and purposes, but not the binary layout. The actual GPU kernel ABI structs must be reverse-engineered from AITER source (`mha_bwd.h` / `mha_bwd.cu`), using CK only as a cross-reference.

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
| Dispatch logic | How does AITER's `mha_bwd.cu` / Python layer choose between backward kernel variants? |
| Decision tree | Which parameters drive selection (dtype, head dim, causal, group mode, platform, accumulator precision)? |
| CSV metadata | How does the codegen CSV define the available variants? |
| Kernel count | How many backward `.co` variants exist for each platform? |
| Multi-kernel orchestration | How does AITER sequence the odo, main bwd, and dq_convert kernels? Are all three always needed? |

### RD-2: CK and ASM Kernel Relationship

Document the relationship between CK (Composable Kernel) tile-based kernels and AITER's hand-written ASM kernels.

| Topic | Expected content |
|-------|-----------------|
| When CK is used | Which attention configurations use CK tile kernels vs. ASM kernels? |
| Backward coverage | Does CK have backward attention kernels? What is their coverage vs. ASM? |
| Fallback behavior | Does AITER fall back to CK when no ASM kernel matches? |
| Performance delta | Qualitative comparison (ASM is faster for specific configs; CK provides broader coverage) |
| Dependency implications | What does using CK kernels mean for build time, binary size, and dependencies? |

### RD-3: Post-POC Roadmap Input

Provide the data needed to create an incremental plan for expanding beyond the POC.

| Topic | Expected content |
|-------|-----------------|
| Priority variants | Which additional backward kernel variants to add next (causal? hd192? FP8? gfx950?) |
| Forward pass | Assessment of AITER's forward ASM kernels and effort to integrate as a complement to IREE/Fusilli |
| Build time impact | Estimated impact of adding CK tile kernels as a fallback |
| Coverage gaps | What backward SDPA configurations would remain uncovered after adding ASM kernels? |
| Maintainability | Risks of maintaining copied ASM binaries vs. building from AITER source |
| Variant suffixes | Document what `pddv`, `pssk`, `psskddv`, `swa` suffixes mean and when each is needed |

---

## Dependencies and Assumptions

| Dependency | Type | Notes |
|------------|------|-------|
| MI300X (gfx942) hardware | Test infrastructure | Required for integration tests; unit tests run on any platform |
| AITER repository access | One-time | Needed to extract `.co` binaries, kernel arg structs, and reference the source files |
| Plugin SDK version | API stability | Assumes current `EnginePluginImpl.inl` and interface versions |
| Existing plugin skeleton | Starting point | Handle, container, context, settings, entry point, build system are complete |
| ROCm toolchain | Build dependency | HIP runtime and ROCm clang compiler |
| Forward-pass outputs for test inputs | Test setup | Integration tests need O and Stats (LSE) tensors as backward inputs; these must be produced by a known-correct forward pass (CPU reference `CpuFpReferenceSdpa::forward()` or equivalent) |
| AITER backward kernel ABI | Reverse engineering | The backward arg structs (`fmha_bwd_v3_args`, odo args, dq_convert args) must be reverse-engineered from AITER source (`mha_bwd.h` / `mha_bwd.cu`) and CK reference (`fmha_bwd.hpp`) |

---

## Out of Scope

The following are explicitly excluded from this POC:

| Item | Rationale |
|------|-----------|
| Forward pass | IREE/Fusilli provides forward coverage; forward can be added post-POC to complement it |
| FP16 / FP8 data types | Additional kernel variants and arg handling; deferred to post-POC |
| Head dimensions other than 128 | Requires additional `.co` binaries and config entries |
| Causal masking | Requires different `.co` variants and mask-aware arg setup |
| Dropout | Requires seed/offset tensors, dropout mask, and additional kernel arg fields |
| Bias / dBias gradients | Requires bias tensor handling and dBias output |
| Paged attention | Requires separate plan builder and different kernel family |
| ALiBi masking | Requires bias tensor handling |
| Variable-length sequences | Requires group-mode `.co` variants and sequence-length tensors |
| Non-gfx942 platforms | Requires platform-specific `.co` binaries |
| Performance benchmarking | POC validates correctness only; performance is a post-POC concern |
| Multi-kernel selection logic | Only one kernel variant per step; no dispatch logic needed |
| Production error handling | POC uses basic error checking; robust error reporting is post-POC |
| Backward-compatible API surface | No public API commitments from the POC |
| Sliding window attention (`swa` variants) | Post-POC expansion |
| Persistent variants (`pddv`, `pssk`, `psskddv`) | Post-POC; need RD-3 analysis first to understand when they apply |

---

## Acceptance Criteria

| ID | Criterion | Verification Method |
|----|-----------|-------------------|
| FR-1 | Plugin reports at least one engine ID | `hipdnnEnginePluginGetAllEngineIds()` returns count >= 1; updated unit tests pass |
| FR-2 | `isApplicable()` returns true for BF16 hd128 non-causal backward SDPA on gfx942 | Unit tests with matching `SdpaBackwardAttributes` graph configurations |
| FR-3 | All three backward kernels load and launch without HIP errors | Integration test completes without `hipModule*` failures |
| FR-4 | CPU backward reference produces correct gradients | Verified against a known analytical case or cross-checked with an independent implementation |
| FR-5 | GPU dQ, dK, dV match CPU reference (atol=1e-2, rtol=1e-2) for all 3 configs | `IntegrationGpuSdpaKernelBwdBfp16` parameterized test suite passes |
| FR-6 | Workspace size equals `D_buffer + dQ_acc_buffer` | Unit test asserts `getWorkspaceSize()` output; integration test allocates and passes workspace |
| FR-7 | Non-matching graphs return zero applicable engines | Unit tests with forward SDPA, non-SDPA, FP16, causal, non-gfx942 graphs |
| ER-1 | No AITER references in CMake or includes | Grep-based check; CI build without AITER installed |
| ER-2 | Full plugin lifecycle works end-to-end | Integration test exercises create -> set stream -> get engines -> build -> execute -> destroy |
| ER-3 | `ninja` builds without errors; all three `.co` files are installed | Build succeeds; `.co` files present at install prefix |
| ER-4 | `ninja check_format`, `ninja tidy` pass; zero `-Werror` warnings | CI quality gates |
| ER-5 | `static_assert` on all backward kernel arg struct sizes compiles | Build succeeds |
| ER-6 | Each adapted file has AITER provenance comment | Code review |
| RD-1 | AITER kernel selection analysis document exists | Document review |
| RD-2 | CK/ASM relationship document exists | Document review |
| RD-3 | Post-POC roadmap input document exists | Document review |

---

## Reference Documents

| Document | Path |
|----------|------|
| POC task breakdown | `dnn-providers/sdpa-kernel-provider/plans/sdpa-poc-tasks.md` |
| hipDNN backward frontend | `projects/hipdnn/frontend/include/hipdnn_frontend/node/SdpaBpropNode.hpp` |
| Backward attributes | `projects/hipdnn/frontend/include/hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp` |
| Backward FlatBuffer schema | `projects/hipdnn/data_sdk/schemas/sdpa_backward_attributes.fbs` |
| CPU forward reference | `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp` |
| CK backward args reference | `projects/composablekernel/example/ck_tile/01_fmha/fmha_bwd.hpp` |
| AITER backward `.co` kernels | `aiter/hsa/gfx942/fmha_v3_bwd/` |
