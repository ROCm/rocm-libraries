# SDPA Kernel Provider — POC Task Breakdown

## Research Tasks (parallelizable, no code dependencies)

### Task R1: AITER Kernel Selection Analysis (RD-1)
**Goal:** Document how AITER selects which forward attention kernel to dispatch.

**Deliverable:** Analysis document covering dispatch logic in `mha_fwd.cu`/Python layer, the decision tree (dtype, head dim, causal, group mode, platform, rounding mode), CSV metadata format (`fmha_fwd.csv`), number of forward `.co` variants per platform (MI300 vs MI308), and MI300/MI308 selection logic.
**Inputs:** AITER repository access.
**No code changes required.**

### Task R2: CK and ASM Kernel Relationship Analysis (RD-2)
**Goal:** Document the relationship between CK tile-based kernels and AITER's hand-written ASM kernels.

**Deliverable:** Analysis document covering when CK vs ASM is used, forward coverage comparison, fallback behavior, qualitative performance delta, and dependency implications (build time, binary size).
**Inputs:** AITER and CK source repositories.
**No code changes required.**

### Task R3: Post-POC Roadmap Input (RD-3)
**Goal:** Provide data for expanding beyond the POC.

**Deliverable:** Document covering priority next forward variants (causal, hd192, FP8, gfx950, group mode), backward pass assessment (3-kernel pipeline: odo + main bwd + dq_convert), CK build time impact, coverage gap analysis, maintainability risks, and MI308 support effort.
**Inputs:** AITER repository, results from R1 and R2.
**Depends on:** R1, R2 (can start in parallel but should incorporate their findings).

---

## Implementation Tasks (ordered by dependencies)

### Task I1: Reverse-Engineer and Define Forward Kernel Arg Struct (ER-5, ER-6 partial)
**Goal:** Define the GPU kernel argument struct that matches the AITER forward kernel ABI.

**Deliverable:** Header file under `src/asm/` containing:
- `fmha_fwd_v3_args` (forward kernel)

**Requirements:**
- Struct uses `__attribute__((packed))` with SGPR-aligned padding
- `static_assert` on `sizeof` matching the AITER-defined size
- Replace `ck_tile::index_t` with `int32_t`; use direct HIP calls instead of CK tile wrappers
- File has an AITER provenance comment block (exact commit hash, source file paths, summary of adaptations)
- No AITER `#include` directives

**Inputs:** AITER source file (`mha_fwd.h`) for the actual GPU kernel ABI struct. The packed `fmha_fwd_v3_args` struct is defined at lines 266-350 of `aiter/csrc/include/mha_fwd.h`. Key fields include output pointers (ptr_o, ptr_lse), input pointers (ptr_q, ptr_k, ptr_v), scalar (attention scale), stride parameters, s_lse flag (whether to compute LSE), and s_gqa (GQA ratio).
**Depends on:** Nothing.

### Task I2: Copy `.co` Binary and Build System Integration (ER-1, ER-3, ER-6 partial)
**Goal:** Add the pre-compiled ASM kernel binary to the provider and wire it into CMake.

**Deliverable:**
- Copy one `.co` file from AITER `hsa/gfx942/fmha_v3_fwd/MI300/` into the provider:
  - `fwd_hd128_bf16_rtne.co`
- `install(DIRECTORY asm_kernels/ ...)` rule in CMake (net-new — no `.co` installation infrastructure exists in any provider today)
- Compile definition `AITER_ASM_DIR` set to install prefix
- Runtime override via `HIPDNN_AITER_ASM_DIR` env var
- AITER provenance documentation (commit hash, source paths)

**Requirements:**
- No `find_package(aiter)`, no AITER include paths, no AITER link targets
- New sources added to `sdpa_kernel_plugin_impl` OBJECT target in `src/CMakeLists.txt`

**Inputs:** AITER repository access.
**Depends on:** Nothing (parallelizable with I1).

### Task I3: Implement Graph Pattern Matching — `isApplicable()` (FR-2, FR-7)
**Goal:** Implement `AsmSdpaFwdPlanBuilder::isApplicable()` to accept/reject graph configurations.

**Deliverable:** Implementation that accepts when ALL conditions hold:
- Single-node graph with `SdpaAttributes`
- Q/K/V tensors are BF16
- Q tensor is rank-4 with `dims[3] == 128`
- `causal_mask == false`, `causal_mask_bottom_right == false` (deprecated fields; also check no `left_bound`/`right_bound` set — the non-deprecated equivalent)
- `dropout_probability` is null or 0.0
- `alibi_mask == false`, `padding_mask == false`
- `seq_len_q` tensor not set (check `!has_value()` on the optional tensor UID, not `== 0`)
- No attention mask / bias tensor (`attn_mask` not set)
- No paged attention (`page_table_k` / `page_table_v` not set)
- Running on gfx942 (requires `hipGetDeviceProperties` device query — not a graph attribute)

Rejects all other configurations (returns zero applicable engine IDs, no crash, no throw).

**Unit tests** covering:
- Accept: BF16 hd128 non-causal forward SDPA on gfx942
- Reject: non-SDPA graph, backward SDPA, FP16 tensors, causal mask, hd != 128, non-gfx942, dropout, attention mask/bias, paged attention

**Depends on:** Existing plugin skeleton (already merged).

### Task I4: Register Engine and Plan Builder (FR-1)
**Goal:** Register the forward engine and plan builder so hipDNN can discover and dispatch work.

**Deliverable:**
- `SdpaKernelEngine` class implementing `IEngine<SdpaKernelHandle, SdpaKernelSettings, SdpaKernelContext>`
- Engine registered via `HIPDNN_REGISTER_ENGINE` in `SdpaKernelContainer`
- `AsmSdpaFwdPlanBuilder` added to the engine via `engine->addPlanBuilder()`; engine registered via `engineManager.addEngine()`
- `hipdnnEnginePluginGetAllEngineIds()` returns at least one ID

**Test updates:**
- Update the two existing `TestSdpaKernelContainer` tests that assert zero engines (`CopyEngineIdsReturnsZeroEngines`, `CopyEngineIdsWithBufferReturnsZero`) to assert one engine

**Depends on:** I3 (plan builder must exist before registration).

### Task I5: Implement Workspace Size Reporting (FR-5)
**Goal:** Report correct workspace size for the forward pass.

**Deliverable:**
- The forward pass is a single kernel with no intermediate buffers — workspace = 0
- `IPlanBuilder::getMaxWorkspaceSize()` returns 0
- `IPlan::getWorkspaceSize()` returns 0

**Unit test:** Assert `getWorkspaceSize()` returns 0.

**Depends on:** I3 (plan builder structure), I4 (engine registration).

### Task I6: Implement ASM Kernel Loading and Dispatch (FR-3)
**Goal:** Load the pre-compiled `.co` binary and launch the forward kernel.

**Deliverable:**
- Kernel loading via `hipModuleLoad` + `hipModuleGetFunction`
- Launch via `hipModuleLaunchKernel` with `HIP_LAUNCH_PARAM_BUFFER_POINTER`
- Proper argument population using the `fmha_fwd_v3_args` struct from I1:
  - Set Q/K/V input pointers and strides from graph tensor descriptors
  - Set O output pointer and strides
  - Set LSE output pointer (if `generate_stats` is true in attributes)
  - Set `scalar` to attention scale value
  - Set `s_gqa` to `num_heads_q / num_heads_kv`
  - Set `s_lse` flag based on whether Stats output is requested
  - Set `s_seq_len`, `s_qk_head_dim` from tensor dimensions
- Grid/block dimensions computed per AITER dispatch logic
- Module lifecycle: load on plan build, unload on plan/context destruction

**Depends on:** I1 (arg struct), I2 (`.co` binary + CMake), I4 (engine/plan framework), I5 (workspace reporting).

### Task I7: Implement Integration Tests for Correct Computation (FR-4, ER-2)
**Goal:** End-to-end tests validating GPU output matches CPU reference.

**Deliverable:** `IntegrationGpuSdpaKernelFwdBfp16` parameterized test suite with three configurations:

| Config | B | H_q | H_kv | S_q | S_kv | D | Description |
|--------|---|-----|------|-----|------|---|-------------|
| 1 | 1 | 1 | 1 | 256 | 256 | 128 | Small MHA |
| 2 | 2 | 8 | 8 | 512 | 512 | 128 | Medium MHA |
| 3 | 1 | 8 | 2 | 256 | 256 | 128 | GQA (ratio 4) |

**Approach:** Use the `IntegrationGraphVerificationHarness` pattern (see `dnn-providers/miopen-provider/integration_tests/IntegrationGraphVerificationHarness.hpp`):
1. Inherit from `IntegrationGraphVerificationHarness<bfloat16, SdpaFwdTestCase>`
2. Build the forward SDPA graph using the frontend API (`graph.sdpa_fprop(...)` or equivalent)
3. Mark O as output (`oAttr->set_output(true)`) and call `registerValidator(oAttr, tolerance)`
4. Call `verifyGraph(graph, seed)` — the harness automatically runs CPU reference via `CpuReferenceGraphExecutor` and compares against GPU

**Requirements:**
- Tolerance: `atol=1e-2`, `rtol=1e-2`
- Test file added to `sdpa_kernel_plugin_integration_tests` target (in `src/integration_tests/`)
- Requires MI300X (gfx942) hardware

**Depends on:** I6 (kernel dispatch).

---

## Dependency Graph

```
I1 (arg struct) ──────────────────┐
                                   ├──→ I6 (kernel dispatch) ──→ I7 (integration tests)
I2 (.co binary + CMake) ─────────┘                  │
                                                     │
I3 (isApplicable) ──→ I4 (engine reg) ──→ I5 (workspace) ──→ I6

R1 (AITER analysis) ──┐
R2 (CK/ASM analysis) ─┼──→ R3 (roadmap input)
```

**Parallelizable from day one:** I1, I2, I3, R1, R2 (five independent streams).
