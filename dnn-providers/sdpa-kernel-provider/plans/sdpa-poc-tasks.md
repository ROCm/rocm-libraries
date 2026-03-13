# SDPA Kernel Provider — POC Task Breakdown

## Research Tasks (parallelizable, no code dependencies)

### Task R1: AITER Kernel Selection Analysis (RD-1)
**Goal:** Document how AITER selects which backward attention kernel to dispatch.
**Deliverable:** Analysis document covering dispatch logic in `mha_bwd.cu`/Python layer, the decision tree (dtype, head dim, causal, group mode, platform, accumulator precision), CSV metadata format, number of backward `.co` variants per platform, and multi-kernel orchestration (odo → main bwd → dq_convert).
**Inputs:** AITER repository access.
**No code changes required.**

### Task R2: CK and ASM Kernel Relationship Analysis (RD-2)
**Goal:** Document the relationship between CK tile-based kernels and AITER's hand-written ASM kernels.
**Deliverable:** Analysis document covering when CK vs ASM is used, backward coverage comparison, fallback behavior, qualitative performance delta, and dependency implications (build time, binary size).
**Inputs:** AITER and CK source repositories.
**No code changes required.**

### Task R3: Post-POC Roadmap Input (RD-3)
**Goal:** Provide data for expanding beyond the POC.
**Deliverable:** Document covering priority next variants (causal, hd192, FP8, gfx950), forward pass assessment, CK build time impact, coverage gap analysis, maintainability risks, and variant suffix meanings (`pddv`, `pssk`, `psskddv`, `swa`).
**Inputs:** AITER repository, results from R1 and R2.
**Depends on:** R1, R2 (can start in parallel but should incorporate their findings).

---

## Implementation Tasks (ordered by dependencies)

### Task I1: Reverse-Engineer and Define Backward Kernel Arg Structs (ER-5, ER-6 partial)
**Goal:** Define the three GPU kernel argument structs that match the AITER kernel ABI.
**Deliverable:** Header file(s) under `src/asm/` containing:
- `fmha_bwd_v3_args` (main backward kernel)
- `fmha_bwd_odo_args` (O*dO precompute kernel)
- `fmha_bwd_dq_convert_args` (dQ conversion kernel)

**Requirements:**
- All structs use `__attribute__((packed))` with SGPR-aligned padding
- `static_assert` on `sizeof` for each struct matching the AITER-defined size
- Replace `ck_tile::index_t` with `int32_t`; use direct HIP calls instead of CK tile wrappers
- Each file has an AITER provenance comment block (exact commit hash, source file paths, summary of adaptations)
- No AITER `#include` directives

**Inputs:** AITER source files (`mha_bwd.h`, `mha_bwd.cu`), CK reference (`fmha_bwd.hpp`).
**Depends on:** Nothing.

### Task I2: Copy `.co` Binaries and Build System Integration (ER-1, ER-3, ER-6 partial)
**Goal:** Add the pre-compiled ASM kernel binaries to the provider and wire them into CMake.
**Deliverable:**
- Copy three `.co` files from AITER `hsa/gfx942/fmha_v3_bwd/` into the provider:
  - `bwd_hd128_odo_bf16.co`
  - `bwd_hd128_bf16_a32_rtne.co`
  - `bwd_hd128_dq_convert_bf16_rtne.co`
- `install(DIRECTORY asm_kernels/ ...)` rule in CMake
- Compile definition `AITER_ASM_DIR` set to install prefix
- Runtime override via `HIPDNN_AITER_ASM_DIR` env var
- AITER provenance documentation (commit hash, source paths)

**Requirements:**
- No `find_package(aiter)`, no AITER include paths, no AITER link targets
- New sources added to `sdpa_kernel_plugin_impl` OBJECT target in `src/CMakeLists.txt`

**Inputs:** AITER repository access.
**Depends on:** Nothing (parallelizable with I1).

### Task I3: Implement Graph Pattern Matching — `isApplicable()` (FR-2, FR-7)
**Goal:** Implement `AsmSdpaBwdPlanBuilder::isApplicable()` to accept/reject graph configurations.
**Deliverable:** Implementation that accepts when ALL conditions hold:
- Single-node graph with `SdpaBackwardAttributes`
- Q/K/V/O/dO tensors are BF16; Stats (LSE) tensor is FLOAT
- Q tensor is rank-4 with `dims[3] == 128`
- `causal_mask == false`, `causal_mask_bottom_right == false`
- No `left_bound`/`right_bound`
- `dropout_probability` is null or 0.0
- `alibi_mask == false`, `padding_mask == false`
- `seq_len_q_tensor_uid == 0`
- No bias/dBias tensors
- Running on gfx942

Rejects all other configurations (returns zero applicable engine IDs, no crash, no throw).

**Unit tests** covering:
- Accept: BF16 hd128 non-causal backward SDPA on gfx942
- Reject: non-SDPA graph, forward SDPA, FP16 tensors, causal mask, hd != 128, non-gfx942, dropout, bias/dBias

**Depends on:** Existing plugin skeleton (already merged).

### Task I4: Register Engine and Plan Builder (FR-1)
**Goal:** Register the backward engine and plan builder so hipDNN can discover and dispatch work.
**Deliverable:**
- Engine registered via `HIPDNN_REGISTER_ENGINE` in `SdpaKernelContainer`
- `AsmSdpaBwdPlanBuilder` added to the engine via `addPlanBuilder()`
- `hipdnnEnginePluginGetAllEngineIds()` returns at least one ID

**Test updates:**
- Update the three existing `TestSdpaKernelContainer` tests that assert zero engines to assert one engine

**Depends on:** I3 (plan builder must exist before registration).

### Task I5: Implement Workspace Size Reporting (FR-6)
**Goal:** Report correct workspace size for intermediate buffers.
**Deliverable:**
- D (O*dO intermediate): `B * H_q * S_q * sizeof(float)`
- dQ accumulator: `B * H_q * S_q * D * sizeof(float)`
- Total workspace = sum of both
- `IPlanBuilder::getMaxWorkspaceSize()` returns total based on graph tensor dimensions
- `IPlan::getWorkspaceSize()` returns same value from built plan
- Plugin manages sub-allocation (D at offset 0, dQ_acc at offset `sizeof_D_buffer`)

**Unit test:** Assert `getWorkspaceSize()` output matches expected formula.

**Depends on:** I3 (plan builder structure), I4 (engine registration).

### Task I6: Implement ASM Kernel Loading and Multi-Kernel Dispatch (FR-3)
**Goal:** Load pre-compiled `.co` binaries and orchestrate the 3-step backward kernel sequence.
**Deliverable:**
- Kernel loading via `hipModuleLoad` + `hipModuleGetFunction` for all three kernels
- Launch via `hipModuleLaunchKernel` with `HIP_LAUNCH_PARAM_BUFFER_POINTER`
- Execution order: (1) O*dO precompute → (2) main backward → (3) dQ type conversion, serialized on the same HIP stream
- Module lifecycle: load all three on plan build, unload on plan/context destruction
- Proper argument population for each kernel using the structs from I1

**Depends on:** I1 (arg structs), I2 (`.co` binaries + CMake), I4 (engine/plan framework), I5 (workspace allocation).

### Task I7: Implement CPU Backward Reference (FR-4)
**Goal:** Create a CPU reference implementation to serve as the test oracle.
**Deliverable:** `CpuFpReferenceSdpa::backward()` (or local test reference) implementing:
- Naive SDPA backward: recompute `S = softmax(Q * K^T * scale)`, then:
  - `dV = S^T * dO`
  - `dS = dO * V^T`
  - `dP = dS * S - S * rowsum(dS * S)` (softmax backward)
  - `dQ = dP * K * scale`
  - `dK = dP^T * Q * scale`
- Templated on input type (BF16), compute in float
- Handle GQA: `num_heads_q != num_heads_kv` by broadcasting K/V heads and reducing dK/dV gradients
- Inputs: Q, K, V, O, dO, Stats (LSE), attn_scale
- Outputs: dQ, dK, dV

**Verification unit test:** A dedicated test that validates the CPU backward reference against a small hand-computable case (e.g., B=1, H=1, S_q=S_kv=2, D=2) where expected dQ, dK, dV values are derived independently (e.g., computed via PyTorch `torch.autograd.grad` on `F.scaled_dot_product_attention`, or worked out analytically). This test must be added to the `sdpa_kernel_plugin_tests` target and run on any platform (no GPU required).

**Depends on:** Nothing (parallelizable with I1–I6).

### Task I8: Implement Integration Tests for Correct Computation (FR-5, ER-2)
**Goal:** End-to-end tests validating GPU output matches CPU reference.
**Deliverable:** `IntegrationGpuSdpaKernelBwdBfp16` parameterized test suite with three configurations:

| Config | B | H_q | H_kv | S_q | S_kv | D | Description |
|--------|---|-----|------|-----|------|---|-------------|
| 1 | 1 | 1 | 1 | 256 | 256 | 128 | Small MHA |
| 2 | 2 | 8 | 8 | 512 | 512 | 128 | Medium MHA |
| 3 | 1 | 8 | 2 | 256 | 256 | 128 | GQA (ratio 4) |

**Requirements:**
- Forward pass first (CPU reference) to produce O and Stats (LSE) tensors
- Compare dQ, dK, dV: GPU vs CPU with `atol=1e-2`, `rtol=1e-2`
- Exercise full plugin lifecycle: create → set stream → get engines → build → execute → destroy
- Test file added to `sdpa_kernel_plugin_integration_tests` target
- Requires MI300X (gfx942) hardware

**Depends on:** I6 (kernel dispatch), I7 (CPU reference).

---

## Dependency Graph

```
I1 (arg structs) ──────────────────┐
                                   ├──→ I6 (kernel dispatch) ──┐
I2 (.co binaries + CMake) ────────┘                            │
                                                               ├──→ I8 (integration tests)
I3 (isApplicable) ──→ I4 (engine reg) ──→ I5 (workspace) ──→ I6│
                                                               │
I7 (CPU reference) ────────────────────────────────────────────┘

R1 (AITER analysis) ──┐
R2 (CK/ASM analysis) ─┼──→ R3 (roadmap input)
```

**Parallelizable from day one:** I1, I2, I3, I7, R1, R2 (six independent streams).
