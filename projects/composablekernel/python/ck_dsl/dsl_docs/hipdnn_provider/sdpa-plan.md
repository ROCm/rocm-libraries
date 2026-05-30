# Plan: Add SDPA (attention) support to ck-dsl-provider

## Goal & scope (confirmed with user)

Add scaled-dot-product-attention support to `dnn-providers/ck-dsl-provider`, mirroring the
existing convolution vertical. **FP16**, masking = **no-mask + causal**. Delivered in two
sequenced milestones:

- **M1 — Forward (provider-only).** FP16 SDPA forward by wrapping the already torch-validated
  CK DSL kernel `build_fmha_fwd_mfma`. No CK DSL changes. Ships independently.
- **M2 — Backward (follow-up).** FP16 SDPA backward by wrapping `build_fmha_bwd` (single fused,
  torch-validated kernel). Requires first closing the **LSE gap**: add `M`/`L` stats outputs to
  the CK DSL forward kernel, since the backward consumes them and no CK DSL forward emits them today.

## Why this approach

Research established the provider is cleanly layered. Reusable **untouched**: the Python bridge
(`src/python/*` — `CompileServiceBridge::compile(opKind, payload)` is already op-generic), the
runtime (`src/runtime/{JitCache,HipModule,LaunchAbi,KernelArtifact}.*` — all op-agnostic; arg
kinds Pointer/I32/I64/F32 already supported), `PerfMeasurement`, and all plugin-entry boilerplate
(`CkDslPluginPublic.cpp`, `CkDslHandle.*`, `CkDslContext.hpp`, `CkDslSettings.hpp`).

Dispatch is **capability-based** (SDK `EngineManager` polls every engine's `isApplicable`), so an
SDPA engine coexists with conv with zero collision — no central op→engine table to edit.

The CK DSL already ships the kernels:
- `build_fmha_fwd_mfma` (`projects/composablekernel/python/ck_dsl/instances/fmha_mfma.py`) —
  FP16 forward, torch-validated tol 5e-3. Arg ABI: `Q,K,V,O ptrs, scale_log2:f32,
  seqlen_q:i32, seqlen_k:i32, stride_{q,k,v,o}_{token,head}:i32`. Grid `(seqlen_q/16,
  num_query_heads, batch)`, block `(64,1,1)`, lds 0. Supports none/causal/sliding masks.
- `build_fmha_bwd` (`.../instances/fmha_bwd.py`) — FP16 backward, **single** fused kernel,
  torch-validated tol 5e-2. dQ/dK/dV are f32 accumulators directly → **no FP32 workspace needed**.
  Consumes separate `M_saved` (log2 row-max) and `L_saved` (sum exp2) from forward. Causal supported.

## Critical contract: the op-kind string

One immutable string must be **identical** in three places, hashed into the JIT cache key — never
rename it (silently invalidates cache):
1. `PlanBuilder::opKind()` (C++)
2. the `GraphSignature` fold input (C++)
3. the Python `compile()` dispatch key

Proposed: `"sdpa_fmha_fwd"` (M1) and `"sdpa_fmha_bwd"` (M2). Separately, the wire engine-name in
`HIPDNN_REGISTER_ENGINE` is its own immutable FNV-hashed string: `"ck_dsl_sdpa_engine"`.

The scale convention is a correctness footgun: the kernel wants `scale_log2 = attn_scale *
log2(e)` where `attn_scale` defaults to `1/sqrt(D_qk)` if `SdpaAttributes.attn_scale_value` is
unset. Convert in the adapter.

---

## Step 0 — Branch & worktree setup

The ck-dsl-provider (conv vertical) that the SDPA code builds on exists **only on the current
branch** `users/dahawkin/ck-dsl-provider`; it is not yet on `develop`. So the SDPA branch is based
on the current branch, not develop (otherwise the provider source is absent and nothing compiles).
Following the project `worktrees/` convention (not the harness `.claude/worktrees`):

```
# from /home/AMD/dahawkin/work/ck_study/worktrees/ck-dsl-prototype
git worktree add -b users/dahawkin/ck-dsl-sdpa \
  /home/AMD/dahawkin/work/ck_study/worktrees/ck-dsl-sdpa \
  users/dahawkin/ck-dsl-provider
```
All implementation happens in `worktrees/ck-dsl-sdpa`. The current worktree's uncommitted edits
(`ConvImplicitGemmAdapter.cpp`, `GraphSignature.cpp`, `WIP/*`) are NOT carried over (they are local
WIP/review artifacts); confirm with user if any are needed as a base.

## Milestone 1 — Forward (FP16)

### Files to ADD (mirror the conv trio)
```
src/adapters/sdpa/SdpaSpec.hpp                 # mirror ConvImplicitGemmSpec.hpp: SdpaProblem (B,Hq,Hkv,Sq,Skv,Dqk,Dv, strides, dtype, mask, scale) + knobs
src/adapters/sdpa/SdpaAdapter.hpp/.cpp         # buildSpec(SdpaAttributes&, TensorMap&): the ONLY flatbuffer reader; validate fp16, rank-4, GQA-divisible, supported head_size, seqlen%16==0, head-dim-contiguous, mask∈{none,causal}, reject bias/dropout/paged/stats; narrow i64→i32; throw HipdnnPluginException on reject
src/adapters/sdpa/SdpaPayload.hpp/.cpp         # sdpaSpecToPayload(spec)->py::dict, field-for-field with Python dataclass; GIL held by caller
src/engines/sdpa/CkDslSdpaEngine.hpp/.cpp      # IEngine boilerplate copy of CkDslConvImplicitGemmEngine; owns SdpaFwdPlanBuilder; getDetails publishes EngineDetails
src/engines/sdpa/SdpaFwdPlanBuilder.hpp/.cpp   # opKind()="sdpa_fmha_fwd"; isApplicable (structural single-node + NodeAttributes::SdpaAttributes guard + buildSpec in try/catch->false); buildPlan (buildSpec -> GraphSignature -> loader closure -> cache.getOrLoad -> construct SdpaFwdPlan)
src/engines/sdpa/SdpaFwdPlan.hpp/.cpp          # IPlan: holds HipModule + Q/K/V/O UIDs + scalars; execute() resolves device ptrs by UID, packs args via LaunchAbi::pack (schema-driven, no pre-pack template needed), launches
```
Graph signature: add overload `GraphSignature::computeForSpec(string_view, const SdpaSpec&)` in
`src/graph/GraphSignature.cpp` (reuse existing fnv1a helpers) — simplest, no new TU.

### Files to EDIT
```
src/CkDslContainer.cpp        # +HIPDNN_REGISTER_ENGINE(CK_DSL_SDPA_ENGINE,"ck_dsl_sdpa_engine"); +1 entry in engineDefinitions()
src/CMakeLists.txt            # add the new adapters/sdpa + engines/sdpa sources to ck_dsl_provider_impl
python/ck_dsl_provider/compile_service.py   # +elif op_kind=="sdpa_fmha_fwd": _compile_sdpa_fwd(payload); + _compile_sdpa_fwd builder (call build_fmha_fwd_mfma + compile_kernel), _sdpa_fwd_arg_schema(), payload key whitelist
```

### Python compile branch (M1)
`_compile_sdpa_fwd(payload)` returns the standard 7-field dict (`hsaco, kernel_name, kind, grid,
block, lds_bytes, arg_schema`). grid from `fmha_fwd_mfma_grid`, block `(64,1,1)`, lds 0.
`arg_schema` matches `_declare_params` of fmha_mfma: 4 Pointer + 1 F32 (scale_log2) + 2 I32
(seqlens) + 8 I32 (strides). Validate against fmha_mfma `is_valid_spec` constraints
(seqlen_q/k %16==0, head_size∈{32,64,128,192,256}, fp16).

### Tests (M1)
```
tests/SdpaAdapterTest.cpp           # buildSpec accept/reject (device-independent): fp16 ok, bf16 reject, bad shape reject, causal vs none, scale_log2 conversion
tests/SdpaFwdPlanBuilderTest.cpp    # isApplicable true on valid SDPA graph, false on conv graph / unsupported features
integration_tests/IntegrationGpuCkDslSdpaFwdFp16.cpp   # gfx950-gated; build SdpaAttributes FB graph (BHSD), CpuFpReferenceSdpa::forward reference, drive PlanBuilder->buildPlan->execute(workspace=nullptr), compare tol ~1e-2; parameterize over shape set (vary B/Hq/Hkv/Sq/Skv/D, GQA ratio, none+causal). Start D=128.
tests/CMakeLists.txt, integration_tests/CMakeLists.txt   # register new sources
```
CPU reference `hipdnn_test_sdk::utilities::CpuFpReferenceSdpa` exists and supports GQA.

### M1 build/validate
Per memory: ctest `-check` does NOT rebuild — build the `ck_dsl_provider_integration_tests` and
`ck_dsl_provider_unit_tests` exes, run binaries directly; gfx950-gated. Integration test will
skip on non-gfx950.

---

## Milestone 2 — Backward (FP16) — follow-up

### Precondition: close the LSE gap (CK DSL change)
`build_fmha_fwd_mfma` computes `ms_final`/`ls_final` but discards them. Add two f32 out-pointers
and epilogue stores, preserving the **separate M (log2 row-max) / L (sum exp2)** convention the
bwd consumes (NOT a single combined LSE):
```
projects/composablekernel/python/ck_dsl/instances/fmha_mfma.py     # add M_out,L_out ptrs to ABI (gated by a stats flag on FmhaMfmaSpec so M1 ABI unchanged)
projects/composablekernel/python/ck_dsl/helpers/mfma_attention.py  # epilogue: store ms_final, ls_final
projects/composablekernel/python/ck_dsl/examples/parity_extended_kernels.py  # extend case_fmha_fwd_mfma to validate M/L outputs
```
Keep stats outputs **opt-in** so the M1 forward ABI/cache key is untouched.

### Files to ADD
```
src/engines/sdpa/SdpaBwdPlanBuilder.hpp/.cpp   # opKind()="sdpa_fmha_bwd"; isApplicable matches SDPA_BWD node (NodeAttributes::SdpaBackwardAttributes); buildSpec for bwd
src/engines/sdpa/SdpaBwdPlan.hpp/.cpp          # IPlan: UIDs q,k,v,o,dO,stats(LSE=M+L) in; dQ,dK,dV out (f32). Single kernel launch. getWorkspaceSize=0 (dQ/dK/dV are the f32 accumulators)
src/adapters/sdpa/SdpaBwdAdapter + SdpaBwdSpec/Payload (or extend SdpaSpec with a direction)     # read SdpaBackwardAttributes; same fp16/shape/mask validation
```
Reuse the SdpaFwdPlanBuilder pattern; add bwd plan builder to the same `CkDslSdpaEngine` via
`addPlanBuilder` (or a second engine — decide at impl time; single engine with two builders
matches the SDK pattern).

### Files to EDIT
```
python/ck_dsl_provider/compile_service.py   # +elif op_kind=="sdpa_fmha_bwd": _compile_sdpa_bwd (call build_fmha_bwd); _sdpa_bwd_arg_schema (24-arg ABI: Q,K,V,dO,M,L f32, dQ,dK,dV f32, scale_log2,scale_inv f32, seqlens, strides)
src/CMakeLists.txt                          # add bwd sources
src/engines/sdpa/CkDslSdpaEngine.cpp        # register bwd plan builder
```
Also wire the forward stats path: when an SDPA_FWD graph requests stats (training), the M1 fwd
builder must request the stats-enabled forward variant and bind the M/L output UIDs.

### Tests (M2)
```
integration_tests/IntegrationGpuCkDslSdpaBwdFp16.cpp   # gfx950-gated; fwd(+stats) then bwd; reference dQ/dK/dV via torch-equivalent CPU autograd or finite-diff; tol ~5e-2. MUST cover causal + GQA (current CK DSL bwd validation is narrow: head=64,seq=16,no-mask,no-GQA)
tests/SdpaBwdPlanBuilderTest.cpp
```

### M2 risks (carry into impl)
- bwd correctness validated only at head=64/seq=16/no-mask/no-GQA upstream → **must** add causal+GQA
  parity coverage before relying on it (do this in the CK DSL examples and/or the integration test).
- The MFMA-tiled bwd helper (`helpers/mfma_attention_bwd.py`) is unwired/unvalidated and uses a
  conflicting LSE convention — do NOT use it; the shipped warp-scalar `build_fmha_bwd` is the target.
- `head_size % WARP_SIZE == 0` required (64/128/256 ok; 32 fails).

---

## Sequencing / execution model (orchestrator)

- **M1** is a single coherent vertical → small parallelizable streams:
  1. Python `_compile_sdpa_fwd` + arg schema.
  2. C++ adapter trio (Spec/Adapter/Payload) + graph signature overload.
  3. C++ engine trio (Engine/FwdPlanBuilder/FwdPlan) + container/CMake registration — depends on (2).
  4. Tests (unit + integration) — depends on (2),(3).
  Implementors write code only; orchestrator commits; Builder merges + builds with redirected logs;
  Reviewer validates plan, each stream, and the integrated set.
- **M2** starts after M1 lands: CK DSL LSE change (+re-validate fwd parity) first, then the bwd
  vertical, then bwd integration tests with causal+GQA coverage.

## Out of scope (rejected in isApplicable → graceful fallback)
bf16/fp8, additive bias/attn_mask tensor, dropout, paged/block-mask, variable-length (GROUP)
sequences, sink tokens, sliding-window (causal only for now), unsupported head sizes / non-%16 seqlens.
