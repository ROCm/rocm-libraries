# M2 toy→real swap — autonomous overnight decision log

Context: user went to bed 2026-09-11 ~02:30, said "continue making your best decisions,
don't ask me to decide, record the decisions so I can ask in the morning, get as far as
you can." This file records every non-trivial decision I make while they sleep.

## Status snapshot
- M2 TOY phase: DONE (both N=64/128 select their HSACO through hipDNN, numerics ~1e-7).
  See memory `m2-rmsnorm-family-toy-done`.
- Now doing: M2 SWAP — replace toy HSACOs with FlyDSL's REAL `build_rmsnorm_module`
  bf16 kernels at N ∈ {4096, 8192}, the final M2 target set the user chose earlier.

## Facts established (from the FlyDSL repo, read tonight)
- Real forward kernel: `kernels/norm/rmsnorm_kernel.py::build_rmsnorm_module(N, dtype_str,
  store_rstd=False, eps=EPS=1e-5, BLOCK_THREADS=256, weight_dtype_str=None)`.
- Device kernel signature: `rmsnorm_kernel(Input, Gamma, Rstd, Output)` — 4 Tensor args.
  Rstd is only written when store_rstd=True (default False → still an arg, but unused).
- Host launch (from test_rmsnorm.py): `compiled_fn(input_dev, gamma_dev, output_dev, M, stream)`
  — i.e. the @jit wrapper takes (Input, Gamma, Output, M, stream); it must synthesize/omit
  Rstd internally when store_rstd=False. MUST confirm exact wrapper arg list + how Rstd is
  passed to the device kernel (dummy buffer? dropped arg?). This drives the kernarg ABI.
- dtype_str for bf16 = "bf16"; weight defaults to same. EPS = 1e-5 (NOT the toy's 1e-6).
- Geometry: grid = (M,1,1), block = (BLOCK_THREADS,1,1) = (256,1,1). One block per row,
  256 threads cooperate per row (LDS reduction), VEC_WIDTH=8, tile_cols=2048.
  N=4096→num_tiles=2, N=8192→num_tiles=4; both hit the bf16 128-bit fast path.
- Reference (torch): out = x/sqrt(mean(x^2)+eps)*gamma, eps=1e-5. bf16 atol used by repo = 2e-2.

## Decisions
1. **ABI discovery, not guessing.** The real kernarg layout (esp. how Rstd + 2D [M,N]
   tensors are encoded) is the #1 risk. I will BUILD one real HSACO and read its AMDGPU HSA
   metadata (`.args` offsets/sizes) from the ELF note to get the exact ABI, rather than
   assuming the toy's (ptr,i32)-per-arg pattern holds for 2D tensors. Decision rationale:
   the plan explicitly flags this as the hardest bit and says bubblegum/hardcode-after-
   discovery is fine.
2. **Metadata-driven dispatch (generalize the pack).** Rather than hardcode a second ABI,
   I'll extend the descriptor `metadata` per kernelDescriptor with: `hsaco` (filename),
   `symbol` (entry point), `block_threads` (launch block.x). The dispatch reads these so the
   SAME pack serves both the toy family and the real family with no C++ per-family branching
   beyond the kernarg builder. Keeps the toy root working unchanged (its descriptors just
   won't set the new fields → fall back to toy defaults). If the real 2D-tensor ABI proves
   materially different from (ptr,i32), I'll add an `abi` metadata tag selecting the packer.
3. **Separate descriptor root + HSACO names for the real family** so the toy proof keeps
   passing: `flydsl_descriptors_rmsnorm_bf16/` and `rmsnorm_real_n<N>_bf16_gfx950.hsaco`.
4. **App**: new `flydsl_rmsnorm_bf16_app.cpp` (bf16 device buffers via hip_bfloat16, fp32
   host reference, atol 2e-2), N ∈ {4096,8192}, ROWS/M = 4 (small M is fine; M is runtime).
   Keep the toy app intact.
5. **If the real kernel won't drive through the ADD carrier** (e.g. frontend rejects the
   shapes/dtype), fall back to the plan's escape: a minimal custom graph in the copied app.
   Won't touch the frontend.
6. Everything stays THROWAWAY; no pushing; branch users/brpepers/hipdnn-flydsl-poc only.

## Log (append as I go)
- 02:3x — toy phase confirmed done; started real-kernel investigation; wrote this file.
- 02:2x — DECISION #1 executed: built real bf16 HSACOs (N=4096 5920B, N=8192 6496B) via
  `build_rmsnorm_real.py <N> bf16` and read the AMDGPU `.args` from the ELF note. Symbol
  `rmsnorm_kernel_0`; kernarg_segment_size 80; group_segment 32B LDS; reqd block [256,1,1].
- 02:3x — REAL 2D ABI FULLY DECODED (from HSACO `.args` + stage-01 `01_fly_rewrite_func_signature.mlir`):
  kernel `rmsnorm_kernel(Input, Gamma, Rstd, Output)`, Rstd slot reuses Gamma ptr.
  * 2D [M,N] tensor by_value = `{i32 dim0, i32 dim1, i64 row_stride}` (16B). Stage-01 builds
    `layout (M,N):(row_stride,1)` — inner/col stride is a compile-time 1, outer/row stride is the
    runtime i64. Row-major contiguous ⇒ row_stride = N.
  * 1D tensor by_value = `{i32 size}` (4B). (Matches the toy's per-arg i32.)
  * Kernarg offsets (all confirmed against `.args`): Input.ptr@0 desc@8(16) Gamma.ptr@24 size@32(4)
    Rstd.ptr@40 size@48(4) Output.ptr@56 desc@64(16). Total 80. grid=(M,1,1) block=(256,1,1).
- 02:3x — DECISION #2 executed: made `FlydslRmsNormNative.cpp` metadata-driven. New optional
  per-kernel metadata `hsaco`/`symbol`/`block_threads`/`abi` ("toy"|"real2d"); absent ⇒ toy defaults
  (`rmsnorm_toy_n<N>_gfx950.hsaco`, symbol `rmsnorm_0`, block 1, toy 44B ABI). `abi=real2d` selects the
  80B 2D packer + block from metadata. One pack, one dispatch, two families, zero frontend changes.
- 02:3x — DECISION #3/#4 executed: separate root `flydsl_descriptors_rmsnorm_bf16/` (id prefix
  f1d5b160-, 2 kernelDescriptors N=4096/8192, metadata dtype=bf16 + hsaco/symbol/block_threads/abi),
  new `flydsl_rmsnorm_bf16_app.cpp` (bf16 device buffers via uint16 + manual RNE convert, fp32 host
  ref rounded through bf16, EPS=1e-5, atol 2e-2, N∈{4096,8192}, M=4), `run_rmsnorm_bf16_app.sh`,
  CMake target added.
- 02:3x — **M2 SWAP DONE / EXIT CRITERION MET.** Rebuilt `hip_kernel_provider` + app clean.
  `run_rmsnorm_bf16_app.sh` on MI355X/gfx950:
    * N=4096 → engine hipkernel:FlydslRmsNorm, PROOF#1 selected, max_abs_err 3.903e-03 (< 2e-2) ✓
    * N=8192 → same engine, PROOF#1 selected, max_abs_err 3.883e-03 (< 2e-2) ✓
  Regression check: toy `run_rmsnorm_app.sh` still passes N=64 (1.192e-7) / N=128 (2.384e-7) — the
  metadata defaults preserve the toy path. DECISION #5 (custom-graph fallback) NOT needed: the ADD
  carrier drove the real bf16 kernel fine. Everything throwaway, branch-only, nothing pushed (#6).
- NEXT (morning): M3 — bf16 prefill attention. Rebuild with `-DHIPDNN_ENABLE_SDPA=ON`, AOT the
  FlyDSL flash_attn_gfx950 family, author `FlydslAttentionNative.cpp` mirroring the SDPA graph_match.

## M3 ABI-DISCOVERY SPIKE (done overnight; main build untouched, no SDPA rebuild yet)
Decision: do the highest-value, lowest-risk M3 step now — decode the attention kernarg ABI
(the plan's flagged "hardest bit") via a standalone AOT build, WITHOUT reconfiguring the main
hipDNN build (protects the M2 morning deliverable). Deferred to daytime: -DHIPDNN_ENABLE_SDPA=ON
rebuild + FlydslAttentionNative.cpp authoring + SDPA graph_match.

- Driver `flydsl_build/build_flash_attn_real.py <H> <D> [causal=1] [bf16]`; built
  `flash_attn_real_h8_d128_causal_bf16_gfx950.hsaco` (24208B). Factory =
  `kernels/attention/flash_attn_gfx950.py::build_flash_attn_dualwave_swp_module(num_heads,
  head_dim, causal, dtype_str, ...)`. Dense launch wrapper: `launch(q,k,v,o,B,S)` with
  Q/K/V/O = [B,S,H,D] bf16. D must be 64 or 128; dtype bf16/f16 only; gfx950-only.
- Kernel symbol `flash_attn_dualwave_swp_gfx950_kernel_0`. kernarg_segment_size = 608.
  block_size = 512 (known_block_size=[512,1,1]). LDS group_segment_fixed_size = 68096 B.
- Kernarg = 12 tensors (each ptr@8B + descriptor@40B, stride 48B ⇒ 576B) + 8 i32 scalars
  (offsets 576,580,...,604). 12*48 + 8*4 = 608. ✓
- 4D tensor descriptor = 40B = {4×i32 dims, 3×i64 outer strides}; innermost stride is the
  compile-time 1 (same elision as the RMSNorm 2D {2×i32,1×i64}=16B). Row-major [B,S,H,D]:
  s2=D, s1=H*D, s0=S*H*D. ALL 12 by_value slots are uniformly 40B (kernel treats every tensor
  as rank-4; placeholders get zero/dummy descriptors).
- Tensor slot order (= device kernel signature, flash_attn_gfx950.py:191): 0 Q, 1 K, 2 V,
  3 O, 4 LSE, 5 DebugCounts, 6 CuSeqQ, 7 CuSeqKv, 8 BlockTable, 9 Bias, 10 AlibiSlopes, 11 Sink.
  Dense causal prefill uses Q/K/V/O; the other 8 are unused-mode placeholders (need dummy
  device buffers + how the launch wrapper's _prep_* helpers fill them — decode when authoring).
- Scalar order (arg24..31): seq_len, seq_len_kv, stride_q_n, stride_kv_n, head_dim_runtime,
  block_table_stride, bias_stride0, alibi_stride_b.
- Baked-into-HSACO (family identity): num_heads, head_dim, causal, dtype (+ knobs). Free at
  runtime: seq_len (=S), batch (=B) — one HSACO spans all prefill lengths for its head-config.
- Dump dir with stage-01/20/21 for full field-layout confirmation:
  /tmp/flydsl-build-fa-h8-d128-6d_vxanb (ephemeral; rebuild to regenerate).

## M3 COMPLETE — flyDSL attention (bf16 prefill) SELECTED + EXECUTED through hipDNN (2026-09-11 ~11:03)
Exit criterion MET on gfx950 (MI355X): for bf16 causal prefill SDPA, `hipkernel:FlydslAttention`
is in `get_ranked_engine_ids()`, the matched HSACO is raw-loaded per head-config, and output
matches an fp32 host causal-SDPA reference.
- **Proof (run_attention_bf16_app.sh):** H=8 → max_abs_err **5.26e-4**; H=16 → **5.75e-4** (both
  ≪ 5e-2 tol). The ~5e-4 error (not ~1e-2) is byte-level confirmation the 608B kernarg ABI decode,
  the BAKED scale (Q pre-scaled by 1/√D), causal masking, and BSHD strides are all correct.
- **Pack authored:** `packs/FlydslAttentionNative.cpp` (graph_match mirrors the dense SDPA
  isApplicable + the flyDSL divergence: REQUIRE attn_scale ≈ 1/√head_size since scale is baked;
  kernel_match pins only {dtype,head_size,num_query_heads,num_kv_heads,causal} — seq_len/batch are
  runtime; dispatch raw-loads the matched HSACO and packs the 608B kernarg: slots 0-3 = Q/K/V/O
  ptr+40B desc, slots 4-11 = O ptr+desc placeholder, scalars @576-604). Wired via IngestorPacks
  .hpp/.cpp + kernel_ingestor_engine/CMakeLists.txt.
- **Descriptor pack:** `flydsl_descriptors_attention/flydsl_attention/` (6 JSON, prefix f1d5b1a7-),
  two kernelDescriptors h8/h16 (metadata {dtype BF16, head_size 128, num_query_heads/kv 8|16,
  causal 1, hsaco, symbol, block_threads 512, block_m 256}).
- **Harness:** `app/flydsl_attention_bf16_app.cpp` builds an SDPA graph via
  `graph->sdpa(q,k,v,attr)` with `attr.set_attn_scale(1/√128).set_causal_mask(true)`, logical
  (B,H,S,D)=(1,H,256,128) + token-major BSHD strides, loops H∈{8,16}. Run:
  `env -u ROCM_PATH bash flydsl_poc_scratch/app/run_attention_bf16_app.sh`.
- **Key finding (StateManager isolation):** each descriptor pack's own graph_match symbol → its own
  KernelIngestorStateManager → its own isolated catalog. FlydslAttention returning nullopt cannot
  empty the dense/tiled/asm SDPA catalogs — they graph_match the same node independently and merge
  at ranking. This is exactly what makes the M4 3-engine bake-off viable.
- Everything branch-only, nothing pushed. Both HSACOs (h8/h16, 24208B) live in flydsl_poc_scratch/.
- NEXT (M4): 3-engine bake-off — enable ASM_SDPA + rocKE alongside flyDSL, de-risk with 2 engines
  first, then confirm all three enumerate on one prefill shape and autotune ranks a winner.

## M4 IN PROGRESS — 2-engine co-enumeration ACHIEVED (2026-09-11 ~11:10)
De-risk milestone (plan "confirm all engines enumerate") half-done: **ASM_SDPA_ENGINE + FlydslAttention
both enumerate for one SDPA shape on gfx950.** Key findings + decisions:
- **ASM_SDPA is already compiled INTO libhip_kernel_provider.so** (AsmSdpaEngine.cpp.o in
  hip_kernel_provider_impl) — no separate plugin, already loaded. rocKE is OFF
  (HIPKERNELPROVIDER_ENABLE_ROCKE=OFF; needs a rebuild to add as 3rd engine — deferred).
- **The gfx950 asm_sdpa FWD catalog has EXACTLY 2 entries, both NON-CAUSAL (mask=0):**
  `fwd_hd128_bf16.co` (hd128) and `fwd_hd192_hd128_bf16.co` (hd192x128). Source:
  build/dnn-providers/.../asm_sdpa_engine/generated/asm_fmha_v3_fwd_configs.hpp (ADD_CFG rows,
  arch="gfx950"). Causal (mask=2) variants exist ONLY for gfx942. asm_sdpa BWD is gfx942-only.
  ⇒ asm_sdpa CANNOT serve a causal gfx950 prefill; my M3 causal shape is exactly why it declined
  ("Could not find matching kernel for parameter combination", SdpaFwdPlanBuilder.cpp:374/565).
- **DECISION: the common bake-off shape = NON-CAUSAL, bf16, head_dim=128.** That's the intersection
  of flyDSL (I can AOT any (H,D,causal) tuple) and the gfx950 asm_sdpa catalog. Built NON-CAUSAL
  flyDSL HSACOs `flash_attn_real_h{8,16}_d128_noncausal_bf16_gfx950.hsaco` (23184B) via
  flydsl_build/build_flash_attn_real.py <H> 128 0 bf16. Same symbol/608B ABI (causal just baked off).
- Added 2 non-causal kernelDescriptors (ids ...0009 h8, ...000a h16, causal=0) to
  flydsl_attention.kdp.json. graph_match ALREADY handles NO_MASK→causal=0 (no code change needed);
  kernel_match now finds the non-causal instances. Pure-JSON change, no rebuild.
- **PROVEN (run: bakeoff_app 0):** for H=8 and H=16, non-causal bf16 D=128, get_ranked_engine_ids()
  returns BOTH `ASM_SDPA_ENGINE` (id 4714091817493728420) and `FlydslAttention` (id
  5031429073904537099). Harness `app/flydsl_attention_bakeoff_app.cpp` (argv[1]=causal flag, no pin,
  prints ranked engines by name via engineNameOrHex). Confirms StateManager-per-graph_match isolation
  lets independent engines co-serve one SDPA node → merged at ranking.
- NEXT (finish M4): (1) execute EACH ranked engine on the non-causal shape + numeric check vs a host
  non-causal SDPA ref (proves both actually run, not just enumerate); (2) autotune (build_plans(ALL) +
  benchmark) to rank a winner per shape; (3) optionally rebuild with ENABLE_ROCKE=ON for the 3rd engine.

### M4 UPDATE — 2-engine EXECUTE + NUMERICS proven (2026-09-11 ~11:14)
Both enumerated engines now EXECUTE the SAME non-causal bf16 d128 (B1 H8 S256) SDPA problem and match
an fp32 host reference: ASM_SDPA_ENGINE max_abs_err **1.51e-4**, FlydslAttention **1.51e-4** (both OK,
< 5e-2). Harness `app/flydsl_attention_bakeoff_exec_app.cpp` (enumerate no-pin → for each ranked id,
build fresh graph pinned to it, execute, compare). This is the 2-engine bake-off fully de-risked:
enumerate + execute + numerics for flyDSL AND asm_sdpa on one problem.
- **REQUIRED RUNTIME ENV for asm_sdpa:** `HIPDNN_AITER_ASM_DIR=$REPO/build/hip_kernel_provider/asm_kernels`.
  Without it, asm_sdpa's buildPlan tries to load its .co from the hardcoded install path
  /opt/rocm/lib/hipdnn_plugins/engines/hip_kernel_provider/asm_kernels/gfx950/fmha_v3_fwd/fwd_hd128_bf16.co
  (doesn't exist in our base-rocm setup) and fails with HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR. The .co
  files ARE built under build/hip_kernel_provider/asm_kernels/; AsmKernelPath.hpp checks HIPDNN_AITER_ASM_DIR
  first. Add this to any bake-off run script.
- Investigating (agent): whether rocKE can serve this SAME non-causal bf16 d128 shape and what enabling
  it (HIPKERNELPROVIDER_ENABLE_ROCKE, currently OFF) requires — user asked to include rocKE as 3rd engine.

### M4 AUTOTUNE DONE (2026-09-11 ~11:22) — ranked winner per shape, 2 engines
New app `flydsl_attention_autotune_app.cpp` (+ CMake target): enumerates engines (no pin), then per
engine builds plan, checks numerics once vs fp32 host non-causal-SDPA ref, warms up 5×, times reps=50
via hipEvent per-iter → median, and prints a per-shape WINNER RANKING (fastest first). Any future
engine that enumerates (rocKE) is timed+ranked with ZERO code change. Run:
`bash run_attention_bakeoff_app.sh ./build/flydsl_attention_autotune_app 50`.

Result on MI355X (gfx950), non-causal bf16 d128, 4 prefill shapes — BOTH engines numerically OK
(err ~1.5e-4), asm_sdpa wins all 4 by ~8%, flyDSL correctly ranked #2 (close):
  B1H8S256D128 : asm 0.0124ms  < flyDSL 0.0135ms
  B1H8S512D128 : asm 0.0167ms  < flyDSL 0.0181ms
  B2H8S512D128 : asm 0.0169ms  < flyDSL 0.0183ms
  B1H16S256D128: asm 0.0127ms  < flyDSL 0.0137ms
FlydslAttention prints as hex id 0x45D33A15C6A70E0B (=5031429073904537099) — EngineNames has no
friendly mapping for ingestor packs; cosmetic only. This satisfies M4 exit "flyDSL sits in the ring,
autotune ranks a winner per shape, flyDSL at worst correctly ranked." Only gap to full M4 = 3rd engine
(rocKE).

### M4 COMPLETE — 3-ENGINE BAKE-OFF ACHIEVED (2026-09-11 ~11:46), rocKE in the ring
User confirmed rocKE should be easy here (this branch IS Brian Harrison's rocke-gfx950 work). It was:
NO C++ changes, NO provider rebuild. Route A (offline pack + runtime env var).

Steps that worked:
1. Fetched the ONE missing dep — rocm_kpack PYTHON packer — blobless-sparse from
   ROCm/rocm-systems@a022846 subdir shared/kpack/python → /home/AMD/brpepers/rocm-dev-work/kpack-src.
   pip install msgpack zstandard into flydsl-venv. (C++ kpack reader already shipped in base rocm.)
2. Authored a 1-kernel source root flydsl_poc_scratch/rocke_pack_src/gfx950_attention_dense/: copied
   the 5 shared matcher/score/dispatch JSONs from descriptor-packaging/examples/.../gfx950_attention_dense
   verbatim, and wrote a .kdp.json with ONE kernelDescriptor cloned from Brian's bf16 non-causal 8/8/128
   template (was batch4/seq1024) with batch=1, seqlen_q=seqlen_kv=256, causal=false. (Brian's 1694-variant
   sweep had NO 8/8/128 non-causal at seqlen 256 — smallest was 2048 — so cloned+reshaped one spec.)
3. Packed offline: rocke_build/build_rocke_attention_pack.sh runs descriptor-packaging/tools/hkp_pack.py
   (PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:kpack, ROCKE_COMGR_LIB=base
   libamd_comgr.so.3, AMD_COMGR_CACHE_DIR=/tmp, --hipcc base hipcc). Output: rocke_pack_out/gfx950/
   {kpack/hip_kernel_provider_gfx950.kpack, gfx950_attention_dense/*.json rewritten kind:kpack}. Symbol
   rocke_attention_dense_d128_hq8_kv8_bn64_bf16_sq256_sk256_full_lazyrs (full=non-causal). Persisted from
   /tmp into flydsl_poc_scratch/rocke_pack_out.
4. Ran with HIPDNN_DESCRIPTOR_RUNTIME_DIR=flydsl_poc_scratch/rocke_pack_out/gfx950 (ADDITIVE to the flyDSL
   HIPDNN_DESCRIPTOR_DIR). No rebuild — Gfx950AttentionDenseNative matcher already compiled in.

RESULT — 3 engines enumerate + EXECUTE + autotune-rank for B1 H8 S256 D128 non-causal bf16, all err 1.51e-4:
   1. ASM_SDPA_ENGINE          0.0124 ms  (winner)   id  4714091817493728420
   2. FlydslAttention          0.0136 ms             id  5031429073904537099 (0x45D33A15C6A70E0B)
   3. Gfx950AttentionDense     0.0166 ms  (rocKE)    id -8518255706404043867 (0x89C9139111D7C3A5)
Other 3 swept shapes stay 2-engine (only packed the H8/S256 rocKE variant). To add rocKE shapes: clone
more spec entries into rocke_pack_src .kdp.json and re-run build_rocke_attention_pack.sh (no rebuild).
Benign warnings: descriptor loader "ignoring provenance" (packer's provenance block) + a stride-order
note — both harmless, numerics correct. NOTHING pushed (branch users/brpepers/hipdnn-flydsl-poc only).

### M4 rocKE FINDING (agent investigation, 2026-09-11 ~11:18) — user asked "can rocKE serve this shape?"
YES, rocKE's dense attention CAN serve our exact shape — but it's real build work, not a flag flip.
- rocKE attention reaches hipDNN as the ingestor native pack **`hipkernel:Gfx950AttentionDense`**
  (IngestorPacks.cpp:51; matcher Gfx950AttentionDenseNative.cpp — the SAME file I mirrored for
  FlydslAttentionNative). Its C++ graph_match ALREADY accepts bf16/fp16 × d64/d128 × {no-mask,
  top-left-causal, bottom-right@Sq==Skv}, BSHD prefill, GQA — i.e. our (1,8,256,128) bf16 non-causal
  graph matches. rocKE's own dispatch cap (rocke/library/dispatch/attention/gfx950.py:190) =
  arches gfx950, dtypes bf16/fp16, causal/full prefill (CK-1 persistent flash-attn).
- WHY it doesn't enumerate now: the pack registers native symbols ONLY; it needs INSTALLED descriptor
  JSON (kind:kpack) on disk, produced by the rocKE packaging pipeline. This POC worktree DELIBERATELY
  disables it: top CMakeLists comments out add_subdirectory(descriptor-packaging) ("skip … our flyDSL
  path uses a compiled-in native pack, not a packed .kpack"). Only conv_fwd+pointwise runtime
  descriptors ship; NO attention descriptors present.
- To enable = TWO flags + deps: HIPKERNELPROVIDER_ENABLE_ROCKE=ON (builds rocke pyenv/wheel + pybind IR
  emitter — NOT an IEngine) AND HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON (produces descriptors),
  needing rocke wheel-env + hipcc + amd_comgr + rocm_kpack, PLUS re-enabling descriptor-packaging (or
  hand-staging lowered kind:kpack descriptors + HIPDNN_DESCRIPTOR_RUNTIME_DIR). The checked-in
  descriptors under descriptor-packaging/examples/.../gfx950_attention_dense/*.json are kind:rocke
  (authoring form) — the runtime loader REJECTS those; hkp_pack must lower them to kind:kpack first.
- **THROWAWAY SHORTCUT (mirrors flyDSL escape hatch):** build ONE rocKE gfx950 dense-attn kernel
  binary via the rocKE python authoring SDK (that ENABLE_ROCKE pyenv), extract its bare kernel binary
  + symbol, and hand-author a native raw-load pack + descriptor (kind:embedded_source) exactly like
  FlydslAttentionNative — avoids the full kpack packaging pipeline. Still needs the rocke pyenv built.
- DECISION PENDING (asked user): full-packaging path vs bubblegum raw-load path vs defer rocKE (2-engine
  bake-off already proves best-of-breed selection). Key files: Container.cpp:50-146 (engine enum),
  Gfx950AttentionDenseNative.cpp:398-600 (gating), CMakeLists.txt:49/341-372, HkpPackaging.cmake:739-822.
