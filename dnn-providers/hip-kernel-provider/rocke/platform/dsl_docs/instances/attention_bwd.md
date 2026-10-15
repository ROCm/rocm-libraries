# Attention Backward (dQ, dK, dV): design of the rocKE SDPA backward kernels

This document explains how the rocKE SDPA backward kernels work and why they are
built the way they are: the math, the kernel decomposition, the per-step tile
program, the memory and register plans, masking and sequence handling, the
per-arch differences, the tuning model, the numerics, and the invariants the
tests hold. The host contract (statistics format, kernarg ABI, workspace,
predicate, declaration) is at the end (section 12).

Source of truth is the code under `library/` (paths below are relative to the
rocke root). Anything described as "not built" exists only as a registered
knob value, a refused value, or a recorded design; no shipped kernel emits it.

## 0. Overview

### 0.1 What exists

* **One family is built: the layout-general backward.** It serves fp16 / bf16,
  head sizes 32 / 64 / 128, any non-negative B / H / S element strides (D
  stride 1), fixed, padded and THD (ragged) lengths, MHA / GQA / MQA and
  `h_k != h_v`, no mask / causal top-left / causal bottom-right / two-sided
  sliding windows, and `s_q = 1`, on gfx942, gfx950, gfx1151 and gfx1201.
* Three kernels per backward on one stream: **prep** (statistics and workspace
  initialisation), **main** (KV-parallel, all five GEMMs), **convert** (fp32
  workspace to the output dtype), plus two more converts when dK / dV are
  accumulated with atomics.
* dQ is accumulated with fp32 atomics into a workspace (the "atomic sink",
  the default). A separate Q-parallel dQ kernel ("split dQ",
  `dq_mode = "split"`, gfx942 / gfx950) is built: the main kernel then
  computes dK / dV only and the dQ kernel writes dQ once, deterministically
  (2.4).
* The **dense** family (shape-specialised, per-arch builders running the
  shared tile step behind a dense shell) is **built** (section 8.2); routing
  takes it only for the dense keys of its tuned table (keys whose dense
  winner passed the bench's confirm stage, 9.3) and the general route for every
  other request. The **unified**
  family (THD queries over a paged KV cache, a rocKE-native entry point) is
  **built** (section 8.3, `attention_bwd_unified.md`).
* Every tile, residency, path, schedule and codegen decision is a defaulted
  spec field or a plan value, registered as a tuning axis
  (`dispatch/attention_bwd/tuning_common.py`). Values whose emission is not
  built are refused before any IR is emitted (`attention_bwd._not_built`).

### 0.2 File map

| File | Content |
|---|---|
| `library/kernels/common/attention_bwd.py` | `AttnBwdSpec`, per-arch default geometry, legality (`validate_attn_bwd_spec`), register model, LDS footprint, kernel names, ABI lists (`attn_bwd_params`), `build_attn_bwd_main` (CTA decode, addressing, q loops, dK / dV epilogue) |
| `library/kernels/common/_attention_bwd_body.py` | `BwdTileGeometry`, `BwdTileStep` (K / V staging, G0..G4, softmax gradient, P / dS hand-off, dQ sink call, epilogue readers), `StatsView`, `BwdLds`, the Q / dO staging copies |
| `library/kernels/common/_attention_bwd_gemm.py` | `BlockGemm` (multi-wave block GEMM over a warp grid), `LdsOperand`, the C-to-A relabel entry points, the inline-asm MFMA clusters (`mma_agpr_asm`, `mma_agpr_asm_steps`) |
| `library/kernels/common/_attention_bwd_frag.py` | lane-map evaluation on an integer builder, atom facts, the checked C-to-A relabel plan, K-permuted B coordinates, the LDS fragment loaders including `ds_read_tr16_b64`, the LDS views (`LdsView`, the line-XOR `XorLdsView`) |
| `library/kernels/common/_attention_bwd_lds.py` | the LDS phase planner (`plan_bwd_lds`), the XT image and row-major XOR layouts (`XtLayout`, `RowXorLayout`) and the XT writer distribution, the DMA-slab layout of the `dma` path, conflict prediction and the per-image swizzle choice |
| `library/kernels/common/_attention_bwd_ring.py` | the Q / dO / stats prefetch ring (`ring_depth`, register or DMA path): prologue, per-step issue, waits and the DMA wait model (3.5) |
| `library/kernels/common/_attention_bwd_counted.py` | the counted in-loop waits (`loop_sync = "counted"`): next-step loads, the `vmcnt(N)` model (3.5) |
| `library/kernels/common/_attention_bwd_valu.py` | the softmax-gradient VALU levers (`bf16_cast`, `bwd_exp2`, `stats_fma`, `stats_vec`; 3.2) |
| `library/kernels/common/_attention_bwd_tiles.py` | legality of the small-tile warp grids (replicated and transposed G0 / G2 grids, 3.1) |
| `library/kernels/common/attention_bwd_dq.py`, `_attention_bwd_dqbody.py` | the split dQ kernel (`AttnBwdDqSpec`, `build_attn_bwd_dq`) and its body (`BwdDqGeometry`, `BwdDqStep`, the key-tile range of a block of query rows; 2.4) |
| `library/kernels/common/attention_bwd_unified.py`, `attention_bwd_unified_plan.py` | the unified family (paged KV; 8.3) |
| `library/kernels/common/_attention_bwd_caps.py` | per-arch capability checks, atom selection, the asm MFMA mnemonics and hazard wait states |
| `library/kernels/common/_attention_bwd_band.py` | inverse band (`q_tile_range`), tile classifier, interior q range, mask policies, dK / dV store rule, `HostInt` (the integer builder on Python ints that evaluates the band and the DMA lane map on the host) |
| `library/kernels/common/_attention_bwd_addr.py` | sequence decode (batched, THD), strided tensor and workspace address generators, packed rows, paged cache addressing; the family's shared emitters `as_i32` / `as_i64` (an int as a constant, a value unchanged) and `deal_chunks` / `chunk_coords` (the chunks of a tile dealt round-robin to the threads, and a chunk's row and column) |
| `library/kernels/common/_attention_bwd_sinks.py` | dQ sinks (atomic, packed atomic; null, register and split-slice sinks recorded) |
| `library/kernels/common/_attention_bwd_worklist.py` | CTA decode, XCD remap, load-balance orders, head ranges, THD work list (layout, order, host model) |
| `library/kernels/common/_attention_bwd_sched.py` | stage table (count models of the synchronous step, the prefetch ring and the merged ring, the counted waits, the split-dQ main kernel's three stages, and the reference placement; sched-group / fence / priority emission hooks; the split dQ kernel's key-step table `DqStageTable`; 3.5a) |
| `library/kernels/common/attention_bwd_aux.py` | `AttnBwdAuxSpec`, prep, convert, the slice reduce (`build_attn_bwd_reduce`, 2.5a) and the paged roles |
| `library/kernels/common/attention_bwd_plan.py` | `AttnBwdRequest`, predicate, workspace, plan, routing, performance classes, declaration, `AttnBwdPolicy` |
| `library/kernels/common/attention_bwd_run.py` | launch glue: compile, pack kernargs, launch on one stream (prep, main, the converts or the dQ kernel) |
| `library/kernels/common/attention_dense_bwd_spec.py` | `AttentionDenseBwdSpec`, `ArchAttentionDenseBwdSpec` and `DENSE_BWD_BODY_VARIANTS` (the per-target spec class and its body variants), mask resolution, `body_spec()`, `validate_attn_dense_bwd_spec`, kernel names, the `rocke.attn_dense_bwd.v1` ABI list (`attn_dense_bwd_params`) |
| `library/kernels/common/_attention_dense_bwd_shell.py` | `DenseFixedAddr` / `BufferRowSlab`, `DiagonalBand` / `DenseNoMask` (compile-time classification), the dense CTA decode (grid, XCD remap, persistent grid), `emit_dense_bwd_main` |
| `library/kernels/{gfx942,gfx950}/attention_dense_bwd.py` | `Gfx942AttentionDenseBwdSpec` / `Gfx950AttentionDenseBwdSpec`: `ArchAttentionDenseBwdSpec` (`attention_dense_bwd_spec.py`, the shared spec plus `body_variant`) bound to the target, and `build_attention_dense_bwd(spec, *, arch)` |
| `library/kernels/common/attention_dense_bwd_plan.py` | `dense_bwd_spec_for`, `attn_dense_bwd_plan` (general prep / converts / workspace, dense main launch, dense dQ launch under split) |
| `library/kernels/common/attention_dense_bwd_dq.py` | `AttentionDenseBwdDqSpec`, `validate_attn_dense_bwd_dq_spec`, `dense_dq_static_classes`, the `rocke.attn_dense_bwd_dq.v1` ABI list, `build_attention_dense_bwd_dq` (the dense split dQ kernel) |
| `library/kernels/common/attention_bwd_comgr.py` | the production comgr of the compile-only paths without importing torch (`ROCKE_COMGR_LIB` pinned by path) and the process-tree comgr cache directory (11) |
| `library/kernels/common/attention_bwd_cache.py` | the on-disk code-object cache shared by the run glue (opt-in, `ROCKE_BWD_HSACO_CACHE`), the tests' pre-build, the sweep and the timing harness: toolchain identity, source hash, keys, atomic entries |
| `library/dispatch/attention_bwd/tuning_common.py` | the knob registry (`KnobAxis`), `backward_spec_legality` and the known-wrong values `BWD_KNOWN_WRONG_KNOBS` (empty) |
| `library/dispatch/attention_bwd/unified_tuning.py` | the unified family's knob axes, `unified_spec_legality` and the tuned page-tile table (8.3) |
| `library/dispatch/attention_bwd/registry.py` | the dispatcher: one candidate per (family, arch) in `ATTENTION_BWD_REGISTRY`, `attention_bwd_ranker`, `dispatch_attention_bwd`, `launch_attention_bwd`, `run_attention_bwd` (8.4) |
| `library/tests/test_attention_bwd_noatomic.py` | the slice sink and the atomic-free backward (2.5a): knob, plan, slice ownership, no atomic instruction in the IR, determinism policy, registry and sweep, device numerics and bitwise repeats |
| `library/tests/test_attention_bwd_access.py` | the memory-access levers (3.9): legality, IR forms, plan narrowing, the resource gate of every new value, device bit-for-bit checks |
| `library/dispatch/attention_bwd/dense_tuning.py`, `routing.py` | the dense knob registry and tuned dense table; `plan_attention_bwd` (routing among the families), `attention_bwd_declaration` (the declaration under the routed policy) |
| `library/dispatch/attention_bwd/gfx942_tuning.py`, `gfx950_tuning.py`, `tuned_tables.py` | the per-arch tuned tables (general, dense, performance status) and the entry check every entry passes, also used on the sweep's emitted entries (9.3) |
| `library/tests/dispatch/attention_bwd/test_serving_policy.py` | the shipped serving policy (9.3): every accepted request served on gfx942 / gfx950, the status recorded and published per class and never steering the plan, the remaining declines, the opt-in performance gate, tuned and untuned keys |
| `library/benchmarks/common/attention_bwd_sweep.py` | the funnel sweep driver; `attention_bwd_compile.py` (compile cache, crash isolation), `attention_bwd_bench.py` (paired timing harness), `attention_bwd_census.json` (shape census, no measurements) |
| `library/benchmarks/common/attention_bwd_rocke_arm.py`, `attention_bwd_unified_bench.py` | the rocKE arm of the timing harness (request from the bench tensors, configured or routed plan, launch); the unified family's bench driver |
| `library/kernels/common/attention_bwd_unified_run.py` | launch glue of the unified family (8.3) |
| `library/kernels/common/_attention_bwd_paged.py` | paged KV addressing of the unified family: block-table reads, the page pointer, the page workspace rows |
| `library/kernels/common/_attention_bwd_refusals.py` | the recorded resource-gate refusals (`RESOURCE_REFUSED`, `LAYOUT_SPILLING_COMBOS`; 11) |
| `library/kernels/common/_lse_store.py` | the natural-log fp32 LSE store of the MFMA / WMMA attention forward bodies (an `AttnFwdExt.epilogue_hook`): the statistics this family consumes (12.1) |
| `platform/python/rocke/helpers/attention_fwd_ext.py`, `attention_band.py`, `wmma_attention.py`, `_attention_shared.py` | forward-side helpers (default off; the forward default IR is unchanged): the opt-in extension points of the forward inner body (`AttnFwdExt`: score and epilogue hooks, used for the LSE output), the runtime two-sided band (`AttnRuntimeBounds`, `BandEmitter`), the wave32 WMMA forward body moved out of `mfma_attention.py` (re-exported there) and the pieces both bodies share |
| `platform/python/rocke/numeric/sdpa_reference.py` | the torch-free float64 SDPA forward and backward oracle of the device tests |
| `platform/python/rocke/runtime/_comgr_fresh_process.py` | the child process of `compile_kernel(..., fresh_process=True)` (3.5a: `iglp_opt(1)` kernels) |
| `library/tests/test_attention_bwd_ir_golden.py`, `tests/golden/attention_bwd_ir_sha256.json` | LLVM-IR golden hashes of the main, split dQ, aux and unified kernels on the four targets at every LLVM flavor (11) |
| `library/tests/parity/barrier_fence_emit.py`, `tests/test_barrier_fence_parity_emit.py` | the Python reference emitter of the barrier `fence` attribute and its checks (11) |

### 0.3 The C++ engine

The backward modules are Python-only family glue: their selectors, specs,
plans, knob registries and routing exist in Python only, and their kernels
are lowered and compiled with the Python lowering backend. The change adds
one thing to the C++ engine, the mirror of a Python change that other families
share, and relies on two that the platform now carries itself:

* **The barrier `fence` attribute** of `tile.sync`, `tile.sync_lds_only` and
  `tile.s_barrier_bare` (every backward barrier carries `fence="compiler"`,
  3.5): `platform/cpp/core/ir/ir_flow.cpp` (`rocke_b_sync_fence`,
  `rocke_b_sync_lds_only_fence`), `platform/cpp/include/rocke/ir.h`,
  `platform/cpp/core/lower_llvm/control.cpp` (the compiler memory clobber
  after `s.barrier`, and the workgroup release / acquire fences around it) and
  `platform/cpp/core/lower_hip/lower_hip_mma.cpp` (the HIP lowering). An
  unfenced barrier keeps its emission, so no other family's IR moves. Tested
  by `platform/tests/core/test_barrier_fence.cpp` (C++ unit test, `ctest`
  label `ckc;unit;host`), `platform/tests/core/test_barrier_fence.py`
  (Python, with the cross-engine comparison when `rocke_engine` is built) and
  the parity emitter `library/tests/parity/barrier_fence_emit.py` (the C++
  engine lowers its serialized probe and backward kernels to the Python
  bytes, `tests/test_barrier_fence_parity_emit.py`).
* **The LLVM flavor of a pinned torch comgr**: both engines resolve an AUTO
  flavor from the LLVM version of the comgr they load (`platform/cpp/core/
  lower_llvm/core.cpp`, `ll_resolve_flavor`; `core/lower_llvm.py`,
  `_detect_llvm_flavor`), so a `ROCKE_COMGR_LIB` naming torch's bundled comgr
  gives one flavor in both (`platform/tests/test_rocke.py`,
  `TestCrossEngineFlavorAutoResolution`). The package's `version.py` still
  dates the library for the toolchain identity
  (`runtime/comgr.py::resolved_lib_rocm_version`, Python only).
* **The gfx942 3-D tiled attention kernel name**: the C++ glue
  (`platform/cpp/instances/gfx942/attention_tiled_3d_gfx942_attn_tiled_3d_public_glue.cpp`)
  adds the `fnuz` name token of an fp8 KV cache that the Python builder
  already emits, so the two engines name that kernel identically (a
  byte-identity fix of the forward family).

**C++ engine waiver.** The attention backward families (general,
split dQ, aux, dense, unified) have no C++ twin in this change: no C++
selector, spec, plan, knob registry, routing or kernel builder. The
dual-engine rule of `platform/AGENTS.md` (byte-identical IR from both
engines) applies to every family the C++ engine builds and serves; the
backward family is not one of them: the C++ engine does not build or serve a
backward kernel and no C++ caller plans a backward request, so the provider's
rocKE path does not offer this backward to hipDNN until a C++ port (or a
Python-serving path) lands; hipDNN serves the backward with its other engines
meanwhile.
What the backward kernels share with the C++ engine, the barrier fence, is
mirrored and tested above, and `check_byte_identity.py` (llvm20, llvm22,
llvm23) holds the IR of every dual-engine family unchanged. The backward
kernels' own IR is pinned by the Python golden hashes
(`test_attention_bwd_ir_golden.py`, `test_attention_dense_bwd_ir.py`).
Follow-up: a C++ port of the backward plan and kernels when a provider serves
the backward without Python. The reviewer's sign-off on this waiver is
recorded in the pull request.

### 0.4 Family glue and the helper layer

The family's emitters live in `library/kernels/common/_attention_bwd_*.py`,
not in `platform/python/rocke/helpers/`. By `platform/AGENTS.md` ("helpers/
placement") an SSA emitter goes to `helpers/` when a second family uses it (or
plausibly will) and it has a C++ port path; single-family glue stays with the
family. The backward has one consumer family and no C++ port (0.3), so nothing
is promoted; a piece moves when a second family needs it. The map against the
existing helpers:

| Family module | What it holds | Nearest helper | Relation |
|---|---|---|---|
| `_attention_bwd_gemm.py` | `BlockGemm` over a warp grid, the C-to-A relabel entry points, the inline-asm MFMA clusters | `helpers/atoms.py`, `helpers/mfma_attention_bwd.py` | uses the atoms; `mfma_attention_bwd.py` is the older per-lane backward inner body of `fmha_bwd.py` (8.4), a different algorithm |
| `_attention_bwd_frag.py` | lane maps, the checked relabel plan, K-permuted B coordinates, LDS fragment loaders (`ds_read_tr16_b64`) | `helpers/layouts.py` (`TransposeLdsReader`) | the forward reader covers K^T tiles of one GEMM; the backward's five-GEMM relabel and permuted coordinates have one user |
| `_attention_bwd_lds.py` | the LDS phase planner, the XT images, the row XOR layouts | `helpers/layouts.py` (`LdsLayout`, `xor_swizzle_bytes`) | same XOR idea; the phase planner (buffers live per step phase) is backward-specific |
| `_attention_bwd_ring.py`, `_attention_bwd_counted.py` | the Q / dO prefetch ring (register and DMA paths) and the counted waits | `helpers/pipeline.py` (`SoftwarePipeline`), `helpers/loads.py` (`AsyncTileLoader`, `AsyncPingPongLoader`) | the ring's wait model counts the step's dQ atomics in `vmcnt`; the helpers have no atomic-aware wait model |
| `_attention_bwd_band.py` | the inverse band (q-tile range of a key tile), the tile classifier, mask policies, the dK / dV store rule | `helpers/attention_band.py` (`AttnRuntimeBounds`, `RowBand`, `BandEmitter`) | the same keep rule; the inverse band and the store rule are backward-only |
| `_attention_bwd_sched.py` | the stage tables and their hint emission | `helpers/schedule.py` (`SchedulePolicy`) | built on the same intrinsics; the tables model this step's matrix-op and atomic counts |
| `_attention_bwd_addr.py`, `_attention_bwd_paged.py`, `_attention_bwd_worklist.py` | strided, THD, workspace and paged addressing; CTA decode, XCD remap, work list | `helpers/tensor_view.py`, `helpers/grid.py` | family-specific forms (64-bit batch / head rebase, clamped workspace rows, block-id guard); the XCD remap matches the chiplet helper's transform (tested) |
| `_attention_bwd_body.py`, `_attention_bwd_dqbody.py`, `_attention_bwd_sinks.py`, `_attention_bwd_valu.py`, `_attention_bwd_tiles.py`, `_attention_bwd_caps.py`, `_attention_bwd_refusals.py`, `_attention_dense_bwd_shell.py` | the tile step, the dQ body, the dQ sinks, the VALU levers, tile legality, arch capabilities, the recorded refusals, the dense shell | none | backward-only |

**Shared code: one copy each.** A helper the family's modules repeat has one
definition. Where it lives follows the placement rule above: a host-only
utility (no IR) may go to a shared helper module, an emitter stays in the
family glue (it would need a C++ port as a `helpers/` emitter, 0.3):

| Helper | Home | Users |
|---|---|---|
| `ceil_div`, `align_up` (host integer math) | `helpers/spec.py` | the plans, the stage tables, the dense forward's `align_up`, the conv families, `helpers/split_k.py`, `helpers/streamk.py`, `ceil_div_grid` |
| `non_default_fields`, `salted_kernel_name` (spec-keyed kernel names: the base key plus the SHA-1 salt of every other non-default field) | `helpers/spec.py` | `AttnBwdSpec`, `AttnBwdDqSpec`, `AttnBwdUnifiedSpec`, `AttentionDenseBwdSpec`, `AttentionDenseBwdDqSpec` |
| `I32_FAR = 2**30` (a limit past every row, key, length and band bound) | `helpers/attention_band.py` | the forward band, the inverse band, the workspace row clamp, the plans' 2**30 bound checks |
| `as_i32`, `as_i64`, `deal_chunks`, `chunk_coords` (emitters) | `_attention_bwd_addr.py` | every backward module and `_lse_store.py` |
| `HostInt` (host evaluation of the integer emitters) | `_attention_bwd_band.py` | the dense compile-time classification, the dense dQ kernel, the DMA lane map |
| `attn_bwd_stage_vec` (the staging width rule, over `used_strides()`) | `attention_bwd_plan.py` | the general and the unified plan |
| `ArchAttentionDenseBwdSpec` (the target-parameterised dense spec) | `attention_dense_bwd_spec.py` | `Gfx942AttentionDenseBwdSpec`, `Gfx950AttentionDenseBwdSpec` |

`BandEmitter._i32` in `helpers/attention_band.py` stays private there: the
platform cannot import the library glue. The tests share
`library/tests/_attention_bwd_test_utils.py` (`kernel_signature`,
`device_arch`, `find_bwd_case`, `distinct_configs`, `device_configs`,
`fake_bench_ok`). Every move above keeps the emitted IR of every family
byte-identical.

**Promotion candidates.** `_attention_bwd_lds.py`, `_ring.py`, `_frag.py`,
`_addr.py` and `_gemm.py` were checked against the helper atoms. Generic
pieces with a plausible second family: the checked C-to-A register relabel
(`_frag.plan_c_to_a` / `emit_c_to_a`, `_gemm.check_relabel_grids`: chained
GEMMs such as a fused forward's P V); the 16-byte-chunk XOR DMA slab layout
and its lane map (`_lds.DmaSlabLayout`, `_ring.dma_lane_source`: an LDS-DMA
tile layout next to `helpers/loads.py`); the atomic-aware `vmcnt` wait model
of the prefetch ring and the counted waits (an extension of
`helpers/pipeline.py`'s wait model); and the round-robin chunk dealing
(`deal_chunks` / `chunk_coords`, close to `helpers/loads.py`'s
`CoalescedTileLoader`). Each is an SSA emitter, so promoting it needs the C++
port the backward does not have (0.3); none is promoted now. The LDS phase
planner, the stage tables, the workspace and paged addressing and `BlockGemm`
(which overlaps `helpers/tiling`'s `TileMma`) are backward-specific or
covered by an existing helper.

## 1. Math

### 1.0 Notation

| Symbol | Meaning |
|---|---|
| `D` | the head dim (`d_qk == d_v`; `head_size` in the specs); never a row statistic: the softmax correction is `Dsum` |
| `S_q`, `S_kv`, `S_max` | sequence lengths (always subscripted; `S_max` a bound) |
| `S`, `S_raw` | in equations, GEMM names and register plans: the score tile (`S_raw = Q K^T`, `S = s * S_raw`); "S / dP" names the two G0 / G2 accumulator tiles |
| `B`, `H`, `S`, `D` in a layout name or stride tuple | tensor dims (`BSHD`, `(B, H, S, D)` strides): batch, head, sequence position, head dim |
| `h_q`, `h_k`, `h_v`, `G` | query / key / value head counts; the query heads per key / value head group (12.3) |
| `s` | the softmax scale (default `1 / sqrt(D)`); `scale_log2 = s * log2(e)` |
| `P`, `dP`, `dS` | the probabilities, `dO V^T`, the gradient of the scores |
| `Dsum` | the per-row softmax correction `rowsum(dO * O)` (`WS_DSUM`) |
| `kM0`, `kN0`, `kK4`, `W` | query rows per step, keys per CTA, the G4 key slice, waves per CTA |

### 1.1 Forward recap

For one (batch, query head) with `Q [S_q x D]`, `K, V [S_kv x D]` and scale
`s` (default `1 / sqrt(D)`):

```
S   = s * Q K^T                       (masked cells: -inf)
P   = softmax_rows(S)
O   = P V
LSE = log(sum_k exp(S[q, k]))         (natural log, fp32, one per row)
```

The backward never recomputes the row statistics: it takes `O` and `LSE` from
the forward (section 12.1). Any forward that stores `O` and a natural-log fp32
LSE is consumable.

### 1.2 Backward equations

Given `dO`, with `S_raw = Q K^T` (unscaled) recomputed tile by tile:

```
P   = exp(s * S_raw - LSE)            (recomputed; masked and dead cells 0)
dP  = dO V^T
Dsum = rowsum(dO * O)                 (= rowsum(P * dP))
dS  = P * (dP - Dsum)
dV  = P^T dO
dK  = s * dS^T Q
dQ  = s * dS K
```

`Dsum` is the per-row correction of the softmax Jacobian; computing it from `dO`
and `O` (one dot product per row) is cheaper than `rowsum(P * dP)` over all
keys, and it is computed once per row by prep instead of once per key tile.

The kernel names the five matrix products of one tile step:

| GEMM | Product | Shape (M x N over K) | Accumulator |
|---|---|---|---|
| G0 | `S_raw = Q K^T` | `kM0 x kN0` over `D` | fresh per step |
| G2 | `dP = dO V^T` | `kM0 x kN0` over `D` | fresh per step |
| G1 | `dV += P^T dO` | `kN0 x D` over `kM0` | carried across steps |
| G3 | `dK += dS^T Q` | `kN0 x D` over `kM0` | carried across steps |
| G4 | `dQ_part = dS K` | `kM0 x D` over `kN0` | fresh per step, sent to the dQ sink |

### 1.3 The log2 domain

The kernels use `exp2`. Prep converts the statistic once:
`lse2 = LSE * log2(e)`; the plan passes `scale_log2 = s * log2(e)`. Then

```
P = exp2(S_raw * scale_log2 - lse2) = exp(s * S_raw - LSE)
```

so the exponent costs one multiply and one subtract per cell
(`BwdTileStep._softmax_grad`), with no per-cell `log2(e)` multiply. A log2 LSE
fed as input is wrong data (it gives wrong gradients); the negative-control
tests feed one on purpose.

### 1.4 Where the scale is applied

The scale `s` must reach dK and dQ but not dV. Three placements are built; the
plan picks one (`scale_placement`, default `fold_ds`) and sets four fp32
multipliers (`attention_bwd_plan._scale_mults`):

| Placement | `ds_mult` (main, on dS) | `dk_mult` (main epilogue) | `dq_mult` (main, before each dQ atomic) | convert `mult` (dQ / dK) |
|---|---|---|---|---|
| `fold_ds` (default) | `s` | 1 | 1 | 1 / 1 |
| `convert` | 1 | `s` in direct mode, 1 in atomic mode | 1 | `s` / `s` in atomic mode |
| `dq_before_atomic` | 1 | as `convert` | `s` | 1 / `s` in atomic mode |

`fold_ds` scales each dS element once, before it is rounded to the I/O dtype,
so dK and dQ come out scaled with no extra multiply per output element. The
other placements keep dS unscaled and multiply later; they round at a
different point and are an order change (section 10).

The main kernel is compiled for its placement (`AttnBwdSpec.static_scale`,
plan-owned): the multipliers that are exactly 1.0 under it are not emitted
(`attention_bwd.static_unit_mults`). `x * 1.0 == x` for every fp32 `x`
(denormals are not flushed), so this is bitwise; `static_scale = "runtime"`
applies all three and serves every placement from one binary. The kernargs
stay in the ABI either way.

### 1.5 Precision and rounding points

* Every accumulation is fp32: S, dP, dK, dV and the dQ partial are MMA
  accumulators; Dsum is an fp32 dot product in prep; dQ (and, in atomic mode,
  dK / dV) are summed with fp32 atomics.
* P and dS are rounded to the I/O dtype (fp16 / bf16) only because they are
  MMA operands: P and dS as the A operands of G1 / G3, dS as the A operand of
  G4 (read back from LDS). Their fp32 values are never stored.
* Before the rounding, each `ds_mult * dS` value passes through an empty
  inline asm (`_rounding_point`). That materialises the fp32 value, so the
  backend cannot fuse the multiply and the conversion into one mixed-precision
  operation in one step variant and not in another; edge and interior steps
  (and every other variant of the step) therefore round dS identically.
* Each output is rounded once: dK / dV in the epilogue (after `dk_mult`), dQ
  (and atomic-mode dK / dV) in the convert kernel (after `mult`).
* `exp2` is the IR's guarded `exp2` (`llvm.exp2.f32`). The single-instruction
  `exp2_fast` exists in the IR and is used by the forward, but the backward
  does not use it (section 13).

### 1.6 Dead rows and "select, never multiply"

A row is **dead** when its LSE is not finite (`!(lse > -inf && lse < +inf)`,
so NaN is dead too) or when it lies outside the sequence. The forward marks
fully masked rows with `-inf`. Prep gates both statistics on one predicate
`row_live = q < len_q && lse > -inf && lse < +inf`:
`lse2 = row_live ? lse * log2(e) : +inf`, `Dsum = row_live ? rowsum(dO * O) : 0`.

The main kernel then forms

```
P  = select(keep,                    exp2(S * scale_log2 - lse2), 0)
dS = select(keep && lse2 < +inf,     P * (dP - Dsum),             0)
```

Selects, never multiplies by a mask: a dead row (`lse2 = +inf`) or a masked
cell contributes an exact 0 whatever `Q`, `dO` or `O` hold there (a NaN in
`O` of a dead row cannot leak through `Dsum`). On a dead row `P` is `+0`
anyway (`exp2(x - inf)`), so G1 adds nothing; the dS select is what blocks
`0 * NaN`. The dead-row term costs one compare per row, folded into the
per-row key limits on edge steps (section 6.3). A test-only build without the
term (`_emit_main(..., sentinel=False)`) is a negative control.

## 2. Kernel decomposition

### 2.1 Launch sequence

```
            per (query head, row)               per (kv tile, head unit, batch)        per (head, row)
 O, dO,  +--------------------+   WS_LSE2   +------------------------------+  WS_DQ   +-----------------+
 LSE --> |  prep              |-- WS_DSUM ->|  main (KV-parallel)          |--------->|  convert(dQ)    |--> dQ
         |  lse2, Dsum,       |   WS_DQ=0   |  G0..G4 per q step           |          +-----------------+
         |  zero WS_DQ        |  (WS_DK=0,  |  dK / dV: on chip (direct)   |  WS_DK   +-----------------+
         |  (zero WS_DK/DV)   |   WS_DV=0)  |   or fp32 atomics (atomic)   |--------->|  convert(dK)    |--> dK  (atomic mode)
         +--------------------+             |  dQ: fp32 atomics            |  WS_DV   +-----------------+
                                            +------------------------------+--------->|  convert(dV)    |--> dV  (atomic mode)
                                                     | direct mode                    +-----------------+
                                                     +--> dK, dV stored once
```

Under `dq_mode = "split"` the sequence is `prep (zero_dq = 0) -> main (dK /
dV only) -> dq`, plus the two dK / dV converts in atomic mode: the dQ kernel
replaces the dQ workspace and its convert (2.4).

All launches are on one stream in plan order (`attention_bwd_run.run_attn_bwd`).
The caller owns every buffer; nothing is allocated and no memset is issued,
because prep initialises every workspace byte a later kernel reads.

### 2.2 Prep (`attention_bwd_aux.build_attn_bwd_prep`)

Grid `(ceil(max(S_q, S_kv) / R), h_q [+ h_k + h_v], B)`, `R` rows per CTA
(default `max(wave, 64) / (D / 8)`, so one row is `D / 8` lanes of 8
elements). Grid y below `h_q` is the q role; above it, in atomic dK / dV
mode, the kv-zero role.

* q role: each lane loads 8 elements of `O` and `dO`, accumulates their
  products in fp32, and the `D / 8` lanes of a row reduce with an XOR-shuffle
  butterfly; the row's first lane stores `lse2` and `Dsum` (gated as in 1.6).
  User tensors are read only in CTAs with a live row; in batched mode the
  other CTAs write the neutral values (`+inf`, `0`) to every row below
  `S_q_max`. It also zeroes the dQ workspace rows (`zero_dq`).
* kv-zero role (`zero_kv`): zeroes the `WS_DK` / `WS_DV` rows.
* work-list role (THD, prep built with `worklist`, runtime `use_worklist`):
  CTA `(0, 0, 0)` also writes the `WORKLIST` items (2.7); its LDS (two i32
  per thread) and barriers exist only in that build.
* Batched mode initialises every row below `S_max`; THD initialises the rows
  inside each segment only (no other row is ever read).

### 2.3 Main: why KV-parallel

One CTA owns one key tile (`kN0` keys) of one head unit and batch, keeps K and
V of that tile on chip, and walks every q tile that attends to it. This is
the FlashAttention-2 backward structure:

* dK and dV of the key tile accumulate in registers (or LDS) across all q
  steps and leave the CTA once, so in direct mode they are deterministic and
  need no workspace.
* K and V are read from global memory once per CTA; Q, dO and the row
  statistics are streamed per q step.
* The price is dQ: every key tile contributes a partial `dS K` to the same dQ
  rows, so dQ needs a cross-CTA reduction. The built design uses fp32 atomics
  into a zeroed workspace (2.4).

In direct mode a CTA owns a kv head and loops over all `G = h_q / h_k` query
heads that share it, summing their dK / dV contributions on chip (no
expand-then-reduce of dK / dV per query head).

### 2.4 dQ strategies

| Strategy | Status | Mechanism |
|---|---|---|
| Atomic sink (`dq_mode = "atomic"`) | built, the only mode | `_attention_bwd_sinks.AtomicWorkspaceSink`: each q step's fp32 partial is added with `global_atomic_add_f32` (agent scope, native no-return) into `WS_DQ`; the convert kernel scales and rounds once |
| Packed atomic sink | built | `PackedAtomicWorkspaceSink`: the same for head-packed steps (per-row head rebase, dead packed rows add an exact 0) |
| Split dQ (`dq_mode = "split"`) | built on gfx942 / gfx950 (`SPLIT_DQ_ARCHS`; default off) | a separate Q-parallel kernel recomputes S and dP for its q rows, forms dS and accumulates `dQ = dS K` in registers over the band, then stores dQ once in the output dtype (no atomics, deterministic, no dQ workspace, no zeroing, no dQ convert); the main kernel takes `NullSink` (no G4, no dS tile, no K^T, no atomics). See 2.4a |
| Register / split-slice sinks | recorded designs (`RegisterAccumulateSink` exists for Q-parallel use; `SplitSliceSink` raises) | dQ in registers across kv tiles; per-split slices with a fixed-order reduce |

The atomic sink is branch-free: a slot row is clamped to `max(len_q - 1, 0)`
(or not, on interior steps), a row at or past `ws_rows_q` adds an exact 0 at a
clamped address, and the workspace pointer is rebased in i64 once per head
(`begin_head`, called where it dominates every step variant) so the in-slab
index stays a small i32. In the token-major workspace layout a row is
`H * D` elements, so the sink rebases per step to the step's first target row.

G4 issue granularity (`dq_issue`): `"block"` computes the whole `kM0 x D`
partial, then issues its atomics; `"fragment"` computes one 16 x 16 fragment
and issues its atomics before the next (the dS row is re-read from LDS per
fragment), so one fragment is live and its atomics overlap the next
fragment's MMAs; `"fragment_regs"` reads dS into registers once, then works
per fragment. Each fragment keeps the block's K order, so the partials are
bit-identical. LDS-accumulator geometries always work per fragment.

### 2.4a The split dQ kernel

Under `dq_mode = "split"`:

* the main kernel computes dK / dV only (a `NullSink`): no G4, no dS tile in
  LDS, no dQ atomics, no `WS_DQ` access (the kernarg stays in the ABI), no
  K^T fragments or K^T LDS buffer; the G4-only knobs (`atom_g4`,
  `warp_grid_g4`, `kt_source`, `dq_issue`) are inert (`INERT_KNOBS`); dK / dV
  come from the same arithmetic as in atomic mode (bit for bit in direct
  mode);
* prep runs with `zero_dq = 0`, and the dQ convert is replaced by the
  **Q-parallel dQ kernel** (`build_attn_bwd_dq(spec, *, arch)`, spec
  `AttnBwdDqSpec` from `AttnBwdSpec.dq_spec(**knobs)`): one CTA owns `kM0`
  query rows, stages Q and dO once into registers (the B operands), and for
  every key tile its band reaches stages K and V, computes `S^T = K Q^T` and
  `dP^T = V dO^T` (`kN0 x kM0`, the waves split the query columns), forms dS
  in the transposed C layout (per-column `lse2` / `Dsum`, the main kernel's
  selects and rounding point; with `stats_prescale`, the default,
  `P = exp2(fma(S, scale_log2, -lse2))` and `P * fma(dP, ds_mult, -ds_mult *
  Dsum)` with the negated stats formed once per CTA), and accumulates
  `dQ += dS K` in fp32 registers: the dS^T accumulator is the G4' A operand by
  the checked register relabel (dS never goes through LDS) and K is the
  K-outer B operand (the transpose read on gfx950, plain LDS reads on gfx942).
  The K / V slab pointers and the dQ slab pointer are formed before the key
  loops and pinned in SGPRs. dQ is scaled by `dq_mult`, rounded once and
  stored as rows (values for `q < len_q`, exact zeros for batched rows
  `[len_q, S_q_max)`, THD rows outside the sequence untouched). No workspace,
  no atomics: **dQ is deterministic**;
* the key range of a CTA is exact (`kv_tile_range`); `edge_tiles` runs it as
  edge / interior / edge loops (`interior_kv_range`) in the same order, with
  the same dQ bits;
* `head_loop = "group"` packs the `G` query heads that share one K head and
  one V head (`G = gcd(h_q / h_k, h_q / h_v)`, the `G` kernarg; the plan
  requires `G | kM0`) onto the rows, so a key tile staged once serves the
  group; `"single"` gives each CTA one query head;
* the scale placement follows the main spec: `fold_ds` scales dS
  (`ds_mult`); `convert` and `dq_before_atomic` scale dQ (`dq_mult = scale`,
  the convert multiplier folded in: there is no dQ convert);
* dQ-kernel knobs reach the plan as `dq_<field>` tuning values (`DQ_KNOBS`:
  `dq_waves`, `dq_block_m`, `dq_block_n`, `dq_atom_g02`, `dq_atom_g4`,
  `dq_transpose_source`, `dq_head_loop`, `dq_stats_prescale`,
  `dq_edge_tiles`, `dq_kv_pad`, `dq_kv_prefetch` (the next key tile's global
  loads issued into registers before a step's matrix work and stored to LDS
  after it; same values), `dq_waves_per_eu`, `dq_agpr_alloc`,
  `dq_scheduler_strategy`, `dq_sched` (the key step's schedule value, 3.5a:
  `none`, `iglp0`, `iglp1`, `fences_only`, `stage_table`,
  `stage_table_setprio`; hints only, dQ bit for bit), and the XCD chunk
  `dq_xcd_chunk` with the request's `num_xcds`); they are refused without
  `dq_mode = "split"`. The
  G4' 16x16x32 atom needs K-permuted B coordinates and is built with the
  plain transpose read only; the start points (four waves, `kM0 = 64`,
  `kN0 = 64`) use 16x16x16 for G4' and the native 16x16x32 atom for G0' / G2'
  on gfx950.
* Resources: 128 keys per step spill on gfx942 (d64 with `stage_vec = 1` or
  `kv_prefetch`) and reach the SGPR ceiling on gfx950 (`edge_tiles`; the
  grouped d64 kernel spills), and `kv_prefetch` with `head_loop = "group"` and
  `edge_tiles` spills at d128 on gfx950. The sweep's compile stage prunes such
  candidates (`static_prune_dq`); the one shipped-tile combination the dQ
  kernel's resource gate found spilling (gfx950, d128, four waves, 64 x 64,
  `kv_prefetch`, `head_loop = "group"`, `edge_tiles`) is refused by its
  validator (`DQ_SPILLING_COMBOS`). The dQ specs that crash the production
  code-object compiler (`dq_scheduler_strategy = "iterative-ilp"`: a comgr
  segfault on part of the problem shapes of each CDNA target) and the dQ
  geometries that spill in the device-test cells are refused exactly
  (`DQ_RESOURCE_REFUSED`, keyed like `RESOURCE_REFUSED`).
* Compositions: the split main kernel takes the prefetch ring and the DMA
  path, the XT images, the VALU levers, the asm G0 / G2 clusters and every
  schedule value but `stage_table_atomics` (its stages A..C, 3.5a); it
  refuses the merged barrier (which sits before G4), the counted waits, the
  G4 key slicing (all of which assume G4 and its atomics in the step) and
  `stage_table_atomics` (no dQ atomics to defer, `SPLIT_DEFERRED`).

Grid of the dQ kernel `(ceil(S_q_max / R), h_q / H, B)`: `R = kM0`, `H = 1`
(`single`) or `R = kM0 / G`, `H = G` (`group`). LDS: stage 0 (Q, dO), the
key loop (K, V with the `kv_pad` row pad) and the epilogue row buffer alias
one region. The tuning registry lists the dQ values as family-8 axes
(`KIND_DQ`); the sweep varies them on its stage T split points and the XCD
chunk in stage R.

### 2.5 dK / dV modes and head mapping

* **Direct** (`h_k == h_v`, `g_split = 1`): unit = kv head `u`; query heads
  `[u * G, u * G + G)`; dK / dV stored once by the CTA (2.7 epilogue).
* **Atomic** (`h_k != h_v`, or a runtime `g_split > 1`): unit = a group of
  `G = gcd(h_q / h_k, h_q / h_v)` query heads, which share one K head and one V
  head because `G` divides both `gk = h_q / h_k` and `gv = h_q / h_v`. Unit `u`
  of split `p` owns the `G / g_split` heads from `u * G + p * G / g_split`,
  with `hk = u * G / gk`, `hv = u * G / gv` (`_attention_bwd_worklist.head_range`).
  The CTA adds its fp32 dK / dV into `WS_DK` / `WS_DV` with atomics; two
  converts follow. `g_split` spreads a large group over more CTAs on
  under-filled grids at the price of dK / dV atomics.

Example: `h_q = 8, h_k = 2, h_v = 4` gives `gk = 4, gv = 2, G = 2`, four units;
unit 3 owns query heads 6 and 7, K head 1, V head 3.

### 2.5a Atomic-free dK / dV (`dkv_sink`)

A defaulted lever of atomic mode (`dkv_sink = "atomic"` stays the default;
every default kernel, plan and workspace is unchanged; inert in direct mode).

* `"atomic"`: every accumulator element of a CTA is one fp32 atomic add into
  the zero-filled `WS_DK` / `WS_DV` workspace; two converts follow.
* `"slices"`: each CTA stores its fp32 partial (dK times `dk_mult`, as the
  atomic epilogue) with plain stores into its own workspace **slice**. The
  CTAs that share K head `hk` own its `gk = h_q / h_k` query heads `G /
  g_split` at a time, so `WS_DK` holds `n_k = gk / (G / g_split)` slices of
  `rows_kv * h_k * D` floats (each laid out like the atomic workspace) and
  the CTA whose first query head is `h0` writes slice `(h0 mod gk) / (G /
  g_split)`; likewise `WS_DV` with `gv = h_q / h_v`. Every slice row of a live
  tile (`k < len_kv`, inside the workspace) is written by exactly one CTA, so
  prep needs no kv zero fill (`zero_kv = 0`; its kv role is off) and no
  workspace byte is accumulated in place. The epilogue form follows
  `dkv_epilogue`: `"lds_staged"` (default) stages each column half of the
  fp32 tile through the `dkv_out` buffer (the I/O-dtype `kN0 x D` buffer
  holds exactly half the columns in fp32) behind a barrier and stores rows of
  `dkv_store_vec` floats (8: two 16-byte stores); `"lds_staged_pair"` stages
  both halves in two buffers behind one barrier; `"direct"` stores every
  accumulator element from its register (4-byte stores, one branch per row);
  LDS accumulators are read from their fp32 images. All forms write the same
  values to the same elements (bit for bit). The two dK / dV converts become
  **reduce** launches (`build_attn_bwd_reduce`, stage `reduce`, the convert's
  spec): `DST = cast(mult * (s_0 + s_1 + ... + s_{n-1}))`, the slices added
  in slice order, one fp32 chain per element, with the convert's row and
  zero rules. No atomics anywhere in dK / dV; dK / dV are **deterministic**.
  The general main kernel builds it; the dense and unified shells keep their
  own epilogues (`dkv_sink = "atomic"` only there: 8.2, 8.3).
* `"last_cta"` (registered, **not built**; `DKV_SINK_NOT_BUILT`): per-tile
  arrival counters, one atomic increment per (kv tile, kv head) and the last
  arriving CTA sums the other slices with plain loads (no reduce launch). It
  needs, in this order: every thread's slice stores; an agent-scope release
  fence in every thread (on gfx942 / gfx950 a `buffer_wbl2` plus the store
  wait, so the stores of all waves reach the coherence point); a workgroup
  barrier that orders memory (a workgroup release fence, `s_barrier`, a
  workgroup acquire fence); one thread's agent-scope `atomicrmw add`
  (acquire-release) on the tile counter; the result broadcast through LDS
  behind the same fenced barrier; in the last CTA an agent-scope acquire
  fence in every thread (L1 / L2 invalidation) before the slice loads; the
  counter reset (or a prep zero fill of the counters, one i32 per tile and kv
  head of each tensor, behind a workspace term). The IR has no fence op and
  `global_atomic_add` carries no sync scope, so the ordering is not
  expressible today. The only formally sound variant with today's
  primitives, one system-scope acquire-release counter increment per thread
  (each thread owns the elements it stores and reduces them when it is the
  last of its release sequence), costs one counter atomic and a system-scope
  write-back and invalidate per wave per CTA; it is recorded, not built.

`dq_mode = "split"` with direct dK / dV, or with `dkv_sink = "slices"`, is
the **atomic-free backward**: no data atomic in any launch and every output a
fixed-order fp32 sum (`AttnBwdPlan.atomic_free` / `.deterministic`; the
`deterministic` request flag is served this way under
`AttnBwdPolicy.deterministic`, see section 10). The plan records
`runtime["dkv_sink"] = "slices"` and `specs["reduce"]` only for slice plans.

### 2.6 Convert (`attention_bwd_aux.build_attn_bwd_convert`)

`DST = cast(mult * WS)` at arbitrary output strides. Grid
`(ceil(S_max / R), heads, B)`; `convert_vec` elements per lane (8, 4 or 16).
Batched rows `[len, S_max)` are written as exact zeros; THD rows outside the
segment are not touched.

### 2.7 Grid and CTA decode

Main grid `(gx, units * g_split, B)` with `gx = n_kv_tiles = ceil(S_kv_max / kN0)`,
`units = h_k` (direct) or `h_q / G` (atomic); block `W * wave_size` threads.

```
 (x, y, z) = block id
   |
   |  [xcd_remap built and xcd_n > 0]   per batch plane z:
   |     P  = x + gx * y                       plane id in [0, gx * gy)
   |     P' = chiplet_transform_chunked(P, num_wgs = gx * gy, n = xcd_n, chunk = xcd_chunk)
   |     x  = P' % gx,  y = P' / gx
   v
 t     = x                                     kv-tile index (fastest)
 unit  = y / g_split,  split = y % g_split     (direct mode: unit = y, split = 0)
 batch = z
   |
   v  load-balance order on t, inside one (unit, batch):
 kt = t                          lb_order natural (0)
 kt = n_kv_tiles - 1 - t         lb_order reverse (1)
 kt = t/2 or n_kv_tiles-1-t/2    lb_order interleave (3; even / odd t), decode built with lb_interleave
 kt = t, then n_kv_tiles-1-t      lb_order mirror (2; runtime pair = 1, grid x = ceil(n_kv_tiles / 2)), built with lb_mirror
   |
   v  [worklist built and use_worklist = 1]   THD work list instead of (batch, kt):
 i = t + n_kv_tiles * z;  (batch, kt) = WORKLIST item i  if i < count,  else (0, n_kv_tiles)  (a dead tile)
```

* **Why kv tile fastest:** consecutive CTAs of one (unit, batch) share Q, dO,
  the statistics and the dQ workspace rows of that head group, so they reuse
  them through L2.
* **XCD remap** (`XcdRemap`, `xcd_remap_id`): consecutive block ids are
  dispatched round-robin over the chiplets; the chunked transform gives each
  chiplet `xcd_chunk` consecutive logical ids, so with a chunk that is a
  multiple of `gx` all kv tiles of one (unit, batch) land on one chiplet and
  its L2. The transform is applied per batch plane because the main ABI
  carries no batch count; it is a bijection of each plane and equals the
  whole-grid transform when `xcd_n * xcd_chunk` divides the plane. The
  transform is monotone inside each chiplet, so a heaviest-first order stays
  heaviest-first per chiplet. A kernel built without `xcd_remap` emits no
  remap IR; the plan sets the field when the request's `num_xcds` hint and a
  non-zero `xcd_chunk` give `xcd_n > 0`.
* **Load-balance orders:** under a causal or window mask the number of q tiles
  per key tile varies with `kt`. `reverse` dispatches the heaviest tiles first
  (bottom-right causal); `interleave` alternates heavy and light tiles. All
  orders visit each kv tile of a (unit, batch) exactly once.
* **Mirror pairing** (`lb_order = "mirror"`; the plan builds main with
  `lb_mirror` and sets the runtime `pair = 1`): one CTA runs kv tile `t` and
  then kv tile `n_kv_tiles - 1 - t` (a heavy and a light tile of a causal
  band). The second tile runs behind a barrier as a second copy of the
  kv-tile code (loop names of its own) that re-derives every value from its
  roots: the integer kernargs and block ids are parked in VGPR lanes before
  the first tile and the thread id goes through an opaque copy, so no value of
  the first tile stays live across its q loops (a CTA-uniform loop over the
  two tiles kept them live and spilled). Every tile runs once with the same
  per-tile code: direct-mode dK / dV are bit for bit those of the unpaired
  decode. The plan pairs only a masked request whose unpaired grid has more
  CTAs than the request's `num_cus`. The CTA that pairs the heaviest kv tile
  does at least that tile's work, so a one-wave unpaired grid cannot get
  shorter; a multi-wave one loses its ragged tail even when the paired grid
  is under `num_cus`. The guard is not a correctness rule: the paired decode
  visits every kv tile once for any grid. The timing harness passes the
  device's CU count and the target's XCD count as request hints by default.
  The work list excludes pairing (it fixes the CTA order).
* **Work list** (THD, runtime `use_worklist`; the plan builds main and prep
  with `worklist = True`): in a skewed THD batch most grid CTAs of the short
  sequences are dead (their tile starts past `len_kv`) and the long
  sequences' tiles run wherever their batch plane falls. A one-CTA role of
  the prep launch (CTA `(0, 0, 0)`, after its q-side rows) writes the live
  `(sequence, kv tile)` items heaviest first into the `WORKLIST` workspace:
  word 0 the count, word 1 zero, item `i` at words `2 + 2i` (sequence) and
  `3 + 2i` (kv tile). Order: sequences by query length descending (each kv
  tile of a sequence loops over at most `len_q` rows; the prep ABI carries no
  band), ties by index; the tiles of a sequence ascending (the heavy end of a
  causal band). Each thread of the role owns sequences `tid, tid + T, ...`
  and sums the live tiles `ceil(len_kv / kN0)` of every sequence sorting
  before its own (`T` (query length, tile count) pairs per LDS pass), then
  stores its items from that start. The lengths are decoded by the same
  `decode_thd` as the main kernel's, so the live set is exactly the grid's.
  The main grid is unchanged: CTA `(t, unit, z)` takes item
  `t + n_kv_tiles * z` (after the XCD remap), so plane 0 runs the heaviest
  `n_kv_tiles` items over every head unit; an index at or past the count is
  the dead tile `(0, n_kv_tiles)`, which in THD mode reads, stores and adds
  nothing, like a dead grid CTA. Every live tile runs once with the same
  per-tile code, so direct-mode dK / dV are bit for bit those of the grid
  decode; dQ is summed by fp32 atomics in another order (tolerance, as for any
  CTA order). The list fixes the order, so the plan refuses it with a
  non-natural `lb_order`; batched requests are refused (a batched CTA past
  `len_kv` stores its tile's zero rows, so every grid CTA runs). No host read:
  the plan sizes the list for the whole grid (12.4) and the role reads the
  offset tables on the device, so the launch sequence is graph-capturable.

## 3. The tile step

### 3.1 Geometry, warp grids and atoms

`BwdTileGeometry.from_spec` turns a resolved spec into three `BlockGemm`s
(`_attention_bwd_gemm.BlockGemm`): a block GEMM `C[M x N] += A[M x K] B[K x N]`
computed by the `W` waves of the CTA on a warp grid `(rows, cols)`; wave `w`
owns the C sub-block at row `(w / cols) * M / rows`, column
`(w % cols) * N / cols`, as a static `frags_m x frags_n` array of 16 x 16
atom tiles, and K advances in `k_steps` atom-K steps. Every fragment index is
a Python int (no dynamic register indexing, so no scratch).

| GEMM | M x N, K | Default warp grid | Waves split |
|---|---|---|---|
| G0 / G2 | `kM0 x kN0`, `D` | `(1, W)` (`warp_grid_g02`) | the keys |
| G1 / G3 | `kN0 x D`, `kM0` | `(W, 1)` (`warp_grid_g13`) | the keys (as rows) |
| G4 | `kM0 x D`, `kN0` | `(1, W)` or `(2, W / 2)` (`warp_grid_g4`) | D (and q rows) |

With the default grids, wave `w` owns the same keys in G0 / G2 (its columns)
as in G1 / G3 (its rows) and every q row of the step. That is the condition
for the register relabel of 3.3. Every other G0 / G2 or G1 / G3 grid needs
`pt_route = "lds"` (`_attention_bwd_tiles`): `(2, W / 2)`, the transposed
grids (G0 / G2 `(W, 1)`: waves split the q rows; G1 / G3 `(1, W)`: waves
split D), and for G0 / G2 a *replicated* grid of `k < W` waves (`k` dividing
`W`): wave `w` computes the sub-block of wave `w mod k` and only the waves
`w < k` store P / dS (a wave-uniform branch; no matrix op is inside it).
These give the small tiles (kN0 or kM0 of 16 or 32 on two or four waves) to
the CTA: on the natural grids every wave owns `kN0 / W >= 16` keys, and a 16 x
16 S tile on two or four waves, or 32 x 16 / 16 x 32 on four, has fewer 16 x
16 fragments than waves, so only a replicated grid splits it (the copies are
discarded; letting those waves skip G0 / G2 is not built). Every grid must
split its tile into whole 16 x 16 atoms; `small_tile_grids(D, kM0, kN0, W)`
lists the grid triples that do. The G1 / G3 grid is never replicated (the
epilogue would count dK / dV twice).

MMA atoms (`_attention_bwd_caps.bwd_atoms`, 16 x 16 only):

| Target | Atoms | Notes |
|---|---|---|
| gfx942 | `16x16x16` (fp16; bf16 through the `_1k` form) | a K = 32 step is two 16x16x16 instructions |
| gfx950 | `16x16x16`, native `16x16x32` | defaults: G0 / G2 / G4 on `16x16x32`, G1 / G3 on `16x16x16` |
| gfx1151, gfx1201 | `wmma16x16x16` (wave32) | |

32 x 32 atoms are excluded: `kM0 = 16` cannot fill them and the relabel is
proven only for 16 x 16 atoms. `atom_g13 = "16x16x32"` needs `kM0 >= 32`.

MFMA instructions per wave per q step (`_attention_bwd_sched.StageTable.mfma`):

```
G0 = G2 = (kM0/16) * (kN0/16) / W02 * (D/k_g02)     (W02: waves of the G0 / G2 grid)
G1 = G3 = (kN0/16) * (D/16) / W * (kM0/k_g13)
G4      = (kM0/(16 g4r)) * (D/(16 g4c)) * (kN0/k_g4)
dQ atomics per lane = kM0 * D / (W * wave_size)
```

For the gfx942 d128 default (16 x 128, W = 8, G4 grid (1, 8), K = 16 atoms)
that is 8 instructions per GEMM per wave and 4 atomics per lane per step.

Build-time proofs (`BwdTileGeometry.check`): every atom's A / B / C lane maps
are evaluated on an integer builder for every lane and slot
(`_attention_bwd_frag.atom_facts`) and must be K-contiguous and affine; the
relabel is proven per atom pair; the transpose read is proven against the
catalog operand map or, for the K-permuted G1 / G3 B of a 16x16x32 consumer,
against the stacked plan (`tr16_read_plan`, 3.3); a K-permuting relabel with
an unpermuted B is refused. An illegal geometry raises `ValueError` before any
IR.

### 3.2 Softmax gradient

`BwdTileStep._softmax_grad` runs per accumulator slot of the G0 / G2
fragments. The row of a slot depends only on `(fragment row, slot)`, so each
lane loads the `(lse2, Dsum)` of its own C rows once per step straight from
the workspace (`StatsView`: per-wave redundant scalar loads; rows at or past
`ws_rows` read `+inf` / `0`; never a multi-dword load across the workspace
end). There is no LDS copy of the statistics. `stats_load = "step_top"` issues
these loads at the top of the step, ahead of the staging and G0 / G2, so
their latency overlaps them; `"at_use"` (default) issues them just before
the softmax gradient.

Per slot: one multiply and one subtract feed `exp2`; on edge steps one
compare against the row's key limits gives `keep`; then the `P` select, `dP -
Dsum`, the multiply by P, the dS select, the `ds_mult` multiply, the rounding
point. The two fp32 fragments P and dS are then handed off (3.3).

Softmax-gradient VALU levers (`_attention_bwd_valu`; each default emits the
shipped IR):

* `bf16_cast`: `"generic"` rounds P / dS to bf16 with `fptrunc` (gfx942
  expands it per element to round to nearest even plus a NaN guard);
  `"rne_noguard"` is the same rounding as integer IR without the guard
  (`u + 0x7fff + ((u >> 16) & 1)` per element, one `v_perm_b32` per pair;
  bf16 only): bit for bit equal for every non-NaN value, infinities and
  subnormals included, while the result of a NaN input is unspecified (it can
  come out as `-0`, `+Inf` or `-Inf`). The sweep does not enumerate it on
  gfx950, whose generic cast is one packed conversion.
* `bwd_exp2`: `"exact"` is `llvm.exp2.f32`, which keeps subnormal results of
  P; `"fast"` is one `v_exp_f32`, as in the forward: P is the same for every
  argument at or above -126 and flushed to 0 below it (P < 2^-126); dead rows
  and NaN behave alike.
* `stats_fma`: P's argument `S * scale_log2 - lse2` as one `fma` with
  `-lse2`, one rounding fewer (within the backward tolerance).
* `stats_vec` (1, 2, 4): one scalar lse2 and Dsum load per C row of the lane,
  or one `<n x f32>` load of each per run of n consecutive C rows; the window
  is moved inside the head-major arrays at their ends and each row's element
  picked by its shift, so the values are bit for bit those of 1 (head-major
  workspace and MFMA targets only; an array of fewer than n floats is read up
  to n - 1 floats past its end, inside the plan's 256-byte rounding of the
  sub-buffer). Under a ring the stats come from the slot (vector reads of the
  same rows).

The sweep enumerates these levers on the MFMA targets; on WMMA targets
`bf16_cast`, `bwd_exp2` and `stats_fma` are legal but not enumerated.

### 3.3 The P^T / dS^T hand-off: relabel or LDS

G1 needs `P^T` and G3 needs `dS^T` as A operands (rows = keys, K = q rows).
They exist only as G0 / G2 C fragments (rows = q, cols = keys).

* **`pt_route = "relabel"`** (CDNA default). For the 16 x 16 MFMA atoms the C
  layout of one tile coincides with the A layout of the transposed product:
  the four rows a lane holds at one column are exactly the four K values its
  A fragment needs at one row. `_attention_bwd_frag.plan_c_to_a` evaluates
  both maps on every lane and slot and returns a `"direct"` plan; the A
  fragment is then just the C registers rounded to the I/O dtype and repacked
  (`emit_c_to_a`). No data moves between lanes and nothing goes through LDS.
  `check_relabel_grids` additionally proves the warp-grid condition of 3.1.
  For a consumer atom with K = 32 (`16x16x32` on gfx950) two C tiles are
  concatenated and the 32 K values arrive in a permuted order; the plan
  records the permutation and the B operand of G1 / G3 is then loaded with
  the same permutation (`kperm_b_coords`), so the K sum is unchanged. On
  `lds_plain` the permuted B is read element by element, and on `xt_lds` the
  permuted K runs are K-inner vector reads of the XT image. On `tr_read` it is
  exactly two 16x16x16 B transpose reads 16 K rows apart (the stacked plan of
  `tr16_read_plan`, proved against the permuted map on every lane and slot at
  build time), so it costs the same number of `ds_read_tr16_b64` as the
  16x16x16 twin and issues half the G1 / G3 MFMAs. The fp32 sum order inside
  the wide MFMA differs from the twin (oracle tolerance, not bit for bit).
  A `lane ^ 16` exchange plan (`permlanex16` + `perm_b32`, proven by replaying
  the program on coordinates) exists for wave32 WMMA, but the family refuses
  the relabel route on WMMA targets.
* **`pt_route = "lds"`** (RDNA default, CDNA option). P and dS are rounded and
  written to the `pt` / `ds` LDS buffers in C layout (`_write_c_tiles`), a
  barrier publishes them, and G1 / G3 read them back as K-outer A operands
  through the transpose source (the plain element reader under `xt_lds`,
  which has no image of P / dS). It frees the warp grids (`(2, W / 2)` becomes
  legal) at the cost of an LDS round trip and a barrier.

dS is always written to LDS once per step (`ds` buffer) because G4 needs it as
an A operand with a different warp split (G4 splits D, not keys).

### 3.4 Operand sources

| Operand | Storage | Read as | Source |
|---|---|---|---|
| G0 A = Q, G2 A = dO | `q`, `do` LDS tiles `[q][d]`, per step | K-inner (vector reads along D) | staged from global every step |
| G0 B = K^T, G2 B = V^T | K, V `[key][d]` | K-inner | registers loaded once per CTA (`kv_residency = "reg"`), or the resident LDS copy every step (`"lds"`) |
| G1 B = dO, G3 B = Q | `do`, `q` tiles `[q][d]`, K = q | K-outer | transpose source |
| G4 A = dS | `ds` tile `[q][key]` | K-inner | written by the step |
| G4 B = K | K `[key][d]`, K = key | K-outer | registers loaded once (`kt_source = "reg"`) or LDS every step (`"lds"`), through the transpose source |
| G1 A = P^T, G3 A = dS^T | registers (relabel) or `pt` / `ds` tiles (LDS route) | K-outer from LDS | 3.3 |

"K^T" in the code names the G4 B operand: K stored key-major is K-outer for
G4, so it needs a transposing read. K-outer reads go through the transpose
source (`transpose_source`):

* `lds_plain`: one 16-bit element per read (gfx942 default, and every
  narrow-access tile);
* `tr_read`: `ds_read_tr16_b64` on gfx950 (default there): one 64-bit read
  delivers 4 K-consecutive values per lane; a K = 16 fragment takes one read,
  K = 32 two. Each lane supplies its own address (`tr16_lane_address`), and
  the delivered coordinates (`tr16_model_coords`) are proven equal to the
  catalog operand map at build time; the per-lane model was validated on the
  device by the transpose probe. The 128-bit form is never used.
* `wmma_lds`: the plain element reader on RDNA;
* `xt_lds` (every target; the gfx942 lever): the published row-major tile is
  copied into a transposed image (`rows = D`, `cols =` the sequence extent:
  `qt` / `dot` per step, `kt_stage` / `kt_res` once per CTA) by
  `_attention_bwd_body.write_xt`, so the K-outer operand becomes K-inner and a
  lane's `atom_k / 4` consecutive K values are one vector read
  (`ds_read_b64` at K = 16, `ds_read_b128` at K = 32). The writer is the
  planner's `default_xt_writer` (each thread reads `k_per_thread` rows of
  `vec_d` elements and writes `vec_d` stores of `k_per_thread` elements). The
  image layout is per buffer (`lds_swizzle`): by default the conflict
  predictor's choice (`planner_xt_swizzle`: the 128-byte line XOR on chunks of
  `atom_k / 4` elements on every gfx942 start point, a row pad for the gfx950
  K^T images, `pad8` on RDNA, which has no predictor profile), or `xor`, `none`,
  `pad8` / `pad16` / `pad32`. P^T / dS^T of the LDS route have no image and are
  read with the plain element reader. `global_path = "dma"` excludes it (a DMA
  cannot write the image). The images move bytes only: every MMA sees the
  operands of the plain reader in the same order, so dK, dV and each dQ
  partial are bit for bit those of the plain reader.

`kt_source = "reg"` requires `kv_residency = "reg"`. Register residency costs
`kN0 * D / (2 * wave * W)` registers per lane per operand (K, V, K^T) and saves
the per-step LDS reads; LDS residency is required on WMMA targets.

### 3.5 The step pipeline and its barriers

`BwdTileStep.run_q_step`, default knobs:

```
 q step (qt, head h), CTA-wide; "|B|" = workgroup barrier
 ----------------------------------------------------------------------------------
 [stats_load = step_top: lse2 / Dsum loads]
 stage Q, dO rows [q0, q0 + kM0) -> LDS q, do       (qdo_stage: split | loads_first | joint)
 |B1|  publish Q, dO
 [xt_lds: q -> qt, do -> dot (XT images); |Bx| before the first G1 / G3]
 G0   S  = Q K^T          A: q (LDS)   B: K (regs | LDS)
 G2   dP = dO V^T         A: do (LDS)  B: V (regs | LDS)
 softmax gradient -> P, ds_mult * dS (fp32, rounding point)
 hand-off   relabel: P^T, dS^T in registers; dS -> LDS ds
            lds:     P -> LDS pt, dS -> LDS ds; |B2| publish P, dS
 G1   dV += P^T dO        B: do via transpose source        \
 G3   dK += dS^T Q        B: q  via transpose source         >  stage_order (default g1g3g4)
 G4   dQ_part = dS K      A: ds (LDS)  B: K (regs | LDS)    /   relabel: |B2| sits right before G4
      (g4_slicing: per atom-K step, or kK4 key slices)
      -> dQ sink (fp32 atomics; block or per fragment)
 |B3|  retire the reads of q, do, ds, pt before the next step overwrites them
 [agpr_fence route: carried dK / dV through an "=a" asm at the back-edge]
```

What each barrier protects:

| Barrier | Protects | Notes |
|---|---|---|
| B1 | RAW: the Q / dO `ds_write`s of all waves before any wave's G0 / G2 / G1 / G3 reads (and the XT writer's reads) | the global loads that feed the writes are waited by data dependency inside the issuing wave |
| Bx | RAW (`xt_lds`, relabel route): the Q^T / dO^T image writes before the first G1 / G3 read | right before the first of G1 / G3; when G4 issues first, B2 publishes the images, and on the LDS route the P / dS barrier does (no Bx) |
| B2 | RAW: dS (and P on the LDS route) written by every wave before G4 (and G1 / G3 on the LDS route) reads it | under relabel it is placed right before G4 in every stage order, so the `ds_write` latency overlaps G1 / G3 (right before G3 under `g4_slicing = "slice_db_g3"`, which reads G4's first dS slice there) |
| B3 | WAR: every read of `q`, `do`, `ds`, `pt` (and `qt`, `dot`) of this step before the next step's staging writes | |

Three barriers per step on both routes (four under `xt_lds` with the relabel
route when G1 or G3 issues before G4). `loop_sync` chooses their flavour:
`"full"` (default) is `s_waitcnt vmcnt(0) lgkmcnt(0)` + `s_barrier`, so B3 also
waits for this step's dQ atomics; `"lds_only"` is `s_waitcnt lgkmcnt(0)` +
`s_barrier`. `lds_only` is correct because no VMEM result crosses an in-loop
barrier from one wave to another: the staging loads are consumed by their own
wave, the statistics are used by the wave that loaded them, and the dQ
atomics are never read inside the kernel. It is proven for the register
staging path only (a DMA path writes LDS from VMEM), hence the coupling to
`global_path = "vgpr"`. The staging and epilogue barriers are always full.

**Barrier fences.** `llvm.amdgcn.s.barrier` does not access memory, so
without more the compiler may move an LDS access across it (machine sinking
did: the stage-0 V-fragment loads of a select-guarded `(2, W/2)`-grid kernel
sank past the barrier that retires them and a fast wave overwrote rows a slow
wave was still reading). Every barrier of the backward kernels (main, aux,
the dQ kernel and the dense / unified shells) therefore carries
`fence = "compiler"` (`BWD_BARRIER_FENCE`): the platform emits an empty
`asm sideeffect "" ~{memory}` after the barrier (both engines), which keeps
every memory access on its side of the barrier at no instruction cost. The
other fence mode, `"workgroup"`, adds release / acquire `syncscope("workgroup")`
fences around the barrier; it is not used here (it costs cache-invalidation
instructions on RDNA and wait states on gfx950). The resource gate checks the
LDS bytes read and written between consecutive prologue barriers, IR against
ISA (`lds_barrier_crossings`).

**Prefetch ring** (`ring_depth` R in 1..3, `global_path`; module
`_attention_bwd_ring`). With `(vgpr, 1)` the step stages synchronously (B1).
Otherwise Q, dO and the step's lse2 / Dsum rows are staged into one of R LDS
ring slots R - 1 steps ahead: the register ring (`vgpr`, R = 2 / 3) issues the
global loads of tile `i + R - 1` at the top of step `i` (its only global
loads, so their wait falls at the slot stores and leaves the dQ atomics in
flight) and stores them to their slot at the end of the step; the DMA ring
(`dma`, gfx950) copies global memory straight into the slot with
`global_load_lds_dwordx4` / `_dword` (inline asm, into the 128-byte slab layout
with a source-side XOR) and waits `vmcnt(N)` with `N` from its wait model. The
copy's 64-bit global base is an SGPR operand (`"s"`): it is the slab pointer
formed again from its base and a `readfirstlane`'d byte offset
(`RowSlab.sgpr_ptr`, `WsSlab.sgpr_ptr`), since the production compiler can keep
the wave-uniform pointer in a VGPR pair and hand that to the operand, which
the assembler rejects (gfx950 debug d128 THD tiles with `stage_table_atomics`).
Every in-step barrier is LDS-only; the step ends with the ring's wait and one
LDS-only barrier that publishes the next slot and retires the reads of this
one. Couplings: `qdo_stage = "split"` and `stats_load = "at_use"` (the ring
owns the issue order), no head packing on the DMA path, 1 KiB DMA pieces, and
not the grouped head-outer form (`step_order = "head_outer"` with
`edge_tiles`): its three range loops nested in the head-group loop, each
carrying the ring's next-step tile, spill SGPRs on every row of both CDNA
targets at any depth above 1 (prefetch ring, DMA ring, merged barrier).

**Barrier merge** (`barrier_merge = "on"`, `ring_issue`; gfx942 / gfx950).
One barrier per q step: Q / dO / lse2 / Dsum live in a 2-slot LDS ring and dS
in 2 slots; step `i` reads slot `i % 2`, and stages the next step into the
other slot before its pre-G4 barrier, which also publishes dS (the hazard
proof is in `BwdTileStep._ring_views`). The next step's global loads are
issued at the step top (`ring_issue = "early"`) or right before the slot
stores (`"late"`), in both cases ahead of this step's dQ atomics. Couplings:
`ring_depth = 2`, the register path, the relabel route, register dK / dV, G4
last in `stage_order`, the default staging order, atomic dQ and (as every
ring) not the grouped head-outer form.

**Counted waits** (`loop_sync = "counted"`, `counted_issue`; CDNA; module
`_attention_bwd_counted`). The next step's Q / dO tile and stats rows are
loaded into registers inside the current step, before its dQ atomics (right
after the staging barrier, `"early"`, or right before G4, `"late"`), carried
across the back-edge and stored at the next step's top; a 0 / 1-trip guard
loop loads the first step of each q loop, followed by one `vmcnt(0)`. Every
in-loop barrier waits `vmcnt(N) lgkmcnt(0)`, `N` the dQ atomics the wave
issued after its last global load, so every load has retired and only the
atomics stay in flight. Couplings: register staging without a ring or merged
barrier, the default `qdo_stage` / `stats_load`, at most 63 dQ atomics per
wave and step, atomic dQ, not the grouped head-outer form; every schedule
value except `stage_table_atomics` with the late issue (3.5a).

The ring, the merged barrier and the counted waits change only the issue
order and the barriers: every accumulator chain is the default's (dK / dV bit
for bit in direct mode). The XT images and the G4 key slicing are built on
the synchronous body only (the validator refuses them with a ring, the
merged barrier or the counted waits); the schedule values run on every one of
these bodies (3.5a).

`stage_order` permutes the issue order of G1, G3 and G4 (six orders; for
example `g4g1g3` issues G4 and its atomics right after dS is published so they
overlap the G1 / G3 matrix ops). Each accumulator's MMA chain is unchanged by
the order, so dK, dV and every dQ partial are bitwise those of the default.

**G4 dS key slicing** (`g4_slicing`, slice width `block_k4` = kK4). G4 reads
its dS A fragments (and, with `kt_source = "lds"`, its K^T B fragments) from
LDS. `"step"` (default) reads one atom-K step and issues its MMAs, step after
step over the whole `kN0` tile; `block_k4` is then inert (`INERT_KNOBS`). The
other values read in kK4 key slices (`BlockGemm.load_k_slice` /
`mma_k_sliced`):

| Value | Order inside G4 | kK4 |
|---|---|---|
| `step` | per atom-K step: read, then MMAs | inert |
| `slice` | read slice `s` (every step of it), then its MMAs; the next slice after them | effective |
| `slice_db` | read slice 0; for each `s`, read slice `s + 1` before the MMAs of slice `s` (two register buffers) | effective |
| `slice_db_g3` | `slice_db`, with slice 0 read before the G3 MMAs; under the relabel route the dS barrier (B2) moves from before G4 to before G3 (still three barriers per step) | effective |

Every accumulator takes its K steps in ascending order under every value, so
the dQ partials are bitwise those of `step` (with one kv tile per query head
and `g_split = 1`, dQ is bitwise too). Couplings: MFMA register accumulators
(`acc_in_lds = False`), `dq_issue = "block"`, kK4 a multiple of the G4 atom K
that divides kN0, and for `slice_db_g3` a `stage_order` with G3 right before G4
(`g1g3g4`, `g3g4g1`). A resolved spec spells a value by the code it emits:
`slice_db` over one slice is `slice`, and `slice` with one atom-K step per
slice is `step`.

Q / dO staging (`copy_row_sets_to_lds`, `qdo_stage`): `"split"` copies each
tensor in turn with each chunk's LDS store right after its global load;
`"loads_first"` issues every global load of both tiles before the first LDS
store (tail chunks load a clamped chunk and only the store is predicated);
`"joint"` gives the Q and dO tiles disjoint whole waves of one pass (legal
only when `kM0 * D / 8` is below the thread count and a multiple of the wave
size). All three move the same bytes to the same cells. Rows are clamped to
`len_q - 1` so MMA operands always come from valid rows; out-of-range rows are
masked by `keep`.

### 3.5a Schedule values (`sched`, `setprio`)

`_attention_bwd_sched.StageTable` (built from the geometry) groups the step
into four stages, A: G0; B: G2 + G1; C: G3; D: G4 (default `stage_order`;
three, A..C, for the split-dQ main kernel, below), and holds the count model
of the built step (`pipeline = "sync"`, or a ring's below): per stage the
matrix ops, VMEM reads / writes and DS reads / writes the body emits. The
body calls its hooks at each stage edge.

| `sched` | `setprio` | Emission per q step |
|---|---|---|
| `none` (default) | False | nothing; the default IR |
| `iglp0` / `iglp1` | False | one `iglp_opt(0 / 1)` right after the staging barrier (in the region of the step's matrix ops) |
| `fences_only` | False | fenced placement, then `sched_barrier(0)` after every stage but the last (A, B and C; A and B under split dQ) |
| `stage_table` | False / True | fenced placement; per stage single-MFMA `sched_group_barrier(mask, n, 0)` groups with the stage's memory work spread between them (the first K step's operand reads and every VMEM read lead; the dS / P writes of B follow G1's MFMAs; the dQ atomics of D follow its last K step); fences; with `setprio`, `s_setprio 1` before and `s_setprio 0` after each stage's ops |
| `stage_table_atomics` | False / True | as `stage_table`, with the deferred dQ sink: each step's dQ partial is loop-carried and its atomics are issued in stage A of the next step (D has none); the first step issues `+0.0` (bitwise neutral: a sum that starts at `+0.0` never becomes `-0.0`), the last partial is flushed after the q loops; no branch encloses an atomic |

Fenced placement (every value with fences): the lse2 / Dsum loads move into A
(after G0; `stats_load = "step_top"` keeps them at the step top), and the
register-fed operands are read one stage ahead: dO^T of G1 at the end of the
stage before G1, Q^T of G3 before G3, an LDS K^T of G4 before G4 (K^T is
resident, so it needs no barrier). Every value only moves work and adds hints:
the matrix ops and every accumulator's chain are the default's, so direct dK /
dV are bit for bit the default's and dQ gets the same values (summed by
atomics in a run-dependent order, as always). The resolved spec, the register
estimate and the LDS plan do not depend on the schedule knobs.

The model's counts equal the IR census of the step body on every shipped
instance and on the knob values that change a stage's traffic (LDS P / dS
route and the `(2, W/2)` grids, LDS K / V / K^T, LDS accumulators,
`dq_issue`, `stats_load`, row pads, AGPR routes, the token-major workspace):
`test_attention_bwd_sched_body.py`. Resources: every value passes the
resource gate on the CDNA instances (zero scratch, no spills, the hot-loop
AGPR copy ceiling). The read-ahead fragments and the deferred dQ partial are
extra live registers; on the RDNA d128 tiles, which already sit at the
256-VGPR edge, the fenced values and even a bare `iglp_opt` can spill, so the
sweep's resource gate drops them there (on gfx1151 d128, `iglp1` also makes
the code-object compile far slower than any other value; the sweep's compile
deadline applies). An `iglp1` kernel compiles in a fresh process of its own
(kernel attribute `comgr_fresh_process`, see `runtime/comgr_and_hipmodule.md`,
"Compiling In A Fresh Process"): LLVM's `iglp_opt(1)` strategy keeps state
across the compiles of a process, so in-process its code object, and its
resource-gate verdict, would depend on the kernels compiled before it. Stage
tables need the default
`stage_order` and exclude `agpr_route = "asm_g13"` (the scheduler cannot see
MFMAs inside asm), as do the iglp values; `setprio` needs a stage table.
The validator refuses `stage_table_atomics` with `dq_mode = "split"` (the
split-dQ main kernel issues no dQ atomics to defer, `SPLIT_DEFERRED`) and
with `loop_sync = "counted"` and `counted_issue = "late"` (below); every other
value is legal on every body; the registry couples the axes the same way.

**Schedule values on the rings and the counted waits.** `StageTable.pipeline`
is the body's count model, picked from the geometry: `"sync"` (the
synchronous step, also under the counted waits), `"ring"` (the Q / dO
prefetch ring of 2.4: the DMA path at `ring_depth` 1 / 2 / 3, the register
path at 2 / 3) and `"merged"` (the merged-barrier ring, both `ring_issue`
points). The stages and their groups are the synchronous ones, with the
staging's own work placed as the body emits it:

| Body | The step's prefetch / staging | Stats | Waits |
|---|---|---|---|
| DMA ring | the copy of the tile `R - 1` ahead at the step top: inline asm in per-wave branches, not an IR memory op a group can take (it is asynchronous and overlaps the whole step anyway); counted per wave | DS reads of the slot (A when fenced, else B) | `vmcnt` at the step top (`R = 1`) or end, the ring's model |
| register ring | the global loads of the tile `R - 1` ahead: in stage A under a stage table (VMEM reads spread over G0's MFMAs; the fence after A keeps them there), else ahead of the step behind their `sched_barrier(0)` as before; the slot stores after D (outside the groups: a branch guards the stats store) | DS reads of the slot | the compiler's |
| merged ring | the next step's loads in stage A under a stage table with `ring_issue = "early"` (else at the step top, or before the merged barrier with `"late"`); the slot stores before the merged barrier, ahead of D | DS reads of the slot | the compiler's |
| counted waits | the staged tile stored at the step top; the next step's loads between their two fences (after the staging barrier, or before G4), never grouped | the staged registers (no load) | `vmcnt(N)` at the three barriers |

`stage_table_atomics` on a ring issues the previous step's dQ atomics in
stage A, after the step's DMA copy. The DMA wait immediates count the VMEM
operations a wave issues after its copy: they are these `n_a` atomics
instead of the step's own (as many, branch-free, zeros on the first step),
so the immediates are unchanged (`StageTable.vm_after_prefetch`, checked
when the ring is built by `QRing.check_step`); the ring's in-order VM queue
replay passes with the atomics in A. Under the counted waits with the early
issue the deferred atomics follow the step's loads, so the dS barrier waits
`vmcnt(n)` instead of `vmcnt(0)` (`StageTable.counted_waits`, the counted
model's recount from the IR agrees). The late issue would load the next step
after the deferred atomics of A, the reverse of the counted issue order, so
that one pair stays refused (`COUNTED_LATE_DEFERRED`); every other value
runs with both issue points.

Every composition only moves work: the twin without a schedule value has
the same resolved spec, register estimate, LDS plan and matrix-op chains,
so direct dK / dV are bit for bit the twin's (device: the composition twin
sets `<ring>_st`, `_stap`, `_fen`, `_ig1` of `COMPOSE_TWIN_SETS`). The CPU
suite `test_attention_bwd_ring_sched.py` checks the hint sequence
(`StageTable.hint_sequence`), every fenced region's census
(`StageTable.regions`: stages, the register ring's prefetch region, the
counted staging / issue regions), the DMA batch per wave and every wait
immediate against the IR, a slot-aware replay of every LDS access (ring
slots evaluated per iteration, DMA copies pending until the wait that
retires them), the deferred atomics and the resource gate per (body, value)
on gfx942 / gfx950. Resources: the read-ahead fragments, the deferred dQ
partial and the ring's own registers (a depth-3 register ring carries a tile)
add live registers; where that exceeds the register file on the production
compiler (mostly the depth-3 rings, `stage_table_atomics` on the debug tile
and the d64 edge / packed tiles of gfx950) the point is refused
(`RESOURCE_REFUSED`, rows `ringsched` of the refusal test), so every legal
(body, value) passes the gate. Three ring twins without a schedule value fail
the gate on their own (gfx950 d64: the DMA rings of depth 1 / 2 with edge
tiles, SGPR spills; the depth-3 register ring with head packing, VGPR spills);
they are refused exactly as well (`RESOURCE_REFUSED`, recorded from the ring
device test's launches), so the gate test exempts no point.

**Schedule values on the split-dQ main kernel.** Under `dq_mode = "split"`
the step has no G4, no dS tile and no dQ atomics, and runs on the synchronous
body (full or LDS-only barriers) and the prefetch rings (register `R` = 2 / 3,
DMA `R` = 1 / 2 / 3); the merged barrier and the counted waits stay refused
there (they are placed against G4 and its atomics). Its stage table
(`StageTable.stage_names` = A, B, C) is the synchronous / ring model above
without D: the relabel hand-off writes no dS tile (the LDS route still writes
P and dS for G1 / G3), no K^T is read ahead, and C closes the step (no fence
after it; the register ring's slot stores fall in C's region as
`StageCounts.fixed`). `iglp0` / `iglp1`, `fences_only` and `stage_table` with
or without `setprio` are built on every split body; `stage_table_atomics`
defers dQ atomics the split-dQ main kernel does not issue and is refused with
the reason (`SPLIT_DEFERRED`; mapping it to the plain table would sweep one
kernel twice). The DMA ring's immediates take `n_a = 0` (no VMEM op follows
the copy before the wait; the ring's VM queue replay with `n_a = 0` is
race-free and its immediates maximal). `test_attention_bwd_split_sched.py`
checks the hint sequence, every region's census and leading reads, the DMA
batch and waits, the slot-aware LDS replay, scheduling only and the resource
gate per (body, value); the composition twin sets `split_<body>_<tag>`
(device) give dK / dV and dQ bit for bit the twin's. Resources: `fences_only`
and the stage tables pass the gate on every split body and tile; `iglp0` /
`iglp1` spill on a few gfx950 d64 default-tile points (gate taken with the
fresh-process compile of `iglp_opt(1)` kernels) and are refused
there (`RESOURCE_REFUSED`, rows `splitsched` of the refusal test); four split
twins without a schedule value fail the gate on their own (gfx950 d64: the
synchronous body with and without LDS-only barriers on THD atomic, the depth-3
register ring with edge tiles, the depth-2 DMA ring; SGPR spills) and are
refused exactly as well (`RESOURCE_REFUSED`), so the gate test exempts no
point; the census keeps the values legal at those points.

**Schedule values of the split dQ kernel (`dq_sched`).** The dQ kernel's key
step runs as stages A (G0' `S^T = K Q^T` with its K reads and, with
`kv_prefetch`, the next tile's global loads), B (G2' `dP^T = V dO^T` with its
V reads, the softmax gradient and the dS^T relabel) and C (G4' `dQ += dS K`
with its K-outer K reads; then the step's barrier and, with `kv_prefetch`, the
next tile's LDS stores outside the groups). `_attention_bwd_sched.DqStageTable`
is its count model and hooks: `iglp0` / `iglp1` (one `iglp_opt` ahead of A),
`fences_only` (`sched_barrier(0)` after A and B), `stage_table` (the fences
and per-stage single-MFMA groups: the first K step's reads lead, the other
reads and the prefetch loads spread) and `stage_table_setprio` (with `s_setprio
1 / 0` around each stage; `s_setprio` is a scheduling boundary, so A's raise
precedes the prefetch loads). No op moves, so dQ is bit for bit the default's
(device: the `dq_*` variants of `test_attention_bwd_split_numeric.py`); the
resolved spec, the LDS plan and the register estimate do not depend on it.
`test_attention_bwd_dq_sched.py` checks the hint sequence and every region's
census against the IR over the dQ suite's variants and the knobs that change
a stage's counts (row pads, prefetch, transpose source, atoms, tiles,
`stage_vec`), the barrier replay of every LDS access, the LLVM hint calls and
the dQ kernel's resource gate per value: every value passes it on both targets
except `iglp0` with `kv_prefetch` at d128 on gfx942 (the start tile spills to
scratch; refused, `DQ_SCHED_SPILLING`); the default dQ kernel at the 128-key
tile (4 waves, 64 x 128) fails it on its own at d128 on gfx950 and d64 on
gfx942 and is refused exactly there (`DQ_RESOURCE_REFUSED`); every value
passes at those rows and stays legal. `iglp1` makes the gfx942 d128 128-key code
object compile for far longer than any other value (the sweep's compile
deadline applies).

### 3.6 One-time K / V staging

`BwdTileStep.stage_kv`, before the q loops. Global reads happen only inside
`scf_if(live)` with `live = k0 < len_kv` (CTA-uniform), so a tile past the
sequence reads nothing.

```
 kv_residency = "reg"                               kv_residency = "lds"
 ------------------------------------------         -------------------------------
 [live] K rows -> LDS (k_stage or k_res)            [live] K -> k_res, V -> v_res
 |B| full                                           |B| full
 K B-fragments for G0 -> registers                  [xt_lds] K -> kt_res; |B| full
 [xt_lds] K -> kt_stage / kt_res; |B| full          (operands read every step)
 [kt_source = reg] K^T fragments -> registers
 |B| full   K reads done before V overwrites the stage-0 region
 [live] V rows -> LDS v_stage
 |B| full
 V B-fragments for G2 -> registers
 |B| full   stage-0 reads done before the loop phase is written
```

The stage-0 K and V buffers and the q-loop buffers alias one LDS region
(section 4.1); these barriers are what make that aliasing safe. The dK / dV
accumulators are zeroed after the q-loop trip counts are known, so the
empty-band case (trip count 0) is decided before any accumulator is live.

### 3.7 dK / dV epilogue

After the q loops (`attention_bwd._direct_epilogue`, `_atomic_epilogue`):

* **Direct mode.** Rows `k < len_kv` get the values (`dk_mult` on dK); batched
  rows `len_kv <= k < S_kv_max` get exact zeros (also for an empty band or a
  tile past the sequence, where the accumulators stayed zero); rows at or past
  `S_kv_max`, and THD rows outside the segment, are not touched
  (`_attention_bwd_band.dkv_store_rule`). `dkv_epilogue` picks the path; every
  path stores the same values to the same elements:
  * `"lds_staged"` (default): round once to the I/O dtype, write the C layout
    to the `dkv_out` LDS buffer, barrier, then store row segments of
    `dkv_store_vec` elements (dK, then dV through the same buffer);
  * `"lds_staged_pair"`: dK and dV in two buffers behind one barrier;
  * `"direct"`: each lane stores its accumulator cells element by element
    (no LDS, no barrier).
  With LDS accumulators the epilogue reads the fp32 images directly.
* **Atomic mode.** Inside `scf_if(live)`, every accumulator element is one
  `global_atomic_add_f32` into `WS_DK` (times `dk_mult`) or `WS_DV`, rebased in
  i64 per head and tile row. `dkv_atomic_guard = "branch"` puts one branch per
  key row around its atomics; `"select"` issues them all and adds `+0.0` for a
  row outside the sequence or the workspace (an fp32 sum that starts at `+0.0`
  never becomes `-0.0`, so this is bitwise).

THD direct mode can form the dK / dV output slabs before the q loops and pin
each 64-bit pointer with an opaque SGPR copy (`dkv_slab_hoist`), so two
pointers instead of the token base, both tensor pointers and their strides
stay live across the loops (an SGPR-pressure fix; same values).

### 3.8 Designed, not built

| Feature | Knob value | What it would do |
|---|---|---|
| Reference ring placement | a stage table that also moves the next tile's LDS writes into B and reads G4's dS in `kK4` slices over C / D | the reference pipeline's placement; `StageTable(pipeline = "ck")` holds its count model (the stage tables on the built rings, 3.5a, keep the rings' own slot-store points and the step-wise G4) |
| Stage tables with the XT images | a fenced `sched` with `transpose_source = "xt_lds"` | publish the Q^T / dO^T images before the stage-A read-ahead and count the XT writer's traffic in the synchronous model; refused today |
| G4 key slices of an LDS K^T image | `g4_slicing != "step"` with `xt_lds` and `kt_source = "lds"` | K-inner key slices of the K^T image; refused today (K^T from registers works) |

The prefetch ring, the DMA path, the merged barrier, the counted waits,
split dQ, mirror pairing and the THD work list are built (2.4, 2.7, 3.5).

### 3.9 Memory-access width and bounds levers

Default-off levers from an audit of every access of the main, prep, convert,
reduce and split-dQ kernels that is not a full-width vector access. Each lever
is a defaulted `AttnBwdSpec` field, registered in the tuning registry (family
6, `numerics = "bitwise"`), and bit for bit equal to the default on the device
tests. The vector `lse2` / `Dsum` loads are `stats_vec` (3.2).

* `ws_bounds = "buffer"`: the main kernel's scalar workspace stats loads
  (`StatsView.rows`: `stats_vec = 1`, unpacked steps, the synchronous step and
  the register and merged rings' row loads; not the counted waits' staged
  loads nor the DMA ring's copies) and the slice sink's stores go through a
  buffer descriptor whose range ends at the last valid row (`num_records` in
  bytes; loads past it read 0 and keep the neutral select, stores past it are
  dropped) instead of address clamps and per-row branches. Head-major
  workspace only (a tile's rows then fit a 32-bit range). Inert elsewhere
  (`INERT_KNOBS`); the registry enumerates it only where it acts.
* `dkv_cell_vec = 4`: the per-slot dK / dV epilogue (`dkv_epilogue =
  "direct"`, direct mode or the slice sink) transposes each 4 x 4 accumulator
  block inside a lane quad (two DPP `quad_perm` exchange stages) so a lane
  stores 4 consecutive columns of one row: 8-byte I/O-dtype stores in direct
  mode, 16-byte fp32 stores into a slice, instead of 2- / 4-byte stores. MFMA
  C layouts only (`c_quad_transposable`).
* `narrow_vec` (1, 2, 4; `stage_vec = 1` instances): user-tensor accesses of
  2 or 4 elements (4- / 8-byte) where every used B / H / S stride is a
  multiple of the width and every base is aligned to its byte size. A tuned
  value is a cap: the plan narrows it to `attn_bwd_narrow_width(req)` (1 gives
  the default kernels) and passes it to the prep, convert / reduce and dQ
  kernels too (`AttnBwdAuxSpec.narrow_vec`, `AttnBwdDqSpec.narrow_vec`).
* `slab_carry`: the Q / dO slab offsets are loop-carried i64 values advanced
  per q step by per-CTA deltas (a head, a q tile of `kM0` rows, or the jump
  at a wrap) instead of forming `h * s_h + q0 * s_t` with two i64 multiplies
  per tensor per step. Every visited q tile has `q0 <= len_q - 1`, so the
  default's base-row clamp is an identity and the addresses are the
  default's. Built for the synchronous step: the prefetch ring (register or
  DMA path, every depth, the merged barrier) and the counted waits form the
  slabs of the steps ahead from their (q tile, head) index, so the validator
  refuses `slab_carry` there; it also excludes `q_rotate`; inert with
  `head_pack` and in the grouped `head_outer` loops of edge tiles (their head
  is loop-invariant inside the range loops already). With
  `stage_table_atomics` the loop carries the offsets before the deferred dQ
  partial, which is read from the values after them. It holds three more i64
  values per tensor in SGPRs across the loop, so on tiles at the SGPR ceiling
  it trades the multiplies for SGPR spills; the resource gate decides per
  instance.

The dense and unified shells build their own addressing, statistics view and
epilogue: `ws_bounds`, `dkv_cell_vec`, `narrow_vec`, `slab_carry` and
`dkv_sink` take only their defaults there (8.2, 8.3).

## 4. Memory layout

### 4.1 LDS phase planner

`_attention_bwd_lds.plan_bwd_lds` places every LDS buffer of a CTA inside one
flat `smem_alloc` (`BwdLds`), and the body reads every buffer through an
`LdsView` at the planner's byte offset, so the planner's footprint is the
kernel's group segment and its aliasing is exactly what the IR does.

Buffers (names are stable):

| Buffer | Phase | Size | Present when |
|---|---|---|---|
| `k_res`, `v_res` | resident | `kN0 x D` | `kv_residency = "lds"` (`k_res` also when K^T is read from LDS with register K / V) |
| `dk_acc`, `dv_acc` | resident | `kN0 x D` fp32 | `acc_in_lds` |
| `kt_res` | resident | XT image `D x kN0` | `xt_lds` with LDS K^T |
| `k_stage` | stage-0 K | `kN0 x D` | register K / V without a resident `k_res` |
| `kt_stage` | stage-0 K | XT image `D x kN0` | `xt_lds` with register K^T |
| `v_stage` | stage-0 V | `kN0 x D` | register K / V |
| `q`, `do` | loop, per ring slot | `kM0 x D` | always |
| `qt`, `dot` | loop, per ring slot | XT images `D x kM0` | `xt_lds` |
| `ds` | loop | `kM0 x kN0` | atomic dQ or LDS route |
| `pt` | loop | `kM0 x kN0` | `pt_route = "lds"` |
| `dkv_out` | epilogue | `kN0 x D` (one or two slots) | direct register accumulators with an LDS-staged epilogue |

Footprint = resident + max(stage-0 K, stage-0 V, loop, epilogue), every buffer
aligned to a 128-byte line so a per-buffer swizzle sees the same address
classes at any offset. `lds_order` puts the resident region below
(`resident_low`, default) or above (`phase_low`) the aliased region; sizes are
equal. `LdsPlan.check` proves that no two buffers live in one phase overlap.

```
 gfx950 d128 default (16 x 128, K / V in registers, K^T from LDS), fp16 / bf16
 offset 0 KiB          32 KiB                                 64 KiB
        +--------------+---------------------------------------+
        |   k_res      |  stage-0 V:  v_stage (32 KiB)         |   one of these
        |  (resident:  |  loop:       q 4 | do 4 | ds 4 KiB    |   at a time,
        |   G4 K^T)    |  epilogue:   dkv_out (32 KiB)         |   barrier-separated
        +--------------+---------------------------------------+
 gfx942 d128 default (16 x 128, K, K^T, V in registers): no resident buffer;
 k_stage, v_stage, the loop tiles and dkv_out share one 32 KiB region.
```

Row layouts (`lds_swizzle`, per buffer, canonical string `"buf=value,..."`):
a row-major buffer is packed (`none`, the default), padded (`pad8`, `pad16`,
`pad32` bytes per row) or line-XOR swizzled (`xor`: `RowXorLayout`, 16-byte
chunks of every 128-byte line permuted by a hash of the row, so the 16-byte
staging stores and K-inner reads stay whole). An XT image takes the same
values, its XOR chunk being the reading atom's `atom_k / 4` elements
(`XtLayout`); without an entry it takes the conflict predictor's choice
(`planner_xt_swizzle`). The body reads every layout through an `LdsView` (an
`XorLdsView` for `xor`, whose index is the planner's offset: the XOR touches
only the chunk bits of a line), which never claims more vector alignment than
the origin, the pitch and the XOR chunk allow (a padded pitch or a chunk
splits wide accesses). The DMA slab XOR belongs to the unbuilt `dma` path.

Per-arch capacity is the catalog's `lds_capacity_bytes`: 64 KiB on gfx942,
gfx1151 and gfx1201, 160 KiB on gfx950. The validator refuses a plan above it.

### 4.2 Register plan

**Model.** `attention_bwd.register_estimates` (per lane, 4-byte registers, two
16-bit operands per register):

| Term | Registers |
|---|---|
| resident K, V (`kv_residency = "reg"`), K^T (`kt_source = "reg"`) | `kN0 * D / (2 * wave * W)` each (K, V on a small-tile G0 / G2 grid: `W` is the grid's columns; the shipped `(2, W/2)` grid keeps `W`) |
| Q / dO step operands | `2 * kM0 * D / (2 * wave)` |
| register prefetch (`vgpr` ring, R >= 2) | `kM0 * D / (wave * W)` per set in flight (`R - 1` sets); the DMA ring holds none; the counted waits carry one tile plus its stats rows |
| P / dS operands | `kM0 * kN0 / (wave * W)` |
| G4 dS operands | `2 * kM0 * k_g4 / (2 * wave)` (`g4_slicing = "step"`: one atom-K step live); under key slicing `2 * kM0 * kK4 / (2 * wave)`, plus `2 * D * kK4 / (2 * wave * g4c)` for the K^T slices when `kt_source = "lds"` |
| dK, dV accumulators | `kN0 * D / (wave * W)` each (none with `acc_in_lds`) |
| S, dP accumulators | `2 * kM0 * kN0 / (wave * W)` (this and P / dS: `W` is the G0 / G2 grid's waves, fewer on a replicated grid) |
| dQ partial | `kM0 * D / (wave * W)` |
| allowance (addresses, loop state, stats, masks) | 24 |

Each accumulator term is counted in the arch-VGPR or the AGPR budget according
to where the emitted code keeps it (`_agpr_placement`, below).

**Budgets.** CDNA has 256 arch VGPRs and 256 AGPRs per lane, one 512-entry
unified file per SIMD. A CTA of `W` waves puts `ceil(W / 4)` waves on each
SIMD, which share the file: an eight-wave CTA gives each wave 256 registers in
total, whatever the AGPR attribute says. The validator applies: arch-VGPR
estimate within 256 (plus a static prune allowance of 32; the compiled
resource notes then decide), AGPR estimate within the reservation, arch +
AGPR within the per-wave share for `W > 4` and for `waves_per_eu` 2 to 4, and
an explicit reservation `(n, n)` must leave at least 24 arch VGPRs of the
share (`(256, 256)` on eight waves is refused). RDNA has 256 VGPRs per lane
and no AGPRs.

**Why AGPR placement matters.** The VALU reads S and dP (softmax gradient)
and the dQ partial (the atomics) every step. When an MFMA writes its result in
AGPR form, each such read is a `v_accvgpr_read` inside the q loop, and an
allocator short of registers adds AGPR spill copies (`v_accvgpr_write` /
`mov`) as well. The backend picks the form per kernel, not per GEMM: with an
`amdgpu-agpr-alloc` reservation (or no attribute and a file that does not fit)
it selects the AGPR form for every MFMA, G0 / G2 / G4 included. Only dK / dV
benefit from AGPRs: they are written by G1 / G3 every step and read by the
VALU only in the epilogue.

**The plans** (`agpr_alloc`, `agpr_route`):

| Plan | Spelling | Arch VGPRs hold | AGPRs hold | Mechanism |
|---|---|---|---|---|
| VGPR form | `agpr_alloc = (0, 0)` | everything | nothing | attribute `0,0` plus the comgr option `-amdgpu-mfma-vgpr-form` |
| attr route ("CK plan") | `(n, n)`, `agpr_route = None` | operands, P / dS | dK, dV, S, dP, dQ partial | the reservation attribute only; the backend selects the AGPR form for every MFMA |
| split, `vgpr_form` | `(n, n)` + `"vgpr_form"` | everything but dK / dV | dK, dV (if the allocator puts them there) | reservation plus the VGPR-form comgr option (`mfma_vgpr_form` kernel attribute) |
| split, `agpr_fence` | `(n, n)` + `"agpr_fence"` | everything but dK / dV | dK, dV | as `vgpr_form`, plus each carried dK / dV fragment passes through an empty asm with an `=a` output at the back-edge (`_agpr_fence`) and once through `=v` after the loop (`finish`) |
| split, `asm_g13` | `(n, n)` + `"asm_g13"` | everything but dK / dV | dK, dV | G1 / G3 issued as AGPR-form inline-asm MFMA clusters (section 5); G0 / G2 / G4 stay VGPR-form intrinsics |
| `asm_g13` + `s_in_agpr` | `(n, n)` + `"asm_g13"` + `s_in_agpr = True` | everything but dK / dV, S, dP | dK, dV, S, dP | G0 / G2 issued as AGPR-form asm clusters too (section 5; the first K step reads srcC `0`, so no AGPR is zeroed; S / dP pinned to the top AGPRs of the reservation); the softmax gradient reads S / dP with `v_accvgpr_read` once per step; G4 stays a VGPR-form intrinsic |

`agpr_alloc = None` is the policy: `(0, 0)` when the whole estimate fits 256
arch VGPRs (always with LDS accumulators), else no attribute (backend choice);
`"backend"` forces no attribute. The shipped CDNA defaults are eight-wave
tiles precisely because of this: on four waves the per-wave state (resident K,
V, K^T and the dK / dV accumulators) fits only with accumulators in AGPRs,
which drags S / dP / dQ into AGPRs too; eight waves halve the per-wave state
and the VGPR form holds all of it.

**Copy budget.** `attention_bwd.agpr_copy_budget` is the one model of the
hot-loop rule: reads allowed per q step are `2 * kM0 * kN0 / (wave * W)` (S, dP)
plus `kM0 * D / (wave * W)` (dQ partial) under the attr route with
`s_in_agpr = True`, the S / dP term only under `asm_g13` with `s_in_agpr`, and 0
everywhere else; writes and moves are always 0 (they can only be spill
traffic). The resource gate counts `v_accvgpr_read / write / mov` in every loop
that issues matrix instructions (edge tiles emit three q loops) against it.

**`s_in_agpr`** declares S / dP in AGPRs. Under the attr route it changes no
code (the backend already puts them there); it selects the register estimate
and the copy budget, so the honest declaration of the attr route is
`s_in_agpr = True`, and the `False` twin always fails the copy rule
(`DECLARATION_ONLY`). Under `asm_g13` it is a code lever: G0 / G2 become
AGPR-form asm clusters, so S / dP sit in AGPRs and the copy budget allows
their reads. It is legal only with the attr route or `asm_g13`.

Other register levers: `waves_per_eu` (kernel attribute `n,n`, 1 to 4 on
CDNA; the arch-VGPR + AGPR estimate must fit `512 / n`;
`_attention_bwd_tiles.attn_bwd_occupancy` gives the waves per SIMD the
estimate and the LDS plan allow, in whole workgroups, and a hint above the LDS
bound is legal but unattainable, so a sweep skips it), `scheduler_strategy` (the backend's machine scheduler; the
one-wave d128 tiles, the CDNA narrow d128 tile and the RDNA d128 tiles resolve
to `iterative-minreg`, because the default scheduler fills the 256-VGPR file
there and some instances spill), `acc_in_lds` (dK / dV as fp32 LDS images:
each 16 x 16 tile is read before its MMA chain and written back after it; each
lane reads back exactly the cells it writes, so no barrier is needed; implies
the VGPR form), and `dkv_slab_hoist` / `interior_clamp` (address and clamp
placement, which shift the allocator's choices).

## 5. Inline-asm MFMA (`asm_g13`)

`_attention_bwd_gemm.mma_agpr_asm_steps` issues the MFMAs of one K step
(`asm_cluster = "k_step"`, default) or of a whole GEMM call (`"gemm"`) as one
`asm sideeffect convergent` block. Under `asm_g13` G1 / G3 are issued this
way; with `s_in_agpr = True` G0 / G2 are too (the same `asm_cluster` and
`asm_tail_nop` apply to all four):

* every accumulator `acc[fm][fn]` is an `=a` output tied to its input, so dK /
  dV live in AGPRs across steps with no copy;
* a GEMM that starts from zero (G0 / G2) passes no accumulator: its first K
  step reads the inline constant `0` as srcC (what the backend emits for an
  intrinsic with a zero accumulator) and its `=a` outputs have no tied input,
  so no `v_accvgpr_write` zeroes the AGPRs every step;
* the G0 / G2 outputs are pinned to physical AGPRs (`={a[lo:hi]}`): S and dP
  take the top `2 * kM0 * kN0 / (wave * W)` AGPRs of the reservation `(n, n)`,
  dK / dV the rest. With free choice the production allocator made room for
  the short-lived S / dP by moving part of dK / dV to arch VGPRs and back every
  step (AGPR copy traffic beyond the budget) although the reservation had
  free AGPRs; pinned, the q loop issues exactly the S / dP reads of the budget;
* the A and B fragments are `v` inputs, each passed once (bitcast to an i32
  vector of the same width), read exactly as on the intrinsic route;
* the block starts with `s_nop 1` (the 2 wait states of
  `ASM_MFMA_VALU_LEAD_WAIT_STATES`), so a VALU write of a source the backend
  scheduled right before the block (the conversion that builds a relabelled
  A fragment) cannot land inside the lead window;
* MFMAs are issued K outer, fragment inner (the intrinsic route's
  accumulation order, so the result is bitwise the intrinsic one); with a
  single fragment per wave the K steps of a `"gemm"` cluster are separated by
  `s_nop`;
* the block starts with `s_nop 1` (the VALU-write lead window, below) and
  ends with `s_nop asm_tail_nop`.

**Hazards.** An asm MFMA is opaque to the backend's hazard recognizer, so the
cluster must cover its own dependencies (`_attention_bwd_caps`):

* a dependent read of an XDL MFMA result that is not the next MFMA's
  full-overlap srcC (a VALU / VMEM / LDS read, `v_accvgpr_read`, srcA / srcB,
  partial srcC) needs `passes + 3` wait states on gfx942 and `passes + 4` on
  gfx950; every 16 x 16 f16 / bf16 atom here is 4-pass, so 7 and 8
  (`mfma_result_wait_states`);
* a full-overlap srcC chain (the next K step on the same accumulator) needs
  none;
* a VALU write of a VGPR an MFMA reads needs 2 wait states before the MFMA
  (`ASM_MFMA_VALU_LEAD_WAIT_STATES`); the backend does not see the asm's
  source reads and may schedule such a write (for example the P^T / dS^T
  relabel conversion) right before a cluster, so every cluster starts with
  `s_nop 1`.

`s_nop n` gives `n + 1` wait states and the instruction right after the
cluster may read the last MFMA's result, so the floor is the result wait
states minus one: **6 on gfx942, 7 on gfx950** (`asm_tail_nop_min`), whatever
the cluster size. The validator refuses a smaller `asm_tail_nop`
(`check_asm_tail_nop`, on the G1 / G3 atom and, with `s_in_agpr`, on the G0 /
G2 atom); the shipped count is 15 (`AGPR_ASM_TAIL_NOP`), and a resolved spec
spells 15 as the default `None`.

G0 / G2 as asm (`s_in_agpr`) add one dependency kind and no new table entry:
the softmax gradient reads S / dP with `v_accvgpr_read` (a VALU read of an asm
MFMA result, the same wait as the epilogue's read of dK / dV), right after the
G2 cluster's tail `s_nop`. Their sources are the Q / dO LDS reads and the
resident K / V registers (no VALU writes them in the step), and the srcC `0`
of the first step reads no register.

**Evidence** (`tests/test_attention_bwd_agpr_probe.py`). A block GEMM cannot
show the hazard (a full-overlap srcC chain needs no wait state), so the probe
uses a negative control that provably reads early: one MFMA and a
`v_accvgpr_read` of its result in one asm block with nothing in between (the
disassembly is checked). On gfx942 and gfx950 that control gives wrong values,
while every count from the floor up is bitwise equal to the intrinsic, and a
chained srcC sequence with nothing in between is bitwise; the zero-accumulator
form of G0 / G2 is bitwise the intrinsic GEMM at every legal count too. An ISA
audit of whole `asm_g13` kernels built at the floor, with and without
`s_in_agpr`, checks that every asm MFMA result
is read no earlier than the table allows and that no VALU write of an asm MFMA
source lands inside the lead window. Before the lead `s_nop`, that audit
flagged a gfx950 d64 `asm_g13` kernel over `xt_lds` (production comgr: a
`v_cvt_pk_f16_f32` one wait state before an asm MFMA), and the kernel gave
non-finite dV on the device; with it the audit is clean and the asm variants
pass under every XT / `xor` configuration on gfx942 and gfx950. The same audit
with the lead removed (`BlockGemm.mma(asm_lead_wait=0)`, a test control only)
also finds such writes on the one-wave tile under other stage orders and under
the operand hoist below (gfx942 `v_pack_b32_f16`, gfx950 `v_cvt_pk_f16_f32`),
and a gfx942 device run of the hoist with G4 between G1 and G3 gave non-finite
dK without the lead; with it the audit (srcC included) is clean in every stage
order, at both granularities, with and without the hoist.

**Costs and couplings.** Each asm block is a scheduling fence (side effect), so
`asm_g13` excludes stage tables and `iglp` (the scheduler cannot see the MFMAs
inside), and `asm_cluster = "gemm"` reduces the number of fences.
`asm_load_hoist` (`asm_g13` with register accumulators only, default
`False`) reads the LDS operands of the G1 / G3 GEMM issued second (G3 in the
default stage order) right after the first GEMM's first-step operands, before
its first asm cluster: an asm block is a scheduling barrier for memory
operations, so without it those reads wait behind the first GEMM's MFMAs. Same
MFMAs on the same values (bit for bit), more live operand registers.

## 6. Masking and iteration

### 6.1 Keep rule

The frozen forward band (`rocke.helpers.attention_band`) plus the backward's
row term:

```
keep(q, k) = q < len_q and k < len_kv
             and (right < 0 or k <= q + off + right)
             and (left  < 0 or k >= q + off - left)
off = 0 (top-left) or len_kv - len_q (bottom-right), per sequence
```

Causal is `left = -1, right = 0`; a sliding window sets either or both bounds
(either alignment); no mask uses `NoMaskKTail` (only the q tail and the k tail
are masked). Bounds and lengths stay below `2^30` (i32-safe arithmetic).

### 6.2 Inverse band (`_attention_bwd_band.q_tile_range`)

A key tile `[k0, k0 + kN0)` is attended by rows `q_lo <= q <= q_hi`:
`q_lo = max(0, k0 - off - right)` (right bounded), `q_hi = min(len_q - 1,
k_last - off + left)` (left bounded), `k_last = min(k0 + kN0 - 1, len_kv - 1)`.
The row range is exact, so the only slack is tile quantisation at both ends.
A tile past `len_kv`, `len_q = 0`, or a band that misses every row gives
`n_qt = 0`. Both results go through `readfirstlane`, so the q-loop trip counts
live in SGPRs. The classifier and the inverse band are checked exhaustively
against a brute-force keep on the host (`tests/test_attention_bwd_band.py`).

### 6.3 Per-row key limits

On edge steps each row gets limits `[lo, hi]` once
(`RuntimeBand._row_lo_hi`: `lo = max(q + lo_off, 0)`, `hi = min(q + hi_off,
len_kv - 1)` with CTA-uniform `hi_off = right < 0 ? 2^30 : off + right` and
`lo_off = left < 0 ? -2^30 : off - left`), with `hi = -1` for a row that keeps
nothing (outside `len_q`, empty band row, dead row, dead packed row). The cell
test is one compare: `min(max(k, lo), hi) == k`.

### 6.4 Edge / interior / edge loops

A (q tile, key tile) pair is **interior** when every row is `< len_q`, every
key `< len_kv` and every cell inside the band (checked at the two corners
that bound the band). The interior tiles of a key tile form one contiguous
sub-range `[i0, i1)` of the visited range (`interior_q_range`), so with
`edge_tiles = True` the q loop is emitted as three loops over
`[qt_start, i0)`, `[i0, i1)`, `[i1, qt_end)`, in the single loop's order:

* interior steps skip the per-cell keep selects and keep only the dead-row
  select of dS (P of a dead row is `+0` anyway), and drop the row clamps of
  the Q / dO staging and the dQ sink (`interior_clamp = True` keeps them, which
  changes only register allocation);
* the accumulation order is unchanged, so dK / dV are bitwise those of
  `edge_tiles = False` in direct mode (a device test checks it).

A head-packed tile is always an edge tile. The default is `edge_tiles = False`
(one loop, selects on every step).

### 6.5 Step order, GQA groups and head packing

The CTA's work is a flattened sequence of (q tile, head group) steps with
`dK / dV` carried through it:

* `step_order = "qtile_outer"` (default): q tiles outer, head groups inner, so
  each q-tile range (edge / interior / edge) is one contiguous run of steps;
* `"head_outer"`: q tiles inner; with several q-tile ranges a loop over the
  head groups runs the range loops of one group per trip. dK / dV are summed in
  a different order when a CTA has more than one head group and more than one
  q tile.
* `q_rotate`: each q-tile range starts at `qt0 + kt mod len` and wraps, so the
  kv-tile CTAs of one head walk its q tiles (and dQ rows) out of lockstep (an
  order change for dK / dV).

**Head packing** (`head_pack`, `PackedRows`, `PackedStep`). A GQA group whose
query span is short (`s_q = 1` decode-like cells) would fill one of the `kM0`
rows of each step. The plan sets `head_pack` when `G > 1` and
`G * S_q <= kM0`, and passes `pack_heads = G`: M row `r` of the single q tile
then holds row `r % rows_per_head` of query head `head0 + r / rows_per_head`
(`rows_per_head = max(len_q, 1)` at runtime). Staging, statistics, the keep and
the dQ sink use each row's own head (per-row i64 head rebase); rows of a head
index past the group are dead in every term. `pack_heads = 0` runs the
unpacked loop in the same binary.

### 6.6 Sequences and addressing

**Decode** (`_attention_bwd_addr.decode_batched`, `decode_thd`):

* batched: `len = has_len ? clamp(SEQ[b * len_stride], 0, S_max) : S_max`; the
  SEQ pointer is read inside a 0/1-trip loop, so a null pointer is never
  dereferenced when `has_len = 0`; workspace rows start at `b * S_max`;
* THD: `o0 = OFF[b]`, `o1 = OFF[b + 1]` (int32, or int64 read as two words, by
  the runtime `off64`; an int32 table is never read past `B + 1` entries);
  `tok0 = o0 * mult / div`, `len = clamp((o1 - o0) * mult / div, 0, S_max)` in
  64-bit arithmetic (`div` is the tensor's token stride for element offsets),
  `min`-ed with the SEQ count under `has_len`. Workspace rows start at `tok0`
  (`ws_seg = 0`, rows by storage token) or at `b * S_max` (`ws_seg != 0`, rows
  by sequence slot); the selected base is `readfirstlane`'d so the backend
  cannot sink it past the loops.

**User tensors** (`StridedTensorAddr`): per-tensor `(batch, head, token)`
element strides, unit D stride. `slab(h, row0)` rebases the pointer once with a
64-bit byte offset (`global_ptr_add`) to (batch or sequence, head, first row of
the tile); the remaining per-element index is a small i32 (`< rows * stride_t +
D`, guaranteed by the predicate's `INDEX_RANGE`). In THD mode the sequence base
`tok0 * stride_t` is folded into the base pointer when the address generator is
built, so one 64-bit pointer (not a pointer and a 64-bit offset) stays live
across the q loops; the gfx950 THD kernels sit at the SGPR ceiling without it.
Rows are clamped to `max(len - 1, 0)` for operand reads (`clamp=True`) and
addressed raw for predicated stores (`clamp=False`). `stage_vec` sets the global
access width of the user tensors only: 8 issues 16-byte vectors (every used
B / H / S stride a multiple of 8 elements, bases 16-byte aligned), 1 issues
2-byte accesses (any element stride, any element-aligned base). LDS, MMA,
workspace and ABI are the same for both.

**Workspace** (`WorkspaceAddr`): fp32 `[h][rows][X]` (`head_major`) or
`[rows][h][X]` (`token_major`), `X = D` or 1. The (head, base row) offset is an
i64 pointer rebase; rows at or past `ws_rows` are clamped to a valid row and
their values replaced (reads), zeroed (atomics) or skipped (epilogue atomics),
so no access leaves the workspace even when a caller's THD bound is too small.

### 6.7 Zero-length spans

* `len_q = 0`: `n_qt = 0` for every key tile, so dK / dV of that batch are
  exact zeros; prep writes neutral statistics; in batched mode the convert
  writes the dQ rows below `S_q_max` as exact zeros, in THD mode it writes
  nothing.
* `len_kv = 0`: every key tile has `k0 >= len_kv`: no global read, no q step;
  in batched direct mode the epilogue still stores exact zeros to dK / dV rows
  below `S_kv_max`; dQ stays exactly 0 (zeroed workspace, no contribution).
* THD zero-length sequences, including the last one: the offset table is read
  at `b` and `b + 1` only; no row of the segment exists, nothing is written.
* An empty band behaves the same (0 trips, zero accumulators, zero rows).

## 7. Arch specifics

| | gfx942 | gfx950 | gfx1151, gfx1201 |
|---|---|---|---|
| Matrix path | MFMA, wave64 | MFMA, wave64 | WMMA, wave32 |
| Atoms | 16x16x16 | 16x16x16, 16x16x32 | wmma 16x16x16 |
| LDS per workgroup | 64 KiB | 160 KiB | 64 KiB |
| Registers per lane | 256 arch VGPR + 256 AGPR (unified 512 per SIMD) | same | 256 VGPR, no AGPR |
| Transpose read | none (`lds_plain`) | `ds_read_tr16_b64` (`tr_read`) | none (`wmma_lds`) |
| LDS DMA | one dword per lane (not used) | 1, 3 or 4 dwords (`dma` path) | none |
| Legal waves | 1, 2, 4, 8 | 1, 2, 4, 8 | 1, 2, 4 |
| Legal ring depth (registered) | 1, 2 | 1, 2, 3 | 1 |
| `waves_per_eu` | 1, 2, 3, 4 | 1, 2, 3, 4 | 1, 2 |
| asm tail `s_nop` floor | 6 | 7 | n/a |
| Native fp32 no-return atomic add | yes | yes | yes |

Default start points (`attn_bwd_default_geometry`; a tuned table overrides
them per class):

| Target, d | kM0 x kN0, W | G4 grid | K / V, K^T | Transpose | Atoms G02 / G13 / G4 |
|---|---|---|---|---|---|
| gfx942 d32 | 32 x 128, 4 | (2, 2) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx942 d64 | 32 x 128, 8 | (2, 4) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx942 d128 | 16 x 128, 8 | (1, 8) | reg, reg | lds_plain | 16 / 16 / 16 |
| gfx950 d32 | 32 x 128, 4 | (2, 2) | reg, reg | tr_read | 32 / 16 / 32 |
| gfx950 d64 | 32 x 256, 8 | (2, 4) | reg, lds | tr_read | 32 / 16 / 32 |
| gfx950 d128 | 16 x 128, 8 | (1, 8) | reg, lds | tr_read | 32 / 16 / 32 |
| CDNA narrow (`stage_vec = 1`) | 16 x 64, 4 (d32: 32 x 64) | (1, 4) / (2, 2) | reg, reg | lds_plain | gfx950 wide where legal |
| RDNA | 16 x 16, 1 | (1, 1) | lds, lds | wmma_lds | wmma |

Every default uses `global_path = "vgpr"`, ring depth 1, no stage table,
`pt_route = "relabel"` on CDNA and `"lds"` on RDNA, register accumulators on
CDNA and LDS accumulators on RDNA, `waves_per_eu = 1` on CDNA. The gfx950 d128
tile is 16 x 128 on eight waves because 192 keys do not split over eight waves.
The narrow tiles exist because scalar global access cannot feed a large tile.

RDNA specifics: one wave runs the whole 16 x 16 tile; K / V and the transposed
operands come from LDS; the relabel route is refused (the WMMA C layout is not
the A layout of the transposed product without a cross-lane exchange); dK / dV
are fp32 LDS images, so G4 always feeds the sink one fragment at a time, which
bounds the live registers. RDNA targets carry no performance claim
(`degraded` in the declaration) and are never declined on performance.

## 8. Families

### 8.1 Layout-general (built)

Everything above. One builder (`build_attn_bwd_main(spec, *, arch)`), one
aux spec shared with prep / convert (`AttnBwdSpec.aux_spec`), one plan. The
compile-time problem class is `(dtype, d, seq_mode, dkv_mode, mask_class,
stage_vec)`; everything else is a knob.

### 8.2 Dense (built; a route inside declared cells)

A per-arch family for contiguous BSHD / BHSD fixed-length problems with the
shape baked at build time. It runs the shared tile step (section 3), mask
policy, stats view, dQ sink, per-kv-tile driver and dK / dV epilogues behind a
dense shell, so it has no predicate, reason codes or declaration of its own:
it only serves cells the general family already declares.

* **Eligibility** (`dense_bwd_eligibility`, a routing tag, never a decline):
  gfx942 / gfx950, every tensor dense BSHD or BHSD with the same layout on the
  q and kv sides, base alignment >= 16 bytes, no padding and no SEQ_LEN
  tensors, `h_k == h_v`, d 64 / 128.
* **Spec.** `Gfx942AttentionDenseBwdSpec` / `Gfx950AttentionDenseBwdSpec`
  (`library/kernels/{gfx942,gfx950}/attention_dense_bwd.py`, base
  `AttentionDenseBwdSpec`): `seqlen_q`, `seqlen_kv`, `num_query_heads`,
  `num_kv_heads`, `head_size` required; `batch = None` (runtime kernarg, grid
  z; an int pins it for A/B only); mask `causal`, `causal_bottom_right`,
  `sliding_window` or an explicit two-sided band (`window_left` /
  `window_right`); `layout` `bshd` / `bhsd`; `ragged` (lengths that are not
  tile multiples: its own kernel; tail tiles masked, rows past the sequence
  load as zero, partial rows dropped by guarded stores).
* **Dense knobs** (defaulted, registered in
  `dispatch/attention_bwd/dense_tuning.py`): `epilogue` (`lds_staged`,
  `lds_staged_pair`, `direct`), `oob` (`buffer_oob`: one buffer resource per
  (batch, head) slab with `num_records` ending at the last row, rows past the
  sequence read as zero; `clamp`: the general clamped slab), `edge_tiles`
  (compile-time band classification over every kv tile; a loop proved empty
  at build time is not emitted), `ws_layout`, `dkv_mode` (`atomic` exactly when
  the runtime `g_split > 1`), `persistent` / `persist_decode` (`kv_tile_major`,
  `hkv_major`; runtime `num_persistent`), `xcd_remap` (whole-grid chiplet remap
  through `xcd_remap_id`, set by the plan when `num_xcds` and a non-zero
  `xcd_chunk` give `xcd_n > 0`), `dq_mode` (`split` registered, not built), the
  per-arch `body_variant` (`general`; the gfx942 XT and gfx950 trload bodies are
  registered, not built) and `body_knobs` (any tuning knob of `AttnBwdSpec`
  except the fields the dense spec owns; an entry equal to its default is
  dropped, so one kernel has one spec and one name). As in the general family,
  the plan derives the body knobs `static_scale` (multipliers that are 1.0
  under the scale placement are not applied) and `lb_interleave` (from
  `lb_order = "interleave"`); `ptr_noalias` marks the dense pointer list by the
  general rule. Kernel names carry the base key and a salt over every
  non-default knob. A hook configuration with `g_split > 1` gets
  `dkv_mode = "atomic"` (the general plan's rule), so one knob set serves
  both routes; `head_pack = False` is accepted and dropped, `head_pack = True`
  is refused as not built (one query head per q step; short-query GQA
  requests stay on the general route, whose plan packs their heads).
* **Addressing.** `DenseFixedAddr`: compile-time element strides, the batch
  and head rebased in 64 bits; the in-slab 32-bit voffset bound is checked at
  build time.
* **Prefetch ring.** The dense shell builds the general shell's Q / dO
  prefetch ring (`QRing`, 3.5: the register path at depth 2 / 3, the DMA path
  at depth 1-3 on gfx950) with the same geometry, LDS plan, stats view and dQ
  atomic count, and hands it to the shared per-kv-tile driver; the merged
  barrier (`barrier_merge = "on"`) stages inside the tile step as in the
  general family. The ring loads Q and dO through a global pointer (its DMA
  copies take a pointer and an i32 offset; its joint register load selects
  between the Q and dO pointers), so under `oob = "buffer_oob"` those two
  tensors use a global-pointer view of the slab: no row clamp in a non-ragged
  kernel (no q row passes the sequence end), the general clamp in a ragged
  one; K and V keep their buffer resources. A direct `buffer_load ... lds`
  from the slab resource is not built (it buys nothing over the view: the
  non-ragged view never reads past the end and the ragged one clamps).
* **Launches** (`attn_dense_bwd_plan`): `prep -> dense main -> convert(dQ)
  [-> convert(dK), convert(dV)]`; prep, the converts, the workspace and the
  scale multipliers are the general family's (same request, `ws_layout`,
  `g_split`, scale placement and aux values).
* **Kernarg ABI `rocke.attn_dense_bwd.v1`** (main only): pointers `Q K V O dO
  LSE dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV` (O and LSE are never
  dereferenced), f32 `scale_log2 ds_mult dk_mult dq_mult`, i32 `batch g_split
  lb_order pair xcd_n xcd_chunk num_persistent` (`pair` must be 0: mirror
  pairing is not built).
* **Numerics.** dK / dV are bit for bit those of the general family on the
  same body configuration with edge tiles on and head packing off; every
  dense knob except `dkv_mode = "atomic"` keeps dK / dV bitwise.
* **Split dQ** (`dq_mode = "split"`, a dense spec field). The dense main
  kernel computes dK / dV only (a `NullSink`, as in the general shell: no G4,
  no dS tile, no `WS_DQ` access; the ring's DMA wait counts no dQ atomics),
  and the dense split dQ kernel (`library/kernels/common/attention_dense_bwd_dq.py`,
  `AttentionDenseBwdDqSpec`, `build_attention_dense_bwd_dq`) produces dQ. It
  is the general dQ kernel's body (2.4a, `BwdDqStep`) behind the dense
  addressing: compile-time lengths, heads, group and strides; under
  `buffer_oob` a buffer resource per K / V slab (and per Q / dO slab of a
  one-head CTA; a head-packed CTA reads Q / dO through the global-pointer
  view); the dense mask policies; and a compile-time classification of every
  (q block, key tile) pair (`dense_dq_static_classes`, the kernel's own range
  emitters on the host) that drops the edge key loops when no visited tile is
  an edge tile and the interior loop when none is interior (with
  `dq_edge_tiles`; the same key order). The tuning values are the general dQ
  kernel's (`AttnBwdDqSpec`, the plan's `dq_*` knobs). Launches: `prep ->
  dense main -> dense dQ [-> convert(dK), convert(dV)]` (stage `"dense_dq"`):
  the general plan in split mode supplies prep without the dQ zero fill, the
  workspace without `WS_DQ` and the dQ multipliers (the convert's folded into
  `dq_mult`). Kernarg ABI `rocke.attn_dense_bwd_dq.v1`: pointers `Q K V dO dQ
  WS_LSE2 WS_DSUM`, f32 `scale_log2 ds_mult dq_mult`, i32 `batch xcd_n
  xcd_chunk`; grid `(ceil(S_q / R), h_q / H, B)` as in 2.4a. dQ, dK and dV
  are bit for bit those of the general family's split run with the same
  values (the device twin tests).
* **Not built.** The per-arch bodies (`body_variant` `xt` / `trload`) and
  mirror pairing raise before any IR is emitted.
* **The shape pin is explicit.** The shape is compiled into the code object
  (strides, slab ranges, the kv-tile count and the compile-time band
  classification fold it into constants), so two shapes are two kernels and
  the shape is part of the kernel name (readable tokens: profiles, traces and
  resource-gate rows stay attributable to a shape, as in the gfx942 dense
  forward). The spec class states it: `SHAPE_BINDING = "compile_time"`,
  `SHAPE_FIELDS` (`seqlen_q`, `seqlen_kv`, `num_query_heads`, `num_kv_heads`,
  `head_size`, `layout`, and `batch` when pinned) and `shape_key()`. A
  runtime-shape dense body would be a new body (the lengths as kernargs lose
  the compile-time classification) and is not built; the general family
  serves every shape outside the tuned dense keys.
* **Tuning.** The sweep's dense slices (9.2) take the general funnel's
  survivors onto the dense route, vary the dense-only knobs in the runtime
  stage and confirm the best dense candidate against the general winner; a
  dense key enters `DENSE_BWD_TUNED` only on a paired win with a clean A/A
  control under the pruning-set gate. The gate (`tests/_attention_bwd_perf_gate.py`)
  runs a dense plan's pruning cases on the dense route with the same
  configuration (one dense kernel per pruning shape, compiled once per job).
* **Body levers of the general shell only.** `dkv_sink`, `ws_bounds`,
  `dkv_cell_vec`, `narrow_vec` and `slab_carry` (2.5a, 3.9) act in the
  general shell's epilogue, statistics view and addressing; the dense shell
  has its own, so they take only their defaults on the dense route
  (`_shell_not_built`, registered as not built there).

### 8.3 Unified (built; paged KV, its own entry point)

THD queries over a paged KV cache (`[num_blocks, page, h_kv, D]` with a block
table), with dK / dV scattered into a paged gradient cache: direct scatter
when the caller asserts that the block table is injective over live slots,
else an fp32 page workspace with atomics (`page_zero` / `page_convert`). It
runs the shared tile step and the per-kv-tile driver behind its own shell and
has its own request, predicate, plan and declaration (version 2); the
contract is `attention_bwd_unified.md`. It is reachable from rocKE dispatch
only (`dispatch/attention_bwd/routing.plan_attention_bwd_unified`; hipDNN
keeps declining `PAGED`). The shell refuses the general shell's knobs it does
not build (the prefetch ring without the merged barrier, the DMA path, the
XCD remap, the interleave and mirror decodes, `ptr_noalias`, the dK / dV slab
hoist, the slice sink `dkv_sink`, `ws_bounds`, `dkv_cell_vec`, `narrow_vec`
and `slab_carry`: its own epilogues, statistics view and addressing); body
knobs (the merged barrier, the counted waits, the schedule
values, the XT images, the G4 key slicing, the VALU levers) pass through to
the tile step.

### 8.4 Routing

```
request -> attn_bwd_support (general predicate) -> decline (reason code) | OK
OK      -> attn_bwd_route: dense-eligible and key in dense_pack -> "dense"   -> attn_dense_bwd_plan
                           otherwise                           -> "general" -> attn_bwd_plan
```

Routing uses request fields only (no device query) and never adds or removes
a declared cell. `dispatch/attention_bwd/routing.py` (`plan_attention_bwd`)
returns the plan of exactly one family: `dense_policy` turns the tuned dense
table (`DENSE_BWD_TUNED`, the per-arch dense tables; a dense key enters only
after its confirm stage) into the policy's `dense_pack` and dense tuning, `arch_tuned_policy`
joins the per-arch general tables into the policy's `tuned_table` and the
per-arch status tables (empty, 9.3) into its `perf_status`, and `route=`
forces a route for gate runs. The routed policy serves every request the
predicate accepts and the plan builds; the performance status of a class
never steers the route or the knobs (9.3). A paged (unified) request is not
routed here: it has its own entry point (8.3).

**Dispatcher.** `dispatch/attention_bwd/registry.py` puts the three
families behind the forward dispatcher's interface (`rocke.dispatch.core`):
one `KernelCandidate` per (family, arch) in `ATTENTION_BWD_REGISTRY`, each
with a `Capability` and the family's exact predicate (the general predicate
under the routed policy, the dense eligibility plus the dense plan, the
unified predicate), the ranker `attention_bwd_ranker` (a candidate with a
tuned entry for the request first, so the dense family wins exactly where
this routing sends the request, then the registered order) and the entry
points `dispatch_attention_bwd`, `dispatch_attention_bwd_all`,
`launch_attention_bwd` and `run_attention_bwd`.

**Production status: dormant.** This routing and the dispatcher are not
load-bearing for production yet. No provider calls it: the hip-kernel-provider serves rocKE
kernels through the C++ engine, which has no backward (0.3), so a hipDNN
backward request is served by the provider's other engines. Its callers today are
the tests (`test_serving_policy.py` plans every accepted case through
`routing.plan_attention_bwd` with the shipped policy and no override;
`tests/dispatch/attention_bwd/test_registry.py` drives the dispatcher); the
timing harness plans its rocKE arms with the configured policy of each arm
(`attention_bwd_rocke_arm.config_policy`). Follow-up: wire the provider's backward path to
`attention_bwd_declaration` and `plan_attention_bwd` (with the C++ port of
0.3, or through a Python-serving provider).

The scalar `fmha_bwd.py` kernel and the `mfma_attention_bwd.py` helpers are a
separate, older backward (CK Tile parity); this family neither uses nor
extends them.

## 9. Tuning model

### 9.1 Knob registry

`dispatch/attention_bwd/tuning_common.py` registers every tuning decision as a
`KnobAxis`: every `AttnBwdSpec` field except the problem class, every plan
runtime value (`RUNTIME_KNOBS`: `g_split`, `lb_order`, `xcd_chunk`,
`scale_placement`, `use_worklist`) and every prep / convert value (`AUX_KNOBS`).
`_check_registry` fails the import if a spec field has no axis. Per axis:

* `choices[0]` is the default (the field's declared default, or the plan's
  value), so leaving an axis alone emits today's code;
* `family`: one of eight lever families in release order (geometry,
  residency / registers, transpose / global path, atoms / operand routes,
  schedule, codegen / minor / swizzle, runtime, split dQ);
* `stage`: the funnel stage that owns it (geometry, structure, schedule,
  codegen, swizzle, runtime);
* `kind`: `compile` (a spec field: recompiles, salted into the kernel name) or
  `runtime` (a kernarg, or an aux-kernel recompile);
* `level`: `production` or `full`; `full_only` values are enumerated only by the
  full (per-shape) sweep;
* `numerics`: `bitwise`, `reorders_fp` (same algorithm, another fp32 order or
  rounding point: within tolerance, not bit-equal), `tile` (another tile,
  atom, route or placement: its own kernel, held to the tolerance) or `n/a`;
* `not_built`: values the emission check or the plan refuses (recorded by the
  sweep, never compiled); `gated`: values a device probe has not cleared (none
  today);
* `plan_owned`: fields the plan derives from a runtime value (`static_scale`,
  `lb_interleave`, `xcd_remap`, `worklist`, `head_pack`), never enumerated on
  their own;
* `requires`: a cheap coupling filter; the kernel validator still decides.

Legality has one source: `backward_spec_legality` = `validate_attn_bwd_spec`,
the known-wrong values, then `_not_built`. `BWD_KNOWN_WRONG_KNOBS` (per
target, the values that build but are known to compute wrong results; the
analogue of the forward's `KNOWN_WRONG_KNOBS`) is empty: every backward value
found wrong was fixed or is refused exactly by its validator. An entry would
be held out of every sweep level, the legality verdicts, the tuned tables and
the dispatcher's candidates. The unified family's axes and legality are
`dispatch/attention_bwd/unified_tuning.py`; the dense family's,
`dense_tuning.py`.

**Lever retention.** A knob that loses everywhere stays registered and off by
default; nothing is deleted for losing. Knob values that change nothing under a
given spec are listed in `INERT_KNOBS` (predicate per spec; `block_k4` under
`g4_slicing = "step"`) and `s_in_agpr` under the attr route in
`DECLARATION_ONLY`; `NOT_YET_EFFECTIVE_KNOBS` (a knob whose emission is not
built at all) is empty. The sweep compiles one representative per set of
configurations that emit the same code. The geometry stage pins `block_k4`
per geometry; the schedule stage enumerates it with the `g4_slicing` values
(`g4_slicing_values`).

### 9.2 Funnel sweep

`benchmarks/common/attention_bwd_sweep.py` sweeps primary slices `(arch, d)`
(bf16, band mask, batched, direct, `stage_vec = 8`) and secondary slices
(fp16, no mask, THD, atomic, `stage_vec = 1`, head packing), stage by stage:

```
geometry (keep 8) -> structure (top 4 geometries; keep 8) -> schedule (4) -> codegen (4)
  -> swizzle (4) -> alternate (re-run structure on the top 16; 4) -> runtime (2) -> confirm (top 2 on the cohort)
```

Each stage: build candidates from the previous stage's merged records, drop
illegal and duplicate candidates (resolved specs equal apart from inert
knobs), compile (on-disk cache keyed by spec, arch, LLVM flavor, comgr and
source hash; knobs that can crash the compiler compile in child processes),
apply the resource gate (section 11) as a static prune, then time the
survivors with the paired harness: rocKE configurations and the competitor
in the same fresh-process attempts, every configuration correctness-gated in
the timing process. A lever family that is not released contributes only its
default. Records are JSONL outside the source tree; stdout carries counts and
verdict tokens only.

**Dense slices** (`--family dense`, gfx942 / gfx950, d 64 / 128) run only the
`runtime` and `confirm` stages; the other stages skip them. The runtime
stage seeds from the general slice's top runtime-stage survivors (with their
runtime values; mirror pairing and the work list have no dense counterpart),
moves them to the dense route and varies the dense-only knobs (`oob`,
`edge_tiles`, `epilogue`, `ws_layout`, `persistent` / `persist_decode`,
`dq_mode`, and the dense grid remap through `xcd_chunk`) one at a time
(`production`) or crossed (`full`). `dq_mode` moves a seed between atomic and
split dQ (an atomic point carries no dQ-kernel values, a split point keeps
the seed's or the defaults, and the split point at the seed's dense values
also tries `dq_edge_tiles = True`, whose empty key runs the dense dQ kernel
drops). Each dense candidate compiles one kernel per cell (the shape is
compiled in; a split candidate its dense dQ kernel too), each through the
resource gate (the dQ kernel through the dQ rules, `pruned:dq_<rule>`). The
cells are the
dense-eligible ones, minus the short-query GQA cells the general plan
head-packs. `confirm` times the general winner and the best dense candidate
in the same attempts on the dense cohort, each with its own A/A twin
(`--aa-control all` of the timing harness: the two arms differ in kernel and
runtime values, so one arm's twin does not bound the other's noise), and
records, per cell, the paired verdict both ways and the A/A of both arms.
`--emit-dense-table PATH` writes the `DENSE_BWD_TUNED` entries those records
earn: a dense key whose every cell is a paired win (dense passes against the
general winner, the general winner fails against dense) with a clean A/A
control of both arms under the pruning-set gate.

**Pruning set of a head split.** The gate runs, before any configuration is
timed, the pruning cases of its main-kernel class (`perf_prune_cases`) with
the configuration's runtime values; a case whose head group `G` the
configuration's `g_split` does not divide is left out (the plan refuses it).
The set's grouped-query cases have `G = 4`, plus one case per further head
group of the census (`PRUNE_HEAD_GROUPS`: `gqa5_br`, `gqa8_br`) in every class
whose `g_split` divides it, so a head group the census times (5 and 8 too)
runs in the gate under each of its splits. A grouped-query case whose `G`
(`gcd(h_q / h_k, h_q / h_v)`) the split does not divide is rescaled to
`G = g_split` (`h_q = h_k * g_split`, `h_k != h_v` kept, ids with `_hq<h_q>`),
so every legal split (a divisor of a census head group: 5 and 8 too) has a
full pruning set with an `h_k != h_v` case; the sets of `g_split` 1 and 2
carry no rescaled case.

**Device hints in the gate.** A configuration's `num_cus` / `num_xcds`
describe the timed device, and the rocKE arm fills either one in when the
configuration leaves it out, so the pruning cases take their request hints
from the timed plan (`case_request`), never from the configuration: the XCD
count of the main and split dQ remaps (`xcd_chunk` values `gx` / `2gx` /
`4gx` then resolve on each pruning grid), `num_cus = 1` for mirror pairing
(its fill guard is a no-gain rule the small pruning grids never pass), and
`num_cus` = the timed persistent grid for a persistent dense kernel. Both
routes take them (the dense route re-plans each case with the timed plan's
hints), and no hint ever reaches the policy as a knob.

### 9.3 Tuned tables and performance classes

* **Performance class** (`attn_bwd_perf_class`, request fields only):
  `(arch, route, d, dtype, mask kind {none, causal_tl, causal_br, window}, head
  mode {mha, gqa, hk_ne_hv}, length relation {square, unequal, s_q_1}, length
  mode {fixed, padded, thd}, stage_vec)`; the dense route's class is its exact
  key. The device hints `num_cus` / `num_xcds` never enter it.
* **Tuning key** (`attn_bwd_tuning_key`): the class plus a kv-tile bucket
  (`ceil(s_kv / kN0)` at the class's default kN0 against 1, 4, 16, 64) and a
  grid-fill bucket (default grid against `num_cus`: under, full, over,
  unknown). `AttnBwdPolicy.tuned_table` is looked up most-specific first (exact,
  then either bucket as `*`, then both), then the `tuning` hook overrides it.
* **Policy** (`shipped_policy`): the kernel-level default is `AttnBwdPolicy()`
  (every declared feature group, no performance gate, no tuned knobs, no
  dense pack); it reads no module and no table. The routing layer adds the
  per-arch tuned tables and the dense pack (`routing.routed_policy`, 8.4).
  Serving is functional only: the shipped and the routed policy serve every
  request the predicate accepts and the plan builds, on every declared
  target, gfx942 and gfx950 included. The declines that remain are the
  functional ones of 12.5 (scope, dtype, shape, layout, stride, range, the
  resource gate, an undeclared feature group).
* **Performance status is information.** Each class carries one of
  `verified`, `unverified` (the default for a class with no entry) or
  `below_baseline` (`AttnBwdPolicy.perf_status`); RDNA classes are `degraded`
  (no performance claim). The plan records the status of its class
  (`AttnBwdPlan.perf_status`) and the declaration publishes it per class
  (12.6). It never changes the verdict, the route or the knobs: the same
  request plans identically under any status. A policy that wants the old
  claim discipline opts into `AttnBwdPolicy.perf_gate`: a `below_baseline`
  class is declined `PERF_BELOW_BASELINE` and an `unverified` one
  `PERF_UNVERIFIED` unless `waive_unverified` (both off in the shipped and
  the routed policy; RDNA is never declined on performance). The status
  machinery is kept as default-off levers (lever retention).
* **No per-class status is shipped.** The in-repo status tables
  (`GENERAL_PERF_STATUS`, `DENSE_PERF_STATUS` of every arch module) are empty
  by policy: a per-class label derived from a comparison with another
  backend is a performance claim and stays out of the repository. Every CDNA
  class therefore reads `unverified` in the routed declaration, and every
  class is served. A deployment that keeps its own measured statuses passes
  them in `AttnBwdPolicy.perf_status` (and, if it wants the claim discipline,
  `perf_gate`); the tests use synthetic statuses.
* **Per-arch tables** (`dispatch/attention_bwd/gfx942_tuning.py`,
  `gfx950_tuning.py`): `GENERAL_TUNED_TABLE` (tuning key -> knobs) and
  `DENSE_TUNED` (`(arch, dense key)` -> dense knobs) of each arch, plus the
  empty status tables above. An entry is a key whose winner passed the
  bench's confirm stage on every measured cell of the key (clean A/A control,
  pruning-set gate on; a dense entry also a paired win over the general
  winner), one configuration on all of them. The routing joins them
  (`routing.ARCH_TUNING`, `routing.arch_tuned_policy` in `routed_policy`, and
  `DENSE_BWD_TUNED`). The fill bucket of a key is the measured one on the
  device (the requests carried the device's `num_cus`), so a request without
  `num_cus` (fill `unknown`) keeps the defaults. gfx942 has no dense entry
  (no dense configuration was confirmed there).
* **Provenance of the tuned entries** (qualitative; the measurements stay in
  the private record). The general entries come from the funnel sweep (9.2:
  `benchmarks/common/attention_bwd_sweep.py`, its confirm stage, timed by
  `attention_bwd_bench.py` with the pruning-set gate passed explicitly as
  `--gate tests._attention_bwd_perf_gate:gate`) on the census cells of
  `attention_bwd_census.json` that fall in each key; the selection rule is
  "one configuration confirmed on every measured cell of the key, against the
  default configuration and the competitor in the same paired attempts, with
  a clean A/A control" (a key whose cells disagree is left out). The dense
  entries come from the sweep's dense slices (`--emit-dense-table`, the
  paired win over the general winner). `general_entries` / `dense_entries`
  turn the sweep's emitted entries into checked table entries, and the entry
  check below runs on every one.
* **Why these buckets.** The kv-tile bucket bounds (1, 4, 16, 64 key tiles
  at the class's default `kN0`) are a design choice fixed before any
  measurement: powers of four from a single key tile (decode-like spans) to
  long sequences, so each bucket spans a four-fold range of key tiles per
  CTA column. The fill bucket compares the default grid with the device's
  CU count. Both read request fields only, so a table entry can never depend
  on its own choice; the census cells reach several kv-tile buckets and all
  three measured fill buckets (`test_attention_bwd_rocke_arm.py`).
* **Entry check** (`dispatch/attention_bwd/tuned_tables.py`): every entry,
  shipped or emitted by the sweep (`dense_entries` / `general_entries`), is
  planned the way the routing consumes it on representative requests of its
  key: for a general key, requests of its class and buckets that vary what
  the key leaves free (head counts and group sizes of the head mode,
  sequence lengths at both edges of the kv-tile bucket, batches and
  `num_cus` at the edges of the fill bucket, with and without `num_xcds`,
  both dense layouts, the band forms of the mask kind); for a dense key, its
  shape over several batches with and without `num_cus` / `num_xcds`. Every
  representative that plans without the entry must plan with it; the first
  that does not raises with the plan's reason (an unknown knob, an illegal
  value, a coupling, a `g_split` that does not divide every head group of
  the class, mirror pairing at a fill edge that does not overfill the
  device, an XCD remap without `num_xcds`).

### 9.4 Knob catalog

Default first. Numerics: **bitwise** (same fp32 operations in the same order),
**order** (another summation order or rounding point; within tolerance),
**tile** (own kernel; tolerance against the oracle, no bitwise claim),
**relaxed** (a declared change of the results outside the default's
contract: `bf16_cast = "rne_noguard"` for NaN inputs, `bwd_exp2 = "fast"` for
P below 2^-126; both default off).

Geometry, residency, paths, atoms:

| Knob | Values | What changes in the generated code | Numerics | Couplings / status |
|---|---|---|---|---|
| `waves` | policy; 1, 2, 4, 8 (WMMA: 1, 2, 4) | CTA size; every warp grid | tile | `kN0 % 16W == 0` on the natural grids (and under a stage table); W = 1 only for 16 x 16 |
| `block_m` (kM0) | policy; 16, 32, 64 | q rows per step | tile | multiple of `16 * g4r`; >= 32 for `atom_g13 = 16x16x32` |
| `block_n` (kN0) | policy; 16, 32, 64, 128, 192, 256 | keys per CTA, grid x | tile | multiple of `16W` on the natural grids, else split by the G0 / G2 and G1 / G3 grids; multiple of `block_k4` |
| `block_k4` (kK4) | policy; 16, 32, 64 | the G4 dS key-slice width under `g4_slicing != "step"`; inert under `"step"` | bitwise | multiple of the G4 atom K; divides kN0 |
| `warp_grid_g4` | policy; `(1, W)`, `(2, W/2)` | G4 wave split | tile | must divide D and kM0 |
| `warp_grid_g02` | `(1, W)`; `(2, W/2)`, `(W, 1)`, replicated `(1, k)` / `(k, 1)` / `(2, k/2)` for `k` dividing `W` | G0 / G2 wave split | tile | non-default needs `pt_route = "lds"`; must split `kM0 x kN0` into 16 x 16 atoms; no stage table with a replicated grid |
| `warp_grid_g13` | `(W, 1)`; `(2, W/2)`, `(1, W)` | G1 / G3 wave split | tile | as above, over `kN0 x D`; never replicated |
| `kv_residency` | policy; reg, lds | K / V B-fragments in registers once vs LDS reads per step | tile | reg illegal on WMMA |
| `kt_source` | policy; reg, lds | G4 K fragments in registers once vs transposed LDS reads per step | tile | reg needs register K / V |
| `transpose_source` | policy; tr_read (gfx950), lds_plain, wmma_lds (WMMA), xt_lds | K-outer LDS reader; `xt_lds`: XT images of Q^T / dO^T / K^T read as vectors (3.4) | tile (`xt_lds`: bitwise the plain reader) | `xt_lds` excludes `global_path = "dma"` |
| `global_path` | policy; vgpr, dma (gfx950) | global-to-LDS path (`dma`: the DMA ring, 3.5) | bitwise | `dma`: `stage_vec = 8`, no head packing, 1 KiB pieces; not with `xt_lds` |
| `ring_depth` | policy; 1, 2 (gfx942), 3 (gfx950) | Q / dO / stats prefetch depth (3.5) | bitwise | ring > 1 owns the staging order; 2 under `barrier_merge` |
| `atom_g02`, `atom_g4` | policy; 16x16x16, 16x16x32 (gfx950), wmma16x16x16 | MFMA instruction of G0 / G2, G4 | tile | arch atom set |
| `atom_g13` | policy; same | MFMA of G1 / G3; K-permuted B under relabel (on `tr_read` the stacked transpose-read plan) | tile | x32 needs kM0 >= 32 |
| `pt_route` | policy; relabel, lds | P^T / dS^T hand-off (3.3) | tile | relabel illegal on WMMA |
| `acc_in_lds` | policy; False, True | dK / dV as fp32 LDS images | order | implies `(0, 0)`; required on WMMA |

Registers, schedule, issue order:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `agpr_alloc` | policy; backend, (0,0), (128,128), (192,192), (256,256) | `amdgpu-agpr-alloc` attribute and comgr option | bitwise | reservation fits the estimate and leaves >= 24 arch VGPRs |
| `agpr_route` | attr; vgpr_form, agpr_fence, asm_g13 | register plan (4.2) | bitwise | needs `(n, n)`, n > 0; asm_g13 excludes stage tables and iglp |
| `s_in_agpr` | policy; False, True | attr route: estimate and copy budget only; asm_g13: G0 / G2 as AGPR-form asm clusters (S / dP in AGPRs) | bitwise | not with `(0, 0)`; not with vgpr_form / agpr_fence |
| `asm_tail_nop` | 15; floor to 14 (gfx942 6, gfx950 7) | the `s_nop` ending each asm cluster | bitwise | asm_g13 only; below the floor refused |
| `asm_cluster` | k_step; gemm | one asm cluster per K step or per GEMM | bitwise | asm_g13 only |
| `asm_load_hoist` | False; True | the second G1 / G3 GEMM's LDS operand reads before the first one's asm cluster | bitwise | asm_g13 with register accumulators only |
| `g4_slicing` | step; slice, slice_db, slice_db_g3 | G4 dS (and LDS K^T) read per atom-K step or in kK4 key slices, single or double buffered, slice 0 under G3 (3.5) | bitwise | MFMA register accumulators; `dq_issue = "block"`; slice_db_g3 needs G3 right before G4; the synchronous body (no ring, merged or counted barriers), atomic dQ, `sched = "none"`; K^T from registers under `xt_lds` |
| `waves_per_eu` | policy; 1, 2, 3, 4 (WMMA: 1, 2) | occupancy attribute | bitwise | estimate within `512 / n`; above the LDS bound unattainable (`attn_bwd_occupancy`) |
| `scheduler_strategy` | backend default; max-ilp, max-memory-clause, iterative-ilp, iterative-minreg, iterative-maxocc | machine scheduler | bitwise | compiled crash-isolated by the sweep |
| `sched` | none; iglp0, iglp1, fences_only, stage_table, stage_table_atomics | iglp hint, or fenced placement + fences (+ sched groups; + deferred dQ atomics), 3.5a | bitwise | stage tables: default `stage_order`; not with asm_g13; every staging body (synchronous, the prefetch / DMA rings, the merged ring, the counted waits) and the split-dQ main kernel's bodies (stages A..C); `stage_table_atomics` not with split dQ nor the counted late issue; fenced values not with `xt_lds` |
| `setprio` | False; True | `s_setprio` around stages | bitwise | stage tables only |
| `stage_order` | g1g3g4; g3g1g4, g4g1g3, g1g4g3, g3g4g1, g4g3g1 | issue order of G1, G3, G4 + sink | bitwise | not with stage tables |
| `loop_sync` | full; lds_only, counted | in-loop barrier waits; `counted`: next-step loads before the dQ atomics and `vmcnt(N)` waits (3.5) | bitwise | lds_only needs `global_path = "vgpr"`; counted: CDNA, synchronous register staging, atomic dQ (`stage_table_atomics` only with `counted_issue = "early"`) |
| `counted_issue` | early; late | where the counted next-step loads issue | bitwise | `loop_sync = "counted"` only; late not with `stage_table_atomics` |
| `barrier_merge` | off; on | one barrier per q step on a 2-slot ring (3.5) | bitwise | gfx942 / gfx950; ring 2, vgpr path, relabel, register dK / dV, G4 last, atomic dQ, not counted |
| `ring_issue` | early; late | where the merged ring's next-step loads issue | bitwise | `barrier_merge = "on"` only |
| `qdo_stage` | split; loads_first, joint | Q / dO staging issue order / thread split | bitwise | ring 1 and vgpr path; joint below one pass |
| `stats_load` | at_use; step_top | where lse2 / Dsum loads issue | bitwise | ring 1 and vgpr path |
| `dq_issue` | block; fragment, fragment_regs | G4 then atomics per block or per fragment | bitwise | inert with acc_in_lds, and fragment over one fragment per wave |

Codegen, epilogue, layout:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `edge_tiles` | False; True | edge / interior / edge q loops | bitwise | |
| `interior_clamp` | False; True | row clamps kept on interior steps | bitwise | edge_tiles only |
| `step_order` | qtile_outer; head_outer | step loop nesting | order | |
| `q_rotate` | False; True | rotated start of each q-tile range | order | |
| `head_pack` | plan; False, True | (head, row) packing on M rows | order | plan-owned; `G * s_q <= kM0` |
| `ws_layout` | head_major; token_major | workspace addressing (main, prep, convert) | bitwise | |
| `dkv_epilogue` | lds_staged; lds_staged_pair, direct | direct-mode dK / dV store path | bitwise | register accumulators for the non-default values |
| `dkv_store_vec` | 8; 4, 2, 1 | row-store width of the staged epilogue | bitwise | direct mode, staged epilogue |
| `dkv_atomic_guard` | branch; select | row guard of the atomic epilogue | bitwise | atomic mode |
| `dkv_slab_hoist` | False; True | dK / dV pointers formed before the loops | bitwise | THD direct only |
| `ptr_noalias` | False; workspace, all | `noalias` on workspace / all main pointers | bitwise | all needs atomic mode |
| `lds_swizzle` | per buffer: none (row-major), the predictor's choice (XT image); pad8, pad16, pad32, xor | LDS row pitch, or the 128-byte line XOR (`XorLdsView`) | bitwise | a buffer the plan does not have is ignored; xor needs rows that divide or are whole 128-byte lines |
| `lds_order` | resident_low; phase_low | LDS region placement | bitwise | inert without resident buffers |
| `static_scale` | fold_ds; dq_before_atomic, convert, runtime | which multipliers are emitted | bitwise | plan-owned (follows `scale_placement`) |
| `lb_interleave` | False; True | decode serves `lb_order = 3` | bitwise | plan-owned |
| `xcd_remap` | False; True | decode reads `xcd_n` / `xcd_chunk` | bitwise | plan-owned |
| `worklist` | False; True | decode reads the THD work list under `use_worklist`; prep emits the work-list role | bitwise | plan-owned; THD only |
| `dq_mode` | atomic; split | dQ strategy (2.4a) | dK / dV bitwise; dQ: split is deterministic | split on gfx942 / gfx950; not with the merged barrier, counted waits, `stage_table_atomics` or G4 key slicing |
| `lb_mirror` | False; True | the mirror-pairing decode and the second kv-tile copy (2.7) | bitwise | plan-owned (`lb_order = "mirror"`); not with the work list |
| `bf16_cast` | generic; rne_noguard | bf16 rounding of P / dS (3.2) | bitwise for non-NaN values (relaxed: NaN result unspecified) | bf16 only |
| `bwd_exp2` | exact; fast | `exp2` of P (3.2) | relaxed: P below 2^-126 flushes to 0 | |
| `stats_fma` | False; True | exponent argument as one fma (3.2) | order | |
| `stats_vec` | 1; 2, 4 | lse2 / Dsum vector loads (3.2) | bitwise | head-major workspace, MFMA |
| `dkv_sink` | atomic; slices; last_cta (not built) | atomic-mode dK / dV: fp32 atomics, or per-CTA slices and a fixed-order reduce (2.5a) | order (slices: deterministic) | atomic mode (inert in direct mode); general family only |
| `ws_bounds` | clamp; buffer | buffer-descriptor bounds for the scalar workspace stats loads and the slice stores (3.9) | bitwise | head-major workspace; only where it acts |
| `dkv_cell_vec` | 1; 4 | in-quad transpose, 4 columns per store in the per-slot epilogue (3.9) | bitwise | `dkv_epilogue = "direct"`; MFMA C layouts |
| `narrow_vec` | 1; 2, 4 | user-tensor access width cap of `stage_vec = 1` (3.9) | bitwise | `stage_vec = 1`; the plan narrows it |
| `slab_carry` | False; True | loop-carried Q / dO slab offsets (3.9) | bitwise | the synchronous step; not `q_rotate` |

Runtime and aux values:

| Knob | Values | What changes | Numerics | Couplings / status |
|---|---|---|---|---|
| `g_split` | 1; divisors of G | atomic dK / dV, more CTAs per group | order | `> 1` recompiles main in atomic mode |
| `lb_order` | natural; reverse, mirror, interleave | kv-tile order of the decode (mirror: two kv tiles per CTA) | bitwise | mirror: masked requests whose unpaired grid overfills `num_cus` |
| `xcd_chunk` | 0; gx, 2gx, 4gx, 64 | XCD remap chunk | bitwise | needs `num_xcds` |
| `scale_placement` | fold_ds; dq_before_atomic, convert | where `s` is applied (1.4) | order | |
| `use_worklist` | False; True | prep role writes the heaviest-first THD work list; main takes its CTA's (sequence, kv tile) from it (2.7) | bitwise | THD, `lb_order` natural |
| `prep_rows_per_cta` | 1x; 2x, 4x | prep CTA size | bitwise | prep recompile |
| `convert_rows_per_cta` | 1x; 2x, 4x | convert CTA size | bitwise | convert recompile |
| `convert_vec` | 8; 4, 16 | convert elements per lane | bitwise | convert recompile |

## 10. Numerics and determinism

* **Oracle and tolerance** (`tests/sdpa/bwd_cases.py`). Gradients are compared
  with a float64 reference evaluated on exactly the stored inputs. The bound is
  norm-relative per gradient tensor over valid rows,
  `max|got - ref| <= atol + rtol * g * max|ref|` with
  `g = max(1, sqrt(kv_len_max / 1024))`: fp16 `rtol 4e-3, atol 1e-4`, bf16
  `rtol 2.5e-2, atol 1e-3`, plus a per-row check floored at 5 % of the tensor
  maximum (catches errors confined to one row, such as a dropped tail key).
  A per-element relative bound is meaningless here: each gradient is a sum of
  many products that can cancel to near zero, while the error of a correct
  fp32-accumulate kernel scales with the tensor's magnitude. The bounds were
  calibrated against a low-precision simulation of the backward and leave
  headroom for other summation orders, `exp` implementations and P / dS
  rounding points.
* **Nondeterminism.** With `dq_mode = "atomic"` (the default) dQ is summed
  with fp32 atomics from many CTAs in a run-dependent order: nondeterministic
  in the last bits. With `dq_mode = "split"` dQ is one fixed-order fp32
  register sum per element, written once: deterministic (a device test checks
  two runs bit for bit). In atomic mode dK / dV are nondeterministic too with
  `dkv_sink = "atomic"` (the default); with `dkv_sink = "slices"` they are
  summed from per-CTA slices in slice order: deterministic (2.5a; a device
  test checks two runs bit for bit). In direct mode dK / dV are
  deterministic: each is summed on chip in a fixed step order and written
  once. `dq_mode = "split"` with direct dK / dV or `dkv_sink = "slices"` is
  the atomic-free backward: every output bit for bit reproducible
  (`AttnBwdPlan.deterministic`). A deterministic backward request is
  declined (`DETERMINISTIC`) by the shipped policy; a policy with
  `deterministic = True` serves it on gfx942 / gfx950 with the atomic-free
  backward: the plan sets `dq_mode = "split"` and, where dK / dV are atomic,
  `dkv_sink = "slices"`, over any tuned value (a tuned table keys on
  performance classes, not on the flag, so an admitted request always
  plans).
* **Bitwise knobs.** Every knob marked bitwise gives direct-mode dK / dV bit for
  bit equal to the default (device tests: runtime variants, stage orders, edge
  tiles, narrow twin, token-major workspace, slab hoist, static scale). Order
  knobs stay within the tolerance.
* **Relaxed numerics.** Two default-off VALU levers relax the contract
  (3.2): `bf16_cast = "rne_noguard"` (the result of a NaN input is
  unspecified) and `bwd_exp2 = "fast"` (P below 2^-126 flushes to 0);
  `stats_fma` changes a rounding point within the tolerance. The device tests
  hold every lever to the oracle and the bitwise levers to the default.
* **Contracted rounding.** `_rounding_point` (1.5) is what keeps every step
  variant's dS rounding identical; removing it can make edge and interior steps
  disagree in the last bit.

## 11. Invariants enforced by the tests

**Which gate proves what.**

| Gate | Where it runs | Proves | Does not prove |
|---|---|---|---|
| Byte identity (`platform/tools/check_byte_identity.py`, at llvm20, llvm22 and llvm23) | CPU, with `rocke_engine` built | the C++ and Python engines emit the same LLVM IR for every dual-engine family, so the shared changes (the barrier fence, the gfx942 3-D kernel name) moved no forward family | anything about the backward kernels: they have no C++ builder (0.3) |
| Backward IR goldens (`tests/test_attention_bwd_ir_golden.py`, `tests/test_attention_dense_bwd_ir.py`) | CPU, every flavor in every run | the LLVM IR of the main, split dQ, aux, unified and dense kernels on their targets equals the blessed hashes, for the default of every problem base and every registered knob value switched on (the uncovered values are listed and kept exact); a refactor that should be byte-identical is | that the IR is correct (numerics) |
| Barrier-fence parity (`tests/parity/barrier_fence_emit.py` with `tests/test_barrier_fence_parity_emit.py`; `platform/tests/core/test_barrier_fence.{py,cpp}`) | CPU; the cross-engine part with `rocke_engine` built | each barrier gets exactly its fence emission in both engines; the serialized IR keeps the fence; the C++ engine lowers the serialized backward kernels to the Python bytes | that LLVM keeps LDS accesses on their side of a barrier (next row) |
| Fence in the code object (`tests/test_attention_bwd_barrier_fence.py`) | compile (comgr + `llvm-objdump`) | LDS bytes read and written between consecutive prologue barriers are the same in the ISA as in the IR; the unfenced kernel shows the crossing (negative control) | |
| IR structure (`tests/test_attention_bwd_ir.py` and the family IR tests; list below) | CPU | ABI order, barrier placement, matrix-op and atomic counts, address widths, knob effects | numerics |
| Resource gate (below) | compile, `llvm22` with the production comgr | zero scratch and spills, register and LDS budgets, AGPR copy budget on every enumerated instance | correctness, speed |
| Device numerics (`tests/test_attention_bwd_numeric.py` and the family numeric tests) | gfx942, gfx950, gfx1151, gfx1201 | results within the declared per-dtype tolerance of an independent float64 oracle (`tests/_attention_bwd_harness.py`, `rocke.numeric.sdpa_reference`), bitwise claims of the bitwise levers, dead rows, negative controls | speed |
| Mutation tests (`tests/test_attention_bwd_mutations.py`, `test_attention_bwd_unified_mutations.py`) | device | the numeric tests catch injected kernel defects | |
| Host checks (`tests/test_attention_bwd_host_checks.py`) and the paged block-id guard (`tests/test_attention_bwd_paged_guard.py`) | CPU; the guard's canary runs on a device | every host-check code is reached by a malformed input and valid inputs pass (12.7); an out-of-range block id never addresses outside the caches | that a caller runs the checks (the run API trusts its pointers) |
| Pruning-set gate (`tests/_attention_bwd_perf_gate.py`, passed to the timing harness as `--gate`, no environment default) | device, in every timing process | every timed rocKE configuration passes the numeric gate on the very code objects it times | a performance claim (results are private; 9.3) |
| Installed golden (`platform/tests/rocke_installed_golden_test.py`) | CPU, from an installed package | the representative IR cases of the platform harness are byte-stable at every flavor in an install | the backward kernels (not in the platform harness) |

* **Resource gate** (`tests/_attention_bwd_resources.evaluate_resource_gate`,
  run on every IR-suite instance by
  `test_attention_bwd_ir.test_zero_scratch_no_spills_and_lds_equal_to_the_plan`):
  zero scratch, zero VGPR and SGPR spills, arch VGPRs and AGPRs each within 256
  (read from the code object's metadata and the kernel descriptor's
  `accum_offset`), occupancy at least 1, LDS within the per-arch capacity and
  not above the planner's total, AGPR copies in every matrix loop within
  `agpr_copy_budget`, and an SGPR watch level whose crossing is treated as a
  failure. A result counts only on the `llvm22` flavor with the production
  comgr vintage (`GATE_FLAVOR`, `GATE_COMGR_ROCM`); other toolchains skip.
  The gate tests compile through `compile_or_skip`: a compile error skips
  only where the loaded toolchain is not the gate's (another flavor, another
  comgr, or a comgr without the target); on the gate's toolchain a kernel the
  validator admits must compile, so the error fails the test.
  The production comgr is the torch wheel's bundled library. The library
  `conftest`, the compile cache's toolchain identity and the gate tests pin it
  by path (`ROCKE_COMGR_LIB`) without importing torch
  (`kernels/common/attention_bwd_comgr.py`; the platform names the ROCm
  vintage of a library in a torch package's `lib` directory from that
  package's `version.py` text, so the flavor follows it here and in the fresh
  compile processes, which inherit the pin); an explicit `ROCKE_COMGR_LIB`
  wins and `ROCKE_BWD_NO_COMGR_PIN=1` opts out. No compile-only path (IR
  checks, resource gate, pre-build, sweep candidate compile) imports torch.
  The same step gives comgr's own code-object cache a directory of the
  process tree: a new `<temp>/rocke-comgr-cache-<user>-*` made by the first
  process that pins it, inherited by its children (pool workers, shards) and
  removed when that process exits (an explicit `AMD_COMGR_CACHE_DIR` is kept;
  `AMD_COMGR_CACHE=0` turns the cache off). Its default under `~/.cache` on a
  network home is shared by every job: looking up, writing and pruning it
  cost more than the compile, and it can return an object another job's
  process compiled (for an `iglp_opt(1)` kernel that object depends on that
  process's compile history; rocKE compiles such a kernel in a fresh process,
  section 3.5a).
  The resolved specs that fail the gate (or crash the code-object compiler)
  on the production compiler, found by compiling every gate row of the
  suites (the IR suite's instances and knob instances, the schedule values
  over the gate variants, the knob value sets on the default tiles and on the
  device test's cases, the counted-wait and mirror-pairing rows and their
  twins, the main kernels of the mirror, counted-wait and composition device
  tests, every main kernel the device suites launch on the four targets, a
  grid of the default-off lever compositions the hardware rounds reported,
  and the cells of the custom configurations those rounds ran), are refused
  by the validator (among them every counted-wait kernel
  with mirror pairing and edge tiles on the CDNA debug and default tiles, and
  many paired kernels whose second tile's scalars spill SGPRs):
  `_attention_bwd_refusals.RESOURCE_REFUSED`, keyed by a digest of the
  resolved spec's non-default fields (exact: it refuses no other kernel), and
  the XT / XOR layout combinations of `LAYOUT_SPILLING_COMBOS`. Every
  enumerated row therefore passes the gate; `test_attention_bwd_refusals.py`
  pins each refusal to its row. The records describe the general main
  kernel's code object: the dense and unified shells validate their body with
  `main_kernel_refusals=False` and gate their own kernels. The device tests
  plan each variant before they run it and skip a refused cell with the
  validator's reason; a CPU test checks that every variant keeps a planned
  cell on each CDNA target (or is named, with its rule, as refused
  everywhere). A refusal of a shipped configuration's own cell fails the
  test; a refused cell of the `custom` configuration (a caller's knob stack)
  skips with the reason. The CPU tests of emitted structure (fences, LDS
  spans, epilogue stores, ring steps, clamps) build through
  `_emit_main(..., main_kernel_refusals=False)` and check that a refused
  value raises from the shipped validator.
* **IR identity of defaults.** Spelled-out defaults emit the default IR; every
  non-default value of a built knob changes the IR beyond the kernel name;
  not-yet-effective and declaration-only values emit the default IR apart
  from the name; asm knobs that emit the default code resolve to the default
  spelling (`_canonical_asm_knobs`). Kernel names are salted by every
  non-default compile-time field (`AttnBwdSpec.kernel_name`).
* **Structure** (`test_attention_bwd_ir.py`): kernarg order equals the ABI
  lists; LDS barriers separate cross-wave accesses; MMA and atomic counts per
  step equal the stage-table model; dQ atomics are agent-scope native adds;
  workspace rebases are 64-bit; global access widths follow `stage_vec`; no
  128-bit transpose read; the q-loop trip count is `readfirstlane`'d, the
  empty-band exit precedes live accumulators, and no matrix instruction sits in
  a divergent region.
* **Proofs on the host:** lane-map facts and relabel plans per atom pair,
  transpose-read coordinates, the inverse band and tile classifier against a
  brute force, LDS plan non-overlap, register model against worked tables,
  validator / planner agreement, predicate reachability of every reason code.
* **Device:** the case table against the oracle on every target, edge /
  interior bitwise equality, dead rows (`-inf` / NaN LSE, poisoned O), runtime
  variants, zero-length and ragged storage forms, understated THD bounds
  (memory-safe, exact zeros past the bound), the asm hazard probe with its
  negative control, the transpose-read lane-map probe, the forward-to-backward
  chain (with a frozen fixture), and negative controls that must fail (log2
  LSE, shifted LSE, flipped bottom-right, missing dead-row term, missing group
  sum, skipped prep, double scale, skipped converts, ignored SEQ_LEN, wrong
  offset width). Compositions of levers built apart (`COMPOSE_TWIN_SETS`:
  split dQ, the rings, the counted waits, the small-tile grids, the XT images,
  the VALU levers, the G4 key slicing, the asm G0 / G2 clusters, the schedule
  values, also on the split-dQ main kernel and the dQ kernel) run against
  their twin without the lever (dK / dV bit for bit; dQ too under split dQ),
  and the mirror pairing, the work list and the dense and unified shells
  each run with the merged barrier, the rings, split dQ, the XT images, the
  deferred dQ stage table and the key slicing where legal.
* **Registry:** every spec field has exactly one axis; marked values are
  non-default choices; plan-derived fields mirror the plan.

### 11.1 Running the device tests without repeated compiles

By default every test process compiles every backward kernel it launches
(in-memory cache only). Three opt-in pieces remove the repeats; none of them
changes a kernel, a plan or a test verdict:

* `ROCKE_BWD_HSACO_CACHE=<dir>`: `attn_bwd_kernels` reads and writes code
  objects in `<dir>` (`kernels/common/attention_bwd_cache.py`). The key is the
  stage, the resolved spec, the arch, the LLVM flavor, the comgr library
  (vintage and content digest), `ROCKE_DEBUG_LOC` and a hash of every `rocke`
  and `library/kernels` source, so a source or toolchain change misses. Entries
  are written atomically and re-checked on read (a corrupt one is rebuilt); a
  stage built with a swapped builder (test-only body switches,
  `main_built_with`) is cached under the variant the builder declares, and
  bypasses the cache when it declares none. While an emitter is patched (a
  negative control's monkeypatch of an emission function or method: code from
  outside the hashed sources, a mock or a partial,
  `attention_bwd_cache.patched_emission`) the disk is neither read nor
  written, so the control builds its mutant and never stores it under the
  shipped kernel's key (`bypassed` counts those builds). The sweep and the
  timing harness use the same entries; timing attempt processes only load
  them.
* `ROCKE_BWD_PREBUILD=1` (with the cache variable): after collection the
  `conftest` dry-runs the selected tests that launch through the harness's
  `run_case` (plans only, no device work, in spawned processes) and compiles
  every distinct kernel once in a process pool
  (`tests/_attention_bwd_prebuild.py`). It prints a `bwd-prebuild:` count line;
  a repeated selection with every entry cached skips the dry run (manifest).
  With the cache variable set, each session ends with a `bwd-hsaco-cache:`
  line (`misses` = kernels compiled in that session).
* `tests/run_device_shards.py`: one pre-build, then one pytest shard per GPU
  (`HIP_VISIBLE_DEVICES`) on the shared cache, side by side in one allocation.
  `ROCKE_TEST_SHARD=i[,j...]/n` (the default sharding) splits any test file by
  module and parameter id in the `conftest`; `--shard-env ROCKE_BWD_SHARD`
  uses the numeric file's case sharding instead. Each shard is one long-lived
  process and the pre-build's pools reuse their workers, so no interpreter is
  started per kernel. Under `PYTHONDONTWRITEBYTECODE` the children share one
  temporary bytecode directory (`PYTHONPYCACHEPREFIX`, outside the tree)
  rather than recompiling every source in every process (`--no-pycache`
  keeps the environment).

```
cd library
python tests/run_device_shards.py --devices 8 -- \
    tests/test_attention_bwd_numeric.py tests/test_attention_bwd_mutations.py \
    tests/test_attention_bwd_run.py tests/test_attention_bwd_fwd_chain.py \
    -p no:cacheprovider -q
```

## 12. Host contract

### 12.1 Statistics (LSE)

* Input: **natural-log** log-sum-exp, **fp32**, one value per query row and
  query head. Accepted forms (`lse_dims` / `lse_strides`; `()` dims are the
  layout's canonical dims, `()` strides their packed strides):
  * dense (batched, padded): `[B, H_q, S_q, 1]` with rank-4 `(b, h, t, 1)`
    strides; packed `(H_q * S_q, S_q, 1, 1)`;
  * THD: `[T_q, H_q, 1]` (any `T_q`) with rank-3 `(t, h, 1)` strides; packed
    `t = H_q`, `h = 1`;
  * THD: the batched shape `[B, H_q, S_q, 1]` with explicit rank-4 strides
    whose B stride is unused (tokens come from the ragged offsets); no packed
    default.
  Every stride is a non-negative integer; the stride of a trailing extent-1
  dim never moves an address. Any other form is declined `LSE_FORMAT` (the
  detail names why); no form is replaced by a default.
* Dead rows: `!(lse > -inf && lse < +inf)` or outside the sequence; on a dead
  row `dQ = 0` and the row contributes nothing to dK / dV, whatever `O` holds.
* Only this contract couples the backward to a forward
  (`tests/test_attention_bwd_fwd_chain.py` checks it against the forward's LSE
  epilogues and a frozen fixture, `tests/sdpa/bwd_fixture_v1.py`).

### 12.2 Coverage

| Axis | Values |
|---|---|
| dtype | fp16, bf16 for every tensor; fp32 accumulation |
| head size | 32, 64, 128 (`d_qk == d_v`); 256 declined `HEAD_DIM_256_BWD` |
| layout | any non-negative B / H / S element strides with D stride 1 (BSHD, BHSD, packed QKV views), per tensor |
| lengths | fixed; padded (`SEQ_LEN_Q/KV` with `padding_mask`); THD (`[B + 1]` int32 / int64 offsets, multiplier) |
| heads | MHA, GQA, MQA; `h_k != h_v` (atomic dK / dV) |
| masks | none, causal top-left / bottom-right, sliding window (left / right, either alignment), deprecated causal booleans |
| `s_q = 1` | general kernel with head packing |
| scale | host f32 (default `1 / sqrt(d)`); pass-by-value scale tensors resolved on the host |

Strides that are not multiples of 8 elements, or a base alignment in
`[2, 16)` bytes, select the narrow-access instance (`stage_vec = 1`).

### 12.3 Kernarg ABI `rocke.attn_bwd.v3`

Pointers are global; inputs `readonly`; by default no pointer is `noalias`
(`ptr_noalias` marks the workspace pointers, or every main pointer in atomic
mode; the ABI is unchanged); unused pointers are 0 and never dereferenced.
Order is exact (`attn_bwd_params`): pointers, then i64, f32 and i32 scalars (the
i64 block is 8-byte aligned without padding):

```
main   ptr : Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dk_b dk_h  dv_b dv_h
       f32 : scale_log2 ds_mult dk_mult dq_mult
       i32 : q_t k_t v_t do_t dk_t dv_t
             h_q h_k h_v G gk gv  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_rows_kv ws_seg
             g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist
prep   ptr : O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST
       i64 : o_b o_h  do_b do_h  l_b l_h
       i32 : o_t do_t l_t  h_q h_k h_v  S_q_max S_kv_max
             has_len len_stride  off64 q_mult q_div kv_mult kv_div  ws_rows_q ws_rows_kv ws_seg
             zero_kv zero_dq use_worklist n_batch wl_kn0
convert ptr : SRC DST SEQ OFF
       i64 : d_b d_h
       f32 : mult
       i32 : H  d_t  S_max  has_len len_stride  off64 tok_mult tok_div  ws_rows ws_seg
reduce ptr : SRC DST SEQ OFF                                            (dkv_sink = "slices")
       i64 : d_b d_h  s_stride
       f32 : mult
       i32 : H  d_t  S_max  has_len len_stride  off64 tok_mult tok_div  ws_rows ws_seg  n_slices
dq     ptr : Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV            (split dQ kernel)
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dq_b dq_h
       f32 : scale_log2 ds_mult dq_mult
       i32 : q_t k_t v_t do_t dq_t
             h_q h_k h_v G  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_seg  xcd_n xcd_chunk
```

Batch and head strides are i64 (a tensor may span `2^31` elements or more
between batches or heads); token strides are i32 (the predicate bounds
`S_max * stride_t`). Strides are in elements; THD tensors ignore the `*_b`
strides. `G = h_q / h_k` in direct mode, `gcd(h_q / h_k, h_q / h_v)` in atomic
mode; the dQ kernel's `G` is always `gcd(h_q / h_k, h_q / h_v)`. Every v2
scalar keeps its name and meaning in v3. Every request the predicate admits
packs without narrowing (`pack_attn_bwd_args` raises, naming the kernarg,
otherwise). Launch sequence: `prep -> main -> convert(dQ)`, plus `convert(dK)`
and `convert(dV)` in atomic mode; under `dq_mode = "split"` `prep (zero_dq =
0) -> main -> dq`, plus the dK / dV converts in atomic mode. With `dkv_sink =
"slices"` the two dK / dV converts are reduces (`s_stride = ws_rows * H * D`,
`n_slices = n_k` / `n_v`) and prep runs with `zero_kv = 0`.

### 12.4 Workspace

`A(x) = round_up(x, 256)`; every array is fp32; sub-buffers are packed in this
order at 256-byte aligned offsets (`attn_bwd_workspace_layout`):

```
rows_q, rows_kv  : fixed / padded: B * S (the tensor dims)
                   THD with both max_total_q and max_total_kv: those bounds (rows by storage token, ws_seg = 0)
                   THD otherwise: B * S_q_max, B * S_kv_max (rows by sequence slot b * S_max + i, ws_seg = 1)
WS_DQ    = A(4 * rows_q * h_q * D)                 dq_mode == "atomic"
WS_LSE2  = A(4 * rows_q * h_q)
WS_DSUM  = A(4 * rows_q * h_q)
WS_DK    = A(4 * rows_kv * h_k * D * n_k)          h_k != h_v or g_split > 1
WS_DV    = A(4 * rows_kv * h_v * D * n_v)          same condition
WORKLIST = A(8 * (1 + B * ceil(S_kv / kN0)))      THD work list only (i32 words: count, 0, then (sequence, kv tile) per item)
workspace_bytes = sum of the above
```

`n_k = n_v = 1` with `dkv_sink = "atomic"`; with `"slices"` `n_k = (h_q /
h_k) / (G / g_split)` and `n_v = (h_q / h_v) / (G / g_split)` (2.5a), so each
of `WS_DK` / `WS_DV` is `4 * rows_kv * D * h_q / (G / g_split)` bytes (one
`rows_kv x D` slab per CTA-row of the grid's y extent). Example: `h_q = 32`,
`h_k = h_v = 8` (`G = 4`), `g_split = 4`: `n_k = n_v = 4`, four times the
atomic arrays.

The work-list term is sized for every CTA of the main grid's batch planes,
not from `rows_kv`: each sequence length is clamped to `S_kv`, so `B *
ceil(S_kv / kN0)` bounds the live items for any offset table, while the sum
of the offset spans of a caller's table may exceed its `max_total_kv` (an
earlier design sized the list as `A(8 * (B + ceil(rows_kv / kN0)))`, which
holds only when the spans do not exceed the row bound). The plan refuses a
list whose words do not fit i32 indexing.

Example: B = 2, S_q = S_kv = 128, h_q = h_k = h_v = 8, D = 128, fixed lengths,
`g_split = 1`: `1,048,576 + 8,192 + 8,192 = 1,064,960` bytes; with
`g_split = 2` add `2,097,152`.

The size is an upper bound for every admitted request without a device query.
Without both caller bounds (hipDNN has no max-total API) the rows of THD
sequence `b` are its slot `[b * S_max, b * S_max + len_b)`, which no storage
layout can push past `B * S_max`. With both bounds the rows are the storage
tokens and the caller asserts that each bound covers the end token of every
segment; a single bound is not used. Caller contract: a 256-byte aligned base;
an understated bound gives wrong gradients for the excess tokens but no
out-of-bounds access (every workspace row is clamped); a supplied
`max_total_q` / `max_total_kv` must be at least 1 for a non-empty problem.

### 12.5 Predicate and reason codes

`attn_bwd_support(req, arch, *, policy=None) -> Verdict(ok, code, detail)` is a
pure function of the request and the in-repo policy tables; the first failing
check wins. Every code has a kind (`DECLINE_KINDS`, `Verdict.kind`, published
per code in the declaration): `out_of_scope` (a permanent decision of this
entry), `not_yet_implemented` (a feature the family can add; the detail starts
with `NOT_YET_IMPLEMENTED:`), `request_limit` (a malformed request, or one past
the kernels' index, grid or format limits) or `policy` (the shipping policy
declines). The request is type-checked at construction: a flag that is not a
`bool`, or an extent, bound or count that is not an integer, raises
`TypeError`. An accepted request plans: after the request checks the predicate
resolves the plan of the request's route under the policy, its tuned knobs
included, and declines `INSTANCE_UNAVAILABLE` when no plan builds (a spec the
family validator refuses, among them the recorded resource-gate failures, or
a tuned value the request cannot take).

| # | Code | Decline when |
|---|---|---|
| 1 | `ARCH_UNSUPPORTED` | arch not in {gfx942, gfx950, gfx1151, gfx1201} |
| 2 | `ARCH_NOT_VALIDATED` | the arch is in the policy's `not_validated_archs` (empty in the shipped policy) |
| 3 | `DROPOUT`, `BIAS_IN_BACKWARD`, `DBIAS`, `ALIBI`, `PAGED`, `FP8`, `DETERMINISTIC`, `UNSUPPORTED_FEATURE` | the feature is requested (in this order; `DETERMINISTIC` not under a policy with `deterministic = True` on gfx942 / gfx950, section 10) |
| 4 | `SCALE_DEVICE_TENSOR` | scale is a device-buffer tensor |
| 5 | `DTYPE` | Q, K, V, O, dO not all fp16 or all bf16 |
| 6 | `OUT_DTYPE` | dQ, dK, dV differ from the input dtype |
| 7 | `LSE_FORMAT` | stats not fp32 or not in a form of 12.1 |
| 8 | `DQK_NE_DV` | `d_qk != d_v` |
| 9 | `HEAD_DIM_256_BWD` | d == 256 (not implemented yet in the backward; the forward serves d = 256) |
| 10 | `HEAD_DIM` | d not in {32, 64, 128} |
| 11 | `GQA_RATIO` | `h_q % h_k != 0` or `h_q % h_v != 0` |
| 12 | `BAND_BOUND` | a bound `< -1` or `>= 2^30`; both causal booleans; a causal boolean with explicit bounds |
| 13 | `STRIDE_FORMAT` | a stride tuple without exactly four integer entries `(B, H, S, D)` |
| 14 | `STRIDE_D` | a D stride != 1 |
| 15 | `STRIDE_NEGATIVE` | a negative stride |
| 16 | `OUTPUT_OVERLAP` | two distinct indices of dQ, dK or dV share an element (a sorted-extent test proves most layouts disjoint, the rest are decided by enumerating index differences; beyond the enumeration budget the layout is declined conservatively) |
| 17 | `STRIDE_ALIGN` | base alignment below 2 bytes |
| 18 | `INDEX_RANGE` | `S_max * stride_s + D >= 2^31` for a tensor; `S_q * stride_t + 1 >= 2^31` for the LSE; an i64 stride or byte offset `>= 2^63`; `rows_q >= 2^31`; `rows_kv >= 2^31` (atomic mode); `S_max * D >= 2^31`; `S_max >= 2^30` |
| 19 | `GRID_LIMIT` | `B > 65535`; main units `> 65535`; prep y `> 65535`; prep / convert x `>= 2^31` |
| 20 | `SEQ_LEN_FORMAT` | SEQ_LEN tensors without `padding_mask`; padding without both tensors; not int32 `(B, 1, 1, 1)` |
| 21 | `RAGGED_FORMAT` | THD offsets not int32 / int64 `[B + 1]`; q-side or kv-side tensors not on one table; multiplier or token stride outside `[1, 2^31)`; a supplied `max_total_*` below 1 for a non-empty problem |
| 22 | `EMPTY` | B, a head count, a sequence dim or D is 0 |
| 23 | `INSTANCE_UNAVAILABLE` | the serving instance failed the resource gate, or the plan of the request's route does not build (see above) |
| 24 | `PERF_BELOW_BASELINE` | opt-in performance gate only (`AttnBwdPolicy.perf_gate`, off in the shipped policy): the request's performance class on its route is `below_baseline` |
| 25 | `PERF_UNVERIFIED` | opt-in performance gate only: the class is `unverified` and the policy does not waive unverified classes |
| 26 | `NOT_YET_DECLARED` | the request needs a functional feature group whose correctness gate has not passed |
| - | `OK` | otherwise |

Functional feature groups, in gate order (all declared in the shipped
policy): `base`, `masks_and_stats` (any mask, explicit scale, d32, `s_q = 1`),
`grouped_heads`, `split_kv_heads`, `padding`, `ragged`, `narrow_access`.
`attn_bwd_feature_groups(req)` lists the groups a request needs. The plan also
refuses mirror pairing outside its scope (an unmasked request, or an unpaired
grid that would not overfill `num_cus`), the work list outside its scope (a batched request, a non-natural `lb_order`, a list whose
words do not fit i32 indexing) and a `g_split` that does not divide the head
group; these are plan errors (`ValueError`), not predicate codes, because a
runtime value is a tuning choice and the request is served without it.

### 12.6 Declaration

`attn_bwd_declaration(arch, *, policy=None)` returns a JSON-serialisable dict:
`family`, `version` (4), `abi` (`rocke.attn_bwd.v3`), `arch`, `status`
(`declared`, `not_validated`, `unsupported`), `rdna_status`, accepted value sets
per axis (`head_dim_declined` carries the code, its kind and the reason), the
ordered `reason_codes` with their `reason_kinds`, the
`not_yet_implemented_prefix`, `feature_groups` and `declared_feature_groups`, the `lse`
contract, the `workspace` formula, the `determinism` statement, the
`perf_policy`, `perf` (one entry per (route, class) with its status), the
`deferral_register` and `notes` (gfx1201: declared on its hardware run; its
forward-chain tests skip because the forward probe shell has no gfx12 body).
`perf_policy` states how the status is used: `gate` is `information_only`,
with `unverified` and `below_baseline` both `served`, in the shipped policy,
and `declines` under the opt-in performance gate. The kernels-level
declaration reads the policy it is given; the shipped default and the routed
policy carry no status table (9.3), so every CDNA class reads `unverified`.
`dispatch/attention_bwd/routing.attention_bwd_declaration` publishes it under
the routed policy (tuned tables, dense pack).

Deferral register (`DEFERRAL_REGISTER`, published in the declaration): every
feature decline with its kind and the consequence for hipDNN, which falls back
to its other engines on any decline.

| Feature | Code | Kind | Consequence for hipDNN |
|---|---|---|---|
| head dim 256 (backward) | `HEAD_DIM_256_BWD` | not yet implemented | d = 256 models are servable for inference but not trainable through this provider; their training attention stays on CK / AOTriton |
| dropout | `DROPOUT` | not yet implemented | training with attention dropout falls back to CK / AOTriton |
| bias / attention-mask tensor in the backward graph | `BIAS_IN_BACKWARD` | out of scope (the backward contract has no bias input) | `F.sdpa(attn_mask=...)` under autograd falls back to CK / AOTriton on every target |
| dBias output | `DBIAS` | out of scope (the contract returns dQ, dK, dV only) | graphs that need dBias train on CK / AOTriton |
| ALiBi | `ALIBI` | not yet implemented | ALiBi training attention falls back to CK / AOTriton |
| paged KV cache (general entry) | `PAGED` | out of scope for this entry (the unified entry serves it, 8.3) | a paged backward through hipDNN falls back to CK / AOTriton |
| fp8 tensors | `FP8` | not yet implemented | fp8 training attention falls back to CK / AOTriton |
| deterministic backward | `DETERMINISTIC` | not yet implemented (an opt-in policy on gfx942 / gfx950, section 10; not built on RDNA) | under the shipped policy a deterministic backward request falls back to CK / AOTriton |
| block masks, sink tokens, extra statistics outputs | `UNSUPPORTED_FEATURE` | not yet implemented | such graphs train on CK / AOTriton |

### 12.7 Host checks and device data

The run API (`attention_bwd_run.run_attn_bwd`) is pointer-only and trusts its
pointers. A caller that holds the tensors checks them first
(`kernels/common/attention_bwd_run.py`; every check raises
`ValueError("<CODE>: detail")` with a code of `HOST_CHECK_CODES`):

* `check_attn_bwd_tensors(req, tensors)`: every role (`q k v o do dq dk dv
  lse`, plus the `SEQ_LEN` tensors with padding and the offsets with THD)
  against the request: shape, strides on every dim of extent above 1, dtype,
  and the largest addressed byte within the tensor (`TensorMeta.of(t)` reads
  a torch-like tensor without importing torch);
* `check_attn_bwd_lengths(req, ...)`: `SEQ_LEN` values in `[0, S_max]` (a
  larger value is rejected, `SEQ_LEN_RANGE`, not clamped) and THD offsets
  that are non-negative, non-decreasing, whole tokens, spans of at most
  `S_max`, inside the `max_total_*` bound and the tensor (`RAGGED_RANGE`);
* `check_attn_dense_bwd_tensors(spec, tensors, batch=)`: the dense family's
  compile-time shape against its tensors.

Device data that reaches a kernel without the checks has a defined behaviour:
a sequence length is clamped into `[0, S_max]`, and every workspace row is
clamped (12.4). The unified family's checks and its block-id guard are in
`attention_bwd_unified.md` section 8.

### 12.8 Declined features

Out of scope (permanent for this entry): dBias (`DBIAS`), a bias or attention
mask in the backward (`BIAS_IN_BACKWARD`), paged KV through this entry
(`PAGED`; paged problems belong to the unified family, a rocKE-native entry
point, 8.3). Not implemented yet: dropout (`DROPOUT`), d = 256
(`HEAD_DIM_256_BWD`; the forward serves d = 256), fp8 (`FP8`), ALiBi
(`ALIBI`), a deterministic backward under the shipped policy
(`DETERMINISTIC`), block masks, sink tokens and extra statistics outputs
(`UNSUPPORTED_FEATURE`).

## 13. Known limitations and performance characteristics

Qualitative only; measurements live outside this repository.

* **The main kernel dominates.** Prep and convert are a small share of the
  backward on all but the smallest problems, and launch gaps are negligible,
  so fusing the auxiliary kernels is not a lever; the per-step efficiency of
  the main kernel is.
* **dQ atomic drain.** With `loop_sync = "full"` the end-of-step barrier waits
  for the step's dQ atomics (`vmcnt` counts no-return atomics). `lds_only`
  removes that wait from the barrier, but the next step's staging loads are
  issued after the atomics and `vmcnt` retires in order, so the wait tends to
  move to the next staging `s_waitcnt` rather than disappear. Hiding it needs
  the next step's loads issued before the atomics (the prefetch ring, the
  merged barrier, the counted waits) or no atomics (split dQ); all are built
  as default-off levers, and which wins per class is a sweep question.
* **Staging latency.** With ring depth 1 every step loads Q / dO
  synchronously, and B1 exposes the global latency once per step unless
  other resident workgroups hide it or a ring / counted lever is selected.
* **Softmax-gradient VALU.** Per cell: the exponent multiply-subtract, the
  guarded `exp2` (a range-reduction sequence), compares and selects, the dS
  arithmetic, and two fp32-to-16-bit casts. On gfx942 the bf16 cast is a
  multi-instruction NaN-preserving round-to-nearest-even sequence (gfx950 has a
  packed convert instruction). The cheaper forms (fast `exp2`, an FMA for
  the exponent argument, a guard-free bf16 cast, vector stats loads) are
  built as default-off levers (3.2); packed fp32 math is not built.
* **AGPR plans.** Four-wave tiles with large key tiles need AGPR accumulators;
  the backend's all-or-nothing AGPR form puts S / dP / dQ there too and adds
  hot-loop copies. The split routes (`vgpr_form`, `agpr_fence`, `asm_g13`) keep
  them out; `asm_g13` costs scheduling freedom (each cluster is a fence).
* **Small tiles and occupancy.** The small-tile grids make every kM0 x kN0
  pair in {16, 32, 64} x {16, 32, 64} legal on two and four waves, with
  `waves_per_eu` up to 4 where the register estimate allows. On a replicated
  G0 / G2 grid the extra waves compute S / dP copies that are discarded. At
  d128 the LDS plan (staged Q / dO, P / dS, K / V when resident in LDS) keeps
  most four-wave tiles below occupancy 3, so the higher hints are reachable
  there only on the smallest tiles.
* **Causal imbalance.** Key tiles near the diagonal carry fewer q steps;
  reverse, interleave and two-tile mirror pairing are built.
* **gfx942 transposes.** Without a transpose read, the default (`lds_plain`)
  reads K-outer operands on gfx942 one element at a time; `xt_lds` (built,
  default off) writes XOR-swizzled transposed images once per step (Q^T,
  dO^T) and once per CTA (K^T) so those reads become vectors, at the cost of
  the writer's LDS traffic and, under the relabel route, one more barrier per
  step. Which wins is a sweep question (lever retained, structure stage).
* **XOR address registers.** A line-XOR address is not an immediate offset
  of one base (the XOR does not commute with the fragment and K-step
  offsets), so every swizzled access the q loop issues keeps its own address
  register. `xt_lds` stays spill-free at every CDNA start point, but the
  row-major `xor` costs enough arch VGPRs to spill on some tiles, and the XT
  images spill the one-wave gfx1151 d64 tile. The validator refuses exactly
  the spilling combinations the IR suite's resource gate found
  (`LAYOUT_SPILLING_COMBOS`: tile, sequence / mask / packing variant and
  layout, per target); other tiles stay legal and the sweep's resource gate
  prunes any that spill. Row `xor` on a ring-slotted buffer (q / do of a
  register ring, ds of the merged barrier) is refused: the XOR view has no
  run-time slot shift. `xt_lds` runs on the synchronous body only (no ring,
  merged or counted barriers) and not with a fenced schedule value (its
  read-ahead would read dO^T before the image's publishing barrier). An
  XOR whose hash depends only on bits the access offsets do not touch (for
  example a K-blocked image hashed on the row inside 16-row groups) would
  keep one base per image; it is not built.
* **Competitor structure** (stated neutrally). The production backward
  kernels this family is compared against on ROCm use one or more of: a
  separate Q-parallel dQ kernel that recomputes S and dP and writes dQ once
  without atomics; small tiles at occupancy above 1 without in-kernel
  pipelining; or hand-scheduled kernels that own a large key tile per
  workgroup at occupancy 1 with DMA streaming, issue the next tile's loads
  before the step's dQ atomics and wait with partial `vmcnt` counts (no full
  drain in the loop), and pair causal key tiles per workgroup. Some resolve
  GQA by expanding dK / dV per query head and reducing afterwards, where this
  family sums the group on chip.
