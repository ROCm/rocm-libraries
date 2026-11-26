# Implicit-GEMM Forward Convolution

## Algorithm

The forward kernel computes a 2-D (or 3-D) convolution:

```
D[n, ho, wo, k] = sum_{y, x, c} A[n, ho*sH - pH + y*dH, wo*sW - pW + x*dW, c] * B[k, y, x, c]
```

This is cast as an implicit-GEMM with the following dimension mapping:

| GEMM dim | Forward meaning              | Size        |
|----------|------------------------------|-------------|
| M        | Output spatial positions     | N×Ho×Wo     |
| N        | Output channels              | K           |
| K (red.) | Filter spatial × input chan  | Y×X×C       |

### Operands

| Role | Tensor | Layout | GEMM shape              |
|------|--------|--------|-------------------------|
| A    | Input activations  | NHWC  | (M, K_gemm) = (N·Ho·Wo, Y·X·C) |
| B    | Weights            | KYXC  | (N_gemm, K_gemm) = (K, Y·X·C)  |
| D    | Output activations | NHWK  | (M, N_gemm) = (N·Ho·Wo, K)     |

### Coordinate-transform descriptor DAG

The A operand requires a non-trivial address computation because the input tensor
is indexed by `(n, hi, wi, c)` while the GEMM loop variable is `(m, k)`.
The transform DAG maps `(m, k) → NHWC offset`:

```
naive(NHWC):  (n, hi, wi, c)
  + unmerge('m' → n, ho, wo)          — decompose flat output position
  + embed((ho, y) → hi, stride=sH, dH, offset=-pH)  — affine spatial map H
  + embed((wo, x) → wi, stride=sW, dW, offset=-pW)  — affine spatial map W
  + unmerge('k' → y, x, c)            — decompose filter+channel index
  + pad('y', lo=0, hi=Y)              — boundary guard (partial K-tile)
  + pad('x', lo=0, hi=X)              — boundary guard (partial K-tile)
```

B and D use simpler descriptors:

- **B (KYXC):** `naive` + `unmerge('k_gemm' → y, x, c)` + `pad(y)` + `pad(x)`.
- **D (NHWK):** `naive` + `unmerge('m' → n, ho, wo)`.

### Pipelines

| Pipeline | Description |
|----------|-------------|
| `mem`    | Single-buffer LDS; no scheduler hints. Correctness-first baseline. |
| `compv3` | Single-buffer LDS + `sched_group_barrier` hints to overlap DS_read with MFMA. |
| `compv4` | Double-buffer LDS (ping-pong) + scheduler hints + `s_setprio`. |
| `unroll_k` | Python-level K-loop unroll with double-buffering; also triggers double-buffer allocation. |
| `async_dma` | Direct DRAM→LDS via `raw_ptr_buffer_load_lds`; paired with `SoftwarePipeline` for overlap. |
| `wavelet`  | Load/math wave specialization (CK Tile PR #8009). **gfx1250/WMMA only.** Appends `num_load_waves` dedicated load waves to the workgroup; those waves handle all DRAM→LDS transfers while the `warp_m × warp_n` math waves run WMMA exclusively. Single-buffer LDS shared by both roles; no scheduler hints. |

### Epilogues

| Path | Condition | Output |
|------|-----------|--------|
| Direct store | `epilogue="default"` | Per-lane scalar write to D via the NHWK descriptor. |
| CShuffleEpilogue | `epilogue="cshuffle"` | LDS-stage accumulators in MFMA layout, re-read coalesced for wide vector global stores. |

### Data types

`ConvDataSpec` sets the storage type of each tensor: `dtype_a` (input), `dtype_b`
(weights), `dtype_d` (output). The accumulator is always fp32.

| Storage A/B | `mma_dtype` | MMA input | Notes |
|-------------|-------------|-----------|-------|
| `fp16` | `None` | fp16 | The default. |
| `bf16` | `None` | bf16 | |
| `fp32` | `None` | fp32 | The fp32 atoms `16x16x4` / `32x32x2` on gfx942 and gfx950. |
| `fp32` | `"tf32"` | TF32 | TF32 compute: see below. MFMA targets only. |

**TF32 compute.** Set `ConvDataSpec(dtype_a="fp32", dtype_b="fp32",
dtype_d="fp32", mma_dtype="tf32")`. TF32 is an fp32-convolution mode: all three
tensors are fp32, and `ConvDataSpec.mma_dtype_reason()` rejects any other `dtype_d`.
`"tf32"` is a compute mode, not a storage type: `dtype_a="tf32"` is rejected.

- A and B stay fp32 in global memory and in LDS. Loaders, vector widths, the LDS
  budget and the kernel ABI are the fp32 ones.
- After each LDS fragment read, the fp32 fragment is reinterpreted as
  `vec<tf32xN>`, once per fragment. This is a bitcast: it emits no instruction
  and does not round.
- The warp tile must be a TF32 atom of the target:

  | Arch | Atoms (`warp_tile_m x n x k`) | Fragment per lane | Lowering |
  |------|-------------------------------|-------------------|----------|
  | gfx942 | `16x16x8`, `32x32x4` | 2 fp32 (`ds_read_b64`) | Native `v_mfma_f32_*_xf32`. |
  | gfx950 | `16x16x32`, `32x32x16` | 8 fp32 (two `ds_read_b128`) | Emulated: three `v_mfma_f32_*_bf16` per atom (BF16x3). |

  WMMA targets, and any arch whose catalog has no TF32 atom, are rejected with a
  reason from `is_valid_spec`.
- The LDS row stride `tile_k + lds_k_pad` must be a multiple of
  `min(fragment, 4)` fp32 elements, so that every fragment read is 8- or 16-byte
  aligned. The default pad of 8 and the unpadded `async_dma` layout both satisfy
  this.
- `a_operand_override` cannot be combined with TF32, because it supplies the A
  operand without the LDS read. The builder rejects the combination.
- Under `compv3` / `compv4` the scheduling hints count the real instruction stream:
  two DS reads per 8-wide fragment and three MFMAs per emulated atom.
- On gfx950, a warp tile with 384 or more fp32 accumulators per lane is
  rejected for TF32. With the BF16x3 split temporaries live, such a tile sits at
  the 512-VGPR ceiling, where the compiler's VGPR-to-AGPR spill path was seen to
  miscompile a two-stage wgrad kernel (the same IR is correct at `-O1`). The
  native gfx942 atom is not affected.
- Kernel names end in `_tf32`. Names of kernels without TF32 are unchanged.

**Numerics.** Each operand is passed as its raw fp32 payload, matching CK on both
arches:

- gfx942 hardware reads the top 19 bits (sign, 8-bit exponent, 10-bit mantissa)
  and ignores the low 13 mantissa bits. Inputs are truncated, not rounded to
  nearest.
- gfx950 splits each fp32 value into a bf16 high part and a bf16 low part
  (`v_cvt_pk_bf16_f32`, round to nearest even). It then accumulates
  `hi*hi + hi*lo + lo*hi` and drops `lo*lo`. The result is much closer to fp32
  than gfx942 TF32.

Results are therefore **not bit-reproducible across arches**. Compare a gfx942
result against a reference computed from inputs masked with `& 0xFFFFE000`.
Compare a gfx950 result against an fp32 (or fp64) reference.

### Additional features

- **Grouped convolution:** `groups > 1` supported on CDNA targets. The descriptor DAG adds a group-unmerge so each group's GEMM stays within its own `(C/groups, K/groups)` slab.
- **Chiplet-aware workgroup swizzle:** `chiplet_swizzle=True` remaps the 2D block grid through `chiplet_aware_super_tile` to improve XCD-local L2 reuse.
- **Accumulator epilogue:** `ConvAccumulatorEpilogue` applies optional bias, scale, ReLU, or clamp directly on MFMA f32 fragments before the store path.

---

## Changelog

### Initial implementation

- Introduced `ImplicitGemmConvSpec` and `build_implicit_gemm_conv`.
- Descriptor DAG covering 2-D NHWC convolution with arbitrary stride, padding, dilation, and groups.
- `mem` pipeline with direct (`default`) and LDS-staged (`cshuffle`) epilogues.
- Arch-aware spec validation via `is_valid_spec` / `is_valid_spec_for_problem`.

### 3-D convolution

- Extended `ConvProblem` with `Di`, `Z`, `sD`, `pD`, `dD` fields.
- `make_a_descriptor` and `make_b_descriptor` emit the 3-D DAG when `p.is_3d` is `True`, adding `embed((do, z) → di)` and `unmerge('k' → z, y, x, c)` steps.

### compv3 / compv4 pipelines

- `compv3`: `sched_group_barrier` hints after each MFMA K-atom step to overlap DS_read + MFMA.
- `compv4`: double-buffer LDS (`A_smem` / `A_smem2`) ping-pong; `s_setprio` to push the K-loop into compute steady state.

### Async DMA

- `async_dma=True` switches the global→LDS path to `raw_ptr_buffer_load_lds` via `AsyncTileLoader`.
- LDS layout must be plain `[block_m, block_k]` (no K-pad); paired with `SoftwarePipeline` for DRAM/MFMA overlap.

### Wavelet pipeline (`wavelet`)

The wavelet pipeline implements the load/math wave specialization introduced in CK Tile PR #8009 (`gemm_pipeline_ag_bg_cr_wavelet.hpp`). It targets **gfx1250 (WMMA/RDNA wave32)** exclusively and is rejected by `is_valid_spec` on MFMA/CDNA targets.

#### Wave role split

The workgroup is enlarged by `num_load_waves` extra waves appended after the standard `warp_m × warp_n` math waves:

| Role | Thread range | Responsibility |
|------|-------------|----------------|
| Math waves | `[0, block_size)` = `warp_m × warp_n × wave_size` threads | WMMA only — never touch global memory |
| Load waves | `[block_size, launch_block_size)` = `num_load_waves × wave_size` threads | DRAM→LDS transfers only |

```
launch_block_size = block_size + num_load_waves * wave_size
                  = (warp_m × warp_n × wave_size) + (num_load_waves × wave_size)
```

With the gfx1250 defaults (`warp_m=2`, `warp_n=2`, `wave_size=32`, `num_load_waves=4`):
- `block_size` = 128 (math waves)
- `launch_block_size` = 256 (math + load waves)

For all other pipelines `launch_block_size == block_size`. The HIP launch and the kernel's `max_workgroup_size` attribute must use `launch_block_size`, not `block_size`.

#### Overlap mechanism: split fetch / store

Because gfx1250 has separate VMEM and WMMA issue queues, DRAM loads (`buffer_load_vN`) and WMMA matrix instructions can be in flight simultaneously — unlike MFMA targets where both share a single issue slot. The wavelet pipeline exploits this by splitting each tile load into two phases:

1. **`wavelet_fetch`** — issues `buffer_load_vN` into registers *before* the synchronisation barrier. The DRAM request is in flight while math waves execute WMMA.
2. **`wavelet_store`** — after `s_waitcnt(vmcnt=0)` confirms data arrival, writes the fetched registers to LDS.

This is the `fetch` / `store_fetched` split on `CoalescedTileLoader` in `helpers/loads.py`.

#### Barrier protocol (`scf_if_else` form)

The kernel emits a single `scf_if_else` block (`tid < block_size`) that drives both roles through matched barrier sequences. The number of barriers per branch must be identical:

```
MATH branch:   barrier_0
               repeat K-2 times: WMMA-atom → barrier_A → barrier_B
               tail WMMA-atom
               epilogue stub barriers

LOAD branch:   fetch[0] → store[0] → barrier_0
               repeat K-2 times: fetch[i+1] → barrier_A → store[i+1] → barrier_B
               N_epi stub barriers (no-op for load role)
```

This uses LLVM `br i1` branching rather than `s_and_saveexec_b64` exec-mask manipulation, so both branches are lowered to ordinary conditional jumps and the separate issue queues remain active for both roles.

#### LDS allocation

The wavelet pipeline uses a **single-buffer** LDS tile (no ping-pong):

- The `_double` flag that enables ping-pong for `compv4` / `async_dma` / `unroll_k` does not include `wavelet`.
- Because both branches of the `scf_if_else` reference A/B LDS simultaneously throughout the kernel body, the liveness packer cannot alias the cshuffle C tile onto the A/B region. The `_no_alias` flag is forced `True` for wavelet regardless of `cshuffle_no_alias`, making the LDS budget always additive: `total = ab + c` (not `max(ab, c)`).
- The cshuffle epilogue requires **two WAR barriers** (`_war_barriers = 2`) instead of one, because two synchronisation points are needed: math waves done (load waves may exit) and load waves exited (A/B LDS safe to overwrite).

#### Scheduler hints

The wavelet schedule (`helpers/schedule.py`) sets `emit_hints=False` and `mode="default"` — no `sched_group_barrier` hints are emitted. Hardware concurrency between VMEM and WMMA is provided by the separate issue queues, not by software scheduling.

#### Known issue: `ds_load_tr16_b128` mis-optimisation (gfx1250)

The AMDGPU LLVM backend may replace a `load <8 x half>, ptr addrspace(3)` that feeds a WMMA instruction with `ds_load_tr16_b128` (a transposed LDS read). This optimisation assumes LDS was stored in column-major layout, but the wavelet kernel stores A_smem / B_smem row-major (for coalesced DRAM→LDS writes). The result is garbage WMMA inputs even though the load-wave transfers completed correctly. Root cause and three candidate fixes are tracked in the debug notes;

### TF32 compute (`ConvDataSpec.mma_dtype="tf32"`)

- The atom, catalog check and op lookup key on the MMA dtype (`data.mma_a` /
  `data.mma_b`). The LDS element type keys on the storage dtype.
- The fp32 fragment is bitcast to `vec<tf32xN>` after the LDS read. An 8-wide
  fragment is read as two 4-wide loads, so no load claims more than 16-byte
  alignment.
- `is_valid_spec` rejects WMMA targets, arches without TF32 atoms, non-TF32 warp
  tiles and misaligning LDS row strides.
- `validate()` applies the same TF32 gate (`_fwd_tf32_reason`) without an
  arch: it rejects what no arch can build, and `is_valid_spec` adds the arch's
  own refusals.
- The compv3/compv4 hint counts are scaled for TF32 only.
- Specs without TF32 emit byte-identical IR.
- **C++ engine.** The C++ conv instance builds TF32 specs too
  (`mma_dtype="tf32"` in its spec), byte-identical to Python: the
  `conv_implicit_gemm` parity family carries TF32 configs, and
  `kernels.conv.engine_bridge.lower_conv_implicit_gemm` hands TF32 specs to
  the engine like any other.
- See [Data types](#data-types).

### Python-level K-loop unroll (`unroll_k`)

- `unroll_k=True` Python-unrolls the K loop at build time with double-buffering, removing the `scf.for_iter` overhead entirely.

### WMMA support (gfx1251/gfx1250)

- Added RDNA wave32 path using the `wmma` MMA family via `_resolve_conv_op`.
- Supports 16×16×4, 16×16×16, and 16×16×32 WMMA atoms; `mem` pipeline and `default`/`cshuffle` epilogues.
- Dedicated `_emit_direct_epilogue_wmma` and `_emit_wgrad_direct_epilogue_wmma` using `op.dst_layout()`.

### Chiplet-aware workgroup swizzle

- `chiplet_swizzle=True` remaps `(block_id_y, block_id_x)` through `chiplet_aware_super_tile` at build time so consecutive workgroups share an XCD (and an L2 slice).

### Grouped convolution

- `ImplicitGemmConvSpec.groups > 1` supported on CDNA. The descriptor adds a group-unmerge to keep each group's GEMM within its `(C/groups, K/groups)` slab.

### Per-group vector-size derivation for grouped convolutions

`default_vector_sizes` and the `is_valid_spec` gate were both called with the
**total** `C` and `K` values instead of the per-group `cpg = C/groups` and
`kpg = K/groups`. For grouped convolutions the output tensor `D` (NHWK) is
indexed within a group's slab — the relevant stride-1 dimension has extent `kpg`,
not `K`. Using `K` produced an over-wide `vec_c` that did not divide `kpg` and
wrote out-of-bounds when `K % vec_c == 0` but `kpg % vec_c != 0`.

### Accumulator epilogue

- `ConvAccumulatorEpilogue` applies optional `bias`, `scale`, `relu`, `clamp_min`/`clamp_max` directly on MFMA f32 fragments before the D-store path. Its `bias` is one compile-time scalar for every channel.

### Per-output-channel bias (`bias`)

`ImplicitGemmConvSpec.bias` (name and label tag `cbias`, `KernelIdentity.bias`) builds the conv + bias
forward: `y[..., g*kpg + k] = round(acc_fp32 + float(Bias[g*kpg + k]))`, one rounding, the bias added to the
fp32 accumulator fragments before the `acc_epilogue` and the store. `Bias` is a vector of K (= groups * kpg)
values in the output dtype, passed as the trailing `Bias` / `Bias_bytes` kernargs after every other one
(`conv_abi.conv_bias_arg_names`); the kernel reads it through a buffer resource of `Bias_bytes`, so a channel
past it reads 0. `ConvArgs.to_launch_values(bias_ptr=, bias_bytes=)` refuses a `bias_bytes` below K 2-byte
values. The helpers are `kernels/conv/conv_bias.py` and their C++ twin `instance_conv_bias.h`; the direct,
Winograd and split-C forward families use the same ones.

The bias is added where the whole fp32 sum exists, exactly once (`fwd_bias_stage`): the unsplit epilogue, the
owner's fold of `split_det` 1 / 2 (and `split_tail`'s whole-K tiles), the cluster fold of `split_k_cluster`,
and the fold kernels of the two-stage workspace split (`conv_fwd_split_k_reduce` with `bias`) and of
`split_det` 3 (`conv_wgrad_workspace_reduce` with `bias`), whose Stage 1 keeps storing raw partials.
`fwd_bias_conflict` (C++ `rocke_conv_fwd_bias_ok`) refuses the atomic split-K forward and the stream-K /
hybrid segment loops, which have no single owner of an output value.

AOT: the bias twins are opt-in, `ROCKE_AOT_FWD_BIAS=0,1` (default `0`, the plain binaries only), and the
kernel cache never offers a twin for a plain problem. The dispatcher and the route tables carry no bias
tensor and select no bias kernel; the hipDNN engine `hipkernel:Gfx950ConvFwdBiasAot` ships the gfx950 twins.

### Pointwise explicit-GEMM fast path

For **pointwise convolutions** (`Y=X=1`, `sH=sW=1`, `pH=pW=0` — and for 3-D: `Z=1`, `sD=1`, `pD=0`) the forward kernel automatically bypasses the full coordinate-transform descriptor DAG and replaces all three operand address computations with flat multiply-add arithmetic.

**Detection:** `ConvProblem.is_pointwise` returns `True`; no user-facing flag is needed. The benchmark prints a note when this path is active.

**Address arithmetic (pointwise path):**

| Operand | Formula | Replaces |
|---------|---------|---------|
| A (input)  | `offset = m * C + g * cpg + k` | `unmerge(m)` + `embed(ho,y→hi)` + `embed(wo,x→wi)` + `unmerge(k→y,x,c)` + `pad(y)` + `pad(x)` |
| B (weight) | `offset = k_out * cpg + k_gemm`, `k_out = g * kpg + n` | `unmerge(k_gemm→y,x,c)` + `pad(y)` + `pad(x)` |
| D (output) | `offset = m * K + k_out` | `unmerge(m→n,ho,wo)` |

`g` is the conv group (`blockIdx.z`), `k < cpg` and `n < kpg` are the
within-group GEMM coordinates, and the B load is bounded by `n < kpg`, not the
absolute `k_out`. With `groups == 1` the group terms vanish (`cpg == C`,
`kpg == K`) and nothing extra is emitted.

**Why this is faster:** With `Y=X=1`, `sH=sW=1`, `pH=pW=0` the embed transforms reduce to identity (`hi=ho`, `wi=wo`), the filter-unmerge produces `y=x=0` trivially, and the pad guards are always true. The implicit descriptor computes the same values but still emits the full VALU sequence (magic divisions, multiplications, bounds comparisons). Flat arithmetic emits one `mul` and one `add` per operand (plus the group-base `add` when `groups > 1`) — zero dead instructions.

### 3-D x-halo input reuse and K-loop knobs (uniform_k_decode + reg_prefetch=1 + lds_xor)

Opt-in identity fields (all default off; AOT env axes in `aot_jobs._fwd_sched_k_loops`, scoped with
`ROCKE_AOT_FWD_KLOOP_SCOPE` / `ROCKE_AOT_FWD_KLOOP_XCD`):

| Field (tag) | Env axis | What it does |
|---|---|---|
| `uk_xhalo=n` (`uxh<n>`) | `ROCKE_AOT_FWD_XHALO=4,8` | For stride-1, pad-0 (any pad with `xh_pad` on the wave-private DMA loop, `fxh_dma >= 1000`), X=3 3-D convs a tile of consecutive output pixels in one (n, d) plane reads one contiguous input range at tap (z, y, 0). The loop runs over (chunk, z, y), stages that range (tile_m + n pixels) once with the three x taps' W tiles side by side, and runs three MFMA phases at row shift x: a third of the A loads, A LDS writes and barriers. Each workgroup checks the conditions (pad, stride, X, packed strides, span) from its kernargs and otherwise runs the plain uniform_k_decode loop. |
| `lds_kpack_read` (`kprd`) | `ROCKE_AOT_FWD_KPRD=1` | MFMA k order permuted so a lane's two kk steps are contiguous: one `ds_read_b128` per fragment pair; the lds_xor key moves to the 128-byte span (conflict-free 16-byte phases). |
| `uk_chunk_group=n` (`cg<n>`) | `ROCKE_AOT_FWD_CG=2` | With `uk_tap_fast`, n channel chunks advance fastest (one 128-B line at tile_k 32), then the tap, then the next group. |
| `rpf_lds_barrier` (`lbar`) | `ROCKE_AOT_FWD_LBAR=1` | rpf1 barriers drain LDS only (no `vmcnt(0)`). |
| `rpf_fetch_late` (`flate`) | `ROCKE_AOT_FWD_FLATE=1` | Next tile's loads behind the first LDS reads. |
| `xh_dpp` (`xdpp`) | `ROCKE_AOT_FWD_XDPP=1` | x-halo: taps 1, 2 shift the previous tap's A fragments through the lanes (DPP) instead of re-reading LDS. Python engine only (see below). |
| `xh_pad` (`xpd`) | `ROCKE_AOT_FWD_XPD=1` | The wave-private x-halo DMA loop (`fxh_dma >= 1000`) on padded inputs: the DMA masks the padding itself (dilation 1 along D / H). |
| `rpf_setprio=n` (`prio<n>`) | `ROCKE_AOT_FWD_PRIO=1` | `s_setprio n` over the load issue and the MFMA phase. |
| `defer_mfma` on the x-halo loop | `ROCKE_AOT_FWD_SCHED=..d1..,..d2..` | The last tap's last kk MFMAs (or all of them) run after an LDS-only barrier, interleaved with the next tile's LDS writes. |
| `waves_per_eu` | `ROCKE_AOT_FWD_WPE=n` | Occupancy target clones of the rpf1 lxor loops (name tag `wpe<n>`). |

`xh_dpp`, `a_direct` (`adir`), `xh_subtile` (`xsub`), `pp_fetch_deep` (`ppfd`) and `prefetch_regs` (`pfr`) are
Python-engine only: the C++ engine has no twin, so the conv binding refuses a spec that sets one (a `ValueError`
naming the field) rather than build another kernel under its name, and the byte-identity gate does not cover them.

The x-halo knobs target large-spatial 3-D shapes on gfx942, on tiles such as `t256x64x32 w4x1 mem cshuffle xcd6 lxor kprd uxh(4|8)`
(+ tapf/cg2 for C >= 128, lah, dfr1/2, prio1).

### K-loop scheduling, split-K and tail split for small-spatial 3-D convs

Opt-in identity fields (default off, byte-identical defaults; AOT env axes in `aot_jobs`, scoped with
`ROCKE_AOT_FWD_KLOOP_SCOPE`):

| Field (tag) | Env axis | What it does |
|---|---|---|
| `lds_ahead` (`lah`) | `ROCKE_AOT_FWD_SCHED=a1..` | LDS fragment reads one kk step ahead of the MFMAs. |
| `defer_mfma=n` (`dfr<n>`) | `ROCKE_AOT_FWD_SCHED=..d1..` | The last n kk MFMAs run after the LDS-only barrier, 1 ds_write : 2 MFMA with the next tile's store (rpf1). |
| `vmem_interleave=n` (`vmi<n>`) | `ROCKE_AOT_FWD_SCHED=..v2/v4` | Next tile's buffer_loads spread one per n MFMAs (rpf1/rpf2/pfr loops). |
| `rpf_setprio=n` | `ROCKE_AOT_FWD_SCHED=..p1` | as above, token form. |
| `uk_nopad` (`nopad`) | `ROCKE_AOT_FWD_NOPAD=1` | Unpadded (halo-form) problems: no per-row, per-tap bounds checks in the uniform_k_decode loop (fewer VALU instructions per iteration); serves pad=0 problems only. |
| `split_k=2` on uniform_k_decode | `ROCKE_AOT_FWD_UKD_SPLIT=1` | Split-K binary of the ukd loops: each slice starts its tap state mid-reduction. |
| `split_tail` (`spt`) | `ROCKE_AOT_FWD_SPLIT_TAIL=1` | `full_tiles` kernarg: the first tiles in launch order run whole (plain store), only the last partial wave is split and atomic-added (xcd_remap permutes the full tiles only, so the host zeroes just the tail rows). |

The benchmark sweeps fwd split degrees that fill whole waves (`floor(n*CUs/tiles)`,
`conv_split_k.fwd_wave_filling_split_ks`) and, for `split_tail` binaries, `full = T - T % (n*CUs)` at degrees 2/3/4/6
(labelled `@split_k=<S>/full<F>`).

These knobs target small-spatial 3-D shapes on gfx942, on tiles such as
`t128x128x64 w2x2 a32 xcd6 lxor nopad lah dfr3/dfr4 vmi2/4 kprd` (k-packed reads with the 128-B XOR key remove every
LDS bank conflict, which is what lets the deeper defer hide its MFMAs); split_tail covers grids a few waves deep and
ukd split-K the grids with fewer tiles than CUs. `defer_mfma_per_write` (`dpw<n>`, `ROCKE_AOT_FWD_DPW`) sets the
ds_write : MFMA ratio of the deferred phase.

---

### 8-wave block ping-pong K loop (`rpf_pingpong`, uniform_k_decode + reg_prefetch=1)

Triton's BlockPingpong form for 8-wave tiles (two waves per SIMD). The high half of the waves runs one extra
`s_barrier` before the loop and the low half one after it, so the two waves of a SIMD run half an iteration apart.
Every K tile is four barrier-separated phases: [mem A] all LDS fragment reads of the tile, [compute A] half of the
MFMAs, [mem B] the LDS store of the next tile, [compute B] the other half -- one wave's memory phase runs under its
SIMD partner's MFMAs. Modes: `pp1` (fragment reads retired before the barrier, loads of the next tile in [mem A]),
`pp4` (global reads of tile t+2 issued in [mem B] after the store of t+1, loop-carried registers), `pp5` (those reads
spread over the [compute B] MFMAs, one per `vmem_interleave` MFMAs). Every mode waits for the wave's fragment reads
before the [mem A] barrier; `rpf_pingpong` 2 / 3, which did not, are refused (a read still queued at the LDS is not
ordered before the partner SIMD's later write). `pp_pre_mfma` (`ppm<n>`) hoists MFMAs above the opening barriers, `pp_b_double` (`ppbd`) keeps B in
two buffers. It targets the 16x16x16 atom on t256x128x64 w4x2 for gfx942 FWD 3-D convs with 128-1024 channels,
where [mem A] is LDS-read bound.
AOT axis: `ROCKE_AOT_FWD_PP=p<mode>[m<pre>][s<prio>][v<vmi>][w<wpe>],...` on `ROCKE_AOT_FWD_PP_SCOPE` geometries.
The same loop shape exists for the x-halo loop (`uk_xhalo` + `rpf_pingpong` 4/5). Related:
`cshuffle_m_chunks` (`mch<n>`, C tile staged in row slabs, also for the split-K atomic store) and, on the x-halo
loop, `xh_wait_smem` (`xws`: retire pending scalar loads before the loop so its LDS waits are partial) and
`vmem_interleave` (AOT `ROCKE_AOT_FWD_XWS=1|only`, `ROCKE_AOT_FWD_XH_VMI=2,4`).

## Next steps

### K0-M-K1 LDS layout

The current LDS layout stores tiles in `(M, K)` row-major order with a small
`lds_k_pad` column pad to break bank conflicts. A `K0-M-K1` layout reorders the
tile as `(K0, M, K1)` where `K = K0 × K1`, making each MFMA atom's K slice
contiguous in LDS. This eliminates the bank-conflict cross-section that padding
only partially mitigates and enables wider `ds_read` instructions. Adding this
requires:

1. A new `LdsLayout` variant encoding the `(K0, M, K1)` stride formula.
2. Updated `CoalescedTileLoader` / `AsyncTileLoader` store-index calculations.
3. Updated `_emit_smem_load` / `_emit_frag_smem_load` read-index expressions.

### Persistent kernel / stream-K partitioning

The current grid maps one CTA to one output tile. For small M×N problems the
grid is too small to saturate the device. Stream-K partitions the work along
the K loop across CTAs (similar to split-K but without the atomic reduction
overhead), potentially delivering better utilisation for small or irregular
shapes.

This has landed for **wgrad** (`streamk` on `WgradConvSpec`; see
`README_conv_implicit_gemm_wgrad.md`), where the lopsided `K_wg = N*Ho*Wo`
reduction makes it most valuable. The forward direction still maps one CTA per
output tile; the partitioner (`rocke.helpers.streamk.StreamKIterPartition`) and
the flag-based fixups are family-agnostic, so porting them here is wiring rather
than new machinery.

### Wider vector loads for A

The A descriptor emits one element offset per thread per load. Enabling vector
loads (`load_vec > 1`) requires that consecutive `k` values within a tile row
map to contiguous memory addresses. For general convolutions this is only true
along the C axis (the innermost dimension of NHWC); enabling it requires either
a K0/K1 split where K1 == C, or a transposing LDS stage.
