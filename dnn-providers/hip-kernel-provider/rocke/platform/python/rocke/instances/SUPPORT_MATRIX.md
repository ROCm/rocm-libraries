# rocke instance support matrix

What each instance supports across the currently-targeted architectures.
A cell is ✅ if `compile_kernel(build_*(spec, arch), arch)` produces a HSACO
for that target, ❌ if the instance's validator rejects the arch (no suitable
matrix atom / unsupported path).

| Arch | µarch | Matrix engine | Wave | f16 fp32-acc atoms |
|---|---|---|---|---|
| gfx942 | CDNA3 | MFMA | 64 | 16x16x16, 32x32x8 |
| gfx950 | CDNA4 | MFMA | 64 | 16x16x16, 16x16x32, 32x32x8, 32x32x16, scaled 32x32x64 |
| gfx1151 | RDNA3.5 | WMMA | 32 | 16x16x16 (`wmma_f32_16x16x16_{f16,bf16}`) |

Matrix-core instances below were exercised with a portable f16 **16x16x16**
atom (CDNA: `pipeline=mem`/`compv*`; gfx1151: `pipeline=mem`, `epilogue=default`,
`wave_size=32`). Generated 2026-05-29; cross-compiled on a gfx1151 box.

---

## Elementwise / norm / reduce / data-movement (no matrix core)

| Instance | gfx942 | gfx950 | gfx1151 |
|---|:--:|:--:|:--:|
| `elementwise` | ✅ | ✅ | ✅ |
| `reduce2d` | ✅ | ✅ | ✅ |
| `rmsnorm2d` | ✅ | ✅ | ✅ |
| `layernorm2d` | ✅ | ✅ | ✅ |
| `add_rmsnorm2d_bf16` | ✅ | ✅ | ✅ |
| `add_rmsnorm2d_rdquant` | ✅ | ✅ | ✅ |
| `smoothquant` | ✅ | ✅ | ✅ |
| `moe_smoothquant` | ✅ | ✅ | ✅ |
| `transpose2d` | ✅ | ✅ | ✅ |
| `batched_transpose2d` | ✅ | ✅ | ✅ |
| `transpose_bc` | ✅ | ✅ | ✅ |
| `permute_nd` | ✅ | ✅ | ✅ |
| `pooling2d` | ✅ | ✅ | ✅ |
| `img2col` | ✅ | ✅ | ✅ |
| `topk_softmax` | ✅ | ✅ | ✅ |
| `moe_sorting` (histogram/scan/scatter/persistent) | ✅ | ✅ | ✅ |

These emit generic AMDGPU IR; arch only sets the comgr target triple.

`elementwise` and `reduce2d` also accept `dtype="f64"` (native f64
compute, `vec=2`). f64 elementwise is limited to the native f64 ops `copy`,
`neg`, `abs`, `relu`, `add`, `sub`, `mul`, `max` and `min`.

---

## gfx950 native scaled GEMM

`gfx950/block_scaled_gemm` supports all 25 ordered homogeneous/mixed FP8 E4M3,
BF8 E5M2, FP6 E2M3, BF6 E3M2 and FP4 E2M1 pairs with native 32x32x64 scaled
MFMA. E8M0 scales cover K32 groups; accumulation is FP32 and output BF16/FP16.
M/N must be multiples of 32 and K a multiple of 64, within signed i32 indexing.
LLVM 20/22/23 lowering is supported on gfx950. Use the spec-driven builder
directly. The [four focused examples](../examples/gfx950/gemm/README.md)
cover F8, F6, F4 and mixed inputs with independent non-unity scales. These are
untuned correctness examples and make no performance claim. Python/C++
IR and LLVM identity, compiler ISA checks and bounded hardware numeric suites
cover this consumer; arbitrary-input rounding and special scales remain outside
the finite verification fixtures.

## gfx1250 native scaled GEMM

`block_scaled_gemm` supports homogeneous and mixed FP8 E4M3, BF8 E5M2,
FP6 E2M3/E3M2, and FP4 E2M1 through SCALE and SCALE16 with the `llvm23`
emission flavor. FP8/BF8/FP6/BF6 operands require E8M0 scales; FP4 accepts
E8M0, E4M3, or E5M3. FP4 × FP4 requires matching scale formats. Host tests
cover admission and rejection; numeric fixtures cover bounded finite inputs,
varying non-unit scales, and fixed-byte selector discrimination through COMGR/HIP.
See the [scale-format contract](../examples/gfx1250/gemm/SCALE_FORMATS.md)
and [packed FP6 input contract](../examples/gfx1250/gemm/FP6.md).

## GEMM family

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `universal_gemm` | ✅ | ✅ | ✅ | gfx1151: `mem`+`default` only |
| `batched_gemm` | ✅ | ✅ | ✅ | |
| `grouped_gemm` | ✅ | ✅ | ✅ | |
| `flatmm` | ✅ | ✅ | ✅ | |
| `gemm_multi_d` | ✅ | ✅ | ❌ | needs `cshuffle` epilogue (not on WMMA) |
| `gemm_multi_abd` | ✅ | ✅ | ❌ | needs `cshuffle` epilogue (not on WMMA) |
| `mfma_gemm` | ❌ | ✅ | ❌ | needs f16 16x16x32 kpack atom (CDNA4) |
| `streamk_gemm` | ✅ | ✅ | ❌ | MFMA stream-K reduction path |
| `block_scale_gemm` | ✅ | ✅ | ❌ | MFMA quantized GEMM |
| `mx_gemm` | ✅ | ✅ | ❌ | MFMA microscaling GEMM |

---

## Convolution

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `conv_implicit_gemm` | ✅ | ✅ | ✅ | gfx1151: WMMA 16x16x16, `mem`+`default`, `wave_size=32`, `groups=1` |
| `conv_implicit_gemm_auto` | ✅ | ✅ | ❌ | MFMA-specialized autotuned path (raw `MfmaAtom`, K=32 kpack); not ported to WMMA |
| `direct_conv_16c` | ❌ | ✅ | ❌ | `fold_k32` needs 16x16x32 atom (CDNA4) |
| `direct_conv_4c` | ✅ | ✅ | ❌ | 4x4x4 MFMA atom not in WMMA catalog; fp16 and bf16 (`_1k` atom) |
| `direct_conv_4c` dgrad (`direct_grouped_dgrad_4c`) | ✅ | ✅ | ❌ | `cpg=kpg=4`, 1x1/3x3 "same" padding, stride 1, fused weights (no pre-pass); LDS-staged weights and `stage_rows` need `ds_read_b64_tr_b16` (gfx950). In the AOT direct cache; not yet wired into dispatch |
| `direct_conv` (generic) dgrad, fused weights (`direct_grouped_dgrad_fused`) | ✅ | ✅ | ❌ | stride 1, "same" padding, 1x1..7x7, `cpg`/`kpg` multiples of 4 up to 32 (`cpg != kpg` too), single kernel (no pre-pass); LDS-staged weights need `ds_read_b64_tr_b16` (gfx950), `fold_k32` the 16x16x32 atom; row-stream knobs (`prefetch_rows`, `lds_only_sync`, `waves_m`, `lds_pad`, `stage_out`, `row_exit`), `xcd_tiles` gfx950 only. In the AOT direct cache; not yet wired into dispatch |
| `direct_depthwise_dgrad_win` | ✅ | ✅ | ❌ | `cpg=kpg=1`, stride 1, any padding; VALU form (scalar FMA, channel vectors); `dot2` tap pairing (`arith.fdot2`) and the Toeplitz MFMA form (`mfma_f32_16x16x32`, `groups % 8 == 0`) gfx950 only. In the AOT direct cache; not yet wired into dispatch |
| `conv_implicit_gemm_dgrad` stride-1 knobs | ✅ | ✅ | ❌ | folded sub-GEMM record (stride capability), `tap_outer_k` (wave64 MFMA, `kpg % tile_k == 0`), `dy_halo` / `dy_halo_setprio` / `dy_halo_kouter_pad` / `halo_max_wo` (on the tap-outer path, `Ho == Hi`, `Wo == Wi`; the pad needs `lds_k_outer`, gfx950); `xcd_tile_order` remaps only on gfx950. AOT grid: stride-1 tap-outer binaries next to the tilde ones, halo sub-grid on gfx950; not yet wired into dispatch |
| `conv_implicit_gemm_wgrad` | ✅ | ✅ | ✅ | gfx1151/gfx1201/gfx1250: WMMA, `default` epilogue, `split_k=1` |
| `conv_implicit_gemm_wgrad` stream-K | ✅ | ✅ | ❌ | CK iteration-balanced partition; `linear`/`tree` deterministic fixups need the cross-workgroup flag ops |
| `conv_direct_nongrouped` | ✅ | ✅ | ❌ | `groups=1`; gfx942 uses 32x32x8 / 16x16x16 atoms, gfx950 also 32x32x16 / 16x16x32; MFMA-only; the only direct kernel with dilation. Built from `library/kernels`; not yet wired into dispatch |
| `direct_wgrad_tiled` | ❌ | ✅ | ❌ | dense and grouped wgrad, stride 1; needs `ds_read_b64_tr_b16` and `mfma_f32_16x16x32`. Built from `library/kernels`; AOT sweep only, not yet wired into dispatch |
| `dw_wgrad_lds` / `dw_wgrad_cols` / `direct_wgrad_gather` | ❌ | ✅ | ❌ | depthwise (`C % 8 == 0` / `C <= 8`) and dense small-C wgrad, partial slots + `dw_wgrad_fold`; validated on gfx950 only (the LDS kernels DMA 16 bytes per lane). AOT sweep only, not yet wired into dispatch |
| `dw_wgrad_mfma` / `grouped_wgrad_mfma` | ❌ | ✅ | ❌ | depthwise (`C % 8 == 0`, any filter) and 4/8-channel grouped wgrad, stride 1, on `mfma_f32_4x4x4`; partial slots + `dw_wgrad_fold`; need `ds_read_b64_tr_b16`. AOT sweep only, not yet wired into dispatch |
| `direct_depthwise_tiled` | ✅ | ✅ | ❌ | `cpg=kpg=1`, filters up to 31x31, any padding; scalar FMA at the target's wave size (wave64, or wave32 on gfx1250; spec `wave_size` must match). Built from `library/kernels`; not yet wired into dispatch |
| `conv_direct_dgrad_wmma` | ❌ | ❌ | ❌ | gfx1250 only: wave32 WMMA 16x16x32 + TDM, stride-1 dgrad, dense and grouped (`cpg`, `kpg` multiples of 8), `PAD` in 0..K-1. Built from `library/kernels`; served by the direct-conv AOT cache, not the dispatcher |

---

## Attention / FMHA

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `fmha_fwd_mfma` | ✅ | ✅ | ✅ | auto-dispatches to WMMA on gfx1151 |
| `fmha_bwd` | ✅ | ✅ | ✅ | |
| `fmha_appendkv` | ✅ | ✅ | ✅ | |
| `fmha_paged_prefill` | ✅ | ✅ | ✅ | |
| `fmha_varlen` | ✅ | ✅ | ❌ | explicit MFMA-atom lookup |
| `fmha_splitkv_decode` | ✅ | ✅ | ❌ | explicit MFMA-atom lookup |
| `sage_attention` | ✅ | ✅ | ❌ | MFMA FMHA atom required |
| `vsa_sparse_attention` | ✅ | ✅ | ❌ | MFMA FMHA atom required |
| `jenga_sparse_attention` | ✅ | ✅ | ❌ | MFMA FMHA atom required |
| `fmha_head_grouping` | ❌ | ❌ | ❌ | config-dependent atom lookup (not built by default config) |
| `fmha_fwd_fp8` | ❌ | ❌ | ❌ | fp8 attention path (config-dependent) |
| `unified_attention_2d` | ✅ | ✅ | ✅ | scalar (no matrix core) |
| `unified_attention_3d` | ✅ | ✅ | ✅ | scalar |
| `unified_attention_reduce` | ✅ | ✅ | ✅ | scalar |
| tiled 3D `kv_layout="strided"` | ✅ | ✅ | ❌ | non-paged Sq=1, fp16/bf16 D64/128/256, aligned unit-D KV views; existing tiled segment/reduce pipeline |
| `mla_prefill_fwd` | ✅ | ❌ | ❌ | bf16 only; paged compressed KV, `W_UK` absorbed into the query and `W_UV` applied in the epilogue (the latent KV is never expanded). Takes an already projected, RoPE-rotated query. Lives in `library/kernels/mla/`, dispatched by the separate `dispatch.mla` registry (not `dispatch.attention`). gfx950 is a later phase. |

---

## Linear attention (chunkwise gated delta rule)

Not part of the 2026-05-29 sweep above; added with the KDA family and verified
as described in the notes.

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `kda_chunk_fused` | ✅ | ✅ | ❌ | bf16 only; fused prefill; gfx942 partitions V, gfx950 owns a full head |
| `kda_chunk_prep` | ✅ | ✅ | ❌ | bf16 only; split path phase 1, one workgroup per chunk |
| `kda_chunk_scan` | ✅ | ✅ | ❌ | bf16 only; split path phase 2, consumes what prep wrote |

## Sparse attention (DSA lightning indexer)

Added with the DSA lightning-indexer family. The kernel is arch-neutral
(`kernels/common/lightning_indexer.py`): the ✅ cells are compile- and
golden-verified (spec, wiring, IR sha, Python/C++ byte-identity); on-GPU numeric
parity runs on the matching device and is not implied by the ✅ here.

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `lightning_indexer` | ✅ | ✅ | ❌ | bf16; score-only DSA scorer (no softmax/value); runtime seqlen_q/seqlen_k; scalar body (sub-tile seqlen_q / single-query decode fallback, correctness oracle) + MFMA body (tail-masked for any seqlen_q/seqlen_k; index_head_dim 16-aligned); fp8 is a later phase; gfx1151 (RDNA wave32) unsupported |

## Linear attention / recurrent-state decode

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `gdn_decode` | ❌ | ✅ | ❌ | gated delta rule, single-token decode over a paged recurrent state; no softmax |
| `gdn_prefill` | ❌ | ✅ | ❌ | gated delta rule, chunkwise prefill, **bf16 only**; the KDA chunkwise pair in `gate_kind="gdn"` mode, two launches (`chunk_prep` then `chunk_scan`), no fused default |
---

## Arch-specific native instances

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `gfx1151/wmma_gemm` | ❌ | ❌ | ✅ | native WMMA GEMM (GPU-verified) |
| `gfx1151/wmma_fmha_fwd_v2` | ❌ | ❌ | ✅ | native WMMA FMHA prefill, gfx11/gfx12 (GPU-verified on gfx1100/gfx1101/gfx1150/gfx1201; not run on gfx1151) |
| `gfx1151/wmma_fmha_decode` | ❌ | ❌ | ✅ | native split-KV WMMA FMHA decode, gfx11/gfx12 (GPU-verified on gfx1100/gfx1101/gfx1150/gfx1201; not run on gfx1151) |

---

## Notes

- **gfx1151 (WMMA) only ships the `mem` pipeline + `default` epilogue** for the
  GEMM family. The `compv3`/`compv4` pipelines and the `cshuffle` epilogue are
  MFMA-only paths, so any instance that mandates them (e.g. `gemm_multi_d`,
  `gemm_multi_abd`) is rejected on gfx1151.
- **Convolution on gfx1151:** `conv_implicit_gemm` runs on WMMA
  (16x16x16, `mem` pipeline, `default` epilogue, `wave_size=32`, `groups=1`).
  `conv_implicit_gemm_auto` is a separate MFMA-specialized autotuned path
  (raw `MfmaAtom`, K=32 kpack) that has not been ported to the WMMA atom
  contract, so it stays MFMA-only.
- **Merged groups (`group_merge`)** is available in the forward
  (`conv_implicit_gemm`) and wgrad (`conv_implicit_gemm_wgrad`) directions for
  **depthwise** problems only (`cpg == kpg == 1`), and is inert at its default
  of `1`. Forward builds it on MFMA (wave64) and WMMA (wave32) targets alike
  (GPU-numeric-verified on gfx1250; the gfx11 WMMA targets are covered by the
  parity gate only); wgrad is `wave_size == 64` (MFMA) only. The degree must be a power of two
  in {2, 4, 8, 16, 32, 64} and divide `groups`. Beyond that the two directions
  restrict differently. Forward additionally rejects `async_dma`, an explicit
  `vector_size_b` other than 1, a `vector_size_a`/`vector_size_c` that does not
  divide the degree, the pointwise fast path (`Y == X == 1`), **3-D
  convolution**, a degree larger than `tile_n` (the merged GEMM-N is `Gm`), and
  an `a_mhw_index_fn` override; since merging widens the default store vector,
  it also requires `epilogue="cshuffle"` unless `vector_size_c=1` is pinned.
  wgrad additionally rejects `split_k > 1` without `two_stage=True`, and a
  merged GEMM that does not fit one tile (`Gm > tile_m`, or `Z*Y*X*Gm > tile_n`).
  Forward merged kernels are built into the AOT kernel cache at every
  admissible degree and selected there by timing; the grouped-conv dispatcher
  does not select a forward merge degree. **Engine coverage differs between the two directions.** Forward is
  implemented in **both** engines — `group_merge` is a field on the C-ABI spec
  struct (`rocke_implicit_gemm_conv_spec`), is carried across the pybind
  boundary, and the fwd parity emitters build eight depthwise configs
  (`conv_implicit_gemm_emit.{py,c}` indices 21–28: 22–27 merged, 21 and 28 the
  unmerged default/cshuffle controls), plus four merged WMMA ones (29–32:
  gfx1250 `mem` and `wavelet` with cshuffle, gfx1250 with the direct epilogue,
  and gfx1151) and three unmerged grouped gfx1250 controls (33–35), spanning
  both epilogues, the 1-wide
  cshuffle store, `unroll_k`, and degrees with `groups/Gm > 1` as well as
  `Gm == groups` (one merged group; the grouped decode stays engaged). Byte-identity is therefore a
  real gate on merged emission, not a vacuous one. **wgrad is Python-engine
  only:** the C++ engine has no merged wgrad configuration to express and its
  parity emitter builds unmerged configs exclusively, so merged *wgrad* specs
  carry no structural parity coverage and are verified by on-GPU numerics
  alone. That gap predates the forward port and is tracked as follow-up work.
- **GPU-numeric verification on gfx1151** (launched on the Radeon 8060S,
  output compared against a numpy reference). **Verified PASS:**
  `elementwise`, `reduce2d`, `rmsnorm2d`, `layernorm2d`, `transpose2d`,
  `batched_transpose2d`, `transpose_bc`, `permute_nd`, `pooling2d`,
  `img2col`, `batched_gemm`, `universal_gemm`,
  `grouped_gemm`, `flatmm`, `conv_implicit_gemm`, `add_rmsnorm2d_bf16`,
  `add_rmsnorm2d_rdquant` (i8 out only on RDNA), `smoothquant`,
  `moe_smoothquant`, `topk_softmax`, `moe_sorting` (all 4 phases),
  `fmha_fwd_mfma`, `fmha_appendkv`, `unified_attention_2d/3d/reduce`,
  `gfx1151/wmma_gemm`.
  **Marginal PASS** (a few elements over a strict 2e-2 but 0 bad at 3e-2;
  their scalar bodies hardcode `WARP_SIZE=64`/`block=(64,1,1)`, i.e. 2 waves
  on wave32, but resolve correctly): `fmha_paged_prefill` (2.43e-2),
  `fmha_bwd` (2.75e-2).
  The matrix-core GEMMs match the numpy f32 reference to ~2e-5 max-abs (WMMA
  accumulates the f32 partials in a different order than numpy, so a handful
  of elements drift ~1 fp16 ULP — judged within tolerance, not bit-exact).
  The reductions (`reduce2d`/`rmsnorm2d`/`layernorm2d`) **must be built with
  `wave_size=32`** on gfx1151: the XOR-butterfly cross-lane reduction emits
  `log2(wave_size)` shuffle stages, so a `wave_size=64` build issues a
  lane-mask-32 shuffle that is invalid on wave32 hardware and silently
  corrupts the result.
- **`add_rmsnorm2d_rdquant` on gfx1151 (RDNA):** only `out_dtype="i8"` is
  supported. fp8/bf8 output needs the CDNA-only `v_cvt_pk_{fp8,bf8}_f32`
  conversion, so fp8/bf8 specs are rejected by the validator on non-CDNA
  families.
- **KDA (`kda_chunk_*`)** has separate gfx942 and gfx950 emitters, each with an
  architecture gate and schedule matched to that ISA. gfx942 partitions a
  logical value head into 64-channel workgroups to fit its LDS budget; gfx950
  uses CDNA4 K-packed bf16 atoms and supports both full-head and value-split
  scans. Both ✅ columns are GPU-numeric-verified by their architecture-specific
  tests, and their emitted IR is pinned by `test_kda_gfx942_golden.py` and
  `test_kda_gfx950_golden.py`. gfx1151 remains an explicit refusal, not an
  untested gap.
- All other ✅ cells remain compile-verified only (HSACO produced for the
  target; not yet GPU-numeric-verified).
- gfx942/gfx950 cells use a portable f16 16x16x16 config; an instance marked ❌
  for a CDNA arch lacks the specific atom that config selects (e.g. `mfma_gemm`
  and `direct_conv_16c` need the CDNA4 16x16x32 atom absent on gfx942).
- **`gdn_decode` dispatch is gfx950-only by registration and a wave64 target
  match.** Its candidates are registered only for gfx950 and create a default
  `GdnDecodeSpec` with `wave_size=64`. `is_valid_spec` requires that value to
  match the target's hardware wave size, rejecting wave32 targets before it
  considers the thread-block limit. The lane mapping and XOR butterfly depend
  on this match. Adding an arch requires a new module under
  `library/dispatch/gdn/` plus a tuning run. This instance is GPU-numeric-verified on
  gfx950 against an fp32 reference, covering both the output and the in-place
  recurrent-state update.
