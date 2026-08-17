# rocke instance support matrix

What each instance supports across the currently-targeted architectures.
A cell is ✅ if `compile_kernel(build_*(spec, arch), arch)` produces a HSACO
for that target, ❌ if the instance's validator rejects the arch (no suitable
matrix atom / unsupported path).

| Arch | µarch | Matrix engine | Wave | f16 fp32-acc atoms |
|---|---|---|---|---|
| gfx942 | CDNA3 | MFMA | 64 | 16x16x16, 32x32x8 |
| gfx950 | CDNA4 | MFMA | 64 | 16x16x16, 16x16x32, 32x32x8, 32x32x16 |
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

---

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
| `direct_conv_4c` | ✅ | ✅ | ❌ | 4x4x4 MFMA atom not in WMMA catalog |

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

---

## Arch-specific native instances

| Instance | gfx942 | gfx950 | gfx1151 | Notes |
|---|:--:|:--:|:--:|---|
| `gfx1151/wmma_gemm` | ❌ | ❌ | ✅ | native WMMA GEMM (GPU-verified) |
| `gfx1151/wmma_fmha_fwd` | ❌ | ❌ | ✅ | native WMMA FMHA fwd (GPU-verified) |

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
- **GPU-numeric verification on gfx1151** (launched on the Radeon 8060S,
  output compared against a numpy reference). **Verified PASS:**
  `elementwise`, `reduce2d`, `rmsnorm2d`, `layernorm2d`, `transpose2d`,
  `batched_transpose2d`, `transpose_bc`, `permute_nd`, `pooling2d`,
  `img2col`, `batched_gemm`, `universal_gemm`,
  `grouped_gemm`, `flatmm`, `conv_implicit_gemm`, `add_rmsnorm2d_bf16`,
  `add_rmsnorm2d_rdquant` (i8 out only on RDNA), `smoothquant`,
  `moe_smoothquant`, `topk_softmax`, `moe_sorting` (all 4 phases),
  `fmha_fwd_mfma`, `fmha_appendkv`, `unified_attention_2d/3d/reduce`,
  `gfx1151/wmma_gemm`, `gfx1151/wmma_fmha_fwd`.
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
- All other ✅ cells remain compile-verified only (HSACO produced for the
  target; not yet GPU-numeric-verified).
- gfx942/gfx950 cells use a portable f16 16x16x16 config; an instance marked ❌
  for a CDNA arch lacks the specific atom that config selects (e.g. `mfma_gemm`
  and `direct_conv_16c` need the CDNA4 16x16x32 atom absent on gfx942).
