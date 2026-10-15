<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# vLLM custom kernels

Wave-split-K skinny GEMM kernels adapted from vLLM's ROCm backend,
`csrc/rocm/skinny_gemms.cu` at commit `9163190dda009a310d8c175d63860ac7c671ca90`
(Apache-2.0), for LLM decode.

## `wvSplitK_vllm_{bf16,f16}_tn_n<N>_y<YTILE>u<UNRL>`

BF16 or FP16 I/O, FP32 accumulate and alpha/beta, hipBLASLt TN with
`n = N` tokens:

```
D[m + n*ldd] = alpha * sum_k A[m*lda + k] * B[n*ldb + k] + beta * C[m + n*ldc]
```

This is the layout `torch.mm(x, w.t())` and `F.linear` reach hipBLASLt with
(`m` = output features, `n` = tokens), and vLLM's naming matches: its `N` is the
token count. Block `(64, 16)`, grid = device CU count, persistent over `m`.
Each wave reduces `YTILE` rows over K with `UNRL` 16-byte loads per lane in
flight; the BF16 kernels accumulate with `v_mfma_f32_4x4x4_16b_bf16`, the FP16
kernels with `v_dot2c_f32_f16`.

The main loop, the DPP and MFMA reductions, the tail handling and the chunked
staging are vLLM's `wvSplitK_hf_sml_` and `wvSplitK_hf_big_`. Each kernel picks
one at run time: when all `N * K` token elements fit the 160 KB of LDS they are
staged once for the workgroup's lifetime (`sml`), otherwise K is walked in
chunks that fit, each staged by the whole workgroup (`big`). vLLM's third
variant, `wvSplitK_hf_`, which reads the tokens that do not fit from global
memory, is slower than `big` on MI355X and is not used. Changes from vLLM:

- alpha/beta epilogue with separate C and D and their own leading dimensions;
  C is not read when `beta == 0`;
- the tokens are staged packed at stride K, reading only the K valid elements
  of each token, so a padded `ldb` neither overflows LDS nor reads past the
  last token; LDS DMA is used only when every token row is 16-byte aligned;
- 64-bit row offsets;
- vLLM's host-side wave balancing (`mindiv`) is computed on the device, in
  closed form;
- the `sml` path handles `m` that is not a multiple of `YTILE`, like the other
  two variants;
- no bias.

| Type | n | Tiles (YTILE, UNRL) |
| ---- | - | ------------------- |
| BF16 | 1, 3, 4 | (1, 4), (1, 2), (3, 2) |
| BF16 | 2 | (1, 4), (1, 2) |
| BF16 | 5 | (1, 2) |
| FP16 | 1 to 5 | (1, 4), (1, 2), (3, 2) |

gfx950 only: the kernels size their LDS stage for its 160 KB per CU.

Predicated in `custom.config`: `batch == 1`, `n == N`, `m > 8`, `K % 8 == 0`,
and unit strides on all four tensors. K needs no bound.

## Kernarg preload

The kernels take the arguments of the rocBLAS wvSpltK kernels, in the same
order: 72 bytes, of which the first 14 dwords (everything the staging and the
K loop read) are preloaded into SGPRs. `n` is passed but unused.

## Library logic

hipBLASLt picks these kernels through two `Range` logic files in
`gfx950/gfx950/Range/`: `gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` (BF16,
shared with the rocBLAS wvSpltK kernels) and
`gfx950_Cijk_Alik_Bljk_HHS_BH_UserArgs.yaml` (FP16). Every row has `batch == 1`,
one `n` from 1 to 5 and `m >= 9`. The rows were measured on MI355X, over decode
shapes of current models, a power-of-two sweep, and random shapes, each with
the weights in cache and streamed from HBM. For each `n` and each cell of

- m: 9 to 1536, 3072, 6144, 12288, 24576, 49152, 98304, and above;
- K: 8 to 448, 1280, 2560, 5120, 10240, 20480, and above;

the file keeps what hipBLASLt picked before (its own kernel, or a rocBLAS
wvSpltK kernel for BF16) unless one of this `n`'s tiles is at least 3% faster
over the cell's shapes (6% for BF16 `n = 5`, whose single tile gains least);
among the tiles, it keeps the one that is best for this `n` overall unless
another is at least 3% faster. Below K = 449 the kernels leave most lanes idle
and get no rows. A cell without measurements takes a choice only if the
measured cells on either side of it in m agree; otherwise it keeps hipBLASLt's
previous choice. Three cells where hipBLASLt runs with the built library showed
no gain also keep it: under sustained load these kernels lose more clock than
hipBLASLt's own on the largest shapes, which standalone timing misses. Cells
where hipBLASLt's own kernel wins have no row. Rows never overlap, since
hipBLASLt takes the first row that contains the problem.

The files carry the plain-GEMM problem type (no bias, activation or scale
vector), because a custom kernel takes the logic file's problem type and these
kernels support none of those.

Within one device's logic, hipBLASLt searches the plain-GEMM placeholder library
before the Bias/SAV ones, and `Equality` before `Range` inside a placeholder.
The BF16 file reuses the header of the shipped plain-GEMM logic
(`gfx950/gfx950/Equality/gfx950_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml`), so both
share a placeholder and its tuned sizes keep their kernels; tuning a size there
is how to move it off these kernels. gfx950 ships no plain-GEMM FP16 TN logic,
so the FP16 file forms a placeholder of its own, with the same header and the
BF16 problem type's data types set to FP16. A matched range whose kernel fails
its own predicates (for example `K % 8 != 0`) returns nothing, so the problem
falls through to the next library. General-batched (pointer-array) problems
skip custom kernels altogether, since these take plain A/B/C/D addresses.

The files use the MI350 (`0x75a0`) header. MI355X searches its own tuned logic
(`gfx950_id75a3`) first, so its tuned sizes keep their kernels. On MI350, the
ranges come before skinny sizes tuned in the same device's Bias libraries.

## Tensile metadata

Each `.s` embeds its Tensile `custom.config`, generated from the Tensile YAML
for its type and `n`, `custom_vllm_wvsplitk_{bf16,f16}_tn_n<N>.yaml`, which
lists that `n`'s tiles. To refresh one, delete the existing block first:
`AddCustomConfig` will not overwrite it. Replacement assembly must stay at code
object version 4 (see `../README.md`) and keep its `.amdgcn_target` /
`.amdhsa_code_object_version` directives.

```bash
python -m tensilelite.AddCustomConfig \
  tensilelite/CustomKernels/vllm/wvSplitK_vllm_bf16_tn_n1_y1u4.s \
  --yaml tensilelite/Tests/custom/custom_vllm_wvsplitk_bf16_tn_n1.yaml \
  --origin vllm \
  --repository https://github.com/vllm-project/vllm \
  --version 9163190dda009a310d8c175d63860ac7c671ca90
```
