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

Each row of D is reduced by one wave with the same split of K over its lanes,
the same loop and the same reduction, whichever wave and tile it falls in, and
StaggerU is 0. The kernels therefore declare `uniformSummationOrder: true` in
`custom.config`, so hipBLASLt can select them under uniform summation order.

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

## gfx1201

The kernels named `*_gfx1201` port vLLM's skinny GEMMs for gfx12 from
`csrc/rocm/skinny_gemms.cuh`, with the configuration of its launchers
(`skinny_gemms_wvsplitk.cu`, `skinny_gemms_wvsplitkq.cu`), at commit
`392ed873c7fed4c41fafaad022f83f7d0fe9f57a` (Apache-2.0). They are
built for gfx1201 only: 32-lane waves and 64 KB of LDS per workgroup. All of
them take hipBLASLt's bias vector, one value of D's type per row of D (bias
source D), added in FP32 before the conversion to D's type. A null bias
address adds nothing, so the same kernels run GEMMs without a bias:

```
D[m + n*ldd] = alpha * sum_k A[m*lda + k] * B[n*ldb + k] + beta * C[m + n*ldc] + bias[m]
```

with `scaleA * scaleB` folded into alpha for FP8. Changes from vLLM, for
hipBLASLt's interface, as on gfx950:

- alpha/beta epilogue with separate C and D and their own leading dimensions;
  C is not read when `beta == 0`;
- hipBLASLt's bias vector rather than vLLM's broadcast one, loaded when a wave
  starts a row tile;
- every leading dimension is a kernarg, and 64-bit row offsets;
- rows past `m` read the last row and are not written, so `m` needs no bound.

### `wvSplitK_vllm_{bf16,f16}_tn_n<N>_y<YTILE>u<UNRL>_gfx1201`

BF16 or FP16, `n = 1` to `5`: the wave32 path of vLLM's `wvSplitK_hf_sml_` and
`wvSplitK_hf_big_`. Block `(32, 16)`, grid = device CU count, persistent over
`m`; each wave reduces `YTILE` rows over K with `UNRL` 16-byte loads per lane
in flight, with `v_dot2_f32_bf16` or `v_dot2_f32_f16` (through the compiler's
built-ins, where vLLM uses inline assembly), then DPP within each row of 16
lanes and a shuffle across the two. The weights are read with plain loads:
vLLM's nontemporal loads skip the Infinity Cache on gfx12, which on the RX 9070
XT made these kernels 2.7 to 2.9 times slower on weights that fit it, while
plain loads cost 0 to 5% on the shapes tried with the weights read from memory.
Each kernel picks `sml` (all `N * K` token elements staged once in LDS) or
`big` (K walked in chunks that fit) at run time; the tokens are staged packed at
stride K, reading only the K valid elements of each, and vLLM's host-side wave
balancing (`mindiv`) is computed on the device, as on gfx950. The tiles are the
ones the measured rows use (see Library logic):

| Type | n | Tiles (YTILE, UNRL) |
| ---- | - | ------------------- |
| BF16 | 1 | (1, 4), (1, 2) |
| FP16 | 1 | (1, 4) |
| BF16, FP16 | 2, 4 | (1, 4), (2, 2) |
| BF16, FP16 | 3 | (1, 4), (4, 2) |
| BF16, FP16 | 5 | (1, 4), (2, 4) |

Predicated in `custom.config`: `batch == 1`, `n == N`, `K % 8 == 0`, and unit
strides on all four tensors.

### `wvSplitKQ_vllm_f8_{bf16,f16}_tn_n<N>_y<YTILE>u<UNRL>_gfx1201`

FP8 (OCP E4M3) A and B with scalar scales or none, BF16 or FP16 C and D,
`n = 1` to `4`: the `__GFX12__` path of vLLM's `wvSplitKQ_hf_sml_` and
`wvSplitKQ_hf_`. Block `(32, 16)`, grid = device CU count; each lane loads 16
FP8 bytes of K at a time and accumulates with `v_dot4_f32_fp8_fp8`, and the
reduction uses DPP built-ins where vLLM uses inline-assembly DPP adds behind
`s_nop`. The weights are read with plain loads, as in wvSplitK. Each kernel
picks `sml` (all `N * K` token bytes staged in LDS) or `hf_` (the bytes that
fit staged, the rest read from global memory) at run time. A null scale
address reads as 1. The tiles are the measured rows':

| D | n | Tiles (YTILE, UNRL) |
| - | - | ------------------- |
| BF16 | 1 | (1, 4), (4, 2) |
| BF16 | 2 | (1, 2), (4, 1) |
| BF16 | 3 | (1, 4), (4, 1) |
| FP16 | 1, 2 | (1, 4), (4, 1) |
| FP16 | 3 | (1, 4), (2, 4) |
| BF16, FP16 | 4 | (1, 4), (2, 4) |

Predicated in `custom.config`: `batch == 1`, `n == N`, `K % 16 == 0`, and unit
strides on all four tensors.

### Summation order and arguments

Each row of D is reduced by one wave, with the same split of K over its lanes,
the same loop and the same reduction, wherever it falls. StaggerU is 0, and the
epilogue is compiled without contraction, so every row rounds alike whether its
row tile is whole or not. All the kernels declare `uniformSummationOrder: true`.

The kernels take the gfx950 wvSplitK kernels' arguments followed by the bias
address (80 bytes); wvSplitKQ appends the scale A, scale B and bias addresses
(96 bytes). gfx12 has no kernarg preload. `n` is passed but unused.

### Library logic

hipBLASLt picks these kernels through eight `Range` logic files in
`gfx1201/Range/`: `gfx1201_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml` and
`gfx1201_Cijk_Alik_Bljk_HHS_BH_UserArgs.yaml` (wvSplitK, without a bias),
`gfx1201_Cijk_Alik_Bljk_BBS_BH_Bias_B_UserArgs.yaml` and
`gfx1201_Cijk_Alik_Bljk_HHS_BH_Bias_H_UserArgs.yaml` (the same with a bias of
D's type), and `gfx1201_Cijk_Alik_Bljk_F8{B,H}S_BH_SAB_UserArgs.yaml` and
`gfx1201_Cijk_Alik_Bljk_F8{B,H}S_BH_Bias_{B,H}_SAB_UserArgs.yaml` (wvSplitKQ).
Their problem types are the gfx950 files': plain TN GEMM, FP8 with scalar A and
B scales, and a bias of D's type in the bias files. gfx1201 ships no plain-GEMM
TN logic, so each file forms a placeholder of its own. Every bias row lies
inside the plain rows, so a GEMM without a bias, which also matches the bias
problem type, never takes a bias row. The solutions refuse a user split-K or
WGM override, since the kernels take neither.

The files carry gfx1201's header (`[Device 73f0]`) with `CUCount: 32`. HIP
counts workgroup processors on RDNA4, so the 64-CU RX 9070 XT and Radeon AI PRO
R9700 report 32 and take the rows, and the 56-CU RX 9070 (28) keeps its
current kernels. hipBLASLt searches a device's logic for its CU count before
the logic for any CU count, which is all of gfx1201's other logic, and a
matched range whose kernel fails its own predicates (for example `K % 8 != 0`)
returns nothing, so other problems reach the libraries they reach today. On
those devices the rows come before the skinny sizes of gfx1201's other TN
logic, among them one tuned size, FP16 1280 x 1 x 1280 in
`gfx1201/Equality/gfx1201_Cijk_Alik_Bljk_HHS_BH_HAS_UserArgs.yaml`, which the
rows take: its cell is 21% faster through hipBLASLt with them.

The rows were measured on an RX 9070 XT, over decode shapes of current models,
a sweep (m = 9 and powers of two from 16 to 131072, K powers of two from 512
to 32768) and random shapes, `n = 1` to `5` (BF16, FP16) or `1` to `4` (FP8),
each with the weights in the Infinity Cache and read from memory, without and
with a bias: every tile by direct launch and hipBLASLt's previous pick through
hipblaslt-bench, twice each. As on gfx950, for each type, `n` and cell of

- m: 9 to 128, 512, 1536, 3072, 6144, 12288, 24576, 49152, 98304, and above;
- K: 512 to 1280, 2560, 5120, 10240, 20480, and above;

the file keeps hipBLASLt's previous pick unless one of this `n`'s tiles is at
least 3% faster over the cell's shapes, hot and cold together; among the
tiles, it keeps the one that is best for this `n` overall unless another is at
least 3% faster. Unlike gfx950, each `n` keeps at most two tiles, the best
overall and the one that adds most over the cells (all tiles would gain less
than 1% more, with three times the kernels), and the bias rows use the plain
rows' tiles. K below 512 was not measured and gets no rows. A cell without
measurements takes a tile only if the measured cells on either side of it in m
agree. With the rows built, every shape went through hipblaslt-bench again on
both libraries, and a cell keeps its row only if hipBLASLt is at least 3%
faster there and none of its shapes is more than 5% slower, after re-timing
such shapes on both libraries in turn. That drops 168 of the 1,445 cells: 29
that gain less than 3% through the library, and 139 with a shape more than 5%
slower, 46 of them at m up to 128, where few waves are busy and each walks all
of K. Bias rows outside the remaining plain rows go too. Rows never overlap,
since hipBLASLt takes the first row that contains the problem.

Through hipblaslt-bench on the measured shapes, without a bias, develop's
time over the new one (geometric mean over the shapes, with the weights in
the Infinity Cache / read from memory; a bias changes these by at most 0.05):

| Type | n = 1 | n = 2 | n = 3 | n = 4 | n = 5 |
| ---- | ----- | ----- | ----- | ----- | ----- |
| BF16 | 1.35 / 1.23 | 1.21 / 1.16 | 1.17 / 1.13 | 1.12 / 1.08 | 1.07 / 1.06 |
| FP16 | 1.32 / 1.22 | 1.20 / 1.14 | 1.16 / 1.12 | 1.11 / 1.08 | 1.07 / 1.06 |
| FP8, BF16 D | 1.82 / 1.61 | 1.83 / 1.63 | 1.59 / 1.45 | 1.45 / 1.38 | |
| FP8, FP16 D | 1.80 / 1.60 | 1.82 / 1.62 | 1.57 / 1.42 | 1.44 / 1.37 | |

Over the model decode shapes alone, which are large and read the weights at
close to the memory's bandwidth either way, BF16 and FP16 gain 1.10 / 1.04
(`n = 1`) to 1.02 / 1.00 (`n = 5`), and FP8 1.42 / 1.19 (`n = 1`) to
1.20 / 1.12 (`n = 4`). On 360 further random shapes, not used to choose the
rows, 1.1% of the GEMMs the rows take are more than 10% slower than before
(the worst 0.81x), at m from 600 to 5000 with K from 10,000 to 30,000.

### Tensile metadata

Each `.s` embeds its Tensile `custom.config`, generated from the Tensile YAML
for its kernel family, type and `n`, which lists that `n`'s kernels:
`custom_vllm_wvsplitk_{bf16,f16}_tn_bias_n<N>_gfx1201.yaml` or
`custom_vllm_wvsplitkq_f8_{bf16,f16}_tn_bias_n<N>_gfx1201.yaml`, whose
`_tn_n<N>_gfx1201.yaml` siblings run the same kernels without a bias.
`AddCustomConfig` writes `SupportsBias: false`; these kernels set it to `true`.

```bash
python -m tensilelite.AddCustomConfig \
  tensilelite/CustomKernels/vllm/wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.s \
  --yaml tensilelite/Tests/custom/custom_vllm_wvsplitk_bf16_tn_bias_n1_gfx1201.yaml \
  --origin vllm \
  --repository https://github.com/vllm-project/vllm \
  --version 392ed873c7fed4c41fafaad022f83f7d0fe9f57a
```
