<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Conv engine design docs

One document per engine family. Each opens with an **At a glance** summary and then covers why the engine exists, how it works, its implementation, strengths and weaknesses, numerical precision, how it compares with MIOpen, the prior art considered, next steps, and its tests.

| Doc | Engines | One line |
|---|---|---|
| [IGEMM](IGEMM.md) | `CONV_IGEMM_FWD/BWD`, `CONV_IGEMM_FWD/BWD_TF32` (K1) | Implicit GEMM on MFMA/WMMA or FMA micro-tiles; the general forward and backward-data workhorse. |
| [IGEMM_NCHW](IGEMM_NCHW.md) | `CONV_IGEMM_NCHW_BWD`, `CONV_IGEMM_NCHW_BWD_TF32` (K7) | Channels-first backward data with col2im fused, in place. |
| [DIRECT](DIRECT.md) | `CONV_DIRECT_FWD/BWD`, `CONV_DIRECT_FWD/BWD_TF32` (K2) | Stride-1 direct convolution on matrix cores: halo tiles, and filter-row reuse with batch packing. |
| [WRW](WRW.md) | `CONV_WRW`, `CONV_WRW_TF32` (K3) | Backward weights as a row-direct implicit GEMM with a deterministic split reduction. |
| [WINOGRAD](WINOGRAD.md) | `CONV_WINOGRAD_FWD/BWD/WRW`, `CONV_WINOGRAD_FWD/BWD_BF16`, `CONV_WINOGRAD_FWD/BWD_F43` (K5) | Winograd 3x3: fused, cooperative and multipass forms in all three directions. |
| [Dilated Winograd](WINOGRAD.md#dilated-winograd) | `CONV_WINOGRAD_ND_FWD` | Native gfx1250 exact-fp32 F(2,3) on dilation-residue subgrids, with bounded spatial workspace. |
| [GROUP](GROUP.md) | `CONV_GROUP_FWD/BWD/WRW`, `CONV_GROUP_FWD/BWD_TF32` (K4) | Grouped convolution on block-diagonal matrix-core tiles. |
| [GROUP_VALU](GROUP_VALU.md) | `CONV_GROUP_VALU_FWD/BWD/WRW` | Grouped and small-group convolution on the vector ALU, lanes along the row; 1-D, 2-D and 3-D. |
| [DEPTHWISE](DEPTHWISE.md) | `CONV_DEPTHWISE_{MFMA,ROWS,VALU}_FWD/BWD`, `CONV_DEPTHWISE_VALU_WRW` | The depthwise engines, with a selection guide. |
| [SMALLC](SMALLC.md) | `CONV_SMALLC_FWD/BWD` | Few-channel stems forward, and stride-1 backward data into few channels. |
| [STEM_BWD](STEM_BWD.md) | `CONV_STEM_BWD` | Backward data for stems (up to 4 input channels) using stride-phase super-pixels. |
| [NARROW](NARROW.md) | `CONV_NARROW_FWD/BWD/WRW` | At most 8 channels on one side (output layers, stem gradients): taps on the GEMM's N axis, the wide tensor read once. |
| [FFT](FFT.md) | `CONV_FFT_FWD/BWD/WRW` | Tiled overlap-save FFT convolution for large filters and large-kernel depthwise. |
| [NAIVE](NAIVE.md) | `CONV_NAIVE_FWD/BWD/WRW` | Coverage-first production kernels for graphs no faster engine serves, admitted within a workload budget. |

The explicit-GEMM engines (`ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` and their siblings) live in the hipblaslt-provider: [EXPLICIT_GEMM](../../../../hipblaslt-provider/docs/engines/EXPLICIT_GEMM.md).

The short names K1 to K5 and K7 in the table are the engine families' names in code: kernel and
host identifiers (`k1ConvMma`, `K3Tiles`, `K5Args`, `K7Block`) and compile-time defines
(`K1_BUFFER`, `CONV_K3_LOOP`) use them, and the docs use them as shorthand. K1 is the implicit
GEMM (`CONV_IGEMM`), K2 the halo-tile direct (`CONV_DIRECT`), K3 backward weights (`CONV_WRW`),
K4 small-group convolution (`CONV_GROUP`), K5 Winograd (`CONV_WINOGRAD`) and K7 channels-first
backward data (`CONV_IGEMM_NCHW_BWD`). The other engines have no short name.

Instance counts per engine are in [HEURISTICS.md](../../HEURISTICS.md). For regenerating catalogs, see [REGEN.md](../../REGEN.md).

Performance figures in these docs come from sweeps over an internal real-model convolution corpus (9,471 shapes) and a 1,998-shape subset of it: autotuned, with only our engines running, compared against MIOpen. Each doc names the MIOpen mode it compares against (search or exhaustive). Figures marked indicative may not match the current catalog.
