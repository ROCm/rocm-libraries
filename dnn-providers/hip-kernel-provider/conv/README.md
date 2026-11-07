<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# The conv engine group

The hip-kernel-provider's convolution engines: plain HIP kernels, compiled at runtime through
HIPRTC for whatever device a plan is built on, and reached through the kernel ingestor. One
source serves every architecture, from gfx906 through current CDNA and RDNA.

The group exists so that every convolution hipDNN is asked for can be served without the
MIOpen provider, and so that each convolution algorithm is its own engine, which hipDNN's
heuristics (UHD) and autotune choose between directly. Kernels brought over from MIOpen
(and split out of `ConvHipConv`) land here, as engines of their own, until faster engines,
in rocKE and elsewhere, replace them.

## Engines

| Engine family | Engines | Status | What it is |
|---|---|---|---|
| Naive | `hipkernel:CONV_NAIVE_FWD`, `hipkernel:CONV_NAIVE_BWD`, `hipkernel:CONV_NAIVE_WRW` | In progress | Direct convolution, one thread per output element, fp32 accumulation. Correct for every layout, group count, padding, stride, dilation, 1-D/2-D/3-D shape and both convolution modes, in fp32, fp16 and bf16. The coverage floor: accepted only for problems small enough that its speed does not matter. |
| Explicit GEMM | lives in the BLAS provider (`hipblaslt-provider`), not here | Planned | im2col/col2im plus a rocBLAS or hipBLASLt GEMM, and every special case that needs no repack. |
| Direct | — | Planned | MIOpen's HIP direct kernels. |
| Depthwise VALU | `hipkernel:CONV_DEPTHWISE_VALU_FWD`, `hipkernel:CONV_DEPTHWISE_VALU_BWD` | In progress | hipconv's hand-written VALU depthwise kernels (from rocm-libraries PR #10945): a W-strip floor for any filter, stride and dilation, plus register-halo, fused and LDS-staged configs for unit-stride 3×3 to 9×9 and 3×3×3, and channels-first LDS plane-tile configs for unit-stride 3×3, 5×5 and 7×7. NHWC and NCHW, 1-D/2-D/3-D, fp32/fp16/bf16, every architecture. Backward data runs as the forward convolution of dy with the filter reflected for unit stride, and at any stride as a gather over the taps (no zero upsampling). Each config is a catalog entry admitted on correctness alone; autotune (UHD later) picks among them, with the floor the default. |
| Depthwise VALU backward weights | `hipkernel:CONV_DEPTHWISE_VALU_WRW` | In progress | hipconv's Hankel depthwise wgrad schedule reimplemented on vector ALU: a partial kernel sums each partition of the reduction (images x output rows) per channel and tap into an fp32 workspace, and a fold kernel adds the partitions in a fixed order and rounds once, so the result is deterministic with no atomics. A floor for any filter, stride, dilation and padding in 1-D/2-D/3-D, plus register-tile configs (3×3, 5×5, 7×7, 1×4, 1×31, 3×3×3) that keep every tap's sum in registers. NHWC and NCHW, fp32/fp16/bf16, every architecture. |
| hipconv families | — | Planned | `ConvHipConv`'s kernel families (direct, direct_l1, direct_wgrad, patch embed, grouped, Toeplitz and Hankel depthwise), one engine each instead of one opaque solver, then generalized beyond CDNA3/4/5. |
| Winograd | — | Planned | F(2,3)/F(3,2), written fresh in HIP or FlyDSL; MIOpen's are hand-written assembly. |

Every engine applies its own poor-fit rule as part of applicability, so it declines problems
it would serve badly. For naive that is the workload budget, `NAIVE_MACS_PER_LANE_BUDGET` in
`ConvNaiveProblem.hpp`.

An engine with several kernel configs lists each in its catalog, admitted by its kernel match
on correctness alone. There is no shape-to-config routing table: the catalog's declared order
(most general first) is the default, and autotune, then UHD, choose per shape and device.

## Layout

| Piece | Path, relative to the provider root |
|---|---|
| This overview | `conv/` |
| Descriptors, per family | `src/engines/kernel_ingestor_engine/descriptors/conv/<family>/` |
| Kernel sources | `src/engines/kernel_ingestor_engine/descriptors/conv/<family>/kernels/`, beside the descriptors that name them. An embedded source must sit at its descriptor's folder plus `source_file`, which `descriptor-packaging/tools/hkp_verify_embedded_sources.py` checks at build time. |
| Native packs (matching, workload rule, dispatch) | `src/engines/kernel_ingestor_engine/packs/Conv<Family>*.cpp` |
| Kernel embedding (`HKP_CONV_KERNEL_SOURCES`, `HKP_CONV_KERNEL_KEYS`) | `src/engines/kernel_ingestor_engine/CMakeLists.txt` |
| Tests | `src/tests/engines/kernel_ingestor_engine/packs/TestConv<Family>*.cpp` |

## Building

Off by default. Enable with `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_CONV=ON`.
With the option OFF, the packs are not compiled, the kernels are not embedded and the
`conv/` descriptor folder is not packed (`HKP_DESCRIPTOR_FAMILIES` in
`descriptor-packaging/cmake/HkpPackaging.cmake`).

Plans built from these engines cannot be saved and loaded: plan serialization covers kpack
code objects only, and these kernels are compiled from embedded source.

## Adding an engine family

1. Put the HIP source under `descriptors/conv/<family>/kernels/`, with every shape a runtime
   argument so that one compile per element type serves every problem.
2. Register it for embedding in `HKP_CONV_KERNEL_SOURCES` and `HKP_CONV_KERNEL_KEYS`. A
   source's key is its path relative to the descriptor (`kernels/<File>.cpp`); a header is
   keyed by its bare file name, so give each header a name unique across the provider.
3. Author the family's descriptors under `descriptors/conv/<family>/`, with
   `kernel_source.kind` `embedded_source` and the key as `source_file`.
4. Add the native pack (`packs/Conv<Family>Native.cpp`) and its row in `IngestorPacks.cpp`,
   inside `#ifdef HIPDNN_ENGINE_CONV`, and its sources to the `HIPKERNELPROVIDER_ENABLE_CONV`
   block of the engine's CMakeLists.txt.
5. Test every field its matcher reads with a refusal case, and its numerics on the device
   against hipDNN's CPU reference convolution.
