<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# CONV_NAIVE_FWD / BWD / WRW — coverage-first production kernels for any convolution the faster engines do not serve

> Engine ids: `hipkernel:CONV_NAIVE_FWD`, `hipkernel:CONV_NAIVE_BWD`, `hipkernel:CONV_NAIVE_WRW` · Generator: none (the three `conv_naive_*.kdp.json` catalogs are hand-written and kept in git) · Kernels: `src/engines/kernel_ingestor_engine/descriptors/conv/naive/kernels/` · Pack: `packs/ConvNaiveProblem.cpp`, `packs/ConvNaiveNative.cpp`

## At a glance

- **What:** direct convolution, one thread per output element (forward, backward data) or one 256-lane block per dw element (backward weights), fp32 accumulation, every shape a runtime argument.
- **Covers:** fwd · bwd-data · wrw; fp32, fp16, bf16 (all three tensors one type); any element strides (NCHW, NHWC, sliced or padded views); 1-D, 2-D, 3-D; any groups (depthwise included); any padding (asymmetric), stride and dilation; both convolution modes. No TF32 sibling. Admitted only within the workload budget: at most `NAIVE_MACS_PER_LANE_BUDGET` = 5000 multiply-accumulates per SIMD lane (about 1 ms at 1 GHz), so the admitted size scales with the device's compute units.
- **Arches:** every architecture HIPRTC targets; plain VALU code, no matrix cores.
- **Wins when:** almost never on time. It is the best engine on 0-4 of the 1998 graphs per GPU (tiny reductions: 1x1 layers with few channels, small 3-D depthwise layers).
- **Loses when:** everywhere a performance-first engine applies. It is a production engine designed for coverage first, not speed first: it runs real workloads wherever no faster engine serves the graph. It is not a test reference; hipDNN's GPU reference kernel and CPU reference cover that.
- **Precision:** fp32 sums in a fixed order, one rounding at the store; padding taps are skipped. Deterministic. No numerical notes.
- **Instances:** 9 configs (3 directions x 3 dtypes), one HIPRTC compile per (direction, dtype) serving every shape.
- **vs MIOpen:** the counterpart of `ConvDirectNaiveConv{Fwd,Bwd,Wrw}`. MIOpen admits those everywhere and lets Find reject them; this engine refuses poor fits up front.

## Why it exists

- Every convolution graph a caller can build must have at least one engine without the MIOpen provider loaded. The naive engines are that floor for real workloads: they read every field of the graph and accept any shape the convolution defines, as long as it fits the workload budget.
- Coverage first, performance second: the other engines are designed around performance and cover the shapes where that pays; naive guarantees that the remaining shapes still run, at a predictable cost.
- Correctness first, at minimal code: about 300 lines of kernel, no workspace, no configuration space, nothing to tune.
- The poor-fit problem MIOpen has with its naive solvers (applicable to almost everything, so selection must keep them out of slow picks) is solved here inside applicability, by the workload budget. hipDNN selection is unchanged.

## How it works

Forward and backward data: thread `t` owns output element `t` (grid-strided, 64-bit index), decodes its coordinates, and loops over the reduction channels of its group and every filter tap:

```
y[n,k,o]  = sum over c in group(k), taps f with in-range input of  x[n,c,o*s - pad + f*dil] * w[k,c,f]
dx[n,c,i] = sum over k in group(c), taps f with (i + pad - f*dil) divisible by s of  dy[n,k,(i + pad - f*dil)/s] * w[k,c,f]
```

Backward data is a gather, not a scatter: each dx element is written exactly once, including those no output reaches (they get 0), with no zero upsampling and no atomics.

Backward weights: one 256-thread block per dw element. Lane `j` sums the (image, output position) pairs whose flat index is `j` mod 256; the block then adds the 256 partial sums in a fixed binary tree in LDS. The reduction is spread over a block without atomics or a workspace, and the result is bitwise repeatable.

True convolution reflects the filter index (`filterIndex`), so both modes run the same loops.

## Implementation design

**Kernel** (`ConvNaive.cpp`, entry points `ConvNaiveFwd`, `ConvNaiveBwd`, `ConvNaiveWrw`):
- Compiled once per element type (`-DCONV_NAIVE_TYPE=float|_Float16|__bf16`); `ConvNaiveArgs` carries every extent, padding, stride, dilation and the three tensors' strides as 64-bit values, so one code object serves every problem of its dtype and direction.
- 1-D and 2-D problems are padded to 3-D on the host (leading extents 1, stride 0).
- Loads convert to fp32; the store converts once to the element type.

**Host gates** (`parseConvNaiveProblem`, every field read and gated):
- one convolution node of the engine's direction; no override-shape graphs (the argument block is fixed at plan build);
- conv mode CROSS_CORRELATION or CONVOLUTION; three distinct tensor uids;
- each tensor: non-virtual, not pass-by-value, not ragged; fp32, fp16 or bf16; alignment at least one element; rank 3 to 5; positive extents; non-negative strides;
- x, w and y share one data type, and the node's compute type is UNSET or FLOAT (HALF also for fp16 tensors), following cuDNN's compute types;
- the written tensor (y, dx or dw) has distinct addresses for distinct elements (an overlap would race);
- padding (pre and post), stride and dilation vectors of the spatial rank, padding >= 0, stride and dilation >= 1;
- groups implied by the filter's channel count (C a multiple of it), K a multiple of the groups, y's batch and channels matching, and y exactly the extents the convolution produces;
- every product and extent checked for 64-bit overflow.

**Workload budget** (`withinWorkloadBudget`):

```
lanes        = compute units x 64            (RDNA reports WGPs; counted as two CUs each;
                                              16 CUs assumed when the device reports none)
outputLanes  = output elements               (fwd, bwd)  |  dw elements x 256  (wrw)
perLane      = ceil(MACs / min(lanes, outputLanes))
admitted    <=> perLane <= 5000
```

5000 MACs per lane is a starting value: about 1 ms at a conservative 1 GHz and 0.005 MACs per lane-cycle, the low end of what MIOpen's naive kernels measured on gfx950 and gfx1151 when their own work gate was calibrated. On a 304-CU MI325X that admits roughly 97 M MACs once the output fills the device; on a 60-CU MI50 about 19 M.

**Launch:** 256-thread blocks; at most 2^20 blocks, the kernels stride over the rest. No workspace, one launch.

## Fusion

`CONV_NAIVE_FWD` applies any fused pointwise epilogue the generator emits
([conv README](../../README.md#fused-pointwise-epilogues)) at its single store, on the fp32
accumulator, with the output channel in scope. It claims a fused graph exactly when it claims
the graph's convolution alone: the workload gate and every other gate read the convolution, so
fusion never widens the engine's coverage. Backward data and backward weights take no epilogue.

**Known cost.** A fused graph this engine wins takes longer than the bare convolution's best time by, geomean with a bias first: MI325X (channels-last) relu 1.10x, gelu 1.09x, silu 1.08x; gfx950 (both layouts) relu 1.09x, gelu 1.10x, silu 1.10x. The exp2-based activations cost the most where the epilogue is a large share of the kernel. Against the convolution plus the pointwise chain as a separate one-pass kernel, the fused graph is faster by 1.52x on MI325X and 1.54x on gfx950 (usage-weighted; the conv README's fusion table).

## Strengths and weaknesses

Measured with autotune (fresh cache), our engines only, against MIOpen in search mode, on a 1,998-shape subset (31 models) of an internal real-model corpus; gfx942 is MI325X.

| GPU | fwd graphs admitted (of 771) | bwd (of 634) | wrw (of 593) | graphs where naive is the best engine |
|---|---|---|---|---|
| gfx950 | 189 | 111 | 107 | 4 |
| gfx942 (MI325X) | 200 | 117 | 113 | 4 |
| gfx90a | 151 | 82 | 80 | 3 |
| gfx908 | 156 | 87 | 81 | 1 |
| gfx906 | 127 | 67 | 62 | 4 |
| gfx1201 | 127 | 67 | 62 | 3 |
| gfx1100 | 130 | 69 | 65 | 0 |

- Admission scales with the device: the same graph can be admitted on a large part and declined on a smaller part of the same architecture.
- Where naive is best the graphs take 8-130 µs (1x1 layers with 1-4 output channels, 3-D depthwise at batch 2); there its speedup over MIOpen (MIOpen time / ours) ranges from 0.3x to 2x, against MIOpen's own naive, 1x1 asm or Winograd kernels. These are noise-level in absolute time.
- No subset graph depends on it: on every one of the 1998 graphs some other hipkernel or BLAS engine is also admitted.

## Numerical precision

- fp32 accumulation (fp16/bf16 operands convert to fp32 exactly), in a fixed order (reduction channel, then depth, row and column taps), one rounding to the output type.
- Padding taps are skipped rather than multiplied by zero, so an infinity or NaN in the input reaches exactly the outputs whose real taps read it. With an infinite filter value, a reference that multiplies zero padding gives NaN at border outputs where naive gives ±inf; no test covers infinite filters.
- Determinism: forward and backward data have one writer per element and a sequential loop; backward weights adds lane partials in a fixed tree. The device suite relaunches every case and compares bits.
- Numerical notes: none. The engine stays exact and has no TF32 sibling.
- Tested against hipDNN's CPU reference with a tolerance of 1e-6 per reduction term plus 2 storage epsilons of the value.

## Comparison with MIOpen

- MIOpen's `ConvDirectNaiveConvFwd/Bwd/Wrw` cover more types (int8, fp8, bf8, and mixed types where two of three tensors share a type) but only packed NCHW/NCDHW and NHWC/NDHWC layouts, and only tensors whose lengths fit 32 bits.
- MIOpen's forward naive kernel launches a workgroup per (image, output channel) for NCHW or per (image, output row) for NHWC and loops over the plane inside; this engine launches one thread per output element (one block per dw element), with any strides.
- MIOpen admits its naive solvers on almost every problem and relies on Find timing to reject them; with immediate mode or a missing find-db entry a slow naive pick is possible. Here the budget declines the poor fits at applicability, so neither autotune nor the heuristic ever sees them.
- Mixed-type convolution is served by MIOpen only through its naive kernels; this engine requires one type (the group's documented gap).

## Prior art considered

- MIOpen's naive kernels (`naive_conv_*`) and their applicability rules; the per-lane workload idea comes from the calibration of a work gate for those kernels.
- No code was copied; the kernel is written fresh for hipDNN's strided-tensor model.

## Next steps and deferred ideas

1. **Calibrate the budget per architecture.** 5000 is a single starting value. Cross-arch timing of the three kernels would let the budget follow measured lane throughput (VALU rate differs between CDNA and RDNA, and wrw differs from fwd). Payoff: fewer slow admissions on small parts, more coverage on large ones.
2. **Make the device dependence explicit in coverage tooling.** The budget counts compute units, so parts of one architecture with fewer CUs decline graphs a larger part admits (seen on gfx906, 60 vs 64 CUs). The coverage census already treats naive-only rows as device-dependent; record each architecture's map on its smallest part.
3. **Cheap speedups that keep the coverage-first design.** Vectorized loads along the contiguous dimension and per-thread register blocking over a few output channels, without adding a configuration space; raising lane throughput lets the budget admit more shapes.
4. **Mixed types and integer types.** If hipDNN takes on mixed-type convolution, the naive kernels are the natural first engine for it (separate element types for x, w and y).

## Tests

- `src/tests/engines/kernel_ingestor_engine/packs/TestConvNaiveProblem.cpp` (host): acceptance in every direction; one refusal per gated field; argument packing (2-D padded to 3-D, strides in N C order, thread counts per direction); compute types; the workload budget (small accepted, over-budget declined, lanes the problem can occupy, fallback CU count, two CUs per reported RDNA WGP); each graph matcher serves only its direction; the kernel matcher requires the graph's dtype and direction; no workspace.
- `TestGpuConvNaive.cpp` (device): NCHW and NHWC 2-D, padded, 1-D with stride and dilation, 3-D channels-last with mixed strides and dilation, grouped, depthwise multiplier 2, strided with asymmetric padding, and true-convolution cases, in fp32, fp16 and bf16 for all three directions, against the CPU reference (filter reflected on the host for convolution mode). Each case is relaunched and must reproduce the same bits.
- `src/integration_tests/kernel_ingestor_engine/IntegrationConvCoverageCensus.cpp`: naive is the structural floor (naive wrw serves every scaled pair whose fwd/bwd original naive serves), with naive-only rows treated as device-dependent.
