# Explicit-GEMM convolution — ROCBLAS_CONV_ENGINE and HIPBLASLT_CONV_ENGINE

> Engine ids: `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE`; siblings `ROCBLAS_CONV_TF32_ENGINE`, `HIPBLASLT_CONV_TF32_ENGINE`, `ROCBLAS_CONV_RPR_ENGINE`, `HIPBLASLT_CONV_RPR_ENGINE`, and the deterministic `ROCBLAS_CONV_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_ENGINE_DETERMINISTIC`, `ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC`, and the opt-in `HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE` · Builder: `engines/conv/ConvGemmPlanBuilder.cpp` · Formulation: `common/ConvGemmFormulation.*`, `common/ConvRepackFormulation.*`, `common/GemmFormulation.*` · Kernels: `kernels/ConvRepack.cpp` (HIPRTC) · Registration: `HipblasltContainer.cpp`

## At a glance
- **What:** a convolution becomes one or more strided-batched BLAS GEMMs. The GEMM reads the tensors directly where it can (1x1, point output). Otherwise HIP repack kernels move data around it: im2col, col2im, channels-last staging and fp32 partial sums. Both engines share one backend-neutral layer; the backend only runs the GEMMs.
- **Covers:** fwd, bwd-data, wrw · fp32/fp16/bf16, with one type for all three tensors · compute type unset or fp32, and fp16 for fp16 tensors (cuDNN's set) · 1-D/2-D/3-D (rank 3-5) · any x/y strides · filter KCRS or KRSC · groups, including depthwise · any padding (asymmetric too), stride and dilation · both conv modes · tensors larger than 4 GiB · forward conv + pointwise epilogue graphs (see Fusion).
- **Arches:** `ROCBLAS_CONV_ENGINE` runs everywhere rocBLAS ships. `HIPBLASLT_CONV_ENGINE` runs wherever hipBLASLt is built (gfx908+, gfx11, gfx12; TheRock excludes gfx906 and gfx103X).
- **Build flags:** `HIPBLASLTPROVIDER_ENABLE_CONV` (default OFF). `HIPBLASLTPROVIDER_ENABLE_ROCBLAS` (default OFF) and `_HIPBLASLT` (default ON) choose the backends.
- **Wins when:** pointwise (1x1) in every direction, and many 3-D, patchify and 1-D shapes. On the 9,471-shape corpus with MIOpen excluded, one of the two is our fastest engine on 37-52% of graphs per GPU and layout.
- **Loses when:** dense 3x3 (Winograd and direct MMA engines win), small-channel stems, large-kernel depthwise (FFT), and tiny shapes where the extra repack launches dominate.
- **Precision:**
  - Accumulation is fp32 with one final rounding.
  - Every multi-GEMM sum goes through fp32 partials in a fixed order.
  - rocBLAS calls that would round partials to 16 bits are declined.
  - The default engines report `NONDETERMINISTIC` (the libraries may split a reduction across workgroups); their `*_DETERMINISTIC` siblings do not (see Numerical precision).
- **Workspace:** the repack workspace is held to 1 GiB. Above that, a 16-bit channels-last wrw runs blocks of w's columns over every image; everything else slices by images, then by bands of rows. hipBLASLt gets up to 128 MiB more. rocBLAS uses its handle's own scratch.
- **Large tensors:** every size is 64-bit on the host. A call is split until each piece fits the libraries' 32-bit arguments: per image when the batch stride passes 2^31, and in chunks when n or k passes 2^30.
- **Siblings:**
  - `*_TF32`: permits TF32 on fp32 GEMMs. On rocBLAS this is xf32, gfx942 only. On hipBLASLt it is FAST_TF32: native on gfx942, bf16x3 on gfx950, fp32 fallback elsewhere. Reports `TENSOR_CORE`.
  - `*_RPR`: MIOpen-style per-image fp16 1x1 wrw. Reports `REDUCED_PRECISION_REDUCTION`.
  - `*_DETERMINISTIC`: the exact and TF32 engines with rocBLAS atomics not allowed, or no hipBLASLt workspace; bitwise reproducible.
  - All are offered by default; a caller excludes them by note through the hipDNN API's cuDNN-compatible `deselect_numeric_notes`.
- **vs MIOpen:** competes with `GemmFwd1x1_0_1/_0_2`, `GemmFwdRest`, `GemmBwd1x1_stride1/2`, `GemmBwdRest`, `GemmWrw1x1_stride1` and `GemmWrwUniversal`. On an 86-shape test set, MIOpen Gemm* time / ours is 1.06x to 2.2x geomean per GPU (indicative).

## Why it exists
- **Every-arch floor.** The engines give every arch, gfx906 included, a correct and reasonably fast path for any convolution shape without CK. MIOpen's GEMM solvers fill this role today, but they are tied to MIOpen and to rocBLAS.
- **Separate engine IDs.** The conv engines are separate from `HIPBLASLT_ENGINE`, so conv can be built, gated and ranked without touching matmul.
- **Redesign, not a port.** The explicit-GEMM design (RFC 0002 §5 rows, §8 predicates) is implemented here in the BLAS provider rather than as MIOpen solvers. It drops MIOpen's per-image loops wherever one GEMM can address every image.
- **Shared layer.** rocBLAS and hipBLASLt each win shapes the other loses, so both backends sit behind one formulation layer.

## How it works
`convGemmCalls` picks a plan in this order. Each step's predicate is shape-only. `gemmForViews` then decides whether a BLAS GEMM can address the operands' strides, so an unreadable layout is declined, never misread.

1. **Direct rows, no workspace.**
   - `ONE_BY_ONE`: 1x1, unit stride, unpadded. The GEMM runs over pixels and channels. Channels-last runs as one call batched over groups. Channels-first batches images when there are no groups, and runs one call per image when there are.
   - `POINT_OUTPUT`: the filter spans the whole unpadded input, giving a GEMM over images.
2. **1x1 channels-first wrw over several images.** No single GEMM reads the (image, pixel) reduction here, so there are two plans:
   - *Batched partials:* one fp32 dw partial per image or pixel chunk, then a cast adds them in order and rounds once.
   - *Staged channels-last:* x and dy are copied channels-last, then one GEMM runs.
   - By default the plan that moves fewer bytes wins.
3. **IM2COL.**
   - **Column matrix:** im2col writes it to the workspace, which turns the problem into a 1x1 over C·RS channels. Column order follows the filter's own layout (KCRS or KRSC), so w is read in place. The column layout follows y's layout by default; channels-first y keeps a long reduction over positions contiguous.
   - **bwd-data:** the GEMM writes fp32 columns, and col2im sums them into dx with one rounding. A filter whose taps never overlap (stride ≥ dilated extent, a single tap included) keeps its columns in dx's type and uses a copying col2im.
   - **y not readable:** if the GEMM cannot read y as given (channels-first wrw over several images), y is staged channels-last.
   - **Over the cap:** when the column matrix exceeds the cap, the plan runs a slice of images at a time. If one image alone exceeds it, the plan runs bands of rows along the outermost spatial axis; bwd-data bands dx rows. A plan is declined if a single row's column matrix exceeds the cap.
   - **16-bit wrw over the cap** (`columnBlockWeights`): blocks of w's columns instead of image slices. Each block's column matrix over every image fits the cap, and one GEMM over every image writes that block of w once, in w's type. Blocks follow the filter's column order:
     - K C R S: ranges of input channels with all their taps;
     - K R S C: ranges of filter rows (outermost axis first), then single taps, then channel ranges of one tap.

     The block's padding is shifted so its taps read the same input positions, in both convolution modes. Image slices would sum through an fp32 accumulator, so every GEMM would be a 16-bit-into-fp32 one, which gfx11 runs far behind its 16-bit-output GEMMs (see Backends). fp32 (w is its own accumulator), grouped convolutions, and a channels-first y over several images keep image slices; image slices stay a benchmarking candidate.
4. **Reduction rewrites** (`batchReductionChains`, `sumLongReductionsInFp32`).
   - **Small wrw output, long reduction:** a chain of accumulating GEMMs, or one GEMM with a long k, becomes batched GEMMs over groups × images × k-chunks into fp32 partials. A cast then sums them in order and rounds once. Chains whose outputs have up to 2^20 elements are rewritten. A long k is also cut into chunks when the output has at most 2^16 elements.
   - **16-bit wrw output:** image-sliced or banded wrw, and any wrw GEMM that rocBLAS would k-chunk, accumulate in an fp32 copy of w that is rounded into w once. Column blocks write w directly: each block's GEMM finishes its reduction.
   - **Split-reduction candidate:** the benchmarking builder may also cut the reduction into enough fp32 partials to fill about 512 output tiles of 128x128.
5. **Library limits** (`fitLibraryLimits`). A batch stride beyond 2^31 elements becomes one call per matrix. n beyond 2^30 becomes column chunks. k beyond 2^30 becomes accumulating chunks, which is allowed only into fp32 d.

## Fusion
Every engine but the RPR siblings (backward weights only) also claims a forward convolution followed by a chain of pointwise nodes, under its existing engine id. The chain is read by the parser every provider shares (`hipdnn_plugin_sdk/ConvEpilogueMatch.hpp`): scale and bias (per-channel or scalar, device or pass-by-value), relu (clipped, clamp, leaky), exact and tanh GELU, swish (silu), sigmoid, tanh, elu, abs and softplus. The convolution and the whole chain stay in fp32 and y is rounded once. `fuseEpilogue` (`common/ConvEpilogueFormulation.*`) places the chain, in order of preference:

| Form | Where the chain runs | When |
|---|---|---|
| Native | The GEMM that writes y, in y's type: a leading per-channel bias, then hipBLASLt's sigmoid (16-bit y) or a relu / clamp tail | a chain of a bias, then at most a sigmoid or a relu / clamp whose bounds y's type holds exactly; every GEMM writing y whole (one group, no accumulation) |
| Fused pass | The plan's existing last pass: the GEMMs write the channels-last staging in fp32, and the copy into y becomes the epilogue's cast | y is staged channels-last (the `STAGED_CHANNELS_LAST` layout candidate for a channels-first y) |
| Extra pass | Every GEMM (and any fp32 partials sum) writes an fp32 copy of y in the workspace; one cast applies the chain and rounds into y | everything else: groups, the other activations, a bias of another type |

- **Native form** (`gemmEpilogueFor`, `fuseEpilogue`):
  - **Bias along the rows.** Where the GEMM's rows are y's channels (channels-last 1x1, point output, im2col into a channels-last y), hipBLASLt's `BIAS` epilogue adds it (fp32 or y's type). rocBLAS has no epilogue; such a plan takes the pass.
  - **Bias as an addend.** Where the GEMM's columns are y's channels (channels-first 1x1 and im2col: y_n = x_n^T w^T, pixels as rows), hipBLASLt's bias, a vector along the rows, does not apply. The plan fills one image's matrix of the bias broadcast along the rows in the workspace (`Broadcast*`, `kernels/ConvRepackEpilogueSteps.h`) and adds it as the GEMM's C (beta 1, batch stride 0): one GEMM, one rounding. Both backends. The bias must have y's type, which the C matrix shares; an fp32 bias on a 16-bit y takes the pass, and so does a GEMM with one row or column (the libraries may run it as a GEMV, which reads C as D). Cost: the GEMM reads one more y-sized matrix per image (+2-7% on compute-bound layers, +30-40% on bandwidth-bound ones on MI325X), so autotune times it against the pass (`epilogue_pass` candidate).
  - **Activation in the GEMM.** Only where hipBLASLt computes the chain's value for every input, special values included: sigmoid, for a 16-bit y (an fp32 y takes the few-ulp form in the pass). hipBLASLt has no sigmoid with a bias, so that bias is an addend.
  - **Relu / clamp tail.** A relu or clamp with no slope, whose bounds y's type holds exactly (0 and infinity always), runs after the GEMM as one in-place pass over y in its own type (`Clamp*`, `kernels/ConvRepackEpilogueSteps.h`). Clamping the rounded value gives the rounding of the clamped one, since the rounding is monotonic and keeps the bounds, so y is still rounded once, and NaN is kept. The pass reads y's 16-bit result and writes back only the runs it changed, where the extra pass reads an fp32 copy.
- **Measured** (MI325X, dnn-benchmarking `--autotune`, 20 iterations, best explicit-GEMM engine, on the 113 layers of a fused-graph benchmark set that an explicit-GEMM engine serves, the native form against the pass it replaces; geomean, bf16 / fp16 / fp32):
  - Channels-first bias 1.15x / 1.10x / 1.03x (up to 2.47x), bias + relu 1.13x / 1.10x / 1.04x (up to 2.01x). 1x1 examples: 384x384 c512 to 256 bf16 bias 1.54 to 0.79 ms, bias + relu 1.65 to 0.99 ms; 128x128 c1024 to 2048 fp16 bias + relu 3.17 to 2.09 ms.
  - Channels-last bias + relu 1.25x / 1.21x / 1.14x (up to 2.20x); a bias alone was native already.
  - Against the bare convolution plus the chain as a separate one-pass kernel, channels-first layers lose in 8 of 114 (bias, from 19) and 14 (bias + relu, from 19), and channels-last bias + relu in none (from 11).
  - The library-activation engine against `HIPBLASLT_CONV_ENGINE_DETERMINISTIC` (the same library settings): channels-last 1.12-1.49x geomean by chain and type (swish and tanh GELU most, up to 3.6x), channels-first 1.00-1.14x.
- **hipBLASLt's own activations** (relu, clamp, swish, tanh GELU): their special values differ from the chain's (relu maps NaN to 0, clamp maps NaN to its upper bound, swish and tanh GELU map -inf to NaN, where the chain keeps NaN and gives 0), and their fp32 accuracy is not the few-ulp form. Finite results stay within the fused tolerance. The opt-in `HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE` (`HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS=1`) runs them in the GEMM; it serves only the graphs whose chain maps to one, with only that plan, and is not registered by default, since no numerical note names the difference. Exact GELU has no hipBLASLt epilogue.
- **Known cost.** Outside the native form, a fused plan costs more than its bare convolution by two things:
  - its GEMM writes y in fp32, twice the bytes of the bare GEMM's 16-bit output;
  - one cast/epilogue pass reads that fp32 result and writes y.

  On MI325X (bias + relu, channels-last), the GEMM grows 1.0-2.8x by kernel time; output-bound GEMMs grow the most. Both are the price of rounding once and of giving every engine the same result for the same graph. Autotune weighs fused explicit GEMM against the other conv engines, which fuse in their kernels.
  Where an explicit-GEMM engine wins the fused graph, it takes 1.35-1.42x the bare convolution's best time with an activation (1.01-1.12x for a bias alone), geomean on MI325X (channels-last) and gfx950 (both layouts). Against the bare convolution plus the chain as a separate one-pass kernel it is faster by 1.10x (MI325X) and 1.02x (gfx950), usage-weighted, and slower on 1x1 and patchify layers where the BLAS GEMM wins the bare convolution by a wide margin (down to 0.84x on MI325X, 0.65x on gfx950): the fp32 output and the pass cost more there than a separate pass over the 16-bit output.
- **The pass.** The epilogue cast (`StridedCastEpilogue*`) runs the chain as a run-time menu of steps.
  - On a dense y, it walks memory flat: four elements a thread, one 16-byte load of the fp32 result and one store of the four outputs. The channel is carried along by additions in 32-bit integers.
  - Otherwise it orders its axes so the axis along which y is densest is innermost, and reads the channel off its index walk.
- **Values at execution.** Operand pointers and pass-by-value scalars are bound at execution, so a plan serves the graph's values whatever they are.
- **Workspace.** The extra pass adds y's size in fp32 to the workspace, above the repack cap; the addend one image of y in y's type.
- **Runnable layout.** A fused plan runs on the first form and column layout whose GEMMs the backend can run: the native form, then the same layout with the epilogue in a pass, then staged channels-last, channels-last and channels-first.
- **Shorter reductions.** The deterministic hipBLASLt engines give the library no workspace, and for some small-n, long-k fp16 layers no layout's fp32-output GEMM has a workspace-free algorithm. The plan then cuts each such GEMM's reduction into 2, 4, 8 or 16 consecutive parts, each after the first adding to the fp32 result in order (`splitFp32OutputReductions`): the same sum, rounded once by the epilogue pass, and still deterministic. A fused graph is declined only if none of those runs.
- **Backward directions.** Backward data and backward weights with pointwise nodes are declined.

## Implementation design
- **Plan shape:** a plan is a list of stages. Each stage has repack steps, then GEMM calls in order, then repack steps. All stages share one workspace, laid out in 256-byte-aligned regions with their own uids. `ConvRepackPlan` runs the stages. A direct row runs as the backend's GEMM plan alone.
- **Repack kernels:** one HIPRTC program, compiled once per device, holds every entry point, and every shape is a runtime argument. The host picks a specialized launch per problem; anything else, and any offset beyond 32 bits, uses the generic grid-stride kernel with 64-bit indices.
  - *im2col:* channels-last rows with 16-byte vectors, or one thread per tap for narrow taps; channels-first tiles staged in LDS with a halo; run-per-thread for disjoint taps.
  - *col2im:* rows or staged groups, 4-wide fp32 reads.
  - *Copies:* an LDS 32x32 tiled transpose for channels-first to channels-last.
  - *Cast:* a wide reduction with a fixed tree over many partials.
  - Every sum has one owning thread, so no kernel uses atomics.
- **Benchmarking:** when `global.benchmarking` is on and a plan has a choice, `BenchmarkedPlan` times each distinct candidate on the first execute and keeps the fastest. Each candidate gets a warm-up, then the best of a few runs. The candidates are the default, channels-first, channels-last, staged channels-last, image slices (a 16-bit wrw over the cap through an fp32 accumulator), split reduction, and the wider wrw cap with column blocks and with image slices. A candidate the backend declines is skipped. Without benchmarking, the default plan runs.
- **Backends:**
  - **hipBLASLt** takes the heuristic's top algorithm per call, with up to 128 MiB of workspace. If hipBLASLt finds no algorithm, the engine is not applicable.
  - **rocBLAS** runs `rocblas_gemm_strided_batched_ex`.
  - **rocBLAS precision declines:** the rocBLAS plan throws, which makes the engine not applicable, when rocBLAS would round more than once.
    - *k-chunking:* `rocblasRoundsReductionChunks`. A non-fp32 d whose reduction walks more than 2^31 bytes of a's leading dimension gets split by rocBLAS itself, with partials kept in d's type. Splitting m, n or the batch cannot help, so the call is declined and hipBLASLt serves it.
    - *bf16 GEMV:* `rocblasGemvRoundsBf16Products`. On gfx906/908/90a, gfx10 and gfx11, bf16 GEMV-shaped calls (K/G = 1 grouped) go to `gemvtsm`, which rounds every product to bf16.
  - **rocBLAS untuned path** (`requireTunedRocblasKernels`, `rocblasFp32OutputIsUntuned`): on gfx11, rocBLAS ships tuned Tensile libraries for 16-bit inputs with a 16-bit output, but only fallback libraries for an fp32 output. The conv engines decline a 16-bit GEMM into fp32 with a not transposed and b transposed. Measured on gfx1101, fp16 and bf16:
    - that form runs at 0.2-1 TFLOP/s, 25-180x behind the 16-bit-output GEMM of the same shape (4608 x 512 x 114688: 2.2 s against 15 ms);
    - the other forms run 3-7x behind and are served.
  - **hipBLASLt on gfx11** has only small-tile kernels for 16-bit into fp32. The not-transposed forms run at 4-6 TFLOP/s on gfx1101, against 23-36 TFLOP/s into 16 bits; both a and b transposed runs at 30. Nothing is declined: column blocks keep 16-bit wrw off that path, and autotune times the rest.
- **Applicability gates** (`parseConvProblem`, a field-by-field copy of the hip-kernel-provider naive rules):
  - one node and no override shapes;
  - compute type in cuDNN's set; mode set;
  - three distinct real device tensors of one type;
  - rank 3-5, positive extents, non-negative strides;
  - consistent channel and group counts, and y exactly the size the parameters produce.
  - The formulation then declines: a filter in neither KCRS nor KRSC order; a column-matrix row over the cap; a leading dimension beyond 32 bits; a reduction beyond 2^30 into a 16-bit output.
- **Narrow partials** (`NarrowPartials`): hipBLASLt on gfx11 rounds the parts of its own long-k split to a 16-bit output. So on gfx11, `HIPBLASLT_CONV_ENGINE` runs fwd and bwd-data GEMMs with k ≥ 2^16 into 16-bit outputs as fp32 partials. rocBLAS never needs this, because its chunking is declined instead.

## Strengths and weaknesses
- **Wins and coverage.** Measured with autotune (fresh cache), our engines only, on the full 9,471-shape internal real-model corpus in NCHW and NHWC: one of the two engines is our fastest on 3,493-4,918 graphs per GPU and layout (gfx950, MI325X, gfx1201, gfx1100):
  - pointwise fwd 1,095-1,806 of 1,838 and pointwise bwd 695-920 of 932;
  - a large share of 3-D in every direction, and of patchify fwd;
  - dense 3x3 fwd only 114-459 of 1,357.
  - The engines serve all but about ten corpus graphs.
- **rocBLAS or hipBLASLt.** Neither backend dominates. hipBLASLt wins more often on CDNA and gfx1201; rocBLAS wins more on gfx1100.
- **Known losses to MIOpen's GEMM solvers.**
  - *NCHW 1x1 fp16 wrw:* MIOpen accumulates per image into the fp16 dw. That trades precision for speed and is offered here only as the RPR sibling.
  - *1x1 stride-2 fwd/bwd:* about 1.5x on gfx942 fp32 (indicative: not measured with the staged channels-last plan).
  - *Tiny shapes:* our extra launches cost time.
  - *16-bit channels-last wrw over the cap on gfx11:* column blocks keep it off the fp32-output path. Measured on gfx1101 (autotuned, iters 20, MIOpen in search mode on the same node):
    - DeepLabV3 c512 k512 3x3 d4 (16 x 128²): rocBLAS 8.3 s -> 51.9 ms and hipBLASLt 399 -> 89 ms, against MIOpen's 51.1;
    - c2048 k256 d24 / d36: rocBLAS 13.1 s -> 173-181 ms and hipBLASLt 315-319 -> 196-212 ms, against MIOpen's 114-125. Single-tap blocks there pay an im2col per tap and a 256-wide GEMM per block; CONV_WRW serves these layers at 68-90 ms.
    - On MI325X the image-sliced candidates keep the BLAS engines at their earlier times.
  - *Some long 1-D and small-channel 16-bit wrw against `GemmWrwUniversal`:* HiFi-GAN 1x16 NCHW fp16 and EfficientNetV2 c24 NHWC fp16 run about 4x slower on MI325X. The cause is not diagnosed.
- **OOTB ranking.** hipDNN's candidate enumeration is alphabetical, so the BLAS engines rank ahead of every `hipkernel:CONV_*` engine without autotune. That is a naming accident, not a measured choice. Measured without autotune on a set of gfx950 ConvTranspose graphs, this ranking runs at 0.36-0.40x of MIOpen. It is a heuristics issue, not an engine one.

## Numerical precision
- **Default engines:**
  - Products are exact and accumulate in fp32 in the BLAS kernel.
  - Every sum spanning several GEMMs (slices, bands, images, k-chunks, split reductions) uses fp32 partials in fixed order with one final rounding.
  - bwd-data col2im sums fp32 columns per dx element in a fixed order.
- **vs MIOpen:** MIOpen's `GemmWrw1x1_stride1` rounds dw to fp16 once per image (beta 1 into the output). This engine rounds once overall, by policy.
- **TF32 siblings** (`*_TF32`, offered by default, `TENSOR_CORE`):
  - They serve only plans that contain an fp32 GEMM.
  - hipBLASLt asks for `HIPBLAS_COMPUTE_32F_FAST_TF32` and falls back to fp32 when no algorithm exists: native TF32 on gfx942, bf16x3 emulation on gfx950, silent fp32 elsewhere.
  - rocBLAS sets `rocblas_xf32_xdl_math_op`, which takes effect only on gfx942.
  - They report `TENSOR_CORE`, cuDNN's note for TF32 engines (PyTorch's cuDNN path excludes fp32 engines with it when TF32 is not allowed). The gtest bound is 2^-9 × reduction length.
  - MIOpen's own rocBLAS xf32 path measures net-negative on gfx942 (geomean 0.88x), and MIOpen does not use it. These siblings are not yet compared in an autotuned sweep.
- **RPR siblings** (`*_RPR`, offered by default, `REDUCED_PRECISION_REDUCTION`, `NONDETERMINISTIC`):
  - One form: 1x1, unit-stride, unpadded fp16 wrw over several channels-first images, run as one GEMM per image adding into the fp16 dw. This is MIOpen's `GemmWrw1x1_stride1` and cuDNN's `REDUCED_PRECISION_REDUCTION`.
  - bf16 has no reduced form. Compute type unset, fp32 or fp16, as cuDNN offers it.
  - Measured deviation from exact is 1-4 fp16 ulps of max |dw|.
  - It wins on gfx1100 (DeepLab c1024 k2048 128²: 16.0 ms against 23.8 ms for the exact engines and MIOpen's 36.0) and on gfx1201 (1.32-1.42x on three graphs). It is neutral on CDNA.
- **Notes and determinism** (RFC 0025, cuDNN's model: determinism is a per-engine note a framework deselects):

  | Engine | Numerical notes |
  |---|---|
  | `ROCBLAS_CONV_ENGINE`, `HIPBLASLT_CONV_ENGINE` | `NONDETERMINISTIC` |
  | `ROCBLAS_CONV_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_ENGINE_DETERMINISTIC` | none |
  | `ROCBLAS_CONV_TF32_ENGINE`, `HIPBLASLT_CONV_TF32_ENGINE` | `TENSOR_CORE`, `NONDETERMINISTIC` |
  | `ROCBLAS_CONV_TF32_ENGINE_DETERMINISTIC`, `HIPBLASLT_CONV_TF32_ENGINE_DETERMINISTIC` | `TENSOR_CORE` |
  | `ROCBLAS_CONV_RPR_ENGINE`, `HIPBLASLT_CONV_RPR_ENGINE` | `REDUCED_PRECISION_REDUCTION`, `NONDETERMINISTIC` |
  | `HIPBLASLT_CONV_LIBRARY_ACTIVATION_ENGINE` (opt-in: `HIPBLASLT_PROVIDER_CONV_LIBRARY_ACTIVATIONS=1`) | none |

  - The default engines run the libraries' defaults: rocBLAS with atomics allowed, hipBLASLt offered up to 128 MiB of workspace. Either may split a reduction across workgroups.
  - The deterministic siblings share the default engines' code; one flag picks the plan's library settings. rocBLAS runs with `rocblas_atomics_not_allowed`, the handle's mode saved and restored around the plan (as MIOpen's `DisableRocblasAtomics`). hipBLASLt is offered no workspace (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` 0, as MIOpen's deterministic GEMMs), so it declines a GEMM it has no workspace-free algorithm for. The repack workspace is separate and unaffected.
  - The provider's own repack and cast steps are deterministic: every sum has one owning thread in a fixed order.
  - `TestGpuConvGemmDeterministic` executes each deterministic sibling's plan six times over a re-filled workspace and requires identical bits, on long-K backward weights and small-tile, long-reduction forward and backward data (where split-K would apply) and a spread of im2col shapes.
  - **Cost.** Deterministic sibling time over default engine time (above 1 means the deterministic sibling is slower), each engine autotuned (dnn-benchmarking `--autotune --iters 20`, median kernel time), on the graphs of the internal real-model corpus where an explicit-GEMM engine is the fastest engine: 886 graphs on MI350X (gfx950), 912 on MI325X (gfx942). TF32 rows cover the fp32 graphs. "Declined" counts graphs the default engine serves and its deterministic sibling does not (hipBLASLt has no workspace-free algorithm).

    | GPU | Pair | Graphs | Declined | Geomean | Total time | Graphs > 1.10x | Worst |
    |---|---|---|---|---|---|---|---|
    | MI350X | rocBLAS | 886 | 0 | 0.99 | 1.00 | 7 | 1.26x |
    | MI350X | hipBLASLt | 886 | 0 | 1.11 | 2.28 | 129 | 68x |
    | MI350X | rocBLAS TF32 | 188 | 0 | 1.00 | 1.00 | 1 | 1.35x |
    | MI350X | hipBLASLt TF32 | 188 | 0 | 1.18 | 2.84 | 42 | 32x |
    | MI325X | rocBLAS | 912 | 0 | 0.98 | 0.99 | 5 | 1.29x |
    | MI325X | hipBLASLt | 896 | 16 | 1.01 | 1.04 | 42 | 17x |
    | MI325X | rocBLAS TF32 | 191 | 0 | 0.98 | 0.98 | 1 | 1.11x |
    | MI325X | hipBLASLt TF32 | 191 | 0 | 1.08 | 1.63 | 17 | 19x |

    - rocBLAS costs nothing measurable without atomics: the geomeans sit within run-to-run noise of 1.
    - hipBLASLt without workspace loses its split-reduction algorithms. The cost is in backward weights with long reductions over few output elements (MI350X backward weights: geomean 1.27, total time 3.6x; forward 1.08; backward data 1.00), worst on stems such as 7x7 three-channel layers over 800x800 images.
    - A caller that deselects `NONDETERMINISTIC` keeps both deterministic siblings, and the faster of the two is close to the faster default engine: best deterministic over best default, geomean 1.00 on both GPUs (total time 1.00 MI350X, 1.00 MI325X; 2 and 5 graphs past 1.10x). With TF32 allowed (fp32 graphs): geomean 1.03 and total 1.04 on MI350X, 1.00 on MI325X.
- **Error model:** correctness runs use the #15481 reduction-length-aware tolerance, including large tensors past 2 GiB on gfx950. Three library behaviours would round partials to 16 bits, and each is handled:
  - rocBLAS k-chunking into bf16 is declined;
  - rocBLAS gfx11 bf16 GEMV is declined;
  - hipBLASLt long-k split on gfx11 runs as narrow partials.

  The correctness runs show no index overflow.
- **Column blocks on gfx11:** a block's GEMM writes 16-bit w straight from its fp32 accumulation over every image. Measured on gfx1101 against an fp64 sum (random operands in [-1, 1], 262,144-term reductions, 600 sampled outputs):
  - fp16 output deviates by at most 4.7-7.3 fp16 ulps on both libraries;
  - hipBLASLt's fp32-output GEMM of the same operands (the image-sliced plan's kind) deviates by 5-6, so the 16-bit output adds no measurable error. The deviation is the matrix-core accumulation's, not a split's;
  - bf16 stays under 1 ulp;
  - rocBLAS's untuned fp32-output kernels (declined) are closer, at 0.04 ulp.

## Comparison with MIOpen
- **Same family:** MIOpen's GEMM solvers are the same family.
  - `GemmFwdRest` and `GemmWrwUniversal` run im2col and a GEMM per image in a loop.
  - `GemmBwdRest` runs a GEMM, then col2im.
  - The 1x1 solvers use transposes or per-image GEMMs.
  - Most accept NCHW only; some 1x1 solvers take NHWC through hipBLASLt.
- **Differences here:**
  - A single GEMM over every image wherever the strides allow, with layout-following and staged columns.
  - Arbitrary strides and 3-D in every row.
  - A capped workspace with image slicing and row bands.
  - fp32 partials instead of per-image output-type accumulation.
  - Two BLAS backends, with candidate layouts timed under autotune.
- **Measured** on an 86-shape test set covering every conv class, direction, dtype and layout, autotuned, with MIOpen in search mode, as MIOpen Gemm* time / our best BLAS time (geomean; indicative):

  | GPU | MIOpen Gemm* / ours |
  |---|---|
  | gfx906 | 1.06-1.14x |
  | gfx908 | 1.45x |
  | gfx90a | 1.45x |
  | gfx942 | 2.2x |

  Per GPU, ours is more than 1.2x faster on 18-50 graphs and more than 1.2x slower on 13-30. These figures do not cover the current layout candidates, split reductions and staged forms, and the full-corpus runs above do not isolate MIOpen's GEMM solvers.

## Prior art considered
- **MIOpen GEMM solvers:** the per-image loop structure, the 1x1 stride-2 staging and the per-image fp16 wrw. That last one is kept only as the RPR sibling.
- **RFC 0002's explicit-GEMM catalog and predicates.**
- **cuDNN:** its compute-type set, TF32 permission semantics and `REDUCED_PRECISION_REDUCTION` note.
- **External RDNA MIOpen PRs:** these showed col2im at 60-67% of `GemmBwdRest` time on long, narrow layers, which motivated the specialized col2im kernels.
- **No code was copied.** All repack kernels are new HIP.

## Next steps and deferred ideas
1. **Ranking.** Replace the alphabetical OOTB order with a measured heuristic or UHD. Measured without autotune, that ordering is the largest gap against MIOpen.
2. **Long 1-D and small-channel 16-bit wrw against `GemmWrwUniversal`** (about 4x on MI325X). This needs diagnosis: a layout candidate, the reduction split threshold, or library algorithm choice.
3. **Templated summing col2im on 16-bit columns.** This would keep 16-bit-output GEMMs for overlapping taps. On RDNA3, the libraries' 16-bit-output GEMMs run 4-5x faster than fp32-output ones on a vocoder upsampler.
4. **Further RPR forms** (rocBLAS 16-bit k-chunks, hipBLASLt fp16 split-K) are measurement-led.
5. **Calibrate the 1 GiB repack cap** against measured workloads.
6. **Single-tap column blocks** (c2048-class 16-bit channels-last wrw, 1.4-1.6x behind MIOpen on gfx1101): batch a block's taps into one strided-batched GEMM, or read a single tap's columns as a shifted view of x where its window needs no padding instead of copying them.
7. **gfx11 fp32-output GEMMs elsewhere:** 16-bit backward data with overlapping taps writes fp32 columns (hipBLASLt 4-6 TFLOP/s on gfx1101; rocBLAS declined there). A summing col2im over 16-bit columns (item 3) would move it to the 16-bit-output kernels.

## Tests
`tests/engines/conv/`:
- **Plan structure and gates:** `TestConvGemmFormulation` covers the direct rows and their layout and group rules, plus cuDNN's compute-type set.
- **Repack formulation:** `TestConvRepackFormulation` covers im2col and col2im, column layout, staging, slicing and banding under a lowered cap, fp32 sums for long and narrow reductions, split reductions, aligned and disjoint workspace, and the decline cases. For 16-bit wrw over the cap, it checks that column blocks write every element of w exactly once with 16-bit GEMM outputs: both filter layouts and modes, dilation and stride, caps down to one column. It also checks that the DeepLab ASPP layers (c512 d4, c2048 d24, 16 x 128^2) plan no fp32-output GEMM, and which cases keep image slices.
- **Launch selection:** `TestConvRepackLaunch` covers the vectorized, LDS-tile, disjoint and generic paths, including offsets past 32 bits.
- **Repack kernels on device:** `TestGpuRepackKernels` checks the specialized kernels bit for bit against the generic ones and against the host, and checks the fixed-tree cast.
- **Device correctness:** `TestGpuConvGemm` runs both backends in fp32, fp16 and bf16 against a CPU reference, with sliced, banded, benchmarked, split, staged, TF32, RPR and deterministic (`TestGpuConvGemmDeterministic`, six executes each) variants.
  - `TestConvGemmRocblasLimits` pins both rocBLAS precision declines; `TestMatmulProblem.RocblasUntunedFp32OutputIsGfx11SixteenBitNt` pins the untuned-path decline per arch, type and transpose form.
  - The sliced cases run column blocks for 16-bit channels-last wrw, a channels-last strided, dilated and asymmetric case among them. `ImageSlicesFp16` and `ImageSlicesBfp16` run the image-sliced candidate.
  - `TestGpuConvGemmLarge` is opt-in (`HIPBLASLT_PROVIDER_LARGE_TESTS=1`). It covers more than 4 GiB, n past 2^30, a batch stride past 2^31 and an image over the cap, checking sampled outputs against a double sum.
- **RPR siblings:** `TestConvReducedPrecision` covers registration and one decline per gate.
- **Fusion:** `TestConvEpilogueFormulation` covers how each chain maps onto the GEMM per support (hipBLASLt, rocBLAS, library activations), each field that rules the native form out, and which form each plan takes; `TestGpuConvEpilogue` runs every form on both backends in fp32, fp16 and bf16 against the CPU convolution followed by the chain in double, with NaN and infinity operands and values on the clip bounds, runs the library-activation engine against the library's special values, and counts hipBLASLt's solutions with and without a native epilogue; `TestGpuRepackKernels` checks the clamp tail kernel.
- **Notes:** `TestGpuHipblasltEnginePluginApi.ConvEnginesReportTheirNumericalNotes` checks every conv engine's notes, and that deselecting `NONDETERMINISTIC` keeps exactly the deterministic siblings.
