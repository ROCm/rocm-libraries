## CK Tile direct convolutions

CK Tile convolutions use warp-level GEMM pipelines to compute forward, backward data convolutions.

### Direct convolution for sparse grouped convolutions

### Direct dense convolution algorithm

Consider a non-grouped forward convolution in the channels last layout

$$
\mathcal{O}(n,h_o, w_o, k) = \sum_{y=0}^{Y-1}\sum_{x=0}^{X-1}\sum_{c=0}^{C_{\text{tot}}} \mathcal{I}(n,s_h \times h_o + d_h \times y - p_h, s_w \times w_o + d_w \times x - p_w, c) \times \mathcal{W}(k, y, x, c)
$$

where

- $n \in \{0,...,N-1\}$ is the batch index and $N$ is the number of batches
- $h_o,\,w_o$ are the height and width of the output
- $d_w,\,d_w$ are the dilation
- $s_h,\,s_w$ are the stride
- $p_h,\,p_w$ are the padding
- $C_{\text{tot}}$ is the total number of input channel (in dense convolution all input channels contribute to all output channels)

and $\mathcal{I}$, $\mathcal{W}$, and $\mathcal{O}$ are the input, weight, and output tensors, respectively.

Denote a slice of the output tensor as $\mathcal{O}(n,h_o, :, :)$ which is a small 2D matrix of size $M_{\text{mfma}}^{}$ and  $N_{\text{mfma}}^{}$.
Then, we can break the sum over the input channels as

$$
\sum_{c=0}^{C_{\text{tot}}} = \sum_{i_C=0}^{N_C - 1}\sum_{\tilde{k}=0}^{K_{\text{mfma}}^{}}
$$

where $N_C = (C_{\text{tot}} + K_{\text{mfma}}^{} - 1) / K_{\text{mfma}}^{}$ (ceiling division). We have basically divided the accumulation over the 
input channel as chunks of size $K_{\text{mfma}}^{}$.

For computing the outptu tensor slice $\mathcal{O}(n,h_o, :, :)$, we use the following notation

$$
\begin{aligned}
&A_{\tilde{m}, \tilde{k}}(n, h_o, y, x, i_C) = \mathcal{I}((n,s_h \times h_o + d_h \times y - p_h, s_w \times \tilde{m} + d_w \times x - p_w, \tilde{k})) \\
&B_{\tilde{k}, \tilde{n}}(y,x, i_c) = \mathcal{W}(\tilde{n}, y, x, \tilde{k}) \\
&C_{\tilde{m}, \tilde{n}}(n,h_o) = \mathcal{O}(n,h_o, \tilde{m}, \tilde{n})
\end{aligned}
$$

we can write the initial convolution as

$$
C_{\tilde{m}, \tilde{n}}(n,h_o) = \sum_{i_C=0}^{N_C - 1} \sum_{y=0}^{Y-1}\sum_{x=0}^{X-1} \sum_{\tilde{k}=0}^{K_{\text{mfma}}^{}} A_{\tilde{m}, \tilde{k}}(n, h_o, y, x, i_C) \times B_{\tilde{k}, \tilde{n}}(y,x, i_c)
$$

The innermost loop 

$$
\sum_{\tilde{k}=0}^{K_{\text{mfma}}^{}} A_{\tilde{m}, \tilde{k}}(n, h_o, y, x, i_C) \times B_{\tilde{k}, \tilde{n}}(y,x, i_c) \leftrightarrow \text{MFMA instruction output}
$$

can be efficiently calculated using the MFMA instruction, where a full lane of 64 threads computes the small matrix product using a device built-in instruction.
Hence, we can map the convolution problem into an accumulation of wavefront level GEMM problems.
The $\sum_{i_C=0}^{N_C - 1}$ loop over the input channel slices can be further broken into 

$$
\sum_{i_C=0}^{N_C - 1} = \sum_{i_{\text{wave}_0} = 0}^{N-1} + \cdots +  \sum_{i_{\text{wave}_{N_w-1}} = 0}^{N-1}
$$

where we have $N_w$ waves of 64 threads such that each wave computes $N$ slices of $K_{\text{mfma}}^{}$ input channels. 
The accumulation within wave happens naturally via MFMA instruction (as it is matrix-fused-multiply-add). 
The cross-wave accumulation takes place via LDS, i.e., each wave writes its final result into LDS with an appropriate synchronization for the in-fligt LDS operations.
By moving more slices to the intrawave comoutation, we save LDS usage and allow more wavefronts to reside on a given CU.

## Supported configurations

- **Data type**: fp16, bf16
- **Layout**: NHWC input, KYXC weights, NHWK output
- **Direction**: Fprop (Forward), Dgrad (Backward Data)
- **Filter**: 3x3, stride 1, dilation 1
- **Group size**: 4, 8, 16, or 32 channels per group
- **Constraint**: input channels == output channels (`c == k`), but can be relaxed by using the padding version of the kernels.

The kernel uses MFMA instructions with buffer-load-to-LDS for input staging to compute direct convolution.

## Supported device architectures

CK Tile direct convolutions are developed and validated on `gfx950` (CDNA4),
which supports the full set of variants and directions. `gfx942` (CDNA3) is
supported for the 4c forward variant only; the remaining variants rely on
CDNA4-only hardware features (see notes below).

| Variant | Direction | fp16 | bf16 |
|---------|-----------|------|------|
| 4c (MFMA 4x4x4) | Fprop | gfx942, gfx950 | gfx942, gfx950 |
| 4c (MFMA 4x4x4) | Dgrad | gfx950 | gfx950 |
| 8c (MFMA 16x16x32) | Fprop, Dgrad | gfx950 | gfx950 |
| 16c (MFMA 16x16x16) | Fprop, Dgrad | gfx950 | gfx950 |
| 32c (MFMA 16x16x32) | Fprop, Dgrad | gfx950 | gfx950 |

### CDNA4-only feature dependencies

- **MFMA 16x16x32** (`gfx950-insts`): required by the 8c (Toeplitz) and 32c
  kernels. No CDNA3 equivalent, so 8c/32c cannot compile on gfx942.
- **`ds_read_b64_tr_b16`** transpose read: required by the Dgrad (backward data)
  weight read path. CDNA4 only, so all Dgrad variants are gfx950-only.
- **LDS footprint > 64KB**: the 16c kernel needs 72KB of LDS, exceeding
  gfx942's 64KB per-workgroup limit.
- **16-byte buffer-load-to-LDS** (dwordx4 async copy): used by every variant for
  performant input/weight staging. This one is not a blocker — the kernels fall back to a
  portable load+store register round-trip on non-gfx950 archs.

On non-gfx950 architectures, only gfx942 is currently tested. 
The dispatcher codegen emits only the supported subset (4c forward, fp16/bf16). 
Unsupported direct-conv instances are filtered out at generation time.

## Project structure

Here's a rough project structure

```
kernel/                                  — kernel implementations
  grouped_4c_tile_conv_impl_v3.hpp  — 4-channel grouped fp16/bf16 kernel (MFMA 4x4x4 batch-16) using CK Tile abstractions.
  grouped_8c_tile_conv_impl_v2.hpp  — 8-channel grouped fp16/bf16 kernel (MFMA 16x16x32 for Toeplitz matrix and S-loop fusion) using CK Tile abstractions.
  grouped_16c_tile_conv_impl_v2.hpp — 16-channel grouped fp16/bf16 kernel (MFMA 16x16x16) using CK Tile abstractions.
  grouped_32c_tile_conv_impl_v2.hpp — 32-channel grouped fp16/bf16 kernel (MFMA 16x16x32) using CK Tile abstractions.
utils/                                   — utilities for kernel implementations
  conv_params.hpp                        — Conv2dParams, SizeView, Direction/DataType enums
  types.hpp                              — fp16/bf16/fp32/fp8 type aliases and mapping
  launch_params.hpp                      — LaunchParams (grid, block, shared memory)
  kernel_variant.hpp                     — KernelVariant dispatch interface
  swizzle.hpp                            — LDS swizzle for bank-conflict-free access
  transpose_lds_layout.hpp               — DS_READ_TR_B16 layout for CDNA4 Dgrad
```

## Building

### Dispatcher codegen

The CK Tile Dispatcher is the way to build and profile direct convolution
kernels. It generates each kernel as a separate compilation unit for better build
parallelism. Enable it with `CK_TILE_DISPATCHER=ON`:

To build only the direct convolution instances (without implicit-GEMM), set
`DISABLE_IMPLICIT_GEMM_INSTANCES=ON`. This flag is a codegen filter in the
Dispatcher pipeline that emits only `kind=direct_conv` instances:

```
-D DISABLE_IMPLICIT_GEMM_INSTANCES=ON                                                           
```

When building the CK Profiler, one may build only the relevant CK Tile convolution profilers by 
CMake flag

```
-D CK_PROFILER_OP_FILTER="_tile"
```

The full configure step for direct convolution instances is 

```
cmake                                                                                             \
  -D CMAKE_PREFIX_PATH=/opt/rocm                                                                  \
  -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc                                                       \
  -D CMAKE_BUILD_TYPE=Release                                                                     \
  -D GPU_TARGETS="gfx950"                                                                         \
  -D CK_EXPERIMENTAL_BUILDER=ON                                                                   \
  -D CK_TILE_DISPATCHER=ON                                                                        \
  -D CMAKE_CXX_STANDARD=20                                                                        \
  -D CK_PROFILER_OP_FILTER="_tile"                                                                \
  -D DISABLE_IMPLICIT_GEMM_INSTANCES=ON                                                           \
  -D DISPATCHER_CONFIG_SET=profiler                                                               \
  -G Ninja                                                                                        \
  ..
```

The dispatcher creates unique kernel names for each instance. To use the CK Tile's `GetName` instance string, 
we specify flag

```
-D CK_EXPERIMENTAL_BUILDER=ON
```

The CK Tile kernel names are required for the benchmarking and regression test script that rely on the CK Tile kernel
name format when parsing the results from the CK Profiler output.

We can toggle the implicit GEMM instance flag on and off depending on whether we want to compare the direct conv also to the implicit GEMM.
Switch 

```
-D DISPATCHER_CONFIG_SET=profiler 
```

ensures that we always use the full set implicit GEMM instances. 

## Testing

The unit and integration tests are located in directory `projects/composablekernel/test/ck_tile/direct_conv`.
The entry point is CMake target `ck_tile_direct_conv_tests` that runs all CK Tile direct convolution tests.
Run the target with command 

```
ninja -j32 ck_tile_direct_conv_tests
```


## Profiler

The CK Tile Profiler is the tool for benchmarking. Build it with

```
ninja -j32 ckProfiler
```

If you have also the implicit GEMM instances enabled, use more threads (64 or 128).

## Performance and regression testing

There is a [python CLI](../../../../script/direct_conv/direct_conv_bench.py) that runs a set of fwd/bwd data [cases](../../../../script/direct_conv/direct_conv_cases.txt).
It has three subcommands:

- `run` — run every case and print a text summary (smoke / correctness).
- `regress` — compare best TFLOPS against per-arch expected values (10% tolerance)
  and write a markdown report; exits nonzero on a regression.
- `compare` — compare implicit-GEMM vs direct-conv performance (markdown table,
  optional `--plot` PNG; needs an implicit-GEMM-enabled build).

This can be used to verify that all instances produce correct results as well as for testing performance regresion/improvement after refactoring.
When the coverage of the CK Tile direct convs is expanded, more cases should be added.

```
python3 script/direct_conv/direct_conv_bench.py run     --bin-path <build>/bin
python3 script/direct_conv/direct_conv_bench.py regress --bin-path <build>/bin
```

For running the performance script, use the Dispatcher codegen build (`CK_TILE_DISPATCHER=ON`).
To focus on direct convolution only, add `-D DISABLE_IMPLICIT_GEMM_INSTANCES=ON` to the
Dispatcher build configuration.