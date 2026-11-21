# GEMM with CK Tile

This example demonstrates matrix multiplication (GEMM) using the CK Tile programming model, focusing on tile-based parallelism and modular kernel design.

---

## Algorithm and Math

GEMM computes:
$$
C = A \times B
$$
where $A$ is $[M, K]$, $B$ is $[N, K]$, and $C$ is $[M, N]$.

- **BlockTile GEMM**: Each Block Tile computes a tile of $C$ by loading tiles of $A$ and $B$, performing blockwise matrix multiply-accumulation, and writing results back with the epilogue.

---

## Tile Programming Model

- **Configuration**: The Configuration of how the kernel going to be initialized with Block Tile Dimension, Warps Layout, Warp Tile Dimension, and other improvements.
- **Block Tile**: Each block tile allocates in the compute unit of AMD GPU grabbing the .
- **Pipeline**: Modular design allows swapping different memory/computation pipelines (e.g., basic, memory-bound, compute).
- **Block GEMM**: Block Level implementation on how to coordinate the warps iteration and memory layout in block tile.
- **Warp GEMM**: Each Warp's GEMM Calculation
- **Epilogue**: Transferring the Accumulated result from register to global memory.

---

## Features

- **Flexible Layouts**: Supports row/column-major and custom strides for $A$, $B$, $C$.
- **Split K**: Split the Block Tile also on K Dimension and add it back after the matrix multiply-accumulation. Have a higher performance when M and N is small and K is large.
- **Preshuffled GEMM**: In inference task, shuffle the GEMM of B (weight) matrix in the warp layout and bypass the shared memory to do the GEMM calculation. Best performance solution for GEMM.
- **Precision**: Supports fp16, bf16, fp8, bf8, int4 (for B Matrix).
- **Validation**: CPU/GPU validation and error tolerance options.

---

## Build & Run

```bash
mkdir build && cd build
# you can replace <arch> with the appropriate architecture (for example gfx90a or gfx942) or leave it blank
../script/cmake-ck-dev.sh  ../ <arch>
# The basic pipeline method on the gemm calculation
make tile_example_gemm_basic -j`nproc`
# The memory bound pipeline on the gemm calculation
make tile_example_gemm_universal -j`nproc`
# The weight preshuffle pipeline on the gemm calculation
make tile_example_gemm_weight_preshuffle -j`nproc`
# gfx125 only: weight preshuffle TDM pipeline with data cache prefetch controls
make tile_example_gemm_weight_preshuffle_tdm_data_cache_prefetch -j`nproc`
# gfx125 only: TDM pipelines with data cache prefetch controls (TdmEpilogue)
make tile_example_gemm_tdm_data_cache_prefetch -j`nproc`
# gfx125 only: the same, with C written through LDS one M iteration at a time
# (TdmEpilogueStagedForPipeline) instead of the whole tile
make tile_example_gemm_tdm_data_cache_prefetch_staged -j`nproc`
```
This will result in an executable `build/bin/tile_example_gemm_basic` & `build/bin/tile_example_gemm_universal`

### Fused GEMM + permute (gfx950)

`tile_example_gemm_permute` fuses two ops into one kernel:
`C[t*B+b, d*H+h] = A[t,b,:] * W[:,d,h]` (BF16, FP32 accumulation) followed by
`Y[h,b,t,d] = C[t,b,d,h]`, at T=768, B=512, K=384, D=256, H=8. It runs two fused paths that take
the same inputs (`A[t,b,k]` and the original weight `W[k,d,h]`, i.e. `[K, N]` row-major with
`n = d*H+h`) and write the same dense `Y`; neither writes an intermediate tensor. BF16 outputs
are converted by truncation, the CK default.

- **Library kernel** (`gemm_permute.hpp`): a thin user of `ck_tile::GemmEPermuteKernel`
  (`include/ck_tile/ops/gemm/kernel/gemm_e_permute_kernel.hpp`). It consumes the weight reordered
  to `Wp[k,h,d]` (1.5 MiB, `prepermute_weight`), so `C[t*B+b, h*D+d]` is stored through the
  `[T, B, H, D]` view of `Y` with strides `(D, T*D, B*T*D, 1)`. The stock CompV3 pipeline,
  `CShuffleEpilogue` and `GemmSpatiallyLocalTilePartitioner` are used unchanged, and
  `IsSupportedArgument` checks the view (ragged T and B are supported).
- **A-stationary kernel** (`gemm_permute_astationary.hpp`): a hand-written kernel built from
  CK-Tile parts. One workgroup per `(b, 128 rows of t)` loads its A tile into LDS once (the only
  workgroup barrier); each wave streams the weight from global memory straight into MFMA
  registers and stores its accumulators directly into the interleaved `Y`, with no LDS shuffle,
  overlapping the store of one output tile with the MFMAs of the next. The weight is repacked
  on every call into the `ck_tile::shuffle_b_permuteN` layout (`repack_weight`, about 6 us).
  Any T, and B up to 21845, are supported (the A window of a workgroup uses 32-bit offsets);
  buffers must be 16-byte aligned.

```sh
cmake --build . --target tile_example_gemm_permute
./bin/tile_example_gemm_permute               # bit-exact checks at four small shapes
./bin/tile_example_gemm_permute --benchmark   # also check and time the trace shape
```

The driver compares every output bit of both fused paths and both unfused baselines with
`reference_gemm_gpu` followed by an independent permute, and checks guard regions around every
output. The benchmark times each path with its weight reorder included on every call, as the
median of 11 interleaved, rotated samples of 20 back-to-back launches, against two existing
CK-Tile op pairs launched back to back: the weight-preshuffle `GemmKernel`
(`WeightPreshufflePipelineAGmemBGmemCRegV2`, the fastest existing GEMM op for this shape) or the
CompV3 `GemmKernel` the library kernel builds on, each followed by `ElementWiseKernel` as a
strided copy. It also reports both GEMM ops alone. Medians on MI350X (ROCm 7.1.1), speedup over
the best two-op pair:

| Path | Time (us) | Speedup |
|---|---:|---:|
| A-stationary fused | 694 | 2.47x |
| Library fused (`GemmEPermuteKernel`) | 1005 | 1.71x |
| Weight-preshuffle `GemmKernel` + `ElementWiseKernel` | 1715 | 1.00x |
| CompV3 `GemmKernel` + `ElementWiseKernel` | 1902 | 0.90x |
| Weight-preshuffle `GemmKernel` alone | 785 | |
| CompV3 `GemmKernel` alone | 993 | |

## example
```
args:
          -m    m dimension (default:1024)
          -n    n dimension (default:2048)
          -k    k dimension (default:64)
   -a_layout    Tensor A data layout (default: R)
   -b_layout    Tensor B data layout (default: C)
   -c_layout    Tensor C data layout (default: R)
   -stride_a    Tensor A stride (default:0)
   -stride_b    Tensor B stride (default:0)
   -stride_c    Tensor C stride (default:0)
          -v    0. No validation, 1. Validation on CPU, 2. Validation on GPU (default:2)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
     -warmup    number of iterations before benchmark the kernel (default:50)
     -repeat    number of iterations to benchmark the kernel (default:100)
      -timer    gpu:gpu timer, cpu:cpu timer (default:gpu)
          -split_k    splitK value (default:1)
       -init    0:random, 1:linear, 2:constant(1) (default:0)
 -persistent    0:non-persistent, 1:persistent (default:0)
       -json    0: No Json, 1: Dump Results in Json format (default:0)
   -jsonfile    json file name to dump results (default:gemm.json)
```


## Source Structure

- **Executables**: `gemm_basic.cpp`, `universal_gemm.cpp` (different kinds of GEMM implementation)
- **Utils**: `gemm_utils.hpp` (helper functions)
- **Build**: `CMakeLists.txt`, `run_gemm_example.inc`
- **Scripts**: `script/` (build and run helpers)

---

## Related CK Tile Examples

- [01_fmha](../01_fmha/README.md): Fused multi-head attention (FMHA)
- [18_flatmm](../18_flatmm/README.md): Preshuffled GEMM alternative solution
- [16_batched_gemm](../16_batched_gemm/README.md): Batched GEMM with tiles

For distribution, see `include/ck_tile/tile_program/tile_distribution/`.

---
[Back to CK Tile Examples](../README.md)
