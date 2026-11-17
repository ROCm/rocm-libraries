/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_gemm_wmma.h -- C++ engine port of the batched
 * GEMM of the Winograd forward (K3) in
 * library/kernels/common/conv_winograd_gemm_wmma.py, and of its split-K slab
 * reduce.
 *
 * K3 is G = n2 * g independent GEMMs M[t][G][Kg] = sum_c V[t][G][c] U[G][Kg][c]
 * on the gfx1250 WMMA 16x16x32 atom: per input-channel chunk, two TDM transfers
 * (V by wave 0, U by wave 1) stage the operands in an LDS ring of `stages`
 * slots, and the epilogue stores the tile with one TDM transfer or per lane.
 *
 *   Python (conv_winograd_gemm_wmma.py)      C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) WinogradGemmWmmaSpec  rocke_winograd_gemm_wmma_spec_t
 *     tile_m / tile_n / threads_per_block /  rocke_winograd_gemm_wmma_{tile_m,tile_n,
 *     lds_bytes() / kernel_name()              threads_per_block,lds_bytes,kernel_name}
 *     tiles() / grid()                       rocke_winograd_gemm_wmma_grid
 *   winograd_gemm_reason                     rocke_winograd_gemm_wmma_is_valid_spec
 *   build_winograd_gemm_wmma(spec, arch)     rocke_build_winograd_gemm_wmma(_new)
 *   WinogradGemmReduceSpec                   rocke_winograd_gemm_reduce_spec_t
 *   build_winograd_gemm_reduce(spec, arch)   rocke_build_winograd_gemm_reduce(_new)
 *   (+ convenience: build -> lower .ll)      rocke_winograd_gemm_{wmma,reduce}_lower_to_llvm
 *
 * T, G, Cg, Kg, the split count, chunks per split and the batch range (g0, nb)
 * are kernargs; the tile geometry and dtype are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_GEMM_WMMA_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_GEMM_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  WinogradGemmWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class WinogradGemmWmmaSpec:
 *      dtype: str = "fp16"
 *      name: str = "wino_gemm_wmma"
 *      waves_m: int = 2; waves_n: int = 4
 *      frags_m: int = 8; frags_n: int = 4
 *      tile_k: int = 128; stages: int = 2
 *      lds_pad: str = "tdm"     # "tdm" | "clip"
 *      epilogue: str = "tdm"    # "tdm" | "direct"
 *      split_k: bool = False
 *      prefetch: bool = True
 *      xcd_remap: bool = True; num_xcds: int = 8
 *      sched: Optional[str] = "iterative-ilp"
 *      wave_size: int = 32
 *      taps: int = 1
 * ===================================================================== */
typedef struct rocke_winograd_gemm_wmma_spec
{
    const char* dtype; /* default "fp16" */
    const char* name; /* default "wino_gemm_wmma" */
    int waves_m; /* default 2 */
    int waves_n; /* default 4 */
    int frags_m; /* default 8 */
    int frags_n; /* default 4 */
    int tile_k; /* default 128 */
    int stages; /* default 2 */
    const char* lds_pad; /* default "tdm" */
    const char* epilogue; /* default "tdm" */
    bool split_k; /* default false */
    bool prefetch; /* default true */
    bool xcd_remap; /* default true */
    int num_xcds; /* default 8 */
    const char* sched; /* default "iterative-ilp"; NULL = None */
    int wave_size; /* default 32 */
    int taps; /* default 1 */
} rocke_winograd_gemm_wmma_spec_t;

/* Spec with the Python dataclass defaults. */
rocke_winograd_gemm_wmma_spec_t rocke_winograd_gemm_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_winograd_gemm_wmma_tile_m(const rocke_winograd_gemm_wmma_spec_t* spec);
int rocke_winograd_gemm_wmma_tile_n(const rocke_winograd_gemm_wmma_spec_t* spec);
int rocke_winograd_gemm_wmma_threads_per_block(const rocke_winograd_gemm_wmma_spec_t* spec);
long rocke_winograd_gemm_wmma_lds_bytes(const rocke_winograd_gemm_wmma_spec_t* spec);
/* spec.grid(prob, splits, batches): resolved splits, `nb` batches (0 = all G). */
void rocke_winograd_gemm_wmma_grid(const rocke_winograd_gemm_wmma_spec_t* spec,
                                   int T,
                                   int G,
                                   int Cg,
                                   int Kg,
                                   int splits,
                                   int nb,
                                   int grid[3]);

/* kernel_name(): kernel_name_join(name, wm, wn, fm, fn, tk, st, p{lds_pad},
 * epilogue, [sk], [pf], [x{num_xcds}], [s{sched}], [taps{taps}], [bf16]). */
rocke_status_t rocke_winograd_gemm_wmma_kernel_name(const rocke_winograd_gemm_wmma_spec_t* spec,
                                                    char* out,
                                                    size_t out_cap);

/* winograd_gemm_reason(spec, arch) is None. `arch` NULL => "gfx1250". */
bool rocke_winograd_gemm_wmma_is_valid_spec(const rocke_winograd_gemm_wmma_spec_t* spec,
                                            const char* arch,
                                            char* reason,
                                            size_t reason_cap);

/* build_winograd_gemm_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_winograd_gemm_wmma(rocke_ir_builder_t* b,
                                                   const rocke_winograd_gemm_wmma_spec_t* spec,
                                                   const char* arch);
/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_winograd_gemm_wmma_new(rocke_ir_builder_t* b,
                                                       const rocke_winograd_gemm_wmma_spec_t* spec,
                                                       const char* arch);
/* Convenience: build + lower to LLVM IR text (*out_ll malloc'd). */
rocke_status_t rocke_winograd_gemm_wmma_lower_to_llvm(const rocke_winograd_gemm_wmma_spec_t* spec,
                                                      const char* arch,
                                                      rocke_llvm_flavor_t flavor,
                                                      char** out_ll,
                                                      char* err,
                                                      size_t err_cap);

/* ===================================================================== *
 *  WinogradGemmReduceSpec: d[i] = sum_s w[s][i] (fp32 slabs -> dtype).
 * ===================================================================== */
typedef struct rocke_winograd_gemm_reduce_spec
{
    const char* dtype; /* default "fp16" */
} rocke_winograd_gemm_reduce_spec_t;

rocke_winograd_gemm_reduce_spec_t rocke_winograd_gemm_reduce_spec_default(void);
/* kernel_name_join("wino_gemm_reduce", flags={"bf16": dtype == "bf16"}). */
rocke_status_t rocke_winograd_gemm_reduce_kernel_name(const rocke_winograd_gemm_reduce_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap);
rocke_kernel_def_t* rocke_build_winograd_gemm_reduce(rocke_ir_builder_t* b,
                                                     const rocke_winograd_gemm_reduce_spec_t* spec,
                                                     const char* arch);
rocke_kernel_def_t* rocke_build_winograd_gemm_reduce_new(
    rocke_ir_builder_t* b, const rocke_winograd_gemm_reduce_spec_t* spec, const char* arch);
rocke_status_t
    rocke_winograd_gemm_reduce_lower_to_llvm(const rocke_winograd_gemm_reduce_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_GEMM_WMMA_H */
