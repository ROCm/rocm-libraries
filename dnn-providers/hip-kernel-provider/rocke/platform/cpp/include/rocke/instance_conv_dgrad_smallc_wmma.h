/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_dgrad_smallc_wmma.h -- C++ engine port of the wave32 WMMA
 * small-C stem dgrad builder in library/kernels/common/conv_dgrad_smallc_wmma.py.
 *
 * For a stem convolution (C <= 4 dX channels, ungrouped, not a patch
 * embedding) the dgrad's natural GEMM leaves almost every MMA tile empty, so
 * the kernel flattens the filter taps and the dX channels into one GEMM
 * dimension, Z[q, (r, s, c)] = dY[q, :] x W[:, (r, s, c)], for the dY window
 * that reaches a block's dX tile, on the gfx1250 WMMA 16x16x32 atom. The
 * operands are staged like the direct WMMA dgrad with a 1x1 filter; Z is
 * staged in LDS (fp32) and overlap-added phase by phase (output row / column
 * mod S, a compile-time tap list per phase) into a dX tile that one TDM
 * transfer stores.
 *
 *   Python (conv_dgrad_smallc_wmma.py)       C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DgradSmallcWmmaSpec   rocke_dgrad_smallc_wmma_spec_t
 *     derived @property geometry             rocke_dgrad_smallc_wmma_{block_c,block_p,
 *                                              dy_h,dy_w,threads_per_block,
 *                                              lds_bytes,grid}
 *     .kernel_name()                         rocke_dgrad_smallc_wmma_kernel_name
 *   dgrad_smallc_reason / is_valid_dgrad_smallc_spec
 *                                            rocke_dgrad_smallc_wmma_is_valid_spec
 *   build_dgrad_smallc_wmma(spec, arch)      rocke_build_dgrad_smallc_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_dgrad_smallc_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("dgrad"). Batch, extents and the dY channel
 * count are runtime; the filter, stride, PAD, the dX channel count, dtype and
 * the tile geometry are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DGRAD_SMALLC_WMMA_H
#define ROCKE_INSTANCE_CONV_DGRAD_SMALLC_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DgradSmallcWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DgradSmallcWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "dgrad_smallc_wmma"
 *      waves_c: int = 1      # waves along GEMM columns (r, s, c)
 *      waves_p: int = 4      # waves along dY pixels
 *      tiles_c: int = 2
 *      tiles_p: int = 5
 *      out_h: int = 32       # dX rows per block (multiple of the stride)
 *      out_w: int = 32       # dX columns per block (multiple of the stride)
 *      ck: int = 32
 *      stages: int = 2
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_dgrad_smallc_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dgrad_smallc_wmma" */
    int waves_c; /* default 1  */
    int waves_p; /* default 4  */
    int tiles_c; /* default 2  */
    int tiles_p; /* default 5  */
    int out_h; /* default 32 */
    int out_w; /* default 32 */
    int ck; /* default 32 */
    int stages; /* default 2  */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_dgrad_smallc_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_dgrad_smallc_wmma_spec_t rocke_dgrad_smallc_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_dgrad_smallc_wmma_block_c(const rocke_dgrad_smallc_wmma_spec_t* spec);
int rocke_dgrad_smallc_wmma_block_p(const rocke_dgrad_smallc_wmma_spec_t* spec);
int rocke_dgrad_smallc_wmma_dy_h(const rocke_dgrad_smallc_wmma_spec_t* spec);
int rocke_dgrad_smallc_wmma_dy_w(const rocke_dgrad_smallc_wmma_spec_t* spec);
int rocke_dgrad_smallc_wmma_threads_per_block(const rocke_dgrad_smallc_wmma_spec_t* spec);
long rocke_dgrad_smallc_wmma_lds_bytes(const rocke_dgrad_smallc_wmma_spec_t* spec);
/* spec.grid() -> (N * ceil(H / out_h) * ceil(W / out_w), 1, 1). */
void rocke_dgrad_smallc_wmma_grid(const rocke_dgrad_smallc_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, s{stride}, p{PAD}, c{C}, wc{},
 * wp{}, tc{}, tp{}, oh{}, ow{}, ck{}, st{}, [pf], [s{sched}], [bf16]). */
rocke_status_t rocke_dgrad_smallc_wmma_kernel_name(const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_dgrad_smallc_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_dgrad_smallc_wmma_is_valid_spec(const rocke_dgrad_smallc_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_dgrad_smallc_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_dgrad_smallc_wmma(rocke_ir_builder_t* b,
                                                  const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_dgrad_smallc_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_dgrad_smallc_wmma_lower_to_llvm(const rocke_dgrad_smallc_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DGRAD_SMALLC_WMMA_H */
