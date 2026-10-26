/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_tapouter_wmma.h -- C++ engine port of the
 * tap-outer direct forward in library/kernels/common/conv_direct_fwd_tapouter_wmma.py.
 *
 * A dilated forward conv on the gfx1250 wave32 16x16x32 WMMA, run as KH * KW
 * shifted 1x1 GEMMs accumulating into one output tile: each (channel chunk,
 * tap) step moves the tap's input window and weights into an LDS ring with TDM
 * transfers; taps whose window lies wholly in the padding are skipped per
 * block. The Y tile is staged in LDS and stored with one TDM transfer.
 *
 *   Python (conv_direct_fwd_tapouter_wmma.py)   C (this header)
 *   ------------------------------------------   ---------------------------------------
 *   @dataclass(frozen) DirectFwdTapOuterSpec     rocke_fwd_tapouter_wmma_spec_t
 *     derived @property / methods                rocke_fwd_tapouter_wmma_{block_k,block_p,
 *                                                  tile_h,threads_per_block,lds_bytes,grid}
 *     .kernel_name()                             rocke_fwd_tapouter_wmma_kernel_name
 *   tapouter_tile_efficiency                     rocke_fwd_tapouter_wmma_tile_efficiency
 *   tapouter_reason / is_valid_tapouter_spec     rocke_fwd_tapouter_wmma_is_valid_spec
 *   build_direct_fwd_tapouter_wmma(spec, arch)   rocke_build_fwd_tapouter_wmma(_new)
 *   (+ convenience: build -> lower .ll)          rocke_fwd_tapouter_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents, the group count and
 * both channel counts are runtime; the filter, padding and dilation are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_TAPOUTER_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_TAPOUTER_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* TAPOUTER_VGPR_BUDGET (conv_direct_fwd_tapouter_wmma.py). */
#define ROCKE_FWD_TAPOUTER_VGPR_BUDGET 512

/* ===================================================================== *
 *  DirectFwdTapOuterSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectFwdTapOuterSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_fwd_tapouter_wmma"
 *      waves_k: int = 2
 *      waves_p: int = 4
 *      tiles_k: int = 4
 *      tiles_p: int = 2
 *      tile_w: int = 16
 *      ck: int = 64
 *      stages: int = 3
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 *      tdm_split: int = 1
 * ===================================================================== */
typedef struct rocke_fwd_tapouter_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_fwd_tapouter_wmma" */
    int waves_k; /* default 2 */
    int waves_p; /* default 4 */
    int tiles_k; /* default 4 */
    int tiles_p; /* default 2 */
    int tile_w; /* default 16 */
    int ck; /* default 64 */
    int stages; /* default 3 */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
    int tdm_split; /* default 1 */
} rocke_fwd_tapouter_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_tapouter_wmma_spec_t rocke_fwd_tapouter_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_tapouter_wmma_block_k(const rocke_fwd_tapouter_wmma_spec_t* spec);
int rocke_fwd_tapouter_wmma_block_p(const rocke_fwd_tapouter_wmma_spec_t* spec);
int rocke_fwd_tapouter_wmma_tile_h(const rocke_fwd_tapouter_wmma_spec_t* spec);
int rocke_fwd_tapouter_wmma_threads_per_block(const rocke_fwd_tapouter_wmma_spec_t* spec);
long rocke_fwd_tapouter_wmma_lds_bytes(const rocke_fwd_tapouter_wmma_spec_t* spec);
/* spec.grid() -> (N * groups * h_tiles * w_tiles * k_tiles, 1, 1); grid[0] is
 * clamped to INT32_MAX (the validator rejects a larger grid). */
void rocke_fwd_tapouter_wmma_grid(const rocke_fwd_tapouter_wmma_spec_t* spec, int grid[3]);
/* tapouter_tile_efficiency(spec). */
double rocke_fwd_tapouter_wmma_tile_efficiency(const rocke_fwd_tapouter_wmma_spec_t* spec);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}, dil{}x{}, wk{}, wp{},
 * tk{}, tp{}, tw{}, ck{}, st{}, [pf], [s{sched}], [ts{}], [bf16]). */
rocke_status_t rocke_fwd_tapouter_wmma_kernel_name(const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                   char* out,
                                                   size_t out_cap);

/* is_valid_tapouter_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_fwd_tapouter_wmma_is_valid_spec(const rocke_fwd_tapouter_wmma_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

/* build_direct_fwd_tapouter_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_tapouter_wmma(rocke_ir_builder_t* b,
                                                  const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                  const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_tapouter_wmma_new(rocke_ir_builder_t* b,
                                                      const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                      const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_tapouter_wmma_lower_to_llvm(const rocke_fwd_tapouter_wmma_spec_t* spec,
                                                     const char* arch,
                                                     rocke_llvm_flavor_t flavor,
                                                     char** out_ll,
                                                     char* err,
                                                     size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_TAPOUTER_WMMA_H */
