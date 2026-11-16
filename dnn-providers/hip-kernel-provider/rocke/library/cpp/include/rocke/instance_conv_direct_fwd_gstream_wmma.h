/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_gstream_wmma.h -- C++ engine port of the
 * small-group forward in library/kernels/common/conv_direct_fwd_gstream_wmma.py.
 *
 * Grouped 3x3 forward with 8-64 channels per group (cpg == kpg) on the gfx1250
 * wave32 16x16x32 WMMA. A block keeps one channel slice and one column strip
 * and walks `walk` row bands through a ring of `stages` TDM halo slots; the
 * weights stay in VGPRs for the walk. At 8 / 16 channels per group the WMMA K
 * is filled with two filter taps; groups that are not whole 16-row M tiles
 * (56 channels) are padded with zero weights and their padding rows not
 * stored.
 *
 *   Python (conv_direct_fwd_gstream_wmma.py)   C (this header)
 *   ----------------------------------------   ---------------------------------------
 *   @dataclass(frozen) FwdGstreamWmmaSpec       rocke_fwd_gstream_wmma_spec_t
 *     derived @property / methods               rocke_fwd_gstream_wmma_{block_k,block_c,
 *                                                 kpad,tile_h,threads_per_block,
 *                                                 lds_bytes,grid}
 *     .kernel_name()                            rocke_fwd_gstream_wmma_kernel_name
 *   fwd_gstream_tile_efficiency                 rocke_fwd_gstream_wmma_tile_efficiency
 *   fwd_gstream_reason / is_valid_fwd_gstream_spec
 *                                               rocke_fwd_gstream_wmma_is_valid_spec
 *   build_fwd_gstream_wmma(spec, arch)          rocke_build_fwd_gstream_wmma(_new)
 *   (+ convenience: build -> lower .ll)         rocke_fwd_gstream_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents and the group count are
 * runtime; the filter, stride, padding and cpg == kpg are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_GSTREAM_WMMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_GSTREAM_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GSTREAM_VGPR_BUDGET (conv_direct_fwd_gstream_wmma.py). */
#define ROCKE_FWD_GSTREAM_VGPR_BUDGET 448

/* ===================================================================== *
 *  FwdGstreamWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdGstreamWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_fwd_gstream_wmma"
 *      waves_k: int = 1
 *      waves_p: int = 4
 *      tiles_k: int = 1
 *      tiles_p: int = 2
 *      tile_w: int = 16
 *      walk: int = 8
 *      stages: int = 2
 *      phase_split: bool = False
 *      prefetch: bool = True
 *      pf_depth: int = 1
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_fwd_gstream_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_fwd_gstream_wmma" */
    int waves_k; /* default 1 */
    int waves_p; /* default 4 */
    int tiles_k; /* default 1 */
    int tiles_p; /* default 2 */
    int tile_w; /* default 16 */
    int walk; /* default 8 */
    int stages; /* default 2 */
    bool phase_split; /* default false */
    bool prefetch; /* default true */
    int pf_depth; /* default 1 */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_fwd_gstream_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_gstream_wmma_spec_t rocke_fwd_gstream_wmma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_gstream_wmma_kpad(const rocke_fwd_gstream_wmma_spec_t* spec);
int rocke_fwd_gstream_wmma_block_k(const rocke_fwd_gstream_wmma_spec_t* spec);
int rocke_fwd_gstream_wmma_block_c(const rocke_fwd_gstream_wmma_spec_t* spec);
int rocke_fwd_gstream_wmma_tile_h(const rocke_fwd_gstream_wmma_spec_t* spec);
int rocke_fwd_gstream_wmma_threads_per_block(const rocke_fwd_gstream_wmma_spec_t* spec);
long rocke_fwd_gstream_wmma_lds_bytes(const rocke_fwd_gstream_wmma_spec_t* spec);
/* spec.grid() -> (n_kb * n_wt * walkers, 1, 1); grid[0] is clamped to INT32_MAX
 * (the validator rejects a larger grid). */
void rocke_fwd_gstream_wmma_grid(const rocke_fwd_gstream_wmma_spec_t* spec, int grid[3]);
/* fwd_gstream_tile_efficiency(spec). */
double rocke_fwd_gstream_wmma_tile_efficiency(const rocke_fwd_gstream_wmma_spec_t* spec);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}, [str{S}], c{cpg}k{kpg},
 * wk{}, wp{}, tk{}, tp{}, tw{}, wa{}, st{}, [ps], [pf], [pd{}], [s{sched}], [bf16]). */
rocke_status_t rocke_fwd_gstream_wmma_kernel_name(const rocke_fwd_gstream_wmma_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_fwd_gstream_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_fwd_gstream_wmma_is_valid_spec(const rocke_fwd_gstream_wmma_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* build_fwd_gstream_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_gstream_wmma(rocke_ir_builder_t* b,
                                                 const rocke_fwd_gstream_wmma_spec_t* spec,
                                                 const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_gstream_wmma_new(rocke_ir_builder_t* b,
                                                     const rocke_fwd_gstream_wmma_spec_t* spec,
                                                     const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_gstream_wmma_lower_to_llvm(const rocke_fwd_gstream_wmma_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_GSTREAM_WMMA_H */
