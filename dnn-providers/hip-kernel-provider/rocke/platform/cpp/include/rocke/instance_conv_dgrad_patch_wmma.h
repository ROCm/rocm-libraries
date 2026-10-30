/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_dgrad_patch_wmma.h -- C++ engine port of the wave32 WMMA
 * patch-embedding dgrad builder in library/kernels/common/conv_dgrad_patch_wmma.py.
 *
 * A patch embedding (stride == square filter P, no padding, ungrouped) has no
 * tap reduction, so its dgrad is one GEMM dY[px, K] x W[K, P*P*C] on the
 * gfx1250 WMMA 16x16x32 atom. The operands are staged like the direct WMMA
 * dgrad with a 1x1 filter (one 3-D TDM transfer each per dY channel chunk, the
 * weights transposed on the LDS read); the epilogue pixel-shuffles the GEMM
 * tile in LDS and stores it with one rank-3 TDM transfer. When P does not
 * divide the image, N extra workgroups at the end of the grid zero the dX
 * border no patch covers.
 *
 *   Python (conv_dgrad_patch_wmma.py)        C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DgradPatchWmmaSpec    rocke_dgrad_patch_wmma_spec_t
 *     derived @property geometry             rocke_dgrad_patch_wmma_{block_c,block_p,
 *                                              block_cols,tile_h,threads_per_block,
 *                                              lds_bytes,grid}
 *     .kernel_name()                         rocke_dgrad_patch_wmma_kernel_name
 *   dgrad_patch_reason / is_valid_dgrad_patch_spec
 *                                            rocke_dgrad_patch_wmma_is_valid_spec
 *   build_dgrad_patch_wmma(spec, arch)       rocke_build_dgrad_patch_wmma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_dgrad_patch_wmma_lower_to_llvm
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("dgrad"). Batch, extents and the dY channel
 * count are runtime; the patch size, the dX channel count C (it shapes the
 * shuffle), dtype and the tile geometry are baked.
 *
 * `sched` is the optional Python scheduler_strategy (NULL = backend default).
 */
#ifndef ROCKE_INSTANCE_CONV_DGRAD_PATCH_WMMA_H
#define ROCKE_INSTANCE_CONV_DGRAD_PATCH_WMMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DgradPatchWmmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DgradPatchWmmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "dgrad_patch_wmma"
 *      waves_c: int = 4      # waves along GEMM columns (r, s, c)
 *      waves_p: int = 2      # waves along dY pixels
 *      tiles_c: int = 3      # 16-column tiles per wave
 *      tiles_p: int = 4      # 16-pixel tiles per wave
 *      tile_w: int = 16      # dY columns per block
 *      r_blk: int = 4        # patch rows per block
 *      ck: int = 32          # dY channels per LDS stage
 *      stages: int = 2
 *      prefetch: bool = False
 *      sched: str | None = None
 *      wave_size: int = 32
 * ===================================================================== */
typedef struct rocke_dgrad_patch_wmma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dgrad_patch_wmma" */
    int waves_c; /* default 4  */
    int waves_p; /* default 2  */
    int tiles_c; /* default 3  */
    int tiles_p; /* default 4  */
    int tile_w; /* default 16 */
    int r_blk; /* default 4  */
    int ck; /* default 32 */
    int stages; /* default 2  */
    bool prefetch; /* default false */
    const char* sched; /* default NULL */
    int wave_size; /* default 32 */
} rocke_dgrad_patch_wmma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_dgrad_patch_wmma_spec_t rocke_dgrad_patch_wmma_spec_default(void);

/* is_patch_dgrad_problem(p): stride == square filter > 1, no padding or
 * dilation, ungrouped. */
bool rocke_is_patch_dgrad_problem(const rocke_direct_conv_problem_t* p);

/* Derived geometry (Python @property / methods). */
int rocke_dgrad_patch_wmma_block_c(const rocke_dgrad_patch_wmma_spec_t* spec);
int rocke_dgrad_patch_wmma_block_p(const rocke_dgrad_patch_wmma_spec_t* spec);
int rocke_dgrad_patch_wmma_block_cols(const rocke_dgrad_patch_wmma_spec_t* spec);
int rocke_dgrad_patch_wmma_tile_h(const rocke_dgrad_patch_wmma_spec_t* spec);
int rocke_dgrad_patch_wmma_threads_per_block(const rocke_dgrad_patch_wmma_spec_t* spec);
long rocke_dgrad_patch_wmma_lds_bytes(const rocke_dgrad_patch_wmma_spec_t* spec);
/* spec.grid() -> (main_blocks + (N if a border is uncovered else 0), 1, 1). */
void rocke_dgrad_patch_wmma_grid(const rocke_dgrad_patch_wmma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, p{P}, c{C}, wc{}, wp{}, tc{}, tp{}, tw{},
 * rb{}, ck{}, st{}, [pf], [s{sched}], [bf16]). */
rocke_status_t rocke_dgrad_patch_wmma_kernel_name(const rocke_dgrad_patch_wmma_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_dgrad_patch_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx1250". */
bool rocke_dgrad_patch_wmma_is_valid_spec(const rocke_dgrad_patch_wmma_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* build_dgrad_patch_wmma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_dgrad_patch_wmma(rocke_ir_builder_t* b,
                                                 const rocke_dgrad_patch_wmma_spec_t* spec,
                                                 const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_dgrad_patch_wmma_new(rocke_ir_builder_t* b,
                                                     const rocke_dgrad_patch_wmma_spec_t* spec,
                                                     const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_dgrad_patch_wmma_lower_to_llvm(const rocke_dgrad_patch_wmma_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DGRAD_PATCH_WMMA_H */
