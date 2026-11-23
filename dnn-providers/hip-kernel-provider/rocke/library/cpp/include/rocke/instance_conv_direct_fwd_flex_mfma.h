/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_flex_mfma.h -- C++ engine port of the gfx950
 * flattened-pixel implicit-GEMM MFMA direct forward builder in
 * library/kernels/common/conv_direct_fwd_flex_mfma.py.
 *
 * A dense forward convolution of any filter, stride, dilation and padding as a
 * GEMM of output channels by flattened output pixels (N * Ho * Wo), reduced
 * over taps x 64-channel chunks. Both operands are staged with buffer_load ...
 * lds into two granule-XOR-swizzled stages; the 8 waves form two ping-pong
 * groups over a persistent grid, taps that read only padding are skipped, and
 * split_c > 1 writes fp32 slabs of the direct_split_reduce workspace.
 *
 *   Python (conv_direct_fwd_flex_mfma.py)    C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) FwdFlexMfmaSpec       rocke_fwd_flex_mfma_spec_t
 *     .kt / .threads_per_block               rocke_fwd_flex_mfma_{kt,threads_per_block,
 *     .items() / .grid()                       items,grid,workspace_bytes}
 *     .workspace_bytes
 *     .kernel_name()                         rocke_fwd_flex_mfma_kernel_name
 *   fwd_flex_mfma_reason                     rocke_fwd_flex_mfma_is_valid_spec
 *   build_fwd_flex_mfma(spec, *, arch)       rocke_build_fwd_flex_mfma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_fwd_flex_mfma_lower_to_llvm
 *
 * AOT: the kernel declares the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents and both channel counts
 * are runtime; the filter, padding, stride, dilation, dtype, bk and split_c
 * are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  FwdFlexMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdFlexMfmaSpec:
 *      problem: DirectConvProblem
 *      bk: int = 256                # output channels per block tile (256 or 128)
 *      split_c: int = 1             # pieces of the channel chunks
 *      name: str = "fwd_flex_mfma"
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_fwd_flex_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    int bk; /* default 256 */
    int split_c; /* default 1 */
    const char* name; /* default "fwd_flex_mfma" */
    int wave_size; /* default 64 */
} rocke_fwd_flex_mfma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_flex_mfma_spec_t rocke_fwd_flex_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_flex_mfma_kt(const rocke_fwd_flex_mfma_spec_t* spec);
int rocke_fwd_flex_mfma_threads_per_block(const rocke_fwd_flex_mfma_spec_t* spec);
/* spec.items() -> ceil(N * Ho * Wo / 256) * ceil(total_k / bk) * split_c. */
long long rocke_fwd_flex_mfma_items(const rocke_fwd_flex_mfma_spec_t* spec);
/* spec.grid() -> (min(items, 256), 1, 1): the persistent grid. */
void rocke_fwd_flex_mfma_grid(const rocke_fwd_flex_mfma_spec_t* spec, int grid[3]);
/* spec.workspace_bytes -> 0, or 4 * split_c * N * Ho * Wo * total_k. */
long long rocke_fwd_flex_mfma_workspace_bytes(const rocke_fwd_flex_mfma_spec_t* spec);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}x{PAD_W}, s{stride},
 * d{dil_h}x{dil_w}, k{bk}, sc{split_c}, [bf16]). */
rocke_status_t rocke_fwd_flex_mfma_kernel_name(const rocke_fwd_flex_mfma_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* fwd_flex_mfma_reason(spec, arch) is None -> (ok, reason). `arch` NULL =>
 * "gfx950"; the reason texts are Python's. */
bool rocke_fwd_flex_mfma_is_valid_spec(const rocke_fwd_flex_mfma_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* build_fwd_flex_mfma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_flex_mfma(rocke_ir_builder_t* b,
                                              const rocke_fwd_flex_mfma_spec_t* spec,
                                              const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_flex_mfma_new(rocke_ir_builder_t* b,
                                                  const rocke_fwd_flex_mfma_spec_t* spec,
                                                  const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_flex_mfma_lower_to_llvm(const rocke_fwd_flex_mfma_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_FLEX_MFMA_H */
