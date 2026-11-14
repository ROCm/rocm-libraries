/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_dgrad_pp_mfma.h -- C++ engine port of the gfx950
 * LDS-DMA ping-pong MFMA direct dgrad builder in
 * library/kernels/common/conv_direct_dgrad_pp_mfma.py.
 *
 * A dense stride-1 dgrad computed as the forward pass over the transposed
 * problem (dY in, dX out, flipped padding) by a channel-chunked direct kernel.
 * Every operand byte is staged with buffer_load ... lds; the 8 waves form two
 * ranks (weights of the next filter tap, input of the next channel chunk) that
 * alternate barrier / setprio 1 / MFMAs / setprio 0 / barrier steps; the
 * original KRSC filter is read flipped and transposed in-kernel with
 * ds_read_b64_tr_b16 over a granule-XOR-swizzled weight tile.
 *
 *   Python (conv_direct_dgrad_pp_mfma.py)    C (this header)
 *   ------------------------------------     ---------------------------------------
 *   @dataclass(frozen) DgradPpMfmaSpec       rocke_dgrad_pp_mfma_spec_t
 *     .lds_bytes() / .threads_per_block      rocke_dgrad_pp_mfma_{lds_bytes,
 *     .grid()                                  threads_per_block,grid}
 *     .kernel_name()                         rocke_dgrad_pp_mfma_kernel_name
 *   dgrad_pp_mfma_reason / is_valid_dgrad_pp_mfma_spec
 *                                            rocke_dgrad_pp_mfma_is_valid_spec
 *   build_dgrad_pp_mfma(spec, arch)          rocke_build_dgrad_pp_mfma(_new)
 *   (+ convenience: build -> lower .ll)      rocke_dgrad_pp_mfma_lower_to_llvm
 *
 * The spec's problem is the forward pass the dgrad runs as
 * (dgrad_pp_mfma_problem): cpg = the dgrad's K (the reduction), kpg = its C
 * (the outputs), PAD = KH - 1 - the dgrad's PAD.
 *
 * AOT: the kernel declares the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents and both channel counts
 * are runtime; the filter, PAD, dtype and block_k are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_DGRAD_PP_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_DGRAD_PP_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  DgradPpMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class DgradPpMfmaSpec:
 *      problem: DirectConvProblem   # the transposed (forward-pass) problem
 *      name: str = "dgrad_pp_mfma"
 *      block_k: int = 128           # output channels per block (128 or 256)
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_dgrad_pp_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "dgrad_pp_mfma" */
    int block_k; /* default 128 */
    int wave_size; /* default 64 */
} rocke_dgrad_pp_mfma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_dgrad_pp_mfma_spec_t rocke_dgrad_pp_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_dgrad_pp_mfma_threads_per_block(const rocke_dgrad_pp_mfma_spec_t* spec);
long rocke_dgrad_pp_mfma_lds_bytes(const rocke_dgrad_pp_mfma_spec_t* spec);
/* spec.grid() -> (N * ceil(Ho / 16) * ceil(Wo / 16) * ceil(total_k / block_k), 1, 1). */
void rocke_dgrad_pp_mfma_grid(const rocke_dgrad_pp_mfma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}, bk{block_k}, [bf16]). */
rocke_status_t rocke_dgrad_pp_mfma_kernel_name(const rocke_dgrad_pp_mfma_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* is_valid_dgrad_pp_mfma_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950". */
bool rocke_dgrad_pp_mfma_is_valid_spec(const rocke_dgrad_pp_mfma_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* build_dgrad_pp_mfma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_dgrad_pp_mfma(rocke_ir_builder_t* b,
                                              const rocke_dgrad_pp_mfma_spec_t* spec,
                                              const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_dgrad_pp_mfma_new(rocke_ir_builder_t* b,
                                                  const rocke_dgrad_pp_mfma_spec_t* spec,
                                                  const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_dgrad_pp_mfma_lower_to_llvm(const rocke_dgrad_pp_mfma_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_DGRAD_PP_MFMA_H */
