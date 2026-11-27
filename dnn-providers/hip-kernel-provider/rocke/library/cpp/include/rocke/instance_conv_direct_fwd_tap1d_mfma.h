/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_tap1d_mfma.h -- C++ engine port of the gfx950
 * 1-D tap-reuse MFMA direct forward builder in
 * library/kernels/common/conv_direct_fwd_tap1d_mfma.py.
 *
 * A dense NHWC forward convolution with a 1 x S filter (any width dilation
 * and padding, stride 1): a block computes BX consecutive output pixels of one
 * row x bk output channels; MFMA columns follow a polyphase lane map, so the
 * input fragments a chunk reads from LDS once are reused by every tap from
 * registers. The input window and the per-tap weights are LDS-DMA'd one step
 * ahead.
 *
 *   Python (conv_direct_fwd_tap1d_mfma.py)    C (this header)
 *   -------------------------------------     ---------------------------------------
 *   @dataclass(frozen) FwdTap1dMfmaSpec       rocke_fwd_tap1d_mfma_spec_t
 *     .nwk / .nwx / .ncol / .bx / .win        rocke_fwd_tap1d_mfma_{nwk,nwx,ncol,bx,win}
 *     .win_dmas / .wgt_dmas / .lds_bytes      rocke_fwd_tap1d_mfma_{win_dmas,wgt_dmas,lds_bytes}
 *     .threads_per_block / .grid()            rocke_fwd_tap1d_mfma_{threads_per_block,grid}
 *     .kernel_name()                          rocke_fwd_tap1d_mfma_kernel_name
 *   fwd_tap1d_mfma_reason                     rocke_fwd_tap1d_mfma_is_valid_spec
 *   build_fwd_tap1d_mfma(spec, *, arch)       rocke_build_fwd_tap1d_mfma(_new)
 *   (+ convenience: build -> lower .ll)       rocke_fwd_tap1d_mfma_lower_to_llvm
 *
 * AOT: the kernel declares the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd"). Batch, extents, C and K are runtime; the
 * filter width, padding, dilation, tile, chunk width and the single-chunk flag
 * are baked.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_TAP1D_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_TAP1D_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================================================================== *
 *  FwdTap1dMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdTap1dMfmaSpec:
 *      problem: DirectConvProblem
 *      bk: int = 128       # output channels per block tile (32, 64 or 128)
 *      nwv: int = 8        # waves per block (4 or 8)
 *      seg: int = 9        # output pixels per MFMA column
 *      ck: int = 64        # channels per chunk (64 or 32)
 *      one: bool = False   # C == ck: a single chunk, one window buffer
 *      name: str = "fwd_tap1d_mfma"
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_fwd_tap1d_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    int bk; /* default 128 */
    int nwv; /* default 8 */
    int seg; /* default 9 */
    int ck; /* default 64 */
    bool one;
    const char* name; /* default "fwd_tap1d_mfma" */
    int wave_size; /* default 64 */
} rocke_fwd_tap1d_mfma_spec_t;

rocke_fwd_tap1d_mfma_spec_t rocke_fwd_tap1d_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_tap1d_mfma_nwk(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_nwx(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_ncol(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_bx(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_win(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_win_dmas(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_wgt_dmas(const rocke_fwd_tap1d_mfma_spec_t* spec);
long long rocke_fwd_tap1d_mfma_lds_bytes(const rocke_fwd_tap1d_mfma_spec_t* spec);
int rocke_fwd_tap1d_mfma_threads_per_block(const rocke_fwd_tap1d_mfma_spec_t* spec);
/* spec.grid() -> (N * Ho * ceil(Wo / bx) * (total_k / bk), 1, 1). */
void rocke_fwd_tap1d_mfma_grid(const rocke_fwd_tap1d_mfma_spec_t* spec, int grid[3]);

/* kernel_name(): kernel_name_join(name, f1x{KW}, p{PAD_W}, d{dil_w}, k{bk},
 * w{nwv}, g{seg}, c{ck}, [one], [bf16]). */
rocke_status_t rocke_fwd_tap1d_mfma_kernel_name(const rocke_fwd_tap1d_mfma_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* fwd_tap1d_mfma_reason(spec, arch) is None -> (ok, reason). `arch` NULL =>
 * "gfx950"; the reason texts are Python's. */
bool rocke_fwd_tap1d_mfma_is_valid_spec(const rocke_fwd_tap1d_mfma_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

rocke_kernel_def_t* rocke_build_fwd_tap1d_mfma(rocke_ir_builder_t* b,
                                               const rocke_fwd_tap1d_mfma_spec_t* spec,
                                               const char* arch);

rocke_kernel_def_t* rocke_build_fwd_tap1d_mfma_new(rocke_ir_builder_t* b,
                                                   const rocke_fwd_tap1d_mfma_spec_t* spec,
                                                   const char* arch);

rocke_status_t rocke_fwd_tap1d_mfma_lower_to_llvm(const rocke_fwd_tap1d_mfma_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_TAP1D_MFMA_H */
