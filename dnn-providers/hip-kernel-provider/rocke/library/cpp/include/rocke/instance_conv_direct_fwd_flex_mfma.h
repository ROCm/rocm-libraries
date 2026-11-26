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
 * lds (im2col through a per-wave LDS offset table); the 8 waves form two
 * ping-pong groups over a persistent grid, taps that read only padding are
 * skipped, and split_c > 1 writes fp32 slabs of the direct_split_reduce
 * workspace. The knobs add an X128 tile, two in-block reduction splits (cs 2 /
 * 4), a dgrad's own KRSC filter (wt), strided dgrads in one launch (ds, h1,
 * zf, zl), channel / output-channel tails, a flattened reduction, an 8-tile
 * wave shape, grouped problems, a positions-inner sweep and XCD-aware walks.
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
 * AOT: the kernel declares the forward (ds: dgrad) direct-conv kernarg block
 * of rocke_conv_direct_arg_names(). Batch, extents, both channel counts and
 * (grp) the group count are runtime; the filter, padding, stride, dilation,
 * dtype and the knobs are baked.
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
 *      bk: int = 256          # output channels per block tile (256/192/128/64)
 *      split_c: int = 1       # pieces of the channel chunks
 *      xcd: bool = False      # XCD-aware walk (2: contiguous runs, 3: no rotation)
 *      kmaj: bool = False     # K-major tile list (with xcd)
 *      c_tail: bool = False   # C % 64 != 0 (C % 8 == 0): the last chunk zero-filled
 *      bx: int = 256          # pixels per block tile (256 or 128)
 *      cs: int = 1            # in-block reduction split (1, 2 or 4)
 *      wt: bool = False       # B is a dgrad's own KRSC filter
 *      ds: bool = False       # a strided dgrad in one launch (needs wt)
 *      h1: bool = False       # ds on a one-row dX
 *      zf: bool = False       # ds with one tapped output phase, zeros stored
 *      zl: int = 0            # zf: light zero units (0: off)
 *      k_tail: bool = False   # K % 8 != 0: 16-bit masked stores
 *      flat: bool = False     # flattened (tap, channel) reduction
 *      xt8: bool = False      # a wave owns 128 pixels x bk / 4 channels
 *      grp: bool = False      # grouped conv (groups > 1)
 *      sw: bool = False       # positions-inner sweep
 *      name: str = "fwd_flex_mfma"
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct rocke_fwd_flex_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    int bk; /* default 256 */
    int split_c; /* default 1 */
    int xcd; /* default 0; 1 (True), 2 or 3 */
    bool kmaj;
    bool c_tail;
    int bx; /* default 256 */
    int cs; /* default 1 */
    bool wt;
    bool ds;
    bool h1;
    bool zf;
    int zl; /* default 0 */
    bool k_tail;
    bool flat;
    bool xt8;
    bool grp;
    bool sw;
    const char* name; /* default "fwd_flex_mfma" */
    int wave_size; /* default 64 */
} rocke_fwd_flex_mfma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_flex_mfma_spec_t rocke_fwd_flex_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_flex_mfma_kt(const rocke_fwd_flex_mfma_spec_t* spec);
int rocke_fwd_flex_mfma_threads_per_block(const rocke_fwd_flex_mfma_spec_t* spec);
/* spec.items(): the persistent grid's work items (tiles x split_c). */
long long rocke_fwd_flex_mfma_items(const rocke_fwd_flex_mfma_spec_t* spec);
/* spec.grid() -> (min(items (+ zl units), 256), 1, 1): the persistent grid. */
void rocke_fwd_flex_mfma_grid(const rocke_fwd_flex_mfma_spec_t* spec, int grid[3]);
/* spec.workspace_bytes -> 0, or the fp32 split-C slabs. */
long long rocke_fwd_flex_mfma_workspace_bytes(const rocke_fwd_flex_mfma_spec_t* spec);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}x{PAD_W}, s{stride},
 * d{dil_h}x{dil_w}, k{bk}, sc{split_c}, [x{bx}], [zl{zl}], flags). */
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
