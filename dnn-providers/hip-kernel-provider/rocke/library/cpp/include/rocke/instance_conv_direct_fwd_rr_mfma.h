/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_fwd_rr_mfma.h -- C++ engine port of the gfx950
 * rows-in-registers MFMA direct forward builder and its weight-format helper
 * in library/kernels/common/conv_direct_fwd_rr_mfma.py.
 *
 * A dense NHWC 3x3 / 3x1 stride-1 forward convolution: a wave keeps the input
 * rows its output rows need in registers across the filter rows, the weights
 * are pre-formatted (by the helper kernel) into MFMA A-operand tiles that load
 * straight into VGPRs, the input halo tile is double-buffered in LDS (filled
 * by buffer_load ... lds), the 8 waves form two ping-pong groups over a
 * persistent grid, and the epilogue stores from registers.
 *
 *   Python (conv_direct_fwd_rr_mfma.py)       C (this header)
 *   ------------------------------------      ---------------------------------------
 *   @dataclass(frozen) FwdRrMfmaSpec          rocke_fwd_rr_mfma_spec_t
 *     .nwk / .tp / .tk / .threads_per_block   rocke_fwd_rr_mfma_{nwk,tp,tk,threads_per_block}
 *     .tiles() / .grid()                      rocke_fwd_rr_mfma_{tiles,grid}
 *     .workspace_bytes()                      rocke_fwd_rr_mfma_workspace_bytes
 *     .kernel_name()                          rocke_fwd_rr_mfma_kernel_name
 *   fwd_rr_weight_bytes(problem)              rocke_fwd_rr_weight_bytes
 *   fwd_rr_mfma_reason                        rocke_fwd_rr_mfma_is_valid_spec
 *   build_fwd_rr_mfma(spec, *, arch)          rocke_build_fwd_rr_mfma(_new)
 *   @dataclass(frozen) FwdRrWfmtSpec          rocke_fwd_rr_wfmt_spec_t
 *     .kernel_name() / .launch_geometry()     rocke_fwd_rr_wfmt_{kernel_name,grid}
 *   build_fwd_rr_wfmt(spec, *, arch)          rocke_build_fwd_rr_wfmt(_new)
 *   (+ convenience: build -> lower .ll)       rocke_fwd_rr_{mfma,wfmt}_lower_to_llvm
 *
 * AOT: the conv declares the forward direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd") (B = the formatted weights). Batch,
 * extents and both channel counts are runtime; the filter, padding, dtype and
 * the knobs are baked. The helper declares A, D, A_bytes, D_bytes, p_cpg,
 * p_kpg.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_FWD_RR_MFMA_H
#define ROCKE_INSTANCE_CONV_DIRECT_FWD_RR_MFMA_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Blocks of the persistent grid (PERSIST). */
#define ROCKE_FWD_RR_MFMA_PERSIST 256

/* ===================================================================== *
 *  FwdRrMfmaSpec
 *
 *  @dataclass(frozen=True)
 *  class FwdRrMfmaSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_rr_mfma"
 *      wave_size: int = 64
 *      kw16: int = 4        # 16-channel MFMA tiles per wave (4, 3 or 2)
 *      odd_c: bool = False  # C / 64 odd (>= 3)
 *      c64: bool = False    # one step per filter column (both halves)
 *      unfold: bool = False # columns are 32 images of a width-1 layer
 *      k4w: bool = False    # 4 waves along K, 8-row tile
 *      c_tail: bool = False # C a multiple of 8 but not of 64
 *      one_c: bool = False  # C <= 64: one chunk per tile
 *      grp: bool = False    # grouped (runtime group count)
 *      xcd: bool = False    # XCD-local tile walk
 *      chunk: bool = False  # 4 slots of 8 columns (k4w)
 * ===================================================================== */
typedef struct rocke_fwd_rr_mfma_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_rr_mfma" */
    int wave_size; /* default 64 */
    int kw16; /* default 4 */
    bool odd_c;
    bool c64;
    bool unfold;
    bool k4w;
    bool c_tail;
    bool one_c;
    bool grp;
    bool xcd;
    bool chunk;
} rocke_fwd_rr_mfma_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default(). */
rocke_fwd_rr_mfma_spec_t rocke_fwd_rr_mfma_spec_default(void);

/* Derived geometry (Python @property / methods). */
int rocke_fwd_rr_mfma_nwk(const rocke_fwd_rr_mfma_spec_t* spec);
int rocke_fwd_rr_mfma_tp(const rocke_fwd_rr_mfma_spec_t* spec);
int rocke_fwd_rr_mfma_tk(const rocke_fwd_rr_mfma_spec_t* spec);
int rocke_fwd_rr_mfma_threads_per_block(const rocke_fwd_rr_mfma_spec_t* spec);
/* spec.tiles(): column tiles x row tiles x K blocks (Python floor semantics). */
long long rocke_fwd_rr_mfma_tiles(const rocke_fwd_rr_mfma_spec_t* spec);
/* spec.grid() -> (min(tiles, PERSIST), 1, 1). */
void rocke_fwd_rr_mfma_grid(const rocke_fwd_rr_mfma_spec_t* spec, int grid[3]);
/* spec.workspace_bytes() -> fwd_rr_weight_bytes(problem). */
long long rocke_fwd_rr_mfma_workspace_bytes(const rocke_fwd_rr_mfma_spec_t* spec);
/* fwd_rr_weight_bytes: 2 * total_k * 3 * KW * ceil(cpg / 64) * 64. */
long long rocke_fwd_rr_weight_bytes(const rocke_direct_conv_problem_t* problem);

/* kernel_name(): kernel_name_join(name, f{KH}x{KW}, p{PAD}[x{PAD_W}], [k{tk}],
 * [oddc] [c64] [unf] [k4w] [ctail] [onec] [grp] [xcd] [chk] [bf16]). */
rocke_status_t
    rocke_fwd_rr_mfma_kernel_name(const rocke_fwd_rr_mfma_spec_t* spec, char* out, size_t out_cap);

/* fwd_rr_mfma_reason(spec, arch) is None -> (ok, reason). `arch` NULL =>
 * "gfx950"; the reason texts are Python's. */
bool rocke_fwd_rr_mfma_is_valid_spec(const rocke_fwd_rr_mfma_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* build_fwd_rr_mfma(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_fwd_rr_mfma(rocke_ir_builder_t* b,
                                            const rocke_fwd_rr_mfma_spec_t* spec,
                                            const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_fwd_rr_mfma_new(rocke_ir_builder_t* b,
                                                const rocke_fwd_rr_mfma_spec_t* spec,
                                                const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t rocke_fwd_rr_mfma_lower_to_llvm(const rocke_fwd_rr_mfma_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

/* ===================================================================== *
 *  FwdRrWfmtSpec -- the weight-format helper
 *
 *  @dataclass(frozen=True)
 *  class FwdRrWfmtSpec:
 *      problem: DirectConvProblem
 *      name: str = "fwd_rr_wfmt"
 *      threads: int = 256
 *      kw16: int = 4        # the conv's FwdRrMfmaSpec.kw16
 *      fw: int = 3          # filter width (3x3 or 3x1)
 *      nwk: int = 2         # the conv's waves along K
 *      dg: bool = False     # read a dgrad's own KRSC filter (flipped)
 * ===================================================================== */
typedef struct rocke_fwd_rr_wfmt_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "fwd_rr_wfmt" */
    int threads; /* default 256 */
    int kw16; /* default 4 */
    int fw; /* default 3 */
    int nwk; /* default 2 */
    bool dg;
} rocke_fwd_rr_wfmt_spec_t;

rocke_fwd_rr_wfmt_spec_t rocke_fwd_rr_wfmt_spec_default(void);

/* kernel_name(): kernel_name_join(name, [k{32 kw16}], [w{fw}], [wk{nwk}], [dg],
 * [bf16]). */
rocke_status_t
    rocke_fwd_rr_wfmt_kernel_name(const rocke_fwd_rr_wfmt_spec_t* spec, char* out, size_t out_cap);

/* launch_geometry()[0]: (ceil(granules / threads), 1, 1), one thread per
 * 16-byte granule of the formatted weights. */
void rocke_fwd_rr_wfmt_grid(const rocke_fwd_rr_wfmt_spec_t* spec, int grid[3]);

/* build_fwd_rr_wfmt(spec, arch) into an already-initialised builder (the
 * Python builder has no validator: any spec builds). */
rocke_kernel_def_t* rocke_build_fwd_rr_wfmt(rocke_ir_builder_t* b,
                                            const rocke_fwd_rr_wfmt_spec_t* spec,
                                            const char* arch);

rocke_kernel_def_t* rocke_build_fwd_rr_wfmt_new(rocke_ir_builder_t* b,
                                                const rocke_fwd_rr_wfmt_spec_t* spec,
                                                const char* arch);

rocke_status_t rocke_fwd_rr_wfmt_lower_to_llvm(const rocke_fwd_rr_wfmt_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_FWD_RR_MFMA_H */
