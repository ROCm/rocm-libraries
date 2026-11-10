/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_direct_nongrouped.h -- C++ engine port of the non-grouped
 * (groups == 1) NHWC direct convolution builder in
 * library/kernels/common/conv_direct_nongrouped.py.
 *
 * The kernel stages the input tile *with its halo* in LDS once per channel
 * chunk and lets all KH*KW filter taps read shifted sub-tiles out of that one
 * copy; weights are staged pre-swizzled into MFMA fragment order. The channel
 * loop is a runtime scf.for whose loop-carried state is the accumulator tiles
 * plus the global-load registers of the next chunk (software prefetch).
 *
 *   Python (conv_direct_nongrouped.py)            C (this header)
 *   ------------------------------------   -------------------------------------
 *   @dataclass(frozen) DirectNongroupedConvSpec   rocke_direct_conv_nongrouped_spec_t
 *     derived @property geometry            rocke_direct_conv_nongrouped_{lds_bytes,
 *                                             acc_vgprs,threads_per_block,grid}
 *     .kernel_name() / .validate()          rocke_direct_conv_nongrouped_kernel_name /
 *                                             rocke_direct_conv_nongrouped_validate
 *     .workspace_bytes                      rocke_direct_conv_nongrouped_workspace_bytes
 *   is_valid_nongrouped_spec(spec, arch)          rocke_direct_conv_nongrouped_is_valid_spec
 *   build_direct_conv_nongrouped(spec, arch)      rocke_build_direct_conv_nongrouped(_new)
 *   (+ convenience: build -> lower .ll)     rocke_direct_conv_nongrouped_lower_to_llvm
 *   @dataclass(frozen) DirectSplitReduceSpec      rocke_direct_conv_split_reduce_spec_t
 *     .kernel_name() / .validate() / .grid() rocke_direct_conv_split_reduce_{kernel_name,
 *                                             validate,grid}
 *   build_direct_conv_split_reduce(spec, arch)    rocke_build_direct_conv_split_reduce(_new)
 *   (+ convenience: build -> lower .ll)     rocke_direct_conv_split_reduce_lower_to_llvm
 *
 * The shared problem struct (rocke_direct_conv_problem_t, Python
 * DirectConvProblem) comes from rocke/instance_conv_direct_grouped.h. With
 * grouped == false this family requires problem.groups == 1, so cpg is the
 * full C and kpg the full K.
 *
 * AOT: the kernel declares the direct-conv kernarg block of
 * rocke_conv_direct_arg_names("fwd") (instance_conv_abi.h). N, H, W, Ho, Wo,
 * C (p_total_c), K (p_total_k) and the NHWC / NHWK strides are runtime; the
 * filter, stride, PAD, dtype and tile geometry are baked. The spec's problem
 * only has to pass validate() -- the emitted IR does not depend on its extents
 * or channel counts. validate() on the problem actually launched is what keeps
 * every tensor below the masked-access offset, so run it per launch shape.
 *
 * Split-C (split_c > 1): the channel loop is split over split_c workgroups per
 * output tile, each writing an fp32 partial sum into slab `split` of a packed
 * [split_c, N, Ho, Wo, K] workspace bound to D; the DirectSplitReduceSpec
 * kernel (same kernarg block, workspace bound to A) folds the slabs into the
 * output. split_c == 1 emits exactly the unsplit kernel.
 *
 * Grouped (grouped == true): groups >= 1 with p_groups and the per-group
 * channel counts runtime. A workgroup owns group_merge consecutive groups and
 * runs them as one dense problem of group_merge * cpg input and
 * group_merge * kpg output channels with block-diagonal weights (off-diagonal
 * fragments staged as zeros). The merged group is the innermost coordinate of
 * the cell axis. grouped == false emits exactly the dense kernel.
 *
 * Optional Python knobs (`iglp: int | None`, `waves_per_eu: int | None`) map to
 * the sentinels ROCKE_DCONV_NONGROUPED_IGLP_NONE / ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE.
 * Python rejects the sentinel values themselves (iglp < 0, waves_per_eu < 1) and
 * this port rejects anything below them, so a spec means the same in both engines.
 */
#ifndef ROCKE_INSTANCE_CONV_DIRECT_NONGROUPED_H
#define ROCKE_INSTANCE_CONV_DIRECT_NONGROUPED_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* `iglp=None`: leave the backend scheduler alone (no iglp_opt emitted). */
#define ROCKE_DCONV_NONGROUPED_IGLP_NONE (-1)
/* `waves_per_eu=None`: no amdgpu-waves-per-eu attribute. */
#define ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE 0

/* ===================================================================== *
 *  DirectNongroupedConvSpec
 *
 *  @dataclass(frozen=True)
 *  class DirectNongroupedConvSpec:
 *      problem: DirectConvProblem          # groups must be 1
 *      name: str = "direct_conv_nongrouped"
 *      tile_h: int = 16                    # output rows per workgroup
 *      tile_w: int = 32                    # output cols per workgroup
 *      tile_k: int = 128                   # output channels per workgroup
 *      ck: int = 16                        # input channels per LDS chunk
 *      waves_m: int = 2                    # waves splitting tile_k
 *      waves_n: int = 4                    # waves splitting tile_h
 *      atom: str = "32x32x16"              # 32x32x16 | 32x32x8 | 16x16x32 | 16x16x16
 *      wave_size: int = 64
 *      lds_pad: int = 8
 *      chiplet_swizzle: bool = True
 *      swizzle_wgm: int = 8
 *      chiplet_chunk: int = 64
 *      num_xcds: int = 8
 *      double_buffer: bool = False
 *      iglp: int | None = None
 *      waves_per_eu: int | None = None
 *      split_c: int = 1
 *      grouped: bool = False
 *      group_merge: int = 1                # groups per workgroup (grouped only)
 * ===================================================================== */
typedef struct rocke_direct_conv_nongrouped_spec
{
    rocke_direct_conv_problem_t problem;
    const char* name; /* default "direct_conv_nongrouped" */
    int tile_h; /* default 16  */
    int tile_w; /* default 32  */
    int tile_k; /* default 128 */
    int ck; /* default 16  */
    int waves_m; /* default 2   */
    int waves_n; /* default 4   */
    const char* atom; /* default "32x32x16" */
    int wave_size; /* default 64  */
    int lds_pad; /* default 8   */
    bool chiplet_swizzle; /* default true  */
    int swizzle_wgm; /* default 8   */
    int chiplet_chunk; /* default 64  */
    int num_xcds; /* default 8   */
    bool double_buffer; /* default false */
    int iglp; /* default ROCKE_DCONV_NONGROUPED_IGLP_NONE */
    int waves_per_eu; /* default ROCKE_DCONV_NONGROUPED_WAVES_PER_EU_NONE */
    int split_c; /* default 1 */
    bool grouped; /* default false */
    int group_merge; /* default 1 */
} rocke_direct_conv_nongrouped_spec_t;

/* Spec with the Python dataclass defaults; problem == problem_default() with
 * groups = 1. Caller fills N,H,W,cpg,kpg (and KH/KW/PAD/stride/dtype; groups
 * too when grouped). */
rocke_direct_conv_nongrouped_spec_t rocke_direct_conv_nongrouped_spec_default(void);

/* Derived geometry (Python @property). Return 0 for an unknown atom. */
int rocke_direct_conv_nongrouped_threads_per_block(const rocke_direct_conv_nongrouped_spec_t* spec);
long rocke_direct_conv_nongrouped_lds_bytes(const rocke_direct_conv_nongrouped_spec_t* spec);
int rocke_direct_conv_nongrouped_acc_vgprs(const rocke_direct_conv_nongrouped_spec_t* spec);
/* spec.grid() -> (n_w_tiles * n_h_tiles * N * n_merged_groups * n_k_tiles * split_c,
 * 1, 1); n_k_tiles spans merged_k = kpg * group_merge (grouped) or kpg, and
 * n_merged_groups is groups / group_merge (grouped) or 1. */
void rocke_direct_conv_nongrouped_grid(const rocke_direct_conv_nongrouped_spec_t* spec,
                                       int grid[3]);
/* spec.workspace_bytes: bytes of the fp32 [split_c, N, Ho, Wo, total_k]
 * partial-sum workspace a split_c > 1 kernel writes (0 when split_c == 1). */
long long
    rocke_direct_conv_nongrouped_workspace_bytes(const rocke_direct_conv_nongrouped_spec_t* spec);

/* kernel_name(): kernel_name_join(name, problem.short(), t{h}x{w}x{k}, ck{ck},
 * w{wm}x{wn}, a{atom}, g{wgm}|gnone, [db], [iglp{n}], [we{n}], [sc{n}],
 * [grp{group_merge}], [bf16]). */
rocke_status_t rocke_direct_conv_nongrouped_kernel_name(
    const rocke_direct_conv_nongrouped_spec_t* spec, char* out, size_t out_cap);

/* validate(): the hard assertions of DirectNongroupedConvSpec.validate. On a violated
 * invariant returns ROCKE_ERR_VALUE and writes the Python ValueError message
 * into reason (if non-NULL); else ROCKE_OK and "ok". */
rocke_status_t rocke_direct_conv_nongrouped_validate(
    const rocke_direct_conv_nongrouped_spec_t* spec, char* reason, size_t reason_cap);

/* is_valid_nongrouped_spec(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Adds to validate(): stride in {1, 2}, the MFMA atom present on arch, wave
 * size / threads-per-block / LDS capacity within the arch limits. */
bool rocke_direct_conv_nongrouped_is_valid_spec(const rocke_direct_conv_nongrouped_spec_t* spec,
                                                const char* arch,
                                                char* reason,
                                                size_t reason_cap);

/* build_direct_conv_nongrouped(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_conv_nongrouped(
    rocke_ir_builder_t* b, const rocke_direct_conv_nongrouped_spec_t* spec, const char* arch);

/* Initialise `b` with the spec's kernel name, then build (the Python
 * IRBuilder(spec.kernel_name()) + build_direct_conv_nongrouped pair). */
rocke_kernel_def_t* rocke_build_direct_conv_nongrouped_new(
    rocke_ir_builder_t* b, const rocke_direct_conv_nongrouped_spec_t* spec, const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t
    rocke_direct_conv_nongrouped_lower_to_llvm(const rocke_direct_conv_nongrouped_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

/* ===================================================================== *
 *  DirectSplitReduceSpec  -- the second pass of a split_c > 1 kernel
 *
 *  @dataclass(frozen=True)
 *  class DirectSplitReduceSpec:
 *      dtype: str = "fp16"
 *      split_c: int = 2
 *      threads: int = 256
 *      name: str = "direct_conv_split_reduce"
 *
 *  Sums the split_c fp32 slabs of the packed [split_c, N, Ho, Wo, K] workspace
 *  (bound to A) and writes the output (D, NHWK through the p_D_stride_*
 *  kernargs) in dtype. Takes the forward direct-conv kernarg block; B unused.
 * ===================================================================== */
typedef struct rocke_direct_conv_split_reduce_spec
{
    const char* dtype; /* default "fp16" */
    int split_c; /* default 2   */
    int threads; /* default 256 */
    const char* name; /* default "direct_conv_split_reduce" */
} rocke_direct_conv_split_reduce_spec_t;

rocke_direct_conv_split_reduce_spec_t rocke_direct_conv_split_reduce_spec_default(void);

/* kernel_name(): kernel_name_join(name, sc{n}, b{threads}, [bf16]). */
rocke_status_t rocke_direct_conv_split_reduce_kernel_name(
    const rocke_direct_conv_split_reduce_spec_t* spec, char* out, size_t out_cap);

/* validate(): ROCKE_ERR_VALUE + the Python ValueError message on a violated
 * invariant, else ROCKE_OK and "ok". */
rocke_status_t rocke_direct_conv_split_reduce_validate(
    const rocke_direct_conv_split_reduce_spec_t* spec, char* reason, size_t reason_cap);

/* spec.grid(problem) -> (ceil(N * Ho * Wo * total_k / (threads * 8)), 1, 1). */
void rocke_direct_conv_split_reduce_grid(const rocke_direct_conv_split_reduce_spec_t* spec,
                                         const rocke_direct_conv_problem_t* problem,
                                         int grid[3]);

/* build_direct_conv_split_reduce(spec, arch) into an already-initialised builder. */
rocke_kernel_def_t* rocke_build_direct_conv_split_reduce(
    rocke_ir_builder_t* b, const rocke_direct_conv_split_reduce_spec_t* spec, const char* arch);

/* Initialise `b` with the spec's kernel name, then build. */
rocke_kernel_def_t* rocke_build_direct_conv_split_reduce_new(
    rocke_ir_builder_t* b, const rocke_direct_conv_split_reduce_spec_t* spec, const char* arch);

/* Convenience: build + lower to LLVM IR text. *out_ll is malloc'd (caller
 * frees); on failure the builder/lowerer message is written to err. */
rocke_status_t
    rocke_direct_conv_split_reduce_lower_to_llvm(const rocke_direct_conv_split_reduce_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DIRECT_NONGROUPED_H */
