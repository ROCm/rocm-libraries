/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_conv_direct_grouped.h -- C99 port of the two direct grouped
 * convolution kernel instance builders in
 * ck_dsl/instances/common/conv_direct_grouped.py.
 *
 * Direct grouped conv is a DSL-native streaming row-by-row pipeline: each output
 * row is computed by streaming the input row through MFMAs without ever
 * materialising an im2col / implicit-GEMM tile. Two kernels share the authoring
 * surface and differ only in BLOCK_GROUPS + the MFMA atom:
 *   - 16c variant (cpg=kpg=16): one wave owns one group; mfma_f32_16x16x16_f16
 *     (and, when fold_k32=True, the wide 16x16x32 f16 atom for S=0/1). 8 waves /
 *     block, LDS double-buffered ping-pong, 3-slot circular accumulator over H.
 *   - 4c  variant (cpg=kpg=4): mfma_f32_4x4x4_f16 emits 16 independent 4x4x4
 *     matmuls per wave, mapping one wave to 16 groups at once. No LDS staging;
 *     per-lane register inputs only.
 *
 *   Python (conv_direct_grouped.py)        C99 (this header)
 *   ------------------------------------   -------------------------------------
 *   @dataclass(frozen) DirectConvProblem    ckc_direct_conv_problem_t
 *     .total_c/.total_k/.flops (@property)   ckc_direct_conv_problem_total_c/...
 *     .short()                               ckc_direct_conv_problem_short
 *   @dataclass(frozen) DirectConv16cSpec     ckc_direct_conv_16c_spec_t
 *     .threads_per_block / .n_acc_slots      ckc_direct_conv_16c_*
 *     .kernel_name() / .validate()           ckc_direct_conv_16c_kernel_name / _validate
 *   @dataclass(frozen) DirectConv4cSpec      ckc_direct_conv_4c_spec_t
 *     .threads_per_block                     ckc_direct_conv_4c_threads_per_block
 *     .kernel_name() / .validate()           ckc_direct_conv_4c_kernel_name / _validate
 *   is_valid_spec_16c(spec, arch)            ckc_direct_conv_16c_is_valid_spec
 *   is_valid_spec_4c(spec, arch)             ckc_direct_conv_4c_is_valid_spec
 *   build_direct_conv_16c(spec, arch)        ckc_build_direct_conv_16c
 *   build_direct_conv_4c(spec, arch)         ckc_build_direct_conv_4c
 *   (+ convenience: build -> lower .ll)      ckc_direct_conv_{16c,4c}_lower_to_llvm
 *
 * SPEC AS EXPLICIT C STRUCTS. The frozen Python dataclasses become value
 * structs; ckc_direct_conv_problem_default() / ckc_direct_conv_16c_spec_default()
 * / ckc_direct_conv_4c_spec_default() install the Python dataclass defaults so a
 * caller overrides only the fields it cares about. Both spec structs embed the
 * shared problem by value (Python `problem: DirectConvProblem`).
 *
 * REUSED PORTED HELPERS (no new helper port required for this instance):
 *   - ckc/helper_ck_dsl.helpers.transforms.h : TensorDescriptor.naive/.transform
 *     /.offset/.unmerge_lower + embed + unmerge_magic (the entire conv address
 *     algebra). All already ported.
 *   - ckc/helper_ck_dsl.helpers.spec.h       : kernel_name_join, ckc_sig_entry_t.
 *   - ckc/helper_ck_dsl.core.arch.h          : ArchTarget.from_gfx + mma.has_shape.
 *
 * Error model mirrors the rest of the C port: build/lower route errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gates return a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 *
 * Internal build-context + phase-function contract live in
 * ckc/instance_conv_direct_grouped_internal.h (included only by the .c TUs).
 */
#ifndef CKC_INSTANCE_CONV_DIRECT_GROUPED_H
#define CKC_INSTANCE_CONV_DIRECT_GROUPED_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ckc_sig_entry; /* fwd (ckc/helper_ck_dsl.helpers.spec.h) */
struct ckc_arena; /* fwd (ckc/arena.h)                      */

/* ===================================================================== *
 *  DirectConvProblem
 *
 *  @dataclass(frozen=True)
 *  class DirectConvProblem:
 *      N, H, W, groups, cpg, kpg          # required
 *      KH=3, KW=3, PAD=1, stride=1
 *
 *  Layouts:
 *    A: NHWC fp16, [N, H, W, groups*cpg]
 *    B: KRSC fp16, [groups*kpg, KH, KW, cpg]
 *    D: NHWK fp16, [N, H, W, groups*kpg]
 * ===================================================================== */
typedef struct ckc_direct_conv_problem
{
    int N;
    int H;
    int W;
    int groups;
    int cpg; /* channels per group */
    int kpg; /* filters per group  */
    int KH; /* default 3 */
    int KW; /* default 3 */
    int PAD; /* default 1 */
    int stride; /* default 1 */
} ckc_direct_conv_problem_t;

/* DirectConvProblem with dataclass defaults (KH=KW=3, PAD=1, stride=1) and the
 * six required dims zeroed. Caller fills N,H,W,groups,cpg,kpg. */
ckc_direct_conv_problem_t ckc_direct_conv_problem_default(void);

/* @property total_c -> groups * cpg. */
int ckc_direct_conv_problem_total_c(const ckc_direct_conv_problem_t* p);
/* @property total_k -> groups * kpg. */
int ckc_direct_conv_problem_total_k(const ckc_direct_conv_problem_t* p);
/* @property flops -> 2*N*H*W*groups*kpg*KH*KW*cpg (returned as int64). */
long long ckc_direct_conv_problem_flops(const ckc_direct_conv_problem_t* p);

/* short(): f"N{N}H{H}W{W}_g{groups}_c{cpg}k{kpg}". Writes NUL-terminated into
 * out (capacity out_cap). Returns CKC_OK or CKC_ERR_VALUE (NULL / too small). */
ckc_status_t
    ckc_direct_conv_problem_short(const ckc_direct_conv_problem_t* p, char* out, size_t out_cap);

/* ===================================================================== *
 *  DirectConv16cSpec  (cpg = kpg = 16)
 *
 *  @dataclass(frozen=True)
 *  class DirectConv16cSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_conv_16c"
 *      block_q: int = 16
 *      block_groups: int = 8
 *      wave_size: int = 64
 *      double_buffer: bool = True
 *      fold_k32: bool = True
 * ===================================================================== */
typedef struct ckc_direct_conv_16c_spec
{
    ckc_direct_conv_problem_t problem;
    const char* name; /* default "direct_conv_16c" */
    int block_q; /* default 16 */
    int block_groups; /* default 8  */
    int wave_size; /* default 64 */
    bool double_buffer; /* default true  */
    bool fold_k32; /* default true  */
} ckc_direct_conv_16c_spec_t;

/* Default 16c spec (name "direct_conv_16c", block_q 16, block_groups 8,
 * wave_size 64, double_buffer true, fold_k32 true, problem == default()). */
ckc_direct_conv_16c_spec_t ckc_direct_conv_16c_spec_default(void);

/* @property threads_per_block -> block_groups * wave_size. */
int ckc_direct_conv_16c_threads_per_block(const ckc_direct_conv_16c_spec_t* spec);
/* @property n_acc_slots -> problem.KH. */
int ckc_direct_conv_16c_n_acc_slots(const ckc_direct_conv_16c_spec_t* spec);

/* kernel_name():
 *   kernel_name_join(name, problem.short(), f"bq{block_q}", f"bg{block_groups}",
 *                    "db" if double_buffer else "sb", flags={"k32": fold_k32})
 * Writes NUL-terminated into out (capacity out_cap). */
ckc_status_t ckc_direct_conv_16c_kernel_name(const ckc_direct_conv_16c_spec_t* spec,
                                             char* out,
                                             size_t out_cap);

/* validate(): the hard assertions of DirectConv16cSpec.validate (cpg==kpg==16,
 * groups % block_groups == 0). On a violated invariant returns CKC_ERR_VALUE and
 * (if reason non-NULL, cap reason_cap) writes the message; else CKC_OK. */
ckc_status_t ckc_direct_conv_16c_validate(const ckc_direct_conv_16c_spec_t* spec,
                                          char* reason,
                                          size_t reason_cap);

/* is_valid_spec_16c(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Checks: ArchTarget.from_gfx(arch) resolves; cpg==kpg==16; groups % block_groups
 * == 0; the 16x16x16 f16 MFMA atom present on arch; and when fold_k32 the
 * 16x16x32 f16 atom present on arch (absent on gfx942 -> clean reject). On reject
 * writes the reason (if non-NULL) and returns false; on accept writes "ok",
 * returns true. */
bool ckc_direct_conv_16c_is_valid_spec(const ckc_direct_conv_16c_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* ===================================================================== *
 *  DirectConv4cSpec  (cpg = kpg = 4)
 *
 *  @dataclass(frozen=True)
 *  class DirectConv4cSpec:
 *      problem: DirectConvProblem
 *      name: str = "direct_conv_4c"
 *      block_q: int = 4
 *      block_groups: int = 16
 *      wave_size: int = 64
 * ===================================================================== */
typedef struct ckc_direct_conv_4c_spec
{
    ckc_direct_conv_problem_t problem;
    const char* name; /* default "direct_conv_4c" */
    int block_q; /* default 4  */
    int block_groups; /* default 16 */
    int wave_size; /* default 64 */
} ckc_direct_conv_4c_spec_t;

/* Default 4c spec (name "direct_conv_4c", block_q 4, block_groups 16,
 * wave_size 64, problem == default()). */
ckc_direct_conv_4c_spec_t ckc_direct_conv_4c_spec_default(void);

/* @property threads_per_block -> (block_groups // 16) * wave_size. */
int ckc_direct_conv_4c_threads_per_block(const ckc_direct_conv_4c_spec_t* spec);

/* kernel_name():
 *   kernel_name_join(name, problem.short(), f"bq{block_q}", f"bg{block_groups}")
 * Writes NUL-terminated into out (capacity out_cap). */
ckc_status_t ckc_direct_conv_4c_kernel_name(const ckc_direct_conv_4c_spec_t* spec,
                                            char* out,
                                            size_t out_cap);

/* validate(): the hard assertions of DirectConv4cSpec.validate (cpg==kpg==4,
 * block_groups % 16 == 0, block_q % 4 == 0, groups % block_groups == 0). On a
 * violated invariant returns CKC_ERR_VALUE + (reason if non-NULL); else CKC_OK. */
ckc_status_t ckc_direct_conv_4c_validate(const ckc_direct_conv_4c_spec_t* spec,
                                         char* reason,
                                         size_t reason_cap);

/* is_valid_spec_4c(spec, arch) -> (ok, reason). `arch` NULL => "gfx950".
 * Checks: ArchTarget.from_gfx(arch) resolves; cpg==kpg==4; block_groups % 16 == 0;
 * block_q % 4 == 0; groups % block_groups == 0. The 4x4x4 f16 atom is NOT gated
 * through has_shape (catalog lists only warp tiles; comgr selects it on both
 * targets). On reject writes the reason and returns false; else "ok" + true. */
bool ckc_direct_conv_4c_is_valid_spec(const ckc_direct_conv_4c_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap);

/* ===================================================================== *
 *  BUILD ENTRIES
 * ===================================================================== */

/* build_direct_conv_16c(spec, arch). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does (validate() then is_valid_spec_16c gate then the streaming
 * pipeline), and returns the kernel (b->kernel) on success or NULL with b's
 * sticky error set. `arch` NULL => "gfx950". Does NOT re-init the builder. */
ckc_kernel_def_t* ckc_build_direct_conv_16c(ckc_ir_builder_t* b,
                                            const ckc_direct_conv_16c_spec_t* spec,
                                            const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_16c.
 * Caller owns `b` and frees it with ckc_ir_builder_free(). Returns kernel/NULL. */
ckc_kernel_def_t* ckc_build_direct_conv_16c_new(ckc_ir_builder_t* b,
                                                const ckc_direct_conv_16c_spec_t* spec,
                                                const char* arch);

/* build_direct_conv_4c(spec, arch). Same contract as the 16c entry for the 4c
 * (mfma_f32_4x4x4_f16) kernel. */
ckc_kernel_def_t* ckc_build_direct_conv_4c(ckc_ir_builder_t* b,
                                           const ckc_direct_conv_4c_spec_t* spec,
                                           const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build_direct_conv_4c. */
ckc_kernel_def_t* ckc_build_direct_conv_4c_new(ckc_ir_builder_t* b,
                                               const ckc_direct_conv_4c_spec_t* spec,
                                               const char* arch);

/* ===================================================================== *
 *  SIGNATURE (manifest)  --  both kernels share the 6-entry ABI:
 *    ptr A:f16, ptr B:f16, ptr D:f16, scalar A_bytes:i32, B_bytes:i32,
 *    D_bytes:i32.
 * ===================================================================== */

/* Writes the 6 manifest entries into out[] (capacity out_cap) and sets
 * *out_count = 6. Strings live in `arena`. Returns CKC_OK or CKC_ERR_VALUE
 * (NULL args / out_cap < 6). One signature serves both 16c and 4c. */
ckc_status_t ckc_direct_conv_signature(struct ckc_arena* arena,
                                       struct ckc_sig_entry* out,
                                       size_t out_cap,
                                       size_t* out_count);

/* ===================================================================== *
 *  CONVENIENCE: build -> lower to LLVM .ll text.
 *  `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd
 *  NUL-terminated string the caller frees with free(); on failure it is left
 *  NULL and (if err!=NULL, cap err_cap) a diagnostic is written. Each owns and
 *  frees its IRBuilder.
 * ===================================================================== */
ckc_status_t ckc_direct_conv_16c_lower_to_llvm(const ckc_direct_conv_16c_spec_t* spec,
                                               const char* arch,
                                               ckc_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap);

ckc_status_t ckc_direct_conv_4c_lower_to_llvm(const ckc_direct_conv_4c_spec_t* spec,
                                              const char* arch,
                                              ckc_llvm_flavor_t flavor,
                                              char** out_ll,
                                              char* err,
                                              size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_CONV_DIRECT_GROUPED_H */
