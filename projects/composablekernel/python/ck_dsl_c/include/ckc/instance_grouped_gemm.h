/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_grouped_gemm.h -- C99 port of the grouped GEMM instance builder
 * ck_dsl/instances/common/grouped_gemm.py (CK Tile 17_grouped_gemm parity).
 *
 * KEY FINDING: GroupedGemmSpec is a thin wrapper around UniversalGemmSpec.
 * build_grouped_gemm() converts GroupedGemmSpec -> UniversalGemmSpec
 * (batched=False) then delegates to build_universal_gemm(). The single-launch
 * variant does the same with batched=True. There is NO new IR generation here;
 * all computation lives in the already-ported gemm_universal machinery, so the
 * lowered IR is byte-identical to the universal-GEMM body.
 *
 *   Python (grouped_gemm.py)                 C99 (this header)
 *   --------------------------------------   -----------------------------------
 *   class GroupedGemmProblem                 ckc_grouped_gemm_problem_t
 *   class GroupedGemmSpec                    ckc_grouped_gemm_spec_t
 *   GroupedGemmSpec._data_spec()             ckc_grouped_gemm_data_spec()
 *   GroupedGemmSpec.to_universal_spec()      ckc_grouped_gemm_to_universal_spec()
 *   GroupedGemmSpec.kernel_name()            ckc_grouped_gemm_kernel_name()
 *   is_valid_spec(spec, arch)                ckc_grouped_gemm_is_valid_spec()
 *   build_grouped_gemm(spec, arch)           ckc_build_grouped_gemm()
 *   grouped_gemm_signature(spec)             ckc_grouped_gemm_signature()
 *   build_grouped_gemm_single_launch(...)    ckc_build_grouped_gemm_single_launch()
 *   grouped_gemm_single_launch_signature()   ckc_grouped_gemm_single_launch_signature()
 *   (+ convenience: build -> lower .ll)      ckc_grouped_gemm_lower_to_llvm()
 *
 * GroupedGemmLauncher / GroupedGemmSingleLaunchRunner / grouped_gemm_problems()
 * are Python host-side runtime launchers (torch / hipModuleLaunchKernel glue);
 * they emit NO IR and are not part of the kernel-build (IR-verification) API,
 * so they are intentionally not ported here.
 */
#ifndef CKC_INSTANCE_GROUPED_GEMM_H
#define CKC_INSTANCE_GROUPED_GEMM_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h"
#include "ckc/helper_ck_dsl.helpers.spec.h" /* ckc_sig_entry_t, SignatureBuilder */
#include "ckc/instance_gemm_universal.h"
#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- problem entry *
 *
 * Mirror of Python GroupedGemmProblem (frozen dataclass). One entry in a
 * grouped GEMM workload. Pointers are device pointers; strides are in elements;
 * layout is RCR (row-major A, col-major B, row-major C). This is host-side
 * launch metadata (no IR), provided for ABI completeness. */
typedef struct ckc_grouped_gemm_problem
{
    int M;
    int N;
    int K;
    unsigned long long A_ptr;
    unsigned long long B_ptr;
    unsigned long long C_ptr;
} ckc_grouped_gemm_problem_t;

/* ----------------------------------------------------------------- the spec *
 *
 * Mirror of Python GroupedGemmSpec(WarpTileBlockSizeMixin). The same tile/trait
 * pair is applied to every group. block_size==0 is derived at finalize() via
 * WarpTileBlockSizeMixin._init_block_size() (warp_m*warp_n*warp_k*wave_size). */
typedef struct ckc_grouped_gemm_spec
{
    const char* name;
    ckc_gemm_tile_spec_t tile;
    ckc_gemm_trait_spec_t trait;
    int wave_size;     /* default 64 */
    int block_size;    /* default 0 => derived at finalize() */
    const char* dtype; /* default "fp16" */
} ckc_grouped_gemm_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `name` and the required `tile` geometry. trait defaults
 * track ckc_gemm_universal_spec_default()'s trait defaults. */
ckc_grouped_gemm_spec_t ckc_grouped_gemm_spec_default(void);

/* WarpTileBlockSizeMixin._init_block_size() (== __post_init__): when
 * block_size==0, derive warp_m*warp_n*warp_k*wave_size. Idempotent. */
void ckc_grouped_gemm_spec_finalize(ckc_grouped_gemm_spec_t* spec);

/* GroupedGemmSpec._data_spec(): dt = "fp16" if dtype in ("f16","fp16") else
 * dtype; returns DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt) with the
 * remaining (acc/layout) fields at their universal-GEMM defaults. */
ckc_gemm_data_spec_t ckc_grouped_gemm_data_spec(const ckc_grouped_gemm_spec_t* spec);

/* GroupedGemmSpec.to_universal_spec(): build the UniversalGemmSpec(batched=False)
 * the per-group build delegates to. The returned spec is already finalized
 * (block_size derived) so its kernel_name()/build are usable directly. */
ckc_gemm_universal_spec_t ckc_grouped_gemm_to_universal_spec(const ckc_grouped_gemm_spec_t* spec);

/* GroupedGemmSpec.kernel_name() == to_universal_spec().kernel_name().
 * Writes NUL-terminated into out (capacity out_cap). */
ckc_status_t
ckc_grouped_gemm_kernel_name(const ckc_grouped_gemm_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec, arch): delegates to ckc_gemm_universal_is_valid_spec on
 * to_universal_spec(). `arch` NULL => "gfx950". On reject writes the structured
 * reason into `reason` (capacity reason_cap) and returns false. */
bool ckc_grouped_gemm_is_valid_spec(const ckc_grouped_gemm_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);

/* build_grouped_gemm(spec, arch): the per-group base kernel. Validates the
 * converted UniversalGemmSpec (raises ValueError in Python -> sets b's sticky
 * error + returns NULL here) then delegates to ckc_build_universal_gemm. Builds
 * into the supplied (already ckc_ir_builder_init'd with spec->kernel_name())
 * builder `b`. `arch` NULL => "gfx950". */
ckc_kernel_def_t*
ckc_build_grouped_gemm(ckc_ir_builder_t* b, const ckc_grouped_gemm_spec_t* spec, const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. Returns kernel or
 * NULL with b's sticky error set. */
ckc_kernel_def_t* ckc_build_grouped_gemm_new(ckc_ir_builder_t* b,
                                             const ckc_grouped_gemm_spec_t* spec,
                                             const char* arch);

/* grouped_gemm_signature(spec): (A, B, C ptrs ; M, N, K i32). ptr_dt is
 * spec->dtype if it is one of "f16"/"fp16"/"bf16" else "f16". Entries are built
 * into `arena` via the SignatureBuilder chain; *out_items / *out_count expose
 * the (arena-owned) array. */
ckc_status_t ckc_grouped_gemm_signature(const ckc_grouped_gemm_spec_t* spec,
                                        ckc_arena_t* arena,
                                        const ckc_sig_entry_t** out_items,
                                        size_t* out_count);

/* build_grouped_gemm_single_launch(spec, arch): the single-launch (batched)
 * grouped GEMM. Converts to UniversalGemmSpec, renames to name+"_single_launch",
 * sets batched=True, validates, then delegates to ckc_build_universal_gemm.
 * Builds into `b` (init'd with the single-launch kernel name). */
ckc_kernel_def_t* ckc_build_grouped_gemm_single_launch(ckc_ir_builder_t* b,
                                                       const ckc_grouped_gemm_spec_t* spec,
                                                       const char* arch);

/* Convenience: init `b` with the single-launch kernel name, then build. */
ckc_kernel_def_t* ckc_build_grouped_gemm_single_launch_new(ckc_ir_builder_t* b,
                                                           const ckc_grouped_gemm_spec_t* spec,
                                                           const char* arch);

/* grouped_gemm_single_launch_signature(spec): (A, B, C ptrs ; M, N, K i32 ;
 * stride_a, stride_b, stride_c i32). Same ptr_dt rule as the per-group sig. */
ckc_status_t ckc_grouped_gemm_single_launch_signature(const ckc_grouped_gemm_spec_t* spec,
                                                      ckc_arena_t* arena,
                                                      const ckc_sig_entry_t** out_items,
                                                      size_t* out_count);

/* Convenience: build the per-group base kernel and lower it to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure *out_ll is NULL and (if
 * err!=NULL) a diagnostic is written. Owns and frees its own IRBuilder. */
ckc_status_t ckc_grouped_gemm_lower_to_llvm(const ckc_grouped_gemm_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GROUPED_GEMM_H */
