/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_gemm_multi_d.h -- task-mandated public facade for the C99 port
 * of ck_dsl/instances/common/gemm_multi_d.py (CK Tile ``19_gemm_multi_d``).
 *
 * This header sits on top of the full faithful port that already lives in
 * ckc/helper_ck_dsl.instances.common.gemm_multi_d.{h,c} (the _MultiDEpilogue
 * apply_vec sequence, is_valid_spec, _build_fused_epilogue, the kernarg
 * signature, the launch grid). It exposes exactly the entry shape the workflow
 * map requires:
 *
 *   ckc_gemm_multi_d_spec_t* ckc_gemm_multi_d_spec_new(base, d_operands, num_d,
 *                                                      d_dtype, name, load_kind);
 *   ckc_kernel_def_t*        ckc_build_gemm_multi_d(spec, arch);
 *
 * Python -> C facade map:
 *   GemmMultiDSpec(base=..., d_operands=(("D0","add"),...),       ckc_gemm_multi_d_spec_new(...)
 *                  d_dtype=..., name=..., d_load_kind=...)
 *   build_gemm_multi_d(spec, arch) -> KernelDef                   ckc_build_gemm_multi_d(spec,
 * arch)
 *   (+ convenience: build -> lower .ll) ckc_gemm_multi_d_lower_to_llvm(...)
 *
 * spec_new() constructs a ckc_gemm_multi_d_spec_t (the same struct the full
 * port uses) from a base UniversalGemmSpec, an array of {param_name, "add"|
 * "mul"} operands, a count, a d_dtype, an optional name, and a load_kind
 * string. Allocation comes from the supplied arena (the spec, the copied
 * operand param-name strings, and the spec's own storage are arena-owned).
 *
 * Build path (mirrors build_gemm_multi_d(spec, arch) exactly):
 *   1. ckc_gemm_multi_d_is_valid_spec(spec, arch)         [guard; ValueError]
 *   2. ckc_gemm_multi_d_build_fused_epilogue(arena, spec) [per-D residual chain]
 *   3. rename a fresh copy of base to spec.kernel_name()  [dataclasses.replace]
 *   4. attach the fused epilogue via the side-channel      [object.__setattr__]
 *   5. ckc_build_universal_gemm(b, base_renamed, arch)     [delegate]
 *
 * ckc_build_gemm_multi_d owns and frees its own IRBuilder + arena, matching the
 * Python entry which returns a self-contained KernelDef.
 */
#ifndef CKC_INSTANCE_GEMM_MULTI_D_H
#define CKC_INSTANCE_GEMM_MULTI_D_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/arena.h"
#include "ckc/ir.h"                      /* ckc_status_t, ckc_kernel_def_t */
#include "ckc/lower_llvm.h"              /* ckc_llvm_flavor_t */
#include "ckc/instance_gemm_universal.h" /* ckc_gemm_universal_spec_t */

/* Pull in the full port's spec/operand/load-kind types + helpers
 * (ckc_gemm_multi_d_spec_t, ckc_gemm_multi_d_op_t, ckc_d_load_kind_t,
 * ckc_gemm_multi_d_is_valid_spec, _build_fused_epilogue, _kernel_name, ...).
 * The full port's own 4-arg ckc_build_gemm_multi_d(b, arena, spec, arch) is
 * renamed away here so the task-mandated 2-arg ckc_build_gemm_multi_d below is
 * the one consumers see (and there is no duplicate-symbol clash). */
#ifdef CKC_HELPER_CK_DSL_INSTANCES_COMMON_GEMM_MULTI_D_H
#error \
    "Include <ckc/instance_gemm_multi_d.h> instead of (or before) the full-port helper header: the facade renames the helper's 4-arg builder so the mandated 2-arg ckc_build_gemm_multi_d(spec, arch) owns the symbol. Including the helper header first leaves its un-renamed declaration in scope and clashes."
#endif
#define ckc_build_gemm_multi_d ckc_build_gemm_multi_d_builder
#include "ckc/helper_ck_dsl.instances.common.gemm_multi_d.h"
#undef ckc_build_gemm_multi_d

#ifdef __cplusplus
extern "C" {
#endif

/* DOp = (param_name, "add" | "mul"). The op kind is the canonical lowercase
 * string ("add"/"mul"), compared by strcmp like the Python tuple. This mirrors
 * the sample configs: ("D0","add"), ("D1","mul"), ... */
typedef struct ckc_gemm_multi_d_operand
{
    const char* param_name; /* e.g. "D0" -- copied into the arena by spec_new */
    const char* op;         /* "add" or "mul"                                 */
} ckc_gemm_multi_d_operand_t;

/* ckc_gemm_multi_d_spec_new(base, d_operands, num_d, d_dtype, name, load_kind)
 *
 * Construct a GemmMultiDSpec into arena-owned storage and return a pointer to
 * it (NULL on a bad argument / allocation failure / unrecognised op string).
 *
 *   base       -- the base UniversalGemmSpec (copied by value into the spec;
 *                 callers should have finalize()d it).
 *   d_operands -- array of {param_name, "add"|"mul"}; param_name strings are
 *                 duplicated into the arena. num_d must be in [1, MAX_D].
 *   num_d      -- number of D operands.
 *   d_dtype    -- element dtype for every D operand; NULL => "fp16".
 *   name       -- kernel base name; NULL => "ck_dsl_gemm_multi_d".
 *   load_kind  -- "stock" | "tiled" | "vector"; NULL/unknown => "vector".
 *
 * Validity of the multi-D knobs (epilogue=='cshuffle', unique non-reserved
 * names, arch-aware base check) is NOT enforced here -- it is enforced at build
 * time by ckc_gemm_multi_d_is_valid_spec, exactly like the Python flow where
 * GemmMultiDSpec is a plain dataclass and build_gemm_multi_d runs is_valid_spec.
 * spec_new only rejects structurally-impossible inputs (NULL/empty operand,
 * count out of range, op not in {"add","mul"}). */
ckc_gemm_multi_d_spec_t* ckc_gemm_multi_d_spec_new(ckc_arena_t* arena,
                                                   const ckc_gemm_universal_spec_t* base,
                                                   const ckc_gemm_multi_d_operand_t* d_operands,
                                                   int num_d,
                                                   const char* d_dtype,
                                                   const char* name,
                                                   const char* d_load_kind);

/* ckc_build_gemm_multi_d(spec, arch) -> KernelDef.
 *
 * The task-mandated entry. Validates the spec, composes the fused epilogue,
 * renames a fresh copy of the base spec to spec.kernel_name(), attaches the
 * epilogue via the side-channel, and delegates to ckc_build_universal_gemm.
 * `arch` NULL => "gfx950".
 *
 * Owns its IRBuilder + a private arena on the heap. The returned KernelDef is
 * the builder's kernel and stays valid until freed with
 * ckc_gemm_multi_d_kernel_free(), which tears down the owning builder/arena.
 * Returns NULL on any failure. */
ckc_kernel_def_t* ckc_build_gemm_multi_d(ckc_gemm_multi_d_spec_t* spec, const char* arch);

/* Free a KernelDef returned by ckc_build_gemm_multi_d (tears down the IRBuilder
 * + arena that own it). No-op on NULL or an unrecognised kernel. */
void ckc_gemm_multi_d_kernel_free(ckc_kernel_def_t* kernel);

/* Build into a caller-supplied builder + arena (no implicit ownership). This is
 * the seam ckc_build_gemm_multi_d is built on, and the one byte-identical to
 * the full port's 4-arg builder. `b` must already be ckc_ir_builder_init'd with
 * spec.kernel_name(). Returns b->kernel or NULL with b's sticky error set. */
ckc_kernel_def_t* ckc_build_gemm_multi_d_into(ckc_ir_builder_t* b,
                                              ckc_arena_t* arena,
                                              const ckc_gemm_multi_d_spec_t* spec,
                                              const char* arch);

/* Convenience: build one multi-D GEMM instance and lower it to LLVM .ll text.
 * `arch` NULL => "gfx950". On CKC_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns its IRBuilder +
 * arena internally. */
ckc_status_t ckc_gemm_multi_d_lower_to_llvm(const ckc_gemm_multi_d_spec_t* spec,
                                            const char* arch,
                                            ckc_llvm_flavor_t flavor,
                                            char** out_ll,
                                            char* err,
                                            size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_GEMM_MULTI_D_H */
