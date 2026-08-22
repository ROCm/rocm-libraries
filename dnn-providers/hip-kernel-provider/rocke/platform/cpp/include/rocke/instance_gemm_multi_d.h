/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_multi_d.h -- task-mandated public facade for the C99 port
 * of rocke/instances/common/gemm_multi_d.py (CK Tile ``19_gemm_multi_d``).
 *
 * This header sits on top of the full faithful port that already lives in
 * rocke/helper_rocke.instances.common.gemm_multi_d.{h,c} (the _MultiDEpilogue
 * apply_vec sequence, is_valid_spec, _build_fused_epilogue, the kernarg
 * signature, the launch grid). It exposes exactly the entry shape the workflow
 * map requires:
 *
 *   rocke_gemm_multi_d_spec_t* rocke_gemm_multi_d_spec_new(base, d_operands, num_d,
 *                                                      d_dtype, name, load_kind);
 *   rocke_kernel_def_t*        rocke_build_gemm_multi_d(spec, arch);
 *
 * Python -> C facade map:
 *   GemmMultiDSpec(base=..., d_operands=(("D0","add"),...),       rocke_gemm_multi_d_spec_new(...)
 *                  d_dtype=..., name=..., d_load_kind=...)
 *   build_gemm_multi_d(spec, arch) -> KernelDef                   rocke_build_gemm_multi_d(spec,
 * arch)
 *   (+ convenience: build -> lower .ll) rocke_gemm_multi_d_lower_to_llvm(...)
 *
 * spec_new() constructs a rocke_gemm_multi_d_spec_t (the same struct the full
 * port uses) from a base UniversalGemmSpec, an array of {param_name, "add"|
 * "mul"} operands, a count, a d_dtype, an optional name, and a load_kind
 * string. Allocation comes from the supplied arena (the spec, the copied
 * operand param-name strings, and the spec's own storage are arena-owned).
 *
 * Build path (mirrors build_gemm_multi_d(spec, arch) exactly):
 *   1. rocke_gemm_multi_d_is_valid_spec(spec, arch)         [guard; ValueError]
 *   2. rocke_gemm_multi_d_build_fused_epilogue(arena, spec) [per-D residual chain]
 *   3. rename a fresh copy of base to spec.kernel_name()  [dataclasses.replace]
 *   4. attach the fused epilogue via the side-channel      [object.__setattr__]
 *   5. rocke_build_universal_gemm(b, base_renamed, arch)     [delegate]
 *
 * rocke_build_gemm_multi_d owns and frees its own IRBuilder + arena, matching the
 * Python entry which returns a self-contained KernelDef.
 */
#ifndef ROCKE_INSTANCE_GEMM_MULTI_D_H
#define ROCKE_INSTANCE_GEMM_MULTI_D_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h"
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t */
#include "rocke/ir.h" /* rocke_status_t, rocke_kernel_def_t */
#include "rocke/lower_llvm.h" /* rocke_llvm_flavor_t */

/* Pull in the full port's spec/operand/load-kind types + helpers
 * (rocke_gemm_multi_d_spec_t, rocke_gemm_multi_d_op_t, rocke_d_load_kind_t,
 * rocke_gemm_multi_d_is_valid_spec, _build_fused_epilogue, _kernel_name, ...).
 * The full port's own 4-arg rocke_build_gemm_multi_d(b, arena, spec, arch) is
 * renamed away here so the task-mandated 2-arg rocke_build_gemm_multi_d below is
 * the one consumers see (and there is no duplicate-symbol clash). */
#ifdef ROCKE_HELPER_ROCKE_INSTANCES_COMMON_GEMM_MULTI_D_H
#error \
    "Include <rocke/instance_gemm_multi_d.h> instead of (or before) the full-port helper header: the facade renames the helper's 4-arg builder so the mandated 2-arg rocke_build_gemm_multi_d(spec, arch) owns the symbol. Including the helper header first leaves its un-renamed declaration in scope and clashes."
#endif
#define rocke_build_gemm_multi_d rocke_build_gemm_multi_d_builder
#include "rocke/helper_rocke.instances.common.gemm_multi_d.h"
#undef rocke_build_gemm_multi_d

#ifdef __cplusplus
extern "C" {
#endif

/* DOp = (param_name, "add" | "mul"). The op kind is the canonical lowercase
 * string ("add"/"mul"), compared by strcmp like the Python tuple. This mirrors
 * the sample configs: ("D0","add"), ("D1","mul"), ... */
typedef struct rocke_gemm_multi_d_operand
{
    const char* param_name; /* e.g. "D0" -- copied into the arena by spec_new */
    const char* op; /* "add" or "mul"                                 */
} rocke_gemm_multi_d_operand_t;

/* rocke_gemm_multi_d_spec_new(base, d_operands, num_d, d_dtype, name, load_kind)
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
 *   name       -- kernel base name; NULL => "rocke_gemm_multi_d".
 *   load_kind  -- "stock" | "tiled" | "vector"; NULL/unknown => "vector".
 *
 * Validity of the multi-D knobs (epilogue=='cshuffle', unique non-reserved
 * names, arch-aware base check) is NOT enforced here -- it is enforced at build
 * time by rocke_gemm_multi_d_is_valid_spec, exactly like the Python flow where
 * GemmMultiDSpec is a plain dataclass and build_gemm_multi_d runs is_valid_spec.
 * spec_new only rejects structurally-impossible inputs (NULL/empty operand,
 * count out of range, op not in {"add","mul"}). */
rocke_gemm_multi_d_spec_t*
    rocke_gemm_multi_d_spec_new(rocke_arena_t* arena,
                                const rocke_gemm_universal_spec_t* base,
                                const rocke_gemm_multi_d_operand_t* d_operands,
                                int num_d,
                                const char* d_dtype,
                                const char* name,
                                const char* d_load_kind);

/* rocke_build_gemm_multi_d(spec, arch) -> KernelDef.
 *
 * The task-mandated entry. Validates the spec, composes the fused epilogue,
 * renames a fresh copy of the base spec to spec.kernel_name(), attaches the
 * epilogue via the side-channel, and delegates to rocke_build_universal_gemm.
 * `arch` NULL => "gfx950".
 *
 * Owns its IRBuilder + a private arena on the heap. The returned KernelDef is
 * the builder's kernel and stays valid until freed with
 * rocke_gemm_multi_d_kernel_free(), which tears down the owning builder/arena.
 * Returns NULL on any failure. */
rocke_kernel_def_t* rocke_build_gemm_multi_d(rocke_gemm_multi_d_spec_t* spec, const char* arch);

/* Free a KernelDef returned by rocke_build_gemm_multi_d (tears down the IRBuilder
 * + arena that own it). No-op on NULL or an unrecognised kernel. */
void rocke_gemm_multi_d_kernel_free(rocke_kernel_def_t* kernel);

/* Build into a caller-supplied builder + arena (no implicit ownership). This is
 * the seam rocke_build_gemm_multi_d is built on, and the one byte-identical to
 * the full port's 4-arg builder. `b` must already be rocke_ir_builder_init'd with
 * spec.kernel_name(). Returns b->kernel or NULL with b's sticky error set. */
rocke_kernel_def_t* rocke_build_gemm_multi_d_into(rocke_ir_builder_t* b,
                                                  rocke_arena_t* arena,
                                                  const rocke_gemm_multi_d_spec_t* spec,
                                                  const char* arch);

/* Convenience: build one multi-D GEMM instance and lower it to LLVM .ll text.
 * `arch` NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated
 * string the caller frees with free(); on failure it is left NULL and (if
 * err!=NULL, capacity err_cap) a diagnostic is written. Owns its IRBuilder +
 * arena internally. */
rocke_status_t rocke_gemm_multi_d_lower_to_llvm(const rocke_gemm_multi_d_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_MULTI_D_H */
