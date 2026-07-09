/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gemm_multi_abd.h -- C99 port of the multi A/B/D GEMM kernel
 * instance builder rocke/instances/common/gemm_multi_abd.py (CK Tile
 * ``22_gemm_multi_abd`` parity).
 *
 * This is the clean public *instance* layer (parallel to
 * rocke/instance_gemm_universal.h). It mirrors the Python module surface and is
 * a thin wrapper over the multi-D path: in v1 (num_a == num_b == 1) the single
 * A/B drive the standard MFMA loop and the D list feeds the fused cshuffle
 * epilogue (delegating to rocke_build_gemm_multi_d). When num_d == 0 the wrapper
 * delegates straight to rocke_build_universal_gemm (a plain GEMM).
 *
 *   Python (gemm_multi_abd.py)            C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   MAX_A / MAX_B                         ROCKE_GEMM_ABD_MAX_A / ROCKE_GEMM_ABD_MAX_B
 *   AOperand = (name, dtype)              rocke_gemm_abd_a_operand_t
 *   BOperand = (name, dtype)              rocke_gemm_abd_b_operand_t
 *   DOperand = (name, "add"|"mul")        rocke_gemm_multi_d_op_t (reused)
 *   class GemmMultiAbdSpec                rocke_gemm_multi_abd_spec_t
 *   is_valid_spec(spec, arch)             rocke_gemm_multi_abd_is_valid_spec(...)
 *   build_gemm_multi_abd(spec, arch)      rocke_build_gemm_multi_abd(...)
 *   (+ convenience: init builder)         rocke_build_gemm_multi_abd_new(...)
 *   (+ convenience: build -> lower .ll)   rocke_gemm_multi_abd_lower_to_llvm(...)
 *   gemm_multi_abd_signature(spec)        rocke_gemm_multi_abd_signature(...)
 *   gemm_multi_abd_grid(spec, m, n, b)    rocke_gemm_multi_abd_grid(...)
 *   spec.kernel_name()                    rocke_gemm_multi_abd_kernel_name(...)
 *
 * TYPE REUSE. The D-operand pool reuses the canonical multi-D op type
 * (rocke_gemm_multi_d_op_t) and DLoadKind enum (rocke_d_load_kind_t) from
 * rocke/helper_rocke.instances.common.gemm_multi_d.h, so the abd spec drops
 * straight into a rocke_gemm_multi_d_spec_t with no per-D conversion.
 *
 * Error model mirrors the rest of the C port: build routes errors through the
 * sticky-error IRBuilder (rocke_b_*); the validity gate returns a bool + a reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GEMM_MULTI_ABD_H
#define ROCKE_INSTANCE_GEMM_MULTI_ABD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (signature storage) */
#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/instance_gemm_multi_d.h"
#include "rocke/instance_gemm_universal.h" /* rocke_gemm_universal_spec_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
/* ^ the multi-D facade. It pulls in the helper header under the
 * `#define rocke_build_gemm_multi_d rocke_build_gemm_multi_d_builder` rename, so the
 * 4-arg rocke_build_gemm_multi_d the multi-ABD wrapper calls resolves to the
 * _builder symbol the helper defines (and exposes the canonical
 * rocke_gemm_multi_d_spec_t / _op_t / rocke_d_load_kind_t / rocke_gemm_multi_d_grid /
 * ROCKE_GEMM_MULTI_D_MAX_D + the 2-arg facade entry). */

#ifdef __cplusplus
extern "C" {
#endif

/* MAX_A = 4, MAX_B = 4 (Python module constants). MAX_D is owned by the multi-D
 * header (ROCKE_GEMM_MULTI_D_MAX_D == 8). */
#define ROCKE_GEMM_ABD_MAX_A 4
#define ROCKE_GEMM_ABD_MAX_B 4

/* ------------------------------------------------------------------ operands */

/* AOperand / BOperand: (param_name, dtype_str). */
typedef struct rocke_gemm_abd_a_operand
{
    const char* name;
    const char* dtype;
} rocke_gemm_abd_a_operand_t;

typedef struct rocke_gemm_abd_b_operand
{
    const char* name;
    const char* dtype;
} rocke_gemm_abd_b_operand_t;

/* ----------------------------------------------------------- GemmMultiAbdSpec
 *
 * Mirror of the Python frozen dataclass. a_operands / b_operands default to a
 * single (name, dtype) entry; the caller supplies the arrays + their counts.
 * The struct stores exactly what is given so the validity gate can flag an
 * empty pool (num_a == 0 / num_b == 0). D operands reuse the canonical multi-D
 * op type (name + op_is_mul). */
typedef struct rocke_gemm_multi_abd_spec
{
    rocke_gemm_universal_spec_t base;
    const rocke_gemm_abd_a_operand_t* a_operands;
    size_t num_a_operands;
    const rocke_gemm_abd_b_operand_t* b_operands;
    size_t num_b_operands;
    rocke_gemm_multi_d_op_t d_operands[ROCKE_GEMM_MULTI_D_MAX_D];
    size_t num_d_operands;
    const char* d_dtype; /* default "fp16"                  */
    const char* name; /* default "rocke_gemm_multi_abd"  */
    rocke_d_load_kind_t d_load_kind; /* default ROCKE_D_LOAD_VECTOR        */
} rocke_gemm_multi_abd_spec_t;

/* Default-constructed spec (Python dataclass defaults). base == universal
 * default; a_operands / b_operands left NULL (count 0) so the caller points
 * them at the single-element defaults ({"A","fp16"} / {"B","fp16"}) or its own
 * operands; no D operands; d_dtype "fp16"; name "rocke_gemm_multi_abd";
 * d_load_kind vector. */
rocke_gemm_multi_abd_spec_t rocke_gemm_multi_abd_spec_default(void);

/* spec.num_a / num_b / num_d (just the stored counts). */
size_t rocke_gemm_multi_abd_num_a(const rocke_gemm_multi_abd_spec_t* spec);
size_t rocke_gemm_multi_abd_num_b(const rocke_gemm_multi_abd_spec_t* spec);
size_t rocke_gemm_multi_abd_num_d(const rocke_gemm_multi_abd_spec_t* spec);

/* GemmMultiAbdSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 *   d_suffix = "_".join(f"{n}{op}" for n,op in d_operands) or "noD"
 *   kernel_name_join(name, base.kernel_name(),
 *                    f"ma{num_a}", f"mb{num_b}", f"md{num_d}", d_suffix, d_dtype)
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (NULL arg / buffer too small). */
rocke_status_t rocke_gemm_multi_abd_kernel_name(const rocke_gemm_multi_abd_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). arch NULL => "gfx950". On reject,
 * reason (if non-NULL, capacity reason_cap) receives the structured message and
 * false is returned. On accept returns true and writes "ok". */
bool rocke_gemm_multi_abd_is_valid_spec(const rocke_gemm_multi_abd_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

/* build_gemm_multi_abd(spec, arch). Builds the IR into the supplied (already
 * rocke_ir_builder_init'd) builder `b` and returns the kernel (b->kernel) on
 * success, or NULL with b's sticky error set. arch NULL => "gfx950". `arena`
 * backs the fused-epilogue + op-chain allocations the multi-D path needs (the
 * num_d == 0 plain-GEMM branch ignores it).
 *
 * v1 restriction: num_a == num_b == 1; otherwise the builder's sticky error is
 * set to ROCKE_ERR_NOTIMPL (Python NotImplementedError) and NULL is returned.
 *
 * Like the Python, this expects the builder to have been created with the
 * spec's kernel_name(); it does not re-init the builder. */
rocke_kernel_def_t* rocke_build_gemm_multi_abd(rocke_ir_builder_t* b,
                                               rocke_arena_t* arena,
                                               const rocke_gemm_multi_abd_spec_t* spec,
                                               const char* arch);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with rocke_ir_builder_free(). Returns the kernel or NULL. */
rocke_kernel_def_t* rocke_build_gemm_multi_abd_new(rocke_ir_builder_t* b,
                                                   rocke_arena_t* arena,
                                                   const rocke_gemm_multi_abd_spec_t* spec,
                                                   const char* arch);

/* Convenience: given a spec, init a builder + arena, build, and lower to LLVM
 * .ll text. arch NULL => "gfx950". On ROCKE_OK *out_ll receives a malloc'd
 * NUL-terminated string the caller frees with free(); on failure it is left
 * NULL and (if err != NULL, capacity err_cap) a diagnostic is written.
 * Internally owns and frees its IRBuilder and arena. */
rocke_status_t rocke_gemm_multi_abd_lower_to_llvm(const rocke_gemm_multi_abd_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

/* gemm_multi_abd_signature(spec): manifest-style kernarg signature, built
 * through the shared SignatureBuilder exactly as the Python builds it. The
 * kernarg order matches the Python:
 *   A0.., B0.., C, M, N, K, [stride_a, stride_b, stride_c?], D0, D1, ...
 * Entries (name + type strings) are arena-owned. On ROCKE_OK *out_items /
 * *out_count point at the accumulated array; on failure they are untouched and
 * the status is returned. */
rocke_status_t rocke_gemm_multi_abd_signature(const rocke_gemm_multi_abd_spec_t* spec,
                                              rocke_arena_t* arena,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count);

/* gemm_multi_abd_grid(spec, m, n, batch): same launch grid as
 * build_universal_gemm. Delegates to rocke_gemm_multi_d_grid with a synthesised
 * GemmMultiDSpec whose d_operands default to a single ("D0","add") when the abd
 * spec has no D operands (matching the Python
 * `spec.d_operands or (("D0","add"),)`). On success out[0..2] hold (x, y, z);
 * returns ROCKE_ERR_VALUE on a non-positive tile (the Python ValueError) or NULL
 * args. */
rocke_status_t rocke_gemm_multi_abd_grid(
    const rocke_gemm_multi_abd_spec_t* spec, int m, int n, int batch, int out[3]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GEMM_MULTI_ABD_H */
