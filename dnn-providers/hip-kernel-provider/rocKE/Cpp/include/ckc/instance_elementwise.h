/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/instance_elementwise.h -- C99 port of the elementwise kernel instance
 * builder ck_dsl/instances/common/elementwise.py (CK Tile ``21_elementwise``
 * parity).
 *
 * Emits a single AMDGPU kernel that walks one contiguous N-element tensor with
 * vectorised global loads/stores and applies a fused unary or binary operation
 * per element. Compute is f32 internally; I/O is f16 or bf16.
 *
 *   Python (elementwise.py)               C99 (this header)
 *   -----------------------------------   --------------------------------------
 *   class ElementwiseSpec                 ckc_elementwise_spec_t
 *   ElementwiseSpec.is_unary()            ckc_elementwise_is_unary(...)
 *   ElementwiseSpec.is_binary()           ckc_elementwise_is_binary(...)
 *   ElementwiseSpec.is_bias()             ckc_elementwise_is_bias(...)
 *   ElementwiseSpec.kernel_name()         ckc_elementwise_kernel_name(...)
 *   ElementwiseSpec.elems_per_block()     ckc_elementwise_elems_per_block(...)
 *   is_valid_spec(spec)                   ckc_elementwise_is_valid_spec(...)
 *   build_elementwise(spec)               ckc_build_elementwise(...)
 *   elementwise_grid(numel, spec)         ckc_elementwise_grid(...)
 *   (+ convenience: build -> lower .ll)   ckc_elementwise_lower_to_llvm(...)
 *
 * SPEC AS AN EXPLICIT C STRUCT. The Python is a frozen dataclass with defaults.
 * In C the caller fills a ckc_elementwise_spec_t; ckc_elementwise_spec_default()
 * returns a struct with every field at the Python dataclass default, and the
 * caller overrides `op` (required) plus whatever else it cares about.
 *
 * Error model mirrors the rest of the C port: build/lower routes errors through
 * the sticky-error IRBuilder (ckc_b_*); the validity gate returns a bool + a
 * reason string; the convenience lower returns a ckc_status_t.
 *
 * ACTIVATION DEPENDENCY. The transcendental ops (silu/swish/tanh/sigmoid/
 * quick_gelu/gelu_tanh and the GLU binaries) reduce to the two exp2-based
 * primitives ckc_sigmoid_via_exp2 / ckc_tanh_via_exp2 from
 * ckc/helper_ck_dsl.helpers.activations.h, exactly as the Python imports
 * _sigmoid_via_exp2 / _tanh_via_exp2.
 */
#ifndef CKC_INSTANCE_ELEMENTWISE_H
#define CKC_INSTANCE_ELEMENTWISE_H

#include <stdbool.h>
#include <stddef.h>

#include "ckc/ir.h"
#include "ckc/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------- ElementwiseSpec */

/* Mirror of Python ElementwiseSpec. `op` is one of the unary or binary op
 * spellings (see the dispatch tables below); `dtype` is "f16" or "bf16". */
typedef struct ckc_elementwise_spec
{
    const char* op; /* required (no default in Python)                  */
    const char* dtype; /* default "f16"                                    */
    int block_size; /* default 256                                      */
    int vec; /* default 8                                        */
    const char* name; /* default "ck_dsl_elementwise"                     */
} ckc_elementwise_spec_t;

/* Default-constructed spec (every field == Python dataclass default). The
 * caller must still set `op`. */
ckc_elementwise_spec_t ckc_elementwise_spec_default(void);

/* ElementwiseSpec.is_unary() / is_binary() / is_bias(). */
bool ckc_elementwise_is_unary(const ckc_elementwise_spec_t* spec);
bool ckc_elementwise_is_binary(const ckc_elementwise_spec_t* spec);
bool ckc_elementwise_is_bias(const ckc_elementwise_spec_t* spec);

/* ElementwiseSpec.elems_per_block() == block_size * vec. */
int ckc_elementwise_elems_per_block(const ckc_elementwise_spec_t* spec);

/* ElementwiseSpec.kernel_name() -> NUL-terminated into out (capacity out_cap).
 * Mirrors kernel_name_join(name, op, dtype, "b{block_size}", "v{vec}").
 * Returns CKC_OK or CKC_ERR_VALUE (buffer too small). */
ckc_status_t
    ckc_elementwise_kernel_name(const ckc_elementwise_spec_t* spec, char* out, size_t out_cap);

/* is_valid_spec(spec) -> (ok, reason). On a reject, `reason` (if non-NULL,
 * capacity reason_cap) receives the Python message; returns false. On accept
 * returns true and writes "ok". */
bool ckc_elementwise_is_valid_spec(const ckc_elementwise_spec_t* spec,
                                   char* reason,
                                   size_t reason_cap);

/* ------------------------------------------------------------- build entry */

/* build_elementwise(spec). Builds the IR into the supplied (already
 * ckc_ir_builder_init'd with spec.kernel_name()) builder `b`, exactly as the
 * Python build does, and returns the kernel (b->kernel) on success or NULL with
 * b's sticky error set.
 *
 * Calling pattern:
 *   ckc_elementwise_spec_t spec = ckc_elementwise_spec_default();
 *   spec.op = "relu";
 *   char nm[256];
 *   ckc_elementwise_kernel_name(&spec, nm, sizeof nm);
 *   ckc_ir_builder_t b;
 *   ckc_ir_builder_init(&b, nm);
 *   ckc_kernel_def_t* kernel = ckc_build_elementwise(&b, &spec);
 *   // kernel == b.kernel on success.
 */
ckc_kernel_def_t* ckc_build_elementwise(ckc_ir_builder_t* b, const ckc_elementwise_spec_t* spec);

/* Convenience: init `b` with spec.kernel_name(), then build. The caller owns
 * `b` and frees it with ckc_ir_builder_free(). Returns the kernel or NULL. */
ckc_kernel_def_t* ckc_build_elementwise_new(ckc_ir_builder_t* b,
                                            const ckc_elementwise_spec_t* spec);

/* elementwise_grid(numel, spec) -> (grid_x, 1, 1). Writes the three components
 * into out_grid[3]. grid_x = ceil(numel / (block_size*vec)). */
void ckc_elementwise_grid(int numel, const ckc_elementwise_spec_t* spec, int out_grid[3]);

/* Convenience: given a spec, init a builder, build, and lower to LLVM .ll text.
 * On CKC_OK *out_ll receives a malloc'd NUL-terminated string the caller frees
 * with free(); on failure it is left NULL and (if err!=NULL, capacity err_cap) a
 * diagnostic is written. `arch` NULL => "gfx950". Internally owns and frees its
 * IRBuilder. */
ckc_status_t ckc_elementwise_lower_to_llvm(const ckc_elementwise_spec_t* spec,
                                           const char* arch,
                                           ckc_llvm_flavor_t flavor,
                                           char** out_ll,
                                           char* err,
                                           size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_INSTANCE_ELEMENTWISE_H */
