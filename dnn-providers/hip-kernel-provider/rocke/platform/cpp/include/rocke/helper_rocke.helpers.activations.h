/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.activations.h -- C99 port of
 * rocke.helpers.activations.
 *
 * Shared scalar activation primitives (exp2-based, AMDGPU-lowerable). The
 * transcendental activations used by the fused-epilogue ops and the elementwise
 * instance reduce to the same two f32 building blocks: an ``exp2``-based
 * ``tanh`` and ``sigmoid``. Both deliberately avoid ``llvm.tanh.f32`` /
 * ``math.exp`` because the AMDGPU backend does not lower them on its own (it
 * produces a ``CODEGEN_BC_TO_RELOCATABLE`` failure in ``amd_comgr``); ``exp2``
 * is the one transcendental the backend always lowers.
 *
 * Faithful translation of:
 *
 *     def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         c_neg_log2e = b.const_f32(-1.4426950408889634)
 *         one = b.const_f32(1.0)
 *         return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))
 *
 *     def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         c_2log2e = b.const_f32(2.0 * 1.4426950408889634)
 *         one = b.const_f32(1.0)
 *         e2x = b.exp2(b.fmul(c_2log2e, x))
 *         return b.fmul(b.fsub(e2x, one), b.rcp(b.fadd(e2x, one)))
 *
 * The builder-call sequence is reproduced in the exact same order as the Python
 * so the emitted IR (and SSA value numbering) stays byte-identical. These are
 * pure value-producing builders: there is no error path in the Python (no
 * ValueError), so the only failure mode is the usual sticky-builder NULL no-op.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ACTIVATIONS_H
#define ROCKE_HELPER_ROCKE_HELPERS_ACTIVATIONS_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 / (1 + e^-x), implemented via exp2.
 *
 * ``exp(-x) = exp2(-x * log2(e))``. Avoids ``math.exp`` (which the AMDGPU
 * backend does not lower on its own). Returns the f32 sigmoid Value, or NULL if
 * the builder is already in an error state. */
rocke_value_t* rocke_sigmoid_via_exp2(rocke_ir_builder_t* b, rocke_value_t* x);

/* tanh(x) = (e^{2x} - 1) / (e^{2x} + 1), via exp2.
 *
 * Deliberately avoids ``math.tanh`` / ``llvm.tanh.f32`` to keep the body
 * AMDGPU-lowerable without extra runtime libraries. Returns the f32 tanh Value,
 * or NULL if the builder is already in an error state. */
rocke_value_t* rocke_tanh_via_exp2(rocke_ir_builder_t* b, rocke_value_t* x);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ACTIVATIONS_H */
