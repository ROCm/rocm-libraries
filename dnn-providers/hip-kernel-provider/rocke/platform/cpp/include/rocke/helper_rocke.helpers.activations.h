/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.activations.h -- C99 port of
 * rocke.helpers.activations.
 *
 * Shared scalar activation primitives (AMDGPU-lowerable). The
 * transcendental activations used by the fused-epilogue ops and the elementwise
 * instance reduce to the same two f32 building blocks. Core ``math.tanh``
 * expands to stable arithmetic plus ``exp2`` instead of ``llvm.tanh.f32``;
 * sigmoid also avoids ``math.exp`` because the AMDGPU backend does not lower
 * those intrinsics on its own.
 *
 * Faithful translation of:
 *
 *     def _sigmoid_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         c_neg_log2e = b.const_f32(-1.4426950408889634)
 *         one = b.const_f32(1.0)
 *         return b.rcp(b.fadd(one, b.exp2(b.fmul(c_neg_log2e, x))))
 *
 *     def _tanh_via_exp2(b: IRBuilder, x: Value) -> Value:
 *         return b.tanh(x)
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

/* Stable AMDGPU-lowerable f32 tanh. Returns NULL if the builder is already in
 * an error state or x is not f32. */
rocke_value_t* rocke_tanh_via_exp2(rocke_ir_builder_t* b, rocke_value_t* x);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ACTIVATIONS_H */
