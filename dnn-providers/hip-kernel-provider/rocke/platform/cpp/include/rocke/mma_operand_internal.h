// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* How an MMA operand reaches the instruction's register type. NOT a public API.
 *
 * Mirrors core/mma_operand.py. An MMA instruction takes its A and B operands as
 * one fixed register type (f32 for the f32 MFMAs, vec<i32x4> for the gfx11 iu8
 * WMMA). A kernel may hold the same bits in another shape: a one-element
 * vec<f32x1>, or the instruction's packed values as vec<i8x16>. Reading such an
 * operand as the register type reinterprets the same bits; it never converts.
 *
 * rocke_mma_operand_needs_reinterpret is the rule, with no output syntax. Each
 * lowerer renders its answer in its own language (LLVM `bitcast`, HIP
 * `__builtin_bit_cast`), so both lowerers of both engines accept and refuse the
 * same operands with the same text.
 */
#ifndef ROCKE_MMA_OPERAND_INTERNAL_H
#define ROCKE_MMA_OPERAND_INTERNAL_H

#include "rocke/arch_target.h"
#include "rocke/ir.h"

/* MMA operand `role` of `op` (Python mma_operand_value): operand 0 for
 * ROCKE_MMA_ROLE_A, operand 1 for ROCKE_MMA_ROLE_B. Throws ckc::ValueError for
 * any other role, or when `op` has fewer than two operands. */
const rocke_value_t* rocke_mma_operand_value(const rocke_op_t* op, rocke_mma_role_t role);

/* Whether MMA operand `role` (A or B) of `op_id`, of type `operand_type`, must be
 * reinterpreted to be passed as `register_type` on target `gfx` (Python
 * mma_operand_needs_reinterpret).
 *
 * False when the operand already has the register type; the target's MMA
 * catalog is not consulted then, so a target without a catalog still lowers
 * operands of the register type. True when one reinterpretation of the same
 * bits gives the register type: the operand holds the register's element type
 * or the element type the target's MMA catalog records for that operand, and
 * has the register's bit width.
 *
 * Throws ckc::ValueError, with the Python text, for an operand holding another
 * number format and for an operand of another bit width; and for a mismatched
 * operand on a target with no MMA catalog. */
bool rocke_mma_operand_needs_reinterpret(const char* op_id,
                                         rocke_mma_role_t role,
                                         const rocke_type_t* operand_type,
                                         const rocke_type_t* register_type,
                                         const char* gfx);

#endif /* ROCKE_MMA_OPERAND_INTERNAL_H */
