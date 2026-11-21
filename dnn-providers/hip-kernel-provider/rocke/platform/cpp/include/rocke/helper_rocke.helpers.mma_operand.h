// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Atom-owned operand ABI preparation; mirrors helpers/mma_operand.py.
 * No memory operations or numerical conversions. Inputs are borrowed;
 * returned SSA value is builder-owned. Fragment retains only live storage.
 */
#ifndef ROCKE_HELPER_MMA_OPERAND_H
#define ROCKE_HELPER_MMA_OPERAND_H

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.helpers.tiling.h"

#ifdef __cplusplus
extern "C" {
#endif

rocke_value_t* rocke_h_prepare_scaled_mma_operand(rocke_ir_builder_t* b,
                                                  const rocke_mma_op_t* atom,
                                                  const char* role,
                                                  const rocke_tiling_fragment_t* fragment);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_MMA_OPERAND_H */
