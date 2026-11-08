// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Fixed unscaled-WMMA lowering ABI checks, mirroring
 * core/wmma.py::validate_unscaled_wmma, for the LLVM and HIP lowerers alike.
 * Inspect actual SSA values: re-resolving the catalog cannot check a caller's
 * independent src2/dst values. Returned errors use the caller's buffer; no
 * ownership is transferred. */
#ifndef ROCKE_WMMA_INTERNAL_H
#define ROCKE_WMMA_INTERNAL_H

#include <stdio.h>

#include "rocke/ir_internal.h"
#include "rocke/mma_operand_internal.h"

/* NULL when `op` fits the unscaled WMMA instruction whose accumulator and
 * result are `vec<accumulator_elem x accumulator_count>` and whose A and B are
 * `source_register` (a vector type); otherwise the Python text in `message`.
 * An i32 `source_register` is a packed register: an operand of another element
 * type passes when one reinterpretation reaches it, and is refused with
 * ckc::ValueError otherwise (rocke_mma_operand_needs_reinterpret on `gfx`). */
static inline const char* rocke_unscaled_wmma_operand_error(const rocke_op_t* op,
                                                            const rocke_type_t* accumulator_elem,
                                                            int accumulator_count,
                                                            const rocke_type_t* source_register,
                                                            const char* gfx,
                                                            char* message,
                                                            size_t capacity)
{
    if(op->num_operands != 3 || op->num_results != 1)
    {
        snprintf(message, capacity, "unscaled WMMA expects 3 operands and 1 result");
        return message;
    }
    const rocke_type_t* accumulator_types[] = {op->operands[2]->type, op->results[0]->type};
    for(const rocke_type_t* type : accumulator_types)
        if(!type || type->kind != ROCKE_TYPE_VECTOR || type->count != accumulator_count
           || !rocke_type_eq(type->elem, accumulator_elem))
        {
            snprintf(message,
                     capacity,
                     "unscaled WMMA requires src2 and dst to be vec<%sx%d>",
                     accumulator_elem->name,
                     accumulator_count);
            return message;
        }
    const rocke_type_t* source_elem = source_register->elem;
    for(int role = 0; role < 2; ++role)
    {
        const rocke_type_t* type = op->operands[role]->type;
        if(rocke_type_eq(source_elem, rocke_i32()))
        {
            /* Integer / FP8 entry points render the same-width register bridge. */
            const char* op_id = rocke_attr_get_str(&op->attrs, "op_id");
            if(!op_id)
                op_id = op->name + 5; /* "tile." */
            rocke_mma_operand_needs_reinterpret(op_id,
                                                role == 0 ? ROCKE_MMA_ROLE_SRC0
                                                          : ROCKE_MMA_ROLE_SRC1,
                                                type,
                                                source_register,
                                                gfx);
        }
        else if(!type || type->kind != ROCKE_TYPE_VECTOR || type->count != source_register->count
                || !rocke_type_eq(type->elem, source_elem))
        {
            snprintf(message,
                     capacity,
                     "unscaled WMMA requires src%d to be vec<%sx%d>",
                     role,
                     source_elem->name,
                     source_register->count);
            return message;
        }
    }
    return NULL;
}

#endif /* ROCKE_WMMA_INTERNAL_H */
