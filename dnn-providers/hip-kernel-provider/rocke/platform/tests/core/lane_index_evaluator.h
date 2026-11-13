// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lane_index_evaluator.h -- evaluates a lane's index arithmetic for a host test.
 *
 * Lane maps (an atom's lane_to_output, an MMA layout's lane -> coordinate map)
 * build integer arithmetic over the lane id: constants, add, mul, div and mod.
 * A host test checks such a map by evaluating that arithmetic with the lane id
 * fixed to each lane of a wave and comparing the result with the documented
 * layout. A new test includes this evaluator; the older lane-map tests in
 * this directory still carry their own copies.
 */
#ifndef ROCKE_TESTS_CORE_LANE_INDEX_EVALUATOR_H
#define ROCKE_TESTS_CORE_LANE_INDEX_EVALUATOR_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/ir.h"

/* Evaluates the index arithmetic rooted at `value` with `lane` fixed to
 * `lane_value` and stores it in `*result`. Returns false, with
 * `*failure_reason` naming what stopped it (an op's name, or a description)
 * so the caller can report it, when the arithmetic reaches a value no op
 * defines other than `lane`, a constant without an integer value, an op other
 * than a constant, add, mul, div or mod, or a div or mod by zero. */
static inline bool rocke_test_evaluate_lane_index(const rocke_value_t* value,
                                                  const rocke_value_t* lane,
                                                  int64_t lane_value,
                                                  int64_t* result,
                                                  const char** failure_reason)
{
    if(value == lane)
    {
        *result = lane_value;
        return true;
    }
    const rocke_op_t* op = value->op;
    if(op == NULL)
    {
        *failure_reason = "an op-less value";
        return false;
    }
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        if(rocke_attr_get_int(&op->attrs, "value", result))
        {
            return true;
        }
        *failure_reason = "a constant without an integer value";
        return false;
    }
    if(op->num_operands != 2)
    {
        *failure_reason = op->name;
        return false;
    }
    int64_t lhs = 0;
    int64_t rhs = 0;
    if(!rocke_test_evaluate_lane_index(op->operands[0], lane, lane_value, &lhs, failure_reason)
       || !rocke_test_evaluate_lane_index(op->operands[1], lane, lane_value, &rhs, failure_reason))
    {
        return false;
    }
    if((op->opcode == ROCKE_OP_ARITH_DIV || op->opcode == ROCKE_OP_ARITH_MOD) && rhs == 0)
    {
        *failure_reason = "a div or mod by zero";
        return false;
    }
    switch(op->opcode)
    {
    case ROCKE_OP_ARITH_ADD:
        *result = lhs + rhs;
        return true;
    case ROCKE_OP_ARITH_MUL:
        *result = lhs * rhs;
        return true;
    case ROCKE_OP_ARITH_DIV:
        *result = lhs / rhs;
        return true;
    case ROCKE_OP_ARITH_MOD:
        *result = lhs % rhs;
        return true;
    default:
        *failure_reason = op->name;
        return false;
    }
}

#endif /* ROCKE_TESTS_CORE_LANE_INDEX_EVALUATOR_H */
