// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * test_mfma_atom_lane_to_output.cpp -- rocke_mfma_atom_lane_to_output, the C++
 * port of MfmaAtom.lane_to_output (rocke/helpers/atoms.py).
 *
 * The helper builds index arithmetic over the lane id. This test evaluates that
 * arithmetic for every lane of a wave and every accumulator slot, and checks:
 *   - each (row, column) against the layout atoms.py documents for the atom's
 *     shape and accumulator type, including the f64 16x16 layout;
 *   - that the lanes and slots of one atom cover each of its cells exactly once
 *     (per batch for the batched 4x4 atom);
 *   - that the f64 16x16 branch emits its builder calls in the Python's order
 *     (the f32 branches are held to that by the byte-identity gate);
 *   - that an atom shape the Python rejects throws ckc::NotImplError with the
 *     Python's message, leaving the out-params untouched.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/ir.h"

static const int WAVE_SIZE = 64;

/* Evaluates the index arithmetic rooted at `value` with the lane id fixed to
 * `lane_value`. Returns false on an op this test does not model. */
static bool evaluate(const rocke_value_t* value,
                     const rocke_value_t* lane,
                     int64_t lane_value,
                     int64_t* result)
{
    if(value == lane)
    {
        *result = lane_value;
        return true;
    }
    const rocke_op_t* op = value->op;
    if(op == NULL)
    {
        return false;
    }
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        return rocke_attr_get_int(&op->attrs, "value", result);
    }
    if(op->num_operands != 2)
    {
        return false;
    }
    int64_t lhs = 0;
    int64_t rhs = 0;
    if(!evaluate(op->operands[0], lane, lane_value, &lhs)
       || !evaluate(op->operands[1], lane, lane_value, &rhs))
    {
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
        return false;
    }
}

/* The (row, column) atoms.py documents for slot `slot` of lane `lane_value`. */
static void documented_cell(
    const rocke_mfma_atom_t* atom, int lane_value, int slot, int64_t* row, int64_t* column)
{
    if(atom->m == 4)
    {
        *row = slot;
        *column = lane_value % 4;
        return;
    }
    int64_t m_block = lane_value / atom->n;
    *column = lane_value % atom->n;
    if(atom->m == 32)
    {
        *row = (slot / 4) * 8 + m_block * 4 + slot % 4;
    }
    else if(strcmp(atom->dtype_out, "f64") == 0)
    {
        *row = slot * 4 + m_block;
    }
    else
    {
        *row = m_block * atom->c_per_lane + slot;
    }
}

/* Checks every lane and slot of `atom`; returns the number of failures. */
static int check_layout(const char* dtype, int m, int n, int k)
{
    const rocke_mfma_atom_t* atom = rocke_mfma_atom(dtype, m, n, k);
    if(atom == NULL)
    {
        fprintf(stderr, "no catalog atom %s %dx%dx%d\n", dtype, m, n, k);
        return 1;
    }
    rocke_ir_builder_t builder;
    if(rocke_ir_builder_init(&builder, "lane_to_output_probe") != ROCKE_OK)
    {
        return 1;
    }
    rocke_value_t* lane = rocke_b_lane_id(&builder);
    /* Cells per batch: one batch for 16x16 / 32x32, WAVE_SIZE / 4 for 4x4. */
    const int batches = (atom->m == 4) ? WAVE_SIZE / 4 : 1;
    int hits[WAVE_SIZE / 4][32 * 32];
    memset(hits, 0, sizeof(hits));
    int failures = 0;
    for(int slot = 0; slot < atom->c_per_lane; ++slot)
    {
        rocke_value_t* row_value = NULL;
        rocke_value_t* column_value = NULL;
        rocke_mfma_atom_lane_to_output(&builder, atom, lane, slot, &row_value, &column_value);
        for(int lane_value = 0; lane_value < WAVE_SIZE; ++lane_value)
        {
            int64_t row = 0;
            int64_t column = 0;
            int64_t want_row = 0;
            int64_t want_column = 0;
            if(!evaluate(row_value, lane, lane_value, &row)
               || !evaluate(column_value, lane, lane_value, &column))
            {
                fprintf(stderr, "%s: unmodelled op in slot %d\n", atom->name, slot);
                ++failures;
                continue;
            }
            documented_cell(atom, lane_value, slot, &want_row, &want_column);
            if(row != want_row || column != want_column)
            {
                fprintf(stderr,
                        "%s: lane %d slot %d gave (%lld, %lld), atoms.py documents (%lld, %lld)\n",
                        atom->name,
                        lane_value,
                        slot,
                        (long long)row,
                        (long long)column,
                        (long long)want_row,
                        (long long)want_column);
                ++failures;
                continue;
            }
            const int batch = (batches == 1) ? 0 : lane_value / 4;
            ++hits[batch][row * atom->n + column];
        }
    }
    for(int batch = 0; batch < batches; ++batch)
    {
        for(int cell = 0; cell < atom->m * atom->n; ++cell)
        {
            if(hits[batch][cell] != 1)
            {
                fprintf(stderr,
                        "%s: batch %d cell (%d, %d) held %d times, want once\n",
                        atom->name,
                        batch,
                        cell / atom->n,
                        cell % atom->n,
                        hits[batch][cell]);
                ++failures;
            }
        }
    }
    rocke_ir_builder_free(&builder);
    return failures;
}

/* The f64 16x16 branch has no kernel in the byte-identity corpus, so its op
 * order is pinned here: atoms.py emits const(n), mod, div, const(slot * 4), add. */
static int check_f64_op_order(void)
{
    const rocke_mfma_atom_t* atom = rocke_mfma_atom("fp64", 16, 16, 4);
    rocke_ir_builder_t builder;
    if(atom == NULL || rocke_ir_builder_init(&builder, "lane_to_output_f64_order") != ROCKE_OK)
    {
        return 1;
    }
    rocke_value_t* lane = rocke_b_lane_id(&builder);
    const rocke_region_t* region = builder.region_stack[builder.region_depth - 1];
    const int first_op = region->num_ops;
    rocke_value_t* row_value = NULL;
    rocke_value_t* column_value = NULL;
    int failures = 0;
    rocke_mfma_atom_lane_to_output(&builder, atom, lane, 3, &row_value, &column_value);
    static const rocke_opcode_t PYTHON_ORDER[] = {ROCKE_OP_ARITH_CONSTANT,
                                                  ROCKE_OP_ARITH_MOD,
                                                  ROCKE_OP_ARITH_DIV,
                                                  ROCKE_OP_ARITH_CONSTANT,
                                                  ROCKE_OP_ARITH_ADD};
    const int expected_ops = (int)(sizeof(PYTHON_ORDER) / sizeof(PYTHON_ORDER[0]));
    if(region->num_ops - first_op != expected_ops)
    {
        fprintf(stderr,
                "f64 atom emitted %d ops, atoms.py emits %d\n",
                region->num_ops - first_op,
                expected_ops);
        failures = 1;
    }
    else
    {
        for(int index = 0; index < expected_ops; ++index)
        {
            if(region->ops[first_op + index]->opcode != PYTHON_ORDER[index])
            {
                fprintf(stderr,
                        "f64 atom op %d is %s, not in atoms.py's order\n",
                        index,
                        region->ops[first_op + index]->name);
                failures = 1;
            }
        }
    }
    rocke_ir_builder_free(&builder);
    return failures;
}

/* atoms.py raises NotImplementedError("no lane_to_output dispatch for atom
 * {m}x{n}") for any other shape; the C++ port throws the matching
 * ckc::NotImplError. */
static int check_refusal(void)
{
    static const rocke_mfma_atom_t NON_SQUARE
        = {16, 32, 8, 4, 4, 8, "f16", "f32", "not_a_catalog_atom"};
    rocke_ir_builder_t builder;
    if(rocke_ir_builder_init(&builder, "lane_to_output_refusal") != ROCKE_OK)
    {
        return 1;
    }
    rocke_value_t* lane = rocke_b_lane_id(&builder);
    rocke_value_t* row_value = lane;
    rocke_value_t* column_value = lane;
    int failures = 0;
    try
    {
        rocke_mfma_atom_lane_to_output(&builder, &NON_SQUARE, lane, 0, &row_value, &column_value);
        fprintf(stderr, "16x32 atom was accepted\n");
        failures = 1;
    }
    catch(const ckc::NotImplError& error)
    {
        if(error.code() != ROCKE_ERR_NOTIMPL
           || strcmp(error.what(), "no lane_to_output dispatch for atom 16x32") != 0)
        {
            fprintf(stderr, "16x32 atom: code %d, message '%s'\n", (int)error.code(), error.what());
            failures = 1;
        }
    }
    if(row_value != lane || column_value != lane)
    {
        fprintf(stderr, "16x32 atom: refusal wrote the out-params\n");
        failures = 1;
    }
    rocke_ir_builder_free(&builder);
    return failures;
}

int main(void)
{
    int failures = 0;
    failures += check_layout("f16", 16, 16, 16);
    failures += check_layout("fp64", 16, 16, 4);
    failures += check_layout("f16", 32, 32, 8);
    failures += check_layout("f16", 4, 4, 4);
    failures += check_f64_op_order();
    failures += check_refusal();
    if(failures != 0)
    {
        fprintf(stderr, "rocke_mfma_atom_lane_to_output: %d failure(s)\n", failures);
        return 1;
    }
    printf("rocke_mfma_atom_lane_to_output: OK\n");
    return 0;
}
