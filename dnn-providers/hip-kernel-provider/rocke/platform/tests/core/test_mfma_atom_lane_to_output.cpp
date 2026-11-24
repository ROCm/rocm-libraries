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
 *   - that the f64 16x16 branch emits the Python's ops in its order, each
 *     taking the Python's operands in its order (the f32 branches are held to
 *     that by the byte-identity gate);
 *   - that an atom shape the Python rejects throws ckc::NotImplError with the
 *     Python's message, leaving the out-params untouched.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "lane_index_evaluator.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.atoms.h"
#include "rocke/ir.h"

static const int WAVE_SIZE = 64;

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
            const char* failure_reason = NULL;
            if(!rocke_test_evaluate_lane_index(row_value, lane, lane_value, &row, &failure_reason)
               || !rocke_test_evaluate_lane_index(
                   column_value, lane, lane_value, &column, &failure_reason))
            {
                fprintf(stderr,
                        "%s: slot %d cannot be evaluated: it reaches %s\n",
                        atom->name,
                        slot,
                        failure_reason);
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

/* An expected operand that is the lane id, not an earlier op's result. */
static const int OPERAND_LANE = -1;

/* One op of an expected emission sequence: a constant and its value, or a
 * two-operand op and the sources of its operands, each the lane id
 * (OPERAND_LANE) or the result of an earlier op of the sequence (its index). */
struct ExpectedOp
{
    rocke_opcode_t opcode;
    int64_t constant_value; /* Read only when opcode is a constant. */
    int lhs; /* Read only when opcode is not a constant. */
    int rhs;
};

/* Checks one emitted operand against its expected source; returns 1 on mismatch. */
static int check_operand(const rocke_op_t* emitted,
                         int index,
                         int operand,
                         int expected_source,
                         rocke_op_t* const* sequence,
                         const rocke_value_t* lane)
{
    const rocke_value_t* want
        = (expected_source == OPERAND_LANE) ? lane : sequence[expected_source]->results[0];
    if(emitted->operands[operand] == want)
    {
        return 0;
    }
    if(expected_source == OPERAND_LANE)
    {
        fprintf(stderr,
                "f64 atom op %d (%s): operand %d is not the lane id, as in atoms.py\n",
                index,
                emitted->name,
                operand);
    }
    else
    {
        fprintf(stderr,
                "f64 atom op %d (%s): operand %d is not op %d's result, as in atoms.py\n",
                index,
                emitted->name,
                operand,
                expected_source);
    }
    return 1;
}

/* The f64 16x16 branch has no kernel in the byte-identity corpus, so its
 * emission is pinned here, op by op and operand by operand. With slot 3,
 * atoms.py emits:
 *   ops[0] = const(16)              the atom's n
 *   ops[1] = mod(lane, ops[0])      the column
 *   ops[2] = div(lane, ops[0])      the lane's row block
 *   ops[3] = const(12)              slot * 4
 *   ops[4] = add(ops[3], ops[2])    the row
 * Operand order matters: swapped add operands give the same row but different
 * IR from the Python engine's. */
static int check_f64_emission(void)
{
    const int slot = 3;
    const int atom_n = 16;
    const ExpectedOp PYTHON_EMISSION[] = {
        {ROCKE_OP_ARITH_CONSTANT, atom_n, 0, 0},
        {ROCKE_OP_ARITH_MOD, 0, OPERAND_LANE, 0},
        {ROCKE_OP_ARITH_DIV, 0, OPERAND_LANE, 0},
        {ROCKE_OP_ARITH_CONSTANT, slot * 4, 0, 0},
        {ROCKE_OP_ARITH_ADD, 0, 3, 2},
    };
    const int expected_ops = (int)(sizeof(PYTHON_EMISSION) / sizeof(PYTHON_EMISSION[0]));
    const int row_op = 4;
    const int column_op = 1;

    const rocke_mfma_atom_t* atom = rocke_mfma_atom("fp64", 16, 16, 4);
    rocke_ir_builder_t builder;
    if(atom == NULL || rocke_ir_builder_init(&builder, "lane_to_output_f64_emission") != ROCKE_OK)
    {
        return 1;
    }
    rocke_value_t* lane = rocke_b_lane_id(&builder);
    const rocke_region_t* region = builder.region_stack[builder.region_depth - 1];
    const int first_op = region->num_ops;
    rocke_value_t* row_value = NULL;
    rocke_value_t* column_value = NULL;
    int failures = 0;
    rocke_mfma_atom_lane_to_output(&builder, atom, lane, slot, &row_value, &column_value);
    if(region->num_ops - first_op != expected_ops)
    {
        fprintf(stderr,
                "f64 atom emitted %d ops, atoms.py emits %d\n",
                region->num_ops - first_op,
                expected_ops);
        rocke_ir_builder_free(&builder);
        return 1;
    }
    rocke_op_t* const* sequence = region->ops + first_op;
    for(int index = 0; index < expected_ops; ++index)
    {
        const rocke_op_t* emitted = sequence[index];
        const ExpectedOp* want = &PYTHON_EMISSION[index];
        if(emitted->opcode != want->opcode)
        {
            fprintf(
                stderr, "f64 atom op %d is %s, not in atoms.py's order\n", index, emitted->name);
            ++failures;
            continue;
        }
        if(want->opcode == ROCKE_OP_ARITH_CONSTANT)
        {
            int64_t value = 0;
            if(!rocke_attr_get_int(&emitted->attrs, "value", &value)
               || value != want->constant_value)
            {
                fprintf(stderr,
                        "f64 atom op %d is const(%lld), atoms.py emits const(%lld)\n",
                        index,
                        (long long)value,
                        (long long)want->constant_value);
                ++failures;
            }
            continue;
        }
        failures += check_operand(emitted, index, 0, want->lhs, sequence, lane);
        failures += check_operand(emitted, index, 1, want->rhs, sequence, lane);
    }
    if(row_value != sequence[row_op]->results[0] || column_value != sequence[column_op]->results[0])
    {
        fprintf(stderr, "f64 atom: row or column is not the op atoms.py returns\n");
        ++failures;
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
    failures += check_f64_emission();
    failures += check_refusal();
    if(failures != 0)
    {
        fprintf(stderr, "rocke_mfma_atom_lane_to_output: %d failure(s)\n", failures);
        return 1;
    }
    printf("rocke_mfma_atom_lane_to_output: OK\n");
    return 0;
}
