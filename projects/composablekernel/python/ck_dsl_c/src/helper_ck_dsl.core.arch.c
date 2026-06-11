/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.core.arch.c -- C99 port of the ck_dsl.core.arch helper symbols
 * ArchTarget and MmaOp. See ckc/helper_ck_dsl.core.arch.h for the rationale.
 *
 * Every function here is a thin forwarder onto the canonical, byte-identical
 * arch port (ckc/arch_target.h, implemented in arch_target_data.c /
 * arch_target_query.c). No data or builder-call sequence is duplicated: the IR
 * the layout maps emit, the catalog the target carries, and the from_gfx
 * lookups all resolve to the single canonical SSOT. This file exists only so the
 * helper closure's `ck_dsl.core.arch { ArchTarget, MmaOp }` binding has a stable
 * helper-namespace home, mirroring the Python module surface.
 */

#include "ckc/helper_ck_dsl.core.arch.h"

#include "ckc/arch_target.h"
#include "ckc/ir.h"

/* ============================== MmaOp ================================== */

void ckc_mmaop_shape(const ckc_mmaop_t* op, int* m, int* n, int* k)
{
    /* MmaOp.shape property -> (m, n, k). */
    ckc_mma_op_shape(op, m, n, k);
}

const ckc_arch_layout_map_t* ckc_mmaop_a_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b)
{
    /* MmaOp.a_layout(): A-operand (row, k) lane/slot map. */
    return ckc_mma_op_a_layout(op, b);
}

const ckc_arch_layout_map_t* ckc_mmaop_b_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b)
{
    /* MmaOp.b_layout(): B-operand (k, col) lane/slot map. */
    return ckc_mma_op_b_layout(op, b);
}

const ckc_arch_layout_map_t* ckc_mmaop_c_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b)
{
    /* MmaOp.c_layout(): accumulator (row, col) lane/slot map. */
    return ckc_mma_op_c_layout(op, b);
}

const ckc_arch_layout_map_t* ckc_mmaop_acc_layout(const ckc_mmaop_t* op, ckc_ir_builder_t* b)
{
    /* MmaOp.acc_layout(): alias for the accumulator (C) map. */
    return ckc_mma_op_acc_layout(op, b);
}

bool ckc_arch_layout_map_coord(const ckc_arch_layout_map_t* m,
                               ckc_ir_builder_t* b,
                               ckc_value_t* lane,
                               int slot,
                               ckc_value_t** out0,
                               ckc_value_t** out1)
{
    /* LayoutMap.coord(builder, lane, slot): validate slot, emit index math. */
    return ckc_layout_map_coord(m, b, lane, slot, out0, out1);
}

/* ============================== ArchTarget ============================= */

const ckc_archtarget_t* ckc_archtarget_from_gfx(const char* gfx)
{
    /* ArchTarget.from_gfx(gfx) -> singleton descriptor (NULL if unknown). */
    return ckc_arch_target_from_gfx(gfx);
}

const ckc_arch_mma_catalog_t* ckc_archtarget_mma(const ckc_archtarget_t* t)
{
    /* target.mma. */
    if (t == NULL)
    {
        return NULL;
    }
    return &t->mma;
}

const ckc_mmaop_t* ckc_archtarget_op_for_shape(const ckc_archtarget_t* t,
                                               const char* family,
                                               const char* a_dtype,
                                               const char* b_dtype,
                                               const char* c_dtype,
                                               int m,
                                               int n,
                                               int k)
{
    /* target.mma.op_for_shape(family=..., a/b/c=..., m, n, k). */
    if (t == NULL)
    {
        return NULL;
    }
    return ckc_mma_catalog_op_for_shape(&t->mma, family, a_dtype, b_dtype, c_dtype, m, n, k);
}

const char* ckc_archtarget_isa_triple(const ckc_archtarget_t* t, char* out, size_t out_cap)
{
    /* ArchTarget.isa_triple property. */
    return ckc_arch_isa_triple(t, out, out_cap);
}

bool ckc_archtarget_fits_lds(const ckc_archtarget_t* t, long bytes_in_use)
{
    /* ArchTarget.fits_lds(bytes_in_use). */
    return ckc_arch_fits_lds(t, bytes_in_use);
}

bool ckc_archtarget_supports_dtype_combo(
    const ckc_archtarget_t* t, const char* a, const char* b, const char* c, const char* family)
{
    /* ArchTarget.supports_dtype_combo(a, b, c, family). */
    return ckc_arch_supports_dtype_combo(t, a, b, c, family);
}

int ckc_archtarget_max_vector_load_dwords(const ckc_archtarget_t* t, const char* dtype)
{
    /* ArchTarget.max_vector_load_dwords(dtype). */
    return ckc_arch_max_vector_load_dwords(t, dtype);
}

int ckc_archtarget_max_threads_per_block(const ckc_archtarget_t* t)
{
    /* ArchTarget.max_threads_per_block property. */
    return ckc_arch_max_threads_per_block(t);
}
