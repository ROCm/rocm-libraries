// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.core.arch.c -- C99 port of the rocke.core.arch helper symbols
 * ArchTarget and MmaOp. See rocke/helper_rocke.core.arch.h for the rationale.
 *
 * Every function here is a thin forwarder onto the canonical, byte-identical
 * arch port (rocke/arch_target.h, implemented in arch_target_data.c /
 * arch_target_query.c). No data or builder-call sequence is duplicated: the IR
 * the layout maps emit, the catalog the target carries, and the from_gfx
 * lookups all resolve to the single canonical SSOT. This file exists only so the
 * helper closure's `rocke.core.arch { ArchTarget, MmaOp }` binding has a stable
 * helper-namespace home, mirroring the Python module surface.
 */

#include "rocke/helper_rocke.core.arch.h"

#include "rocke/arch_target.h"
#include "rocke/ir.h"

/* ============================== MmaOp ================================== */

void rocke_mmaop_shape(const rocke_mmaop_t* op, int* m, int* n, int* k)
{
    /* MmaOp.shape property -> (m, n, k). */
    rocke_mma_op_shape(op, m, n, k);
}

const rocke_arch_layout_map_t* rocke_mmaop_a_layout(const rocke_mmaop_t* op, rocke_ir_builder_t* b)
{
    /* MmaOp.a_layout(): A-operand (row, k) lane/slot map. */
    return rocke_mma_op_a_layout(op, b);
}

const rocke_arch_layout_map_t* rocke_mmaop_b_layout(const rocke_mmaop_t* op, rocke_ir_builder_t* b)
{
    /* MmaOp.b_layout(): B-operand (k, col) lane/slot map. */
    return rocke_mma_op_b_layout(op, b);
}

const rocke_arch_layout_map_t* rocke_mmaop_c_layout(const rocke_mmaop_t* op, rocke_ir_builder_t* b)
{
    /* MmaOp.c_layout(): accumulator (row, col) lane/slot map. */
    return rocke_mma_op_c_layout(op, b);
}

const rocke_arch_layout_map_t* rocke_mmaop_acc_layout(const rocke_mmaop_t* op,
                                                      rocke_ir_builder_t* b)
{
    /* MmaOp.acc_layout(): alias for the accumulator (C) map. */
    return rocke_mma_op_acc_layout(op, b);
}

bool rocke_arch_layout_map_coord(const rocke_arch_layout_map_t* m,
                                 rocke_ir_builder_t* b,
                                 rocke_value_t* lane,
                                 int slot,
                                 rocke_value_t** out0,
                                 rocke_value_t** out1)
{
    /* LayoutMap.coord(builder, lane, slot): validate slot, emit index math. */
    return rocke_layout_map_coord(m, b, lane, slot, out0, out1);
}

/* ============================== ArchTarget ============================= */

const rocke_archtarget_t* rocke_archtarget_from_gfx(const char* gfx)
{
    /* ArchTarget.from_gfx(gfx) -> singleton descriptor (NULL if unknown). */
    return rocke_arch_target_from_gfx(gfx);
}

const rocke_arch_mma_catalog_t* rocke_archtarget_mma(const rocke_archtarget_t* t)
{
    /* target.mma. */
    if(t == NULL)
    {
        return NULL;
    }
    return &t->mma;
}

const rocke_mmaop_t* rocke_archtarget_op_for_shape(const rocke_archtarget_t* t,
                                                   const char* family,
                                                   const char* a_dtype,
                                                   const char* b_dtype,
                                                   const char* c_dtype,
                                                   int m,
                                                   int n,
                                                   int k)
{
    /* target.mma.op_for_shape(family=..., a/b/c=..., m, n, k). */
    if(t == NULL)
    {
        return NULL;
    }
    return rocke_mma_catalog_op_for_shape(&t->mma, family, a_dtype, b_dtype, c_dtype, m, n, k);
}

const rocke_mmaop_t* rocke_archtarget_by_op_id(const rocke_archtarget_t* t, const char* op_id)
{
    /* target.mma.by_op_id(op_id): look up an atom by its op_id handle. */
    if(t == NULL)
    {
        return NULL;
    }
    return rocke_mma_catalog_by_op_id(&t->mma, op_id);
}

const char* rocke_archtarget_isa_triple(const rocke_archtarget_t* t, char* out, size_t out_cap)
{
    /* ArchTarget.isa_triple property. */
    return rocke_arch_isa_triple(t, out, out_cap);
}

bool rocke_archtarget_fits_lds(const rocke_archtarget_t* t, long bytes_in_use)
{
    /* ArchTarget.fits_lds(bytes_in_use). */
    return rocke_arch_fits_lds(t, bytes_in_use);
}

bool rocke_archtarget_supports_dtype_combo(
    const rocke_archtarget_t* t, const char* a, const char* b, const char* c, const char* family)
{
    /* ArchTarget.supports_dtype_combo(a, b, c, family). */
    return rocke_arch_supports_dtype_combo(t, a, b, c, family);
}

int rocke_archtarget_max_vector_load_dwords(const rocke_archtarget_t* t, const char* dtype)
{
    /* ArchTarget.max_vector_load_dwords(dtype). */
    return rocke_arch_max_vector_load_dwords(t, dtype);
}

int rocke_archtarget_max_threads_per_block(const rocke_archtarget_t* t)
{
    /* ArchTarget.max_threads_per_block property. */
    return rocke_arch_max_threads_per_block(t);
}
