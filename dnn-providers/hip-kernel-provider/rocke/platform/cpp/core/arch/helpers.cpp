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

#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/ir.h"
#include "rocke/ir_internal.h"

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
    /* MmaOp.c_layout(): accumulator-input (row, col) lane/slot map. */
    return rocke_mma_op_c_layout(op, b);
}

const rocke_arch_layout_map_t* rocke_mmaop_d_layout(const rocke_mmaop_t* op, rocke_ir_builder_t* b)
{
    /* MmaOp.d_layout(): result (row, col) lane/slot map. */
    return rocke_mma_op_d_layout(op, b);
}

const rocke_arch_layout_map_t* rocke_mmaop_acc_layout(const rocke_mmaop_t* op,
                                                      rocke_ir_builder_t* b)
{
    /* MmaOp.acc_layout(): compatibility spelling for the result D map. */
    return rocke_mma_op_acc_layout(op, b);
}

rocke_status_t rocke_mmaop_require_recurrence(rocke_ir_builder_t* b,
                                              const rocke_mmaop_t* op,
                                              const char* where)
{
    bool layout_mismatch = false;
    if(!rocke_i_live(b))
        return b ? b->status : ROCKE_ERR_VALUE;
    if(op == NULL || op->c_dtype == NULL || op->d_dtype == NULL)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "MMA recurrence requires C/D metadata");
        return b->status;
    }
    if(op->c_layout != NULL || op->d_layout != NULL)
    {
        layout_mismatch = op->c_layout == NULL || op->d_layout == NULL;
        if(op->c_layout != NULL && op->d_layout != NULL)
        {
            layout_mismatch = op->c_layout->frag_len != op->d_layout->frag_len
                              || op->c_layout->wave_size != op->d_layout->wave_size
                              || op->c_layout->fn != op->d_layout->fn;
        }
    }
    if(op->c_frag_len != op->d_frag_len || strcmp(op->c_dtype, op->d_dtype) != 0 || layout_mismatch)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "%s: cannot feed MMA D back as C because the C and D contracts "
                        "differ (C=%s[%d], D=%s[%d])",
                        where ? where : "MMA recurrence",
                        op->c_dtype,
                        op->c_frag_len,
                        op->d_dtype,
                        op->d_frag_len);
        return b->status;
    }
    return ROCKE_OK;
}

rocke_value_t* rocke_mmaop_zero_c(rocke_ir_builder_t* b, const rocke_mmaop_t* op)
{
    const rocke_type_t* elem = NULL;
    if(!rocke_i_live(b))
        return NULL;
    if(op == NULL || op->c_dtype == NULL)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "MMA C metadata is incomplete");
    if(strcmp(op->c_dtype, "f16") == 0 || strcmp(op->c_dtype, "fp16") == 0)
        elem = rocke_f16();
    else if(strcmp(op->c_dtype, "bf16") == 0)
        elem = rocke_bf16();
    else if(strcmp(op->c_dtype, "f32") == 0 || strcmp(op->c_dtype, "fp32") == 0)
        elem = rocke_f32();
    else if(strcmp(op->c_dtype, "i32") == 0)
        elem = rocke_i32();
    else
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unsupported MMA accumulator input dtype '%s'", op->c_dtype);
    return rocke_b_zero_vec(b, elem, op->c_frag_len);
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
                                                   const char* d_dtype,
                                                   int m,
                                                   int n,
                                                   int k)
{
    /* target.mma.op_for_shape(family=..., a/b/c/d=..., m, n, k). */
    if(t == NULL)
    {
        return NULL;
    }
    return rocke_mma_catalog_op_for_shape(
        &t->mma, family, a_dtype, b_dtype, c_dtype, d_dtype, m, n, k);
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

bool rocke_archtarget_supports_dtype_combo(const rocke_archtarget_t* t,
                                           const char* a,
                                           const char* b,
                                           const char* c,
                                           const char* d,
                                           const char* family)
{
    /* ArchTarget.supports_dtype_combo(a, b, c, d, family). */
    return rocke_arch_supports_dtype_combo(t, a, b, c, d, family);
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
