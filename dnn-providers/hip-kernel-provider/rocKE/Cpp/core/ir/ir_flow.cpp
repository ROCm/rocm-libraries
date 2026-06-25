// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * ir_ir_flow.c -- C99 port of the "ir_flow" bucket of rocke.core.ir:
 *   - cross-lane / dpp / permute + uniform-scalar helpers
 *   - transpose LDS reads
 *   - LDS pointer arithmetic + async/global DRAM->LDS
 *   - buffer rsrc + buffer loads/stores
 *   - barriers / scheduling hints
 *   - scf / cf control flow
 *
 * Pure C99 (libc only). All allocation goes through the builder arena via the
 * shared internal helpers (rocke_i_*); all error reporting goes through
 * rocke_i_set_err. Text/IR contract is rocke/ir.h -- no new IR types are invented.
 *
 * The shared helpers (rocke_i_op, rocke_i_op1, rocke_i_op0, rocke_i_attrs,
 * rocke_i_value_named, rocke_i_new_value, rocke_i_set_err, rocke_i_live, rocke_i_emit,
 * rocke_i_new_region, rocke_i_type_is, ...) are defined in bucket 0 (ir_core.c);
 * here we only call them via the internal header.
 */
#include "rocke/ir_internal.h"

#include <stdio.h>
#include <string.h>

/* ----- small local conveniences -------------------------------------------- */

/* Return true and leave the builder healthy if `t` is the i32 scalar. */
static bool rocke_flow_is_i32(const rocke_type_t* t)
{
    return rocke_i_type_is(t, "i32");
}

/* ============================ uniform / wave-scalar ====================== */

rocke_value_t* rocke_b_readfirstlane(rocke_ir_builder_t* b, rocke_value_t* v)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "readfirstlane: NULL operand");
    return rocke_i_op1(b, ROCKE_OP_TILE_READFIRSTLANE, &v, 1, v->type, NULL, "ufm");
}

rocke_value_t* rocke_b_pin_sgpr(rocke_ir_builder_t* b, rocke_value_t* v)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "pin_sgpr: NULL operand");
    return rocke_i_op1(b, ROCKE_OP_TILE_PIN_SGPR, &v, 1, v->type, NULL, "sgpr");
}

rocke_value_t* rocke_b_to_sgpr_u32(rocke_ir_builder_t* b, rocke_value_t* v)
{
    /* Convenience: pin_sgpr(readfirstlane(v)). */
    if(!rocke_i_live(b))
        return NULL;
    return rocke_b_pin_sgpr(b, rocke_b_readfirstlane(b, v));
}

rocke_value_t* rocke_b_lane_id(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
        return NULL;
    return rocke_i_op1(b, ROCKE_OP_TILE_LANE_ID, NULL, 0, rocke_i32(), NULL, "lane");
}

rocke_value_t* rocke_b_wave_all(rocke_ir_builder_t* b, rocke_value_t* predicate)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!predicate)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "wave_all: NULL predicate");
    return rocke_i_op1(b, ROCKE_OP_TILE_WAVE_ALL, &predicate, 1, rocke_i32(), NULL, "wave_all");
}

rocke_value_t* rocke_b_wave_any(rocke_ir_builder_t* b, rocke_value_t* predicate)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!predicate)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "wave_any: NULL predicate");
    return rocke_i_op1(b, ROCKE_OP_TILE_WAVE_ANY, &predicate, 1, rocke_i32(), NULL, "wave_any");
}

rocke_value_t* rocke_b_wave_ballot(rocke_ir_builder_t* b, rocke_value_t* predicate)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!predicate)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "wave_ballot: NULL predicate");
    return rocke_i_op1(b, ROCKE_OP_TILE_WAVE_BALLOT, &predicate, 1, rocke_i64(), NULL, "ballot");
}

/* ============================ cross-lane permute / dpp =================== */

rocke_value_t* rocke_b_ds_bpermute(rocke_ir_builder_t* b, rocke_value_t* addr, rocke_value_t* data)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!addr || !data)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "ds_bpermute: NULL operand");
    if(!rocke_flow_is_i32(addr->type) || !rocke_flow_is_i32(data->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_bpermute requires i32 addr + i32 data");
    ops[0] = addr;
    ops[1] = data;
    return rocke_i_op1(b, ROCKE_OP_TILE_DS_BPERMUTE, ops, 2, rocke_i32(), NULL, "bp");
}

rocke_value_t*
    rocke_b_ds_bpermute_b64(rocke_ir_builder_t* b, rocke_value_t* addr, rocke_value_t* data)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!addr || !data)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "ds_bpermute_b64: NULL operand");
    if(!rocke_flow_is_i32(addr->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_bpermute_b64 requires i32 addr");
    if(!rocke_i_type_is(data->type, "i64"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_bpermute_b64 requires i64 data");
    ops[0] = addr;
    ops[1] = data;
    return rocke_i_op1(b, ROCKE_OP_TILE_DS_BPERMUTE_B64, ops, 2, rocke_i64(), NULL, "bp64");
}

rocke_value_t* rocke_b_ds_swizzle_xor(rocke_ir_builder_t* b, rocke_value_t* data, int xor_mask)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return NULL;
    if(!data)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "ds_swizzle_xor: NULL data");
    if(!rocke_flow_is_i32(data->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_swizzle_xor requires i32 data");
    if(!(xor_mask >= 1 && xor_mask <= 31))
        return (rocke_value_t*)rocke_i_set_err(
            b,
            ROCKE_ERR_VALUE,
            "ds_swizzle_xor xor_mask must be 1..31 (intra-32-lane), got %d",
            xor_mask);
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "xor_mask", (int64_t)xor_mask);
    return rocke_i_op1(b, ROCKE_OP_TILE_DS_SWIZZLE_XOR, &data, 1, rocke_i32(), &attrs, "sw");
}

rocke_value_t* rocke_b_mov_dpp(
    rocke_ir_builder_t* b, rocke_value_t* data, int row_shr, int row_shl, bool bound_ctrl)
{
    /* C ABI: exactly one of row_shr/row_shl must be >= 0 (other < 0 = unset). */
    rocke_attr_map_t attrs;
    bool has_shr, has_shl;
    int amt;
    if(!rocke_i_live(b))
        return NULL;
    if(!data)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "mov_dpp: NULL data");
    has_shr = (row_shr >= 0);
    has_shl = (row_shl >= 0);
    if(has_shr == has_shl)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "mov_dpp requires exactly one of row_shr / row_shl to be set");
    amt = has_shr ? row_shr : row_shl;
    if(!(amt == 1 || amt == 2 || amt == 4 || amt == 8 || amt == 15))
        return (rocke_value_t*)rocke_i_set_err(b,
                                               ROCKE_ERR_VALUE,
                                               "mov_dpp %s must be in {1,2,4,8,15}, got %d",
                                               has_shr ? "row_shr" : "row_shl",
                                               amt);
    if(!rocke_flow_is_i32(data->type))
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "mov_dpp requires i32 data");
    attrs = rocke_i_attrs(b);
    rocke_attr_set_bool(b, &attrs, "bound_ctrl", bound_ctrl);
    if(has_shr)
        rocke_attr_set_int(b, &attrs, "row_shr", (int64_t)row_shr);
    else
        rocke_attr_set_int(b, &attrs, "row_shl", (int64_t)row_shl);
    return rocke_i_op1(b, ROCKE_OP_TILE_MOV_DPP, &data, 1, rocke_i32(), &attrs, "dpp");
}

void rocke_b_permlane32_swap(rocke_ir_builder_t* b,
                             rocke_value_t* lo,
                             rocke_value_t* hi,
                             rocke_value_t** out_lo,
                             rocke_value_t** out_hi)
{
    rocke_value_t* ops[2];
    const rocke_type_t* rtys[2];
    rocke_op_t* op;
    if(out_lo)
        *out_lo = NULL;
    if(out_hi)
        *out_hi = NULL;
    if(!rocke_i_live(b))
        return;
    if(!lo || !hi)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "permlane32_swap: NULL operand");
        return;
    }
    if(!rocke_flow_is_i32(lo->type) || !rocke_flow_is_i32(hi->type))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "permlane32_swap requires i32 operands");
        return;
    }
    ops[0] = lo;
    ops[1] = hi;
    rtys[0] = rocke_i32();
    rtys[1] = rocke_i32();
    op = rocke_i_op(b, ROCKE_OP_TILE_PERMLANE32_SWAP, ops, 2, rtys, 2, NULL, NULL, 0, "psw", NULL);
    if(!op)
        return;
    if(out_lo)
        *out_lo = op->results[0];
    if(out_hi)
        *out_hi = op->results[1];
}

rocke_value_t* rocke_b_perm_b32(rocke_ir_builder_t* b,
                                rocke_value_t* src0,
                                rocke_value_t* src1,
                                rocke_value_t* sel)
{
    rocke_value_t* ops[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!src0 || !src1 || !sel)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "perm_b32: NULL operand");
    if(!rocke_flow_is_i32(src0->type) || !rocke_flow_is_i32(src1->type)
       || !rocke_flow_is_i32(sel->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "perm_b32 requires i32 operands");
    ops[0] = src0;
    ops[1] = src1;
    ops[2] = sel;
    return rocke_i_op1(b, ROCKE_OP_TILE_PERM_B32, ops, 3, rocke_i32(), NULL, "perm");
}

rocke_value_t* rocke_b_permlanex16(rocke_ir_builder_t* b, rocke_value_t* v)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "permlanex16: NULL operand");
    if(!rocke_flow_is_i32(v->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "permlanex16 requires an i32 operand");
    return rocke_i_op1(b, ROCKE_OP_TILE_PERMLANEX16, &v, 1, rocke_i32(), NULL, "plx16");
}

rocke_value_t*
    rocke_b_byte_perm(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* bb, int64_t sel)
{
    rocke_value_t* ops[2];
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return NULL;
    if(!a || !bb)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "byte_perm: NULL operand");
    if(!rocke_flow_is_i32(a->type) || !rocke_flow_is_i32(bb->type))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "byte_perm requires i32 operands");
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "sel", (int64_t)((uint64_t)sel & 0xFFFFFFFFu));
    ops[0] = a;
    ops[1] = bb;
    return rocke_i_op1(b, ROCKE_OP_TILE_BYTE_PERM, ops, 2, rocke_i32(), &attrs, "bperm");
}

rocke_value_t* rocke_b_warp_shuffle_xor(rocke_ir_builder_t* b, rocke_value_t* v, int lane_xor)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!v)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "warp_shuffle_xor: NULL operand");

    if(lane_xor >= 1 && lane_xor <= 31)
    {
        /* Intra-32-lane XOR -> ds_swizzle (1 LDS op, no addr-compute). */
        if(rocke_i_type_is(v->type, "f32"))
        {
            rocke_value_t* v_i = rocke_b_bitcast(b, v, rocke_i32());
            rocke_value_t* r = rocke_b_ds_swizzle_xor(b, v_i, lane_xor);
            return rocke_b_bitcast(b, r, rocke_f32());
        }
        if(rocke_flow_is_i32(v->type))
            return rocke_b_ds_swizzle_xor(b, v, lane_xor);
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "warp_shuffle_xor: unsupported type %s", v->type->name);
    }

    /* Wave-wide XOR (lane_xor >= 32): swizzle is intra-32-lane only, fall back
     * to ds_bpermute with an explicit per-lane source-lane address. */
    {
        rocke_value_t* lane = rocke_b_lane_id(b);
        rocke_value_t* xor_const = rocke_b_const_i32(b, (int64_t)lane_xor);
        /* Python uses result_name_hint="lxor"/"laddr" for these two ops (not the
         * default "xor"/"shl" of rocke_b_xor/rocke_b_shl), so emit via rocke_i_binop
         * with the matching hints to keep the SSA names byte-identical. */
        rocke_value_t* addr = rocke_i_binop(b, ROCKE_OP_ARITH_XOR, lane, xor_const, "lxor");
        rocke_value_t* addr_shl
            = rocke_i_binop(b, ROCKE_OP_ARITH_SHL, addr, rocke_b_const_i32(b, 2), "laddr");
        if(rocke_i_type_is(v->type, "f32"))
        {
            rocke_value_t* v_i = rocke_b_bitcast(b, v, rocke_i32());
            rocke_value_t* r = rocke_b_ds_bpermute(b, addr_shl, v_i);
            return rocke_b_bitcast(b, r, rocke_f32());
        }
        if(rocke_flow_is_i32(v->type))
            return rocke_b_ds_bpermute(b, addr_shl, v);
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "warp_shuffle_xor: unsupported type %s", v->type->name);
    }
}

/* ============================ transpose LDS reads ======================= */

/* Shared helper for the three ds_read_tr* ops: [smem, *indices] operands,
 * <count x dtype> result, attrs depend on the variant. */
static rocke_value_t* rocke_flow_ds_read_tr(rocke_ir_builder_t* b,
                                            rocke_opcode_t opcode,
                                            rocke_value_t* smem,
                                            rocke_value_t* const* indices,
                                            int num_indices,
                                            const rocke_type_t* dtype,
                                            int vec_count,
                                            const char* hint,
                                            bool attr_rank_elem)
{
    rocke_value_t** ops;
    const rocke_type_t* rty;
    rocke_attr_map_t attrs;
    int i;
    if(!smem)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "ds_read_tr: NULL smem");
    ops = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                             sizeof(rocke_value_t*) * (size_t)(num_indices + 1));
    if(!ops)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_OOM, "ds_read_tr: OOM");
    ops[0] = smem;
    for(i = 0; i < num_indices; ++i)
        ops[i + 1] = indices[i];
    rty = rocke_vector_type(b, dtype, vec_count);
    attrs = rocke_i_attrs(b);
    if(attr_rank_elem)
    {
        rocke_attr_set_int(b, &attrs, "rank", (int64_t)num_indices);
        rocke_attr_set_str(b, &attrs, "elem_type", dtype->name);
    }
    else
    {
        rocke_attr_set_str(b, &attrs, "dtype", dtype->name);
    }
    return rocke_i_op1(b, opcode, ops, num_indices + 1, rty, &attrs, hint);
}

rocke_value_t* rocke_b_ds_read_tr16_b64(rocke_ir_builder_t* b,
                                        rocke_value_t* smem,
                                        rocke_value_t* const* indices,
                                        int num_indices,
                                        const rocke_type_t* dtype)
{
    if(!rocke_i_live(b))
        return NULL;
    if(num_indices <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_read_tr16_b64 needs at least one index");
    if(!dtype)
        dtype = rocke_f16();
    return rocke_flow_ds_read_tr(
        b, ROCKE_OP_TILE_DS_READ_TR16_B64, smem, indices, num_indices, dtype, 4, "tr16", true);
}

rocke_value_t* rocke_b_ds_read_tr16_b128(rocke_ir_builder_t* b,
                                         rocke_value_t* smem,
                                         rocke_value_t* const* indices,
                                         int num_indices,
                                         const rocke_type_t* dtype)
{
    if(!rocke_i_live(b))
        return NULL;
    if(num_indices <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_read_tr16_b128 needs at least one index");
    if(!dtype)
        dtype = rocke_f16();
    return rocke_flow_ds_read_tr(
        b, ROCKE_OP_TILE_DS_READ_TR16_B128, smem, indices, num_indices, dtype, 8, "tr16w", true);
}

rocke_value_t* rocke_b_ds_read_tr_b8(rocke_ir_builder_t* b,
                                     rocke_value_t* smem,
                                     rocke_value_t* const* indices,
                                     int num_indices,
                                     const rocke_type_t* dtype)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!dtype)
        dtype = rocke_fp8e4m3();
    if(!rocke_i_type_is(dtype, "fp8e4m3") && !rocke_i_type_is(dtype, "bf8e5m2")
       && !rocke_i_type_is(dtype, "i8"))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_read_tr_b8 expects fp8/bf8/i8 dtype, got %s", dtype->name);
    if(num_indices <= 0)
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "ds_read_tr_b8 needs at least one index");
    return rocke_flow_ds_read_tr(
        b, ROCKE_OP_TILE_DS_READ_TR_B8, smem, indices, num_indices, dtype, 8, "tr8", false);
}

/* ============================ LDS pointer arithmetic ==================== */

rocke_value_t* rocke_b_smem_addr_of(rocke_ir_builder_t* b, rocke_value_t* smem)
{
    if(!rocke_i_live(b))
        return NULL;
    if(!smem)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_addr_of: NULL smem");
    return rocke_i_op1(b, ROCKE_OP_TILE_SMEM_ADDR_OF, &smem, 1, rocke_i64(), NULL, "lds_addr");
}

rocke_value_t*
    rocke_b_smem_ptr_add(rocke_ir_builder_t* b, rocke_value_t* lds_addr, rocke_value_t* byte_off)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!lds_addr || !byte_off)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "smem_ptr_add: NULL operand");
    ops[0] = lds_addr;
    ops[1] = byte_off;
    return rocke_i_op1(b, ROCKE_OP_TILE_SMEM_PTR_ADD, ops, 2, rocke_i64(), NULL, "lds_addr");
}

void rocke_b_async_buffer_load_lds_addr(rocke_ir_builder_t* b,
                                        rocke_value_t* rsrc,
                                        rocke_value_t* lds_addr,
                                        rocke_value_t* voffset,
                                        rocke_value_t* soffset,
                                        int dwords,
                                        int coherency)
{
    rocke_value_t* ops[4];
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(!(dwords == 1 || dwords == 3 || dwords == 4))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "async_buffer_load_lds_addr dwords must be 1, 3, or 4 (got %d)",
                        dwords);
        return;
    }
    if(!(coherency >= 0 && coherency <= 3))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "coherency must be 0..3 (got %d)", coherency);
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "dwords", (int64_t)dwords);
    rocke_attr_set_int(b, &attrs, "aux", (int64_t)coherency);
    ops[0] = rsrc;
    ops[1] = lds_addr;
    ops[2] = voffset;
    ops[3] = soffset;
    rocke_i_op0(b, ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS_ADDR, ops, 4, &attrs);
}

void rocke_b_async_buffer_load_lds(rocke_ir_builder_t* b,
                                   rocke_value_t* rsrc,
                                   rocke_value_t* lds_ptr,
                                   rocke_value_t* voffset,
                                   rocke_value_t* soffset,
                                   int dwords,
                                   int coherency)
{
    rocke_value_t* ops[4];
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(!(dwords == 1 || dwords == 3 || dwords == 4))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "async_buffer_load_lds dwords must be 1, 3, or 4 (got %d)", dwords);
        return;
    }
    if(!(coherency >= 0 && coherency <= 3))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "coherency must be 0..3 (got %d)", coherency);
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "dwords", (int64_t)dwords);
    rocke_attr_set_int(b, &attrs, "aux", (int64_t)coherency);
    ops[0] = rsrc;
    ops[1] = lds_ptr;
    ops[2] = voffset;
    ops[3] = soffset;
    rocke_i_op0(b, ROCKE_OP_TILE_ASYNC_BUFFER_LOAD_LDS, ops, 4, &attrs);
}

void rocke_b_global_load_lds(rocke_ir_builder_t* b,
                             rocke_value_t* src_ptr,
                             rocke_value_t* byte_off,
                             rocke_value_t* lds_addr,
                             int size_bytes,
                             int coherency)
{
    rocke_value_t* ops[3];
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(!(size_bytes == 1 || size_bytes == 2 || size_bytes == 4 || size_bytes == 12
         || size_bytes == 16))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "global_load_lds size_bytes must be 1, 2, 4, 12, or 16 (got %d)",
                        size_bytes);
        return;
    }
    if(!(coherency >= 0 && coherency <= 3))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "coherency must be 0..3 (got %d)", coherency);
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "size_bytes", (int64_t)size_bytes);
    rocke_attr_set_int(b, &attrs, "aux", (int64_t)coherency);
    ops[0] = src_ptr;
    ops[1] = byte_off;
    ops[2] = lds_addr;
    rocke_i_op0(b, ROCKE_OP_TILE_GLOBAL_LOAD_LDS, ops, 3, &attrs);
}

/* ============================ global ptr + buffer rsrc ================== */

rocke_value_t*
    rocke_b_global_ptr_add(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* byte_off)
{
    rocke_value_t* ops[2];
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !byte_off)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "global_ptr_add: NULL operand");
    ops[0] = ptr;
    ops[1] = byte_off;
    return rocke_i_op1(b, ROCKE_OP_TILE_GLOBAL_PTR_ADD, ops, 2, ptr->type, NULL, "gptr");
}

rocke_value_t*
    rocke_b_buffer_rsrc(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* num_bytes)
{
    rocke_value_t* ops[2];
    const rocke_type_t* rty;
    if(!rocke_i_live(b))
        return NULL;
    if(!ptr || !num_bytes)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "buffer_rsrc: NULL operand");
    rty = rocke_vector_type(b, rocke_i32(), 4);
    ops[0] = ptr;
    ops[1] = num_bytes;
    return rocke_i_op1(b, ROCKE_OP_TILE_BUFFER_RSRC, ops, 2, rty, NULL, "rsrc");
}

rocke_value_t* rocke_b_buffer_load_vN_f16(rocke_ir_builder_t* b,
                                          rocke_value_t* rsrc,
                                          rocke_value_t* voffset,
                                          rocke_value_t* soffset,
                                          int dwords)
{
    rocke_value_t* ops[3];
    const rocke_type_t* rty;
    rocke_attr_map_t attrs;
    int halves;
    char hint[16];
    if(!rocke_i_live(b))
        return NULL;
    if(!(dwords == 1 || dwords == 2 || dwords == 4))
        return (rocke_value_t*)rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "buffer_load dwords must be 1, 2, or 4 (got %d)", dwords);
    halves = dwords * 2;
    rty = rocke_vector_type(b, rocke_f16(), halves);
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "dwords", (int64_t)dwords);
    /* result_name_hint = f"bl{halves}" */
    snprintf(hint, sizeof(hint), "bl%d", halves);
    ops[0] = rsrc;
    ops[1] = voffset;
    ops[2] = soffset;
    return rocke_i_op1(b, ROCKE_OP_TILE_BUFFER_LOAD_VN_F16, ops, 3, rty, &attrs, hint);
}

rocke_value_t* rocke_b_buffer_load_f16(rocke_ir_builder_t* b,
                                       rocke_value_t* rsrc,
                                       rocke_value_t* voffset,
                                       rocke_value_t* soffset)
{
    rocke_value_t* ops[3];
    if(!rocke_i_live(b))
        return NULL;
    if(!rsrc || !voffset || !soffset)
        return (rocke_value_t*)rocke_i_set_err(b, ROCKE_ERR_VALUE, "buffer_load_f16: NULL operand");
    ops[0] = rsrc;
    ops[1] = voffset;
    ops[2] = soffset;
    return rocke_i_op1(b, ROCKE_OP_TILE_BUFFER_LOAD_F16, ops, 3, rocke_f16(), NULL, "bl1");
}

void rocke_b_buffer_store_vN_f16(rocke_ir_builder_t* b,
                                 rocke_value_t* rsrc,
                                 rocke_value_t* voffset,
                                 rocke_value_t* soffset,
                                 rocke_value_t* value,
                                 int dwords)
{
    rocke_value_t* ops[4];
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(!(dwords == 1 || dwords == 2 || dwords == 4))
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "buffer_store dwords must be 1, 2, or 4 (got %d)", dwords);
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "dwords", (int64_t)dwords);
    ops[0] = rsrc;
    ops[1] = voffset;
    ops[2] = soffset;
    ops[3] = value;
    rocke_i_op0(b, ROCKE_OP_TILE_BUFFER_STORE_VN_F16, ops, 4, &attrs);
}

void rocke_b_buffer_store_f16(rocke_ir_builder_t* b,
                              rocke_value_t* rsrc,
                              rocke_value_t* voffset,
                              rocke_value_t* soffset,
                              rocke_value_t* value)
{
    rocke_value_t* ops[4];
    if(!rocke_i_live(b))
        return;
    ops[0] = rsrc;
    ops[1] = voffset;
    ops[2] = soffset;
    ops[3] = value;
    rocke_i_op0(b, ROCKE_OP_TILE_BUFFER_STORE_F16, ops, 4, NULL);
}

/* ============================ barriers / scheduling ===================== */

void rocke_b_sync(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
        return;
    rocke_i_op0(b, ROCKE_OP_TILE_SYNC, NULL, 0, NULL);
}

void rocke_b_s_barrier_bare(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
        return;
    rocke_i_op0(b, ROCKE_OP_TILE_S_BARRIER_BARE, NULL, 0, NULL);
}

void rocke_b_sync_half_block(rocke_ir_builder_t* b, rocke_value_t* half_selector)
{
    if(!rocke_i_live(b))
        return;
    if(!half_selector)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "sync_half_block: NULL selector");
        return;
    }
    rocke_i_op0(b, ROCKE_OP_TILE_SYNC_HALF_BLOCK, &half_selector, 1, NULL);
}

void rocke_b_sync_lds_only(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
        return;
    rocke_i_op0(b, ROCKE_OP_TILE_SYNC_LDS_ONLY, NULL, 0, NULL);
}

void rocke_b_s_waitcnt(rocke_ir_builder_t* b, int vmcnt, int lgkmcnt, int expcnt)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "vmcnt", (int64_t)vmcnt);
    rocke_attr_set_int(b, &attrs, "lgkmcnt", (int64_t)lgkmcnt);
    rocke_attr_set_int(b, &attrs, "expcnt", (int64_t)expcnt);
    rocke_i_op0(b, ROCKE_OP_TILE_S_WAITCNT, NULL, 0, &attrs);
}

void rocke_b_s_setprio(rocke_ir_builder_t* b, int level)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(level < 0 || level > 3)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "s_setprio level must be in 0..3");
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "level", (int64_t)level);
    rocke_i_op0(b, ROCKE_OP_TILE_S_SETPRIO, NULL, 0, &attrs);
}

void rocke_b_iglp_opt(rocke_ir_builder_t* b, int level)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "level", (int64_t)level);
    rocke_i_op0(b, ROCKE_OP_TILE_IGLP_OPT, NULL, 0, &attrs);
}

void rocke_b_sched_barrier(rocke_ir_builder_t* b, int mask)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "mask", (int64_t)mask);
    rocke_i_op0(b, ROCKE_OP_TILE_SCHED_BARRIER, NULL, 0, &attrs);
}

void rocke_b_sched_group_barrier(rocke_ir_builder_t* b, int mask, int count, int group)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "mask", (int64_t)mask);
    rocke_attr_set_int(b, &attrs, "count", (int64_t)count);
    rocke_attr_set_int(b, &attrs, "group", (int64_t)group);
    rocke_i_op0(b, ROCKE_OP_TILE_SCHED_GROUP_BARRIER, NULL, 0, &attrs);
}

/* ============================ scf / cf control flow ===================== */

/* scf.for: build the op directly (with a "body" region) and a fresh induction
 * variable Value named "%<iv_name>" whose .op points at the for op. Mirrors
 * IRBuilder.scf_for: it does NOT funnel through _op because the iv Value is
 * constructed explicitly and is not an op result.
 *
 * In the C contract rocke_i_op creates the op, copies operands/attrs/regions into
 * the arena and emits it; we then back-patch the iv. rocke_i_op with 0 result
 * types matches the Python results=[] for the no-iter-args case. */
rocke_for_t rocke_b_scf_for(rocke_ir_builder_t* b,
                            rocke_value_t* lo,
                            rocke_value_t* hi,
                            rocke_value_t* step,
                            const char* iv_name)
{
    rocke_for_t out;
    rocke_value_t* ops[3];
    rocke_region_t* body;
    rocke_region_t* regions[1];
    rocke_attr_map_t attrs;
    rocke_op_t* op;
    char iv_full[128];
    const char* nm = iv_name ? iv_name : "k0";

    memset(&out, 0, sizeof(out));
    if(!rocke_i_live(b))
        return out;
    if(!lo || !hi || !step)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_for: NULL bound/step");
        return out;
    }

    body = rocke_i_new_region(b, "body");
    if(!body)
        return out;

    snprintf(iv_full, sizeof(iv_full), "%%%s", nm);

    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "iv", iv_full);
    rocke_attr_set_str(b, &attrs, "iv_type", lo->type->name);

    ops[0] = lo;
    ops[1] = hi;
    ops[2] = step;
    regions[0] = body;
    op = rocke_i_op(b, ROCKE_OP_SCF_FOR, ops, 3, NULL, 0, &attrs, regions, 1, "v", NULL);
    if(!op)
        return out;

    out.op = op;
    out.iv = rocke_i_value_named(b, iv_full, lo->type);
    if(out.iv)
        out.iv->op = op;
    out.body = body;
    out.iter_vars = NULL;
    out.num_iter_vars = 0;
    return out;
}

rocke_for_t rocke_b_scf_for_iter(rocke_ir_builder_t* b,
                                 rocke_value_t* lo,
                                 rocke_value_t* hi,
                                 rocke_value_t* step,
                                 const rocke_iter_arg_t* iter_args,
                                 int num_iter_args,
                                 const char* iv_name,
                                 bool unroll,
                                 bool elide_trailing_barrier)
{
    rocke_for_t out;
    rocke_value_t** ops;
    rocke_value_t** iter_vars;
    const rocke_type_t** rtys;
    rocke_region_t* body;
    rocke_region_t* regions[1];
    rocke_attr_map_t attrs;
    rocke_attr_map_t iter_list;
    rocke_op_t* op;
    char iv_full[128];
    int i;
    const char* nm = iv_name ? iv_name : "k0";

    memset(&out, 0, sizeof(out));
    if(!rocke_i_live(b))
        return out;
    if(!lo || !hi || !step)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_for_iter: NULL bound/step");
        return out;
    }
    if(num_iter_args < 0)
        num_iter_args = 0;
    if(num_iter_args > 0 && !iter_args)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_for_iter: NULL iter_args");
        return out;
    }

    body = rocke_i_new_region(b, "body");
    if(!body)
        return out;

    snprintf(iv_full, sizeof(iv_full), "%%%s", nm);

    /* operands = [lo, hi, step, *iter_inits] */
    ops = (rocke_value_t**)rocke_arena_alloc(&b->arena,
                                             sizeof(rocke_value_t*) * (size_t)(3 + num_iter_args));
    if(!ops)
    {
        rocke_i_set_err(b, ROCKE_ERR_OOM, "scf_for_iter: OOM");
        return out;
    }
    ops[0] = lo;
    ops[1] = hi;
    ops[2] = step;

    /* iter_vars: fresh Values named "%<arg_name>" carrying init type. These are
     * not op results; they are back-patched to point at the op like the iv. */
    iter_vars = NULL;
    if(num_iter_args > 0)
    {
        iter_vars = (rocke_value_t**)rocke_arena_alloc(
            &b->arena, sizeof(rocke_value_t*) * (size_t)num_iter_args);
        if(!iter_vars)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "scf_for_iter: OOM");
            return out;
        }
    }

    /* result types: one per iter arg, type = init type. rocke_i_op makes the
     * result Values, named with the "for" hint => "%for<N>" matching
     * Python's self._fresh("for"). */
    rtys = NULL;
    if(num_iter_args > 0)
    {
        rtys = (const rocke_type_t**)rocke_arena_alloc(
            &b->arena, sizeof(const rocke_type_t*) * (size_t)num_iter_args);
        if(!rtys)
        {
            rocke_i_set_err(b, ROCKE_ERR_OOM, "scf_for_iter: OOM");
            return out;
        }
    }

    /* Build the iter_args metadata list attr: [{name, type}, ...]. */
    iter_list = rocke_i_attrs(b); /* used as a list container below */

    attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "iv", iv_full);
    rocke_attr_set_str(b, &attrs, "iv_type", lo->type->name);

    /* Assemble iter_args metadata + iter_inits + iter_vars + rtys. */
    {
        struct rocke_attr_map** items = NULL;
        if(num_iter_args > 0)
        {
            items = (struct rocke_attr_map**)rocke_arena_alloc(
                &b->arena, sizeof(struct rocke_attr_map*) * (size_t)num_iter_args);
            if(!items)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "scf_for_iter: OOM");
                return out;
            }
        }
        for(i = 0; i < num_iter_args; ++i)
        {
            char vn[128];
            const char* arg_name = iter_args[i].name ? iter_args[i].name : "";
            rocke_value_t* init = iter_args[i].init;
            struct rocke_attr_map* meta;
            if(!init)
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_for_iter: NULL iter init");
                return out;
            }

            snprintf(vn, sizeof(vn), "%%%s", arg_name);
            iter_vars[i] = rocke_i_value_named(b, vn, init->type);
            if(!iter_vars[i])
                return out;
            ops[3 + i] = init;
            rtys[i] = init->type;

            /* meta entry: {"name": vn, "type": init.type.name} */
            meta = (struct rocke_attr_map*)rocke_arena_calloc(&b->arena, sizeof(*meta));
            if(!meta)
            {
                rocke_i_set_err(b, ROCKE_ERR_OOM, "scf_for_iter: OOM");
                return out;
            }
            rocke_attr_map_init(meta);
            rocke_attr_set_str(b, meta, "name", vn);
            rocke_attr_set_str(b, meta, "type", init->type->name);
            items[i] = meta;
        }
        /* attach the list attr "iter_args" */
        {
            rocke_attr_value_t lv;
            /* set a placeholder int so the entry exists, then overwrite kind */
            rocke_attr_set_int(b, &attrs, "iter_args", 0);
            {
                /* find the just-added entry and convert it to a LIST */
                int e;
                for(e = attrs.count - 1; e >= 0; --e)
                {
                    if(strcmp(attrs.entries[e].key, "iter_args") == 0)
                    {
                        lv.kind = ROCKE_ATTR_LIST;
                        lv.u.list.items = items;
                        lv.u.list.count = num_iter_args;
                        attrs.entries[e].value = lv;
                        break;
                    }
                }
            }
        }
    }
    (void)iter_list;

    rocke_attr_set_int(b, &attrs, "num_iter_args", (int64_t)num_iter_args);
    rocke_attr_set_bool(b, &attrs, "unroll", unroll);
    rocke_attr_set_bool(b, &attrs, "elide_trailing_barrier", elide_trailing_barrier);

    regions[0] = body;
    op = rocke_i_op(b,
                    ROCKE_OP_SCF_FOR,
                    ops,
                    3 + num_iter_args,
                    rtys,
                    num_iter_args,
                    &attrs,
                    regions,
                    1,
                    "for",
                    NULL);
    if(!op)
        return out;

    /* Back-patch iv + iter_vars to point at the op. Results already linked by
     * rocke_i_op. */
    out.op = op;
    out.iv = rocke_i_value_named(b, iv_full, lo->type);
    if(out.iv)
        out.iv->op = op;
    for(i = 0; i < num_iter_args; ++i)
    {
        if(iter_vars[i])
            iter_vars[i]->op = op;
    }
    out.body = body;
    out.iter_vars = iter_vars;
    out.num_iter_vars = num_iter_args;
    return out;
}

void rocke_b_scf_yield(rocke_ir_builder_t* b, rocke_value_t* const* values, int num_values)
{
    rocke_attr_map_t attrs;
    if(!rocke_i_live(b))
        return;
    if(num_values < 0)
        num_values = 0;
    if(num_values > 0 && !values)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_yield: NULL values");
        return;
    }
    attrs = rocke_i_attrs(b);
    rocke_attr_set_int(b, &attrs, "num", (int64_t)num_values);
    rocke_i_op0(b, ROCKE_OP_SCF_YIELD, values, num_values, &attrs);
}

rocke_if_t rocke_b_scf_if(rocke_ir_builder_t* b, rocke_value_t* cond)
{
    rocke_if_t out;
    rocke_region_t* then_r;
    rocke_region_t* regions[1];
    rocke_op_t* op;

    memset(&out, 0, sizeof(out));
    if(!rocke_i_live(b))
        return out;
    if(!cond)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "scf_if: NULL cond");
        return out;
    }

    then_r = rocke_i_new_region(b, "then");
    if(!then_r)
        return out;
    regions[0] = then_r;
    op = rocke_i_op(b, ROCKE_OP_SCF_IF, &cond, 1, NULL, 0, NULL, regions, 1, "v", NULL);
    if(!op)
        return out;
    out.op = op;
    out.then_region = then_r;
    return out;
}

void rocke_b_ret(rocke_ir_builder_t* b)
{
    if(!rocke_i_live(b))
        return;
    rocke_i_op0(b, ROCKE_OP_CF_RETURN, NULL, 0, NULL);
}
