// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_fwd_wmma_tdm.cpp -- the TDM staging helpers of the wave32 WMMA forward
 * families: the rank-3 TDM descriptor, the ring wait and the LDS fragment read
 * (see rocke/instance_conv_fwd_wmma_tdm_internal.h).
 */
#include <stdint.h>

#include "rocke/instance_conv_fwd_wmma_tdm_internal.h"
#include "rocke/ir.h"

void rocke_fwd_tdm_descriptor_3d(rocke_ir_builder_t* b,
                                 rocke_value_t* global_ptr,
                                 rocke_value_t* lds_addr,
                                 rocke_value_t* d0,
                                 rocke_value_t* d1,
                                 rocke_value_t* d2,
                                 rocke_value_t* s1,
                                 rocke_value_t* s2,
                                 int tile0,
                                 int tile1,
                                 rocke_value_t* tile2,
                                 rocke_value_t* out[5],
                                 rocke_value_t* tile1_v)
{
    rocke_value_t* c16 = rocke_b_const_i32(b, 16);
    rocke_value_t* gaddr = rocke_b_global_ptr_to_i64(b, global_ptr);
    rocke_value_t* c32 = rocke_b_const_i64(b, 32);
    rocke_value_t* sh = rocke_b_lshr(b, gaddr, c32);
    rocke_value_t* g_hi = rocke_b_trunc(b, sh, rocke_i32());
    rocke_value_t* g0[4];
    rocke_value_t* g1[8];
    rocke_value_t* g2[4];
    g0[0] = rocke_b_const_i32(b, 1);
    g0[1] = lds_addr;
    g0[2] = rocke_b_trunc(b, gaddr, rocke_i32());
    {
        rocke_value_t* m = rocke_b_const_i32(b, 0x1FFFFFF);
        rocke_value_t* lo = rocke_b_land(b, g_hi, m);
        rocke_value_t* ty = rocke_b_const_i32(b, -((int64_t)1 << 31));
        g0[3] = rocke_b_lor(b, lo, ty);
    }
    g1[0] = rocke_b_const_i32(b, 1 << 16);
    g1[1] = rocke_b_shl(b, d0, c16);
    {
        rocke_value_t* x = rocke_b_lshr(b, d0, c16);
        rocke_value_t* y = rocke_b_shl(b, d1, c16);
        g1[2] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = rocke_b_lshr(b, d1, c16);
        rocke_value_t* y = rocke_b_const_i32(b, (int64_t)tile0 << 16);
        g1[3] = rocke_b_lor(b, x, y);
    }
    {
        rocke_value_t* x = tile1_v != NULL ? tile1_v : rocke_b_const_i32(b, tile1);
        rocke_value_t* y = rocke_b_shl(b, tile2, c16);
        g1[4] = rocke_b_lor(b, x, y);
    }
    g1[5] = s1;
    g1[6] = rocke_b_shl(b, s2, c16);
    g1[7] = rocke_b_lshr(b, s2, c16);
    g2[0] = d2;
    g2[1] = rocke_b_const_i32(b, 0);
    g2[2] = rocke_b_const_i32(b, 0);
    g2[3] = rocke_b_const_i32(b, 0);
    rocke_value_t* r[8];
    for(int i = 0; i < 4; ++i)
        r[i] = rocke_b_readfirstlane(b, g0[i]);
    out[0] = rocke_b_vec_pack(b, r, 4, rocke_i32());
    for(int i = 0; i < 8; ++i)
        r[i] = rocke_b_readfirstlane(b, g1[i]);
    out[1] = rocke_b_vec_pack(b, r, 8, rocke_i32());
    for(int i = 0; i < 4; ++i)
        r[i] = rocke_b_readfirstlane(b, g2[i]);
    out[2] = rocke_b_vec_pack(b, r, 4, rocke_i32());
    out[3] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out[4] = rocke_b_zero_vec(b, rocke_i32(), 8);
}

void rocke_fwd_tdm_ring_wait(rocke_ir_builder_t* b,
                             rocke_value_t* step,
                             int stages,
                             rocke_value_t* n_steps)
{
    if(stages == 2)
    {
        rocke_b_s_wait_tensorcnt(b, 0);
        return;
    }
    rocke_value_t* c = rocke_b_const_i32(b, stages - 2);
    rocke_value_t* x = rocke_b_add(b, step, c);
    rocke_value_t* more = rocke_b_cmp_lt(b, x, n_steps);
    rocke_if_else_t ie = rocke_b_scf_if_else(b, more);
    rocke_b_region_enter(b, ie.then_region);
    rocke_b_s_wait_tensorcnt(b, stages - 2);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, ie.else_region);
    rocke_b_s_wait_tensorcnt(b, 0);
    rocke_b_region_leave(b);
}

rocke_value_t* rocke_fwd_lds_frag16(rocke_ir_builder_t* b,
                                    rocke_value_t* lds,
                                    rocke_value_t* base,
                                    rocke_value_t* off,
                                    int64_t const_off,
                                    rocke_value_t* c0,
                                    rocke_value_t* c8,
                                    const rocke_type_t* io_type)
{
    rocke_value_t* addr = base;
    if(off != NULL)
    {
        rocke_value_t* k = rocke_b_const_i32(b, const_off);
        rocke_value_t* t = rocke_b_add(b, off, k);
        addr = rocke_b_add(b, base, t);
    }
    rocke_value_t* i0[2] = {c0, addr};
    rocke_value_t* lo = rocke_b_smem_load_vN(b, lds, i0, 2, io_type, 8);
    rocke_value_t* a8 = rocke_b_add(b, addr, c8);
    rocke_value_t* i1[2] = {c0, a8};
    rocke_value_t* hi = rocke_b_smem_load_vN(b, lds, i1, 2, io_type, 8);
    return rocke_b_vec_concat(b, lo, hi);
}
