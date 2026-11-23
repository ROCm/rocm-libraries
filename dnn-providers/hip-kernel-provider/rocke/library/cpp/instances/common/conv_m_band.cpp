/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C port of library/kernels/conv/conv_m_band.py (emit_m_band_order). Every
 * operand is bound to a temporary in Python's left-to-right order: the SSA ids
 * follow it.
 */
#include "rocke/instance_conv_m_band.h"

rocke_value_t* rocke_conv_emit_m_band_order(rocke_ir_builder_t* b,
                                            rocke_value_t* mt,
                                            rocke_value_t* num_m_tiles,
                                            rocke_value_t* plane,
                                            int tile_m,
                                            rocke_value_t* tile_m_val,
                                            int m_band)
{
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_mb = rocke_b_const_i32(b, m_band);
    rocke_value_t* c_tm;
    rocke_value_t* R;
    if(tile_m_val == NULL)
    {
        c_tm = rocke_b_const_i32(b, tile_m);
        rocke_value_t* half = rocke_b_const_i32(b, tile_m / 2);
        rocke_value_t* sum = rocke_b_add(b, plane, half);
        R = rocke_b_smax(b, rocke_b_div(b, sum, c_tm), c1);
    }
    else
    {
        c_tm = rocke_b_smax(b, tile_m_val, c1);
        rocke_value_t* c2 = rocke_b_const_i32(b, 2);
        rocke_value_t* half = rocke_b_div(b, c_tm, c2);
        rocke_value_t* sum = rocke_b_add(b, plane, half);
        R = rocke_b_smax(b, rocke_b_div(b, sum, c_tm), c1);
    }
    rocke_value_t* rows = rocke_b_div(b, num_m_tiles, R);
    rocke_value_t* full = rocke_b_mul(b, rows, R);
    rocke_value_t* nb = rocke_b_div(b, R, c_mb); /* whole bands per row */
    rocke_value_t* w_last = rocke_b_smax(b, rocke_b_mod(b, R, c_mb), c1);
    rocke_value_t* span = rocke_b_mul(b, rows, c_mb);
    rocke_value_t* in_whole = rocke_b_mul(b, nb, span);
    /* whole bands: L = (band * rows + row) * mb + i */
    rocke_value_t* band = rocke_b_div(b, mt, rocke_b_smax(b, span, c1));
    rocke_value_t* r = rocke_b_mod(b, mt, rocke_b_smax(b, span, c1));
    rocke_value_t* rq = rocke_b_mul(b, rocke_b_div(b, r, c_mb), R);
    rocke_value_t* bm = rocke_b_mul(b, band, c_mb);
    rocke_value_t* rm = rocke_b_mod(b, r, c_mb);
    rocke_value_t* t_whole = rocke_b_add(b, rq, rocke_b_add(b, bm, rm));
    /* the narrow last band: L - in_whole = row * w_last + i */
    rocke_value_t* l2 = rocke_b_sub(b, mt, in_whole);
    rocke_value_t* lq = rocke_b_mul(b, rocke_b_div(b, l2, w_last), R);
    rocke_value_t* nm = rocke_b_mul(b, nb, c_mb);
    rocke_value_t* lm = rocke_b_mod(b, l2, w_last);
    rocke_value_t* t_last = rocke_b_add(b, lq, rocke_b_add(b, nm, lm));
    rocke_value_t* t = rocke_b_select(b, rocke_b_cmp_lt(b, mt, in_whole), t_whole, t_last);
    rocke_value_t* ok_full = rocke_b_cmp_lt(b, mt, full);
    rocke_value_t* ok_rows = rocke_b_cmp_lt(b, c0, rows);
    rocke_value_t* ok = rocke_b_land(b, ok_full, ok_rows);
    return rocke_b_select(b, ok, t, mt);
}
