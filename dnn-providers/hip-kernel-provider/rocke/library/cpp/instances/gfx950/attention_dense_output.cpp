// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "attention_dense_internal.hpp"

#include "rocke/instance_lse_store_internal.h"

namespace rocke_dense_gfx950
{
// Mirrors _dense_lse_output: one half-wave owns LSE; the two halves both store O.
void dense_lse_output(Builder* b,
                      V dst,
                      V index,
                      V m,
                      V l,
                      V row_exists,
                      V row_has_key,
                      V lane_half,
                      V sinks,
                      V head,
                      Type dtype)
{
    V owner = DENSE_OP(land, row_exists, DENSE_OP(cmp_eq, lane_half, DENSE_OP(const_i32, 0)));
    rocke_b_region_enter(b, rocke_b_scf_if(b, owner).then_region);
    rocke_b_region_enter(b, rocke_b_scf_if(b, row_has_key).then_region);
    rocke_lse_store(b, dst, index, m, l, row_has_key);
    rocke_b_region_leave(b);
    rocke_b_region_enter(b, rocke_b_scf_if(b, rocke_b_lnot(b, row_has_key)).then_region);
    if(sinks != nullptr)
    {
        V sink = DENSE_OP(cast_to_f32, DENSE_OP(global_load, sinks, head, dtype, 2));
        V sink_m = DENSE_OP(fmul, sink, DENSE_OP(const_f32, log2e));
        V mass = DENSE_OP(cmp_eq, DENSE_OP(const_i32, 1), DENSE_OP(const_i32, 1));
        rocke_lse_store(b, dst, index, sink_m, rocke_b_const_f32(b, 1.), mass);
    }
    else
    {
        V zero = rocke_b_const_f32(b, 0.);
        V one = rocke_b_const_f32(b, 1.);
        rocke_lse_store(b, dst, index, zero, one, row_has_key);
    }
    rocke_b_region_leave(b);
    rocke_b_region_leave(b);
}

void emit_o_store(Builder* b,
                  V o,
                  V q_row_byte,
                  V d_half,
                  std::span<V const> o_acc,
                  V rcp_l,
                  Type dtype,
                  int width,
                  V row_has_key)
{
    for(size_t dt = 0; dt < o_acc.size(); ++dt)
        for(int g = 0; g < 4; ++g)
            for(int c = 0; c < 4; c += width)
            {
                V d0 = DENSE_OP(add, DENSE_OP(const_i32, dt * 32 + g * 8 + c), d_half);
                V addr = rocke_b_add(b, q_row_byte, d0);
                V vals[4]; // Supports admits store widths 1, 2 and 4.
                for(int kk = 0; kk < width; ++kk)
                    vals[kk] = DENSE_OP(
                        cast_f32_to,
                        DENSE_OP(fmul, DENSE_OP(vec_extract, o_acc[dt], g * 4 + c + kk), rcp_l),
                        dtype);
                V packed = rocke_b_vec_pack(b, vals, width, dtype);
                if(row_has_key != nullptr)
                    packed = rocke_zero_keyless_row(b, packed, row_has_key, dtype, width);
                rocke_b_global_store_vN(b, o, addr, packed, width, 2 * width);
            }
}

// Device half of common/attention_dense_kv_split.py. The host contract already
// admits the split and provides the ordered workspace names and element bound.
std::array<V, 3> segment_params(Builder* b)
{
    std::array<V, 3> params;
    for(size_t i = 0; i < params.size(); ++i)
        params[i]
            = pointer_param(b, ROCKE_DENSE_SEGMENT_PARAMS[i], rocke_f32(), true, i == 0 ? 16 : 4);
    return params;
}

std::pair<V, V> segment_tile_range(Builder* b, V seg, V lo, V hi, int segments)
{
    V chunk = DENSE_OP(div,
                       DENSE_OP(add, DENSE_OP(sub, hi, lo), DENSE_OP(const_i32, segments - 1)),
                       DENSE_OP(const_i32, segments));
    V start = DENSE_OP(add, lo, DENSE_OP(mul, seg, chunk));
    start = DENSE_OP(select, DENSE_OP(cmp_lt, start, hi), start, hi);
    V end = rocke_b_add(b, start, chunk);
    end = DENSE_OP(select, DENSE_OP(cmp_lt, end, hi), end, hi);
    return {start, end};
}

void emit_segment_store(Builder* b,
                        const std::array<V, 3>& params,
                        V token,
                        V head,
                        V num_heads,
                        V seg,
                        int segments,
                        int head_size,
                        std::span<V const> o_acc,
                        V m,
                        V denom,
                        V lane_h)
{
    V ml_idx = DENSE_OP(add,
                        DENSE_OP(mul,
                                 DENSE_OP(add, DENSE_OP(mul, token, num_heads), head),
                                 DENSE_OP(const_i32, segments)),
                        seg);
    V row_base = DENSE_OP(mul, ml_idx, DENSE_OP(const_i32, head_size));
    V d_half = DENSE_OP(mul, lane_h, DENSE_OP(const_i32, 4));
    for(size_t dt = 0; dt < o_acc.size(); ++dt)
        for(int g = 0; g < 4; ++g)
        {
            V d0 = DENSE_OP(add, DENSE_OP(const_i32, dt * 32 + g * 8), d_half);
            V vals[4];
            for(int kk = 0; kk < 4; ++kk)
                vals[kk] = rocke_b_vec_extract(b, o_acc[dt], g * 4 + kk);
            V idx = rocke_b_add(b, row_base, d0);
            V packed = rocke_b_vec_pack(b, vals, 4, rocke_f32());
            rocke_b_global_store_vN(b, params[0], idx, packed, 4, 16);
        }
    V owner = DENSE_OP(cmp_eq, lane_h, DENSE_OP(const_i32, 0));
    rocke_b_region_enter(b, rocke_b_scf_if(b, owner).then_region);
    rocke_b_global_store(b, params[1], ml_idx, m, 4);
    rocke_b_global_store(b, params[2], ml_idx, denom, 4);
    rocke_b_region_leave(b);
}
} // namespace rocke_dense_gfx950
