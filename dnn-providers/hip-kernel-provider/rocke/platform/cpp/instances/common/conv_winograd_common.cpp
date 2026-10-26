// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_winograd_common.cpp -- the Toom-Cook tables and IR helpers the C++
 * ports of the Winograd forward kernels share (see
 * rocke/instance_conv_winograd_internal.h).
 *
 * Every helper emits in Python evaluation order, so the SSA sequence matches
 * the Python helpers it ports.
 */
#include "rocke/instance_conv_winograd_internal.h"

#include <string.h>

namespace rocke_wino
{
namespace
{

/* toom_cook(m) of every conv_winograd_fwd._POINTS entry (r = 3), exact. The
 * byte-identity parity pairs of the Winograd families pin these values
 * against the Python construction. */
const ToomCook kTables[] = {
    /* F(2,3) */
    {2,
     3,
     4,
     {{1, 1}, {1, 1}, {1, 1}, {0, 1}, {0, 1}, {1, 1}, {-1, 1}, {1, 1}},
     {{-1, 1},
      {0, 1},
      {0, 1},
      {1, 2},
      {1, 2},
      {1, 2},
      {1, 2},
      {-1, 2},
      {1, 2},
      {0, 1},
      {0, 1},
      {1, 1}},
     {{-1, 1},
      {0, 1},
      {1, 1},
      {0, 1},
      {0, 1},
      {1, 1},
      {1, 1},
      {0, 1},
      {0, 1},
      {-1, 1},
      {1, 1},
      {0, 1},
      {0, 1},
      {-1, 1},
      {0, 1},
      {1, 1}}},
    /* F(4,3) */
    {4,
     3,
     6,
     {{1, 1},  {1, 1}, {1, 1},  {1, 1}, {1, 1},  {0, 1}, {0, 1},  {1, 1},
      {-1, 1}, {2, 1}, {-2, 1}, {0, 1}, {0, 1},  {1, 1}, {1, 1},  {4, 1},
      {4, 1},  {0, 1}, {0, 1},  {1, 1}, {-1, 1}, {8, 1}, {-8, 1}, {1, 1}},
     {{1, 4},
      {0, 1},
      {0, 1},
      {-1, 6},
      {-1, 6},
      {-1, 6},
      {-1, 6},
      {1, 6},
      {-1, 6},
      {1, 24},
      {1, 12},
      {1, 6},
      {1, 24},
      {-1, 12},
      {1, 6},
      {0, 1},
      {0, 1},
      {1, 1}},
     {{4, 1},  {0, 1},  {-5, 1}, {0, 1}, {1, 1}, {0, 1},  {0, 1},  {-4, 1}, {-4, 1},
      {1, 1},  {1, 1},  {0, 1},  {0, 1}, {4, 1}, {-4, 1}, {-1, 1}, {1, 1},  {0, 1},
      {0, 1},  {-2, 1}, {-1, 1}, {2, 1}, {1, 1}, {0, 1},  {0, 1},  {2, 1},  {-1, 1},
      {-2, 1}, {1, 1},  {0, 1},  {0, 1}, {4, 1}, {0, 1},  {-5, 1}, {0, 1},  {1, 1}}},
    /* F(6,3) */
    {6,
     3,
     8,
     {{1, 1},  {1, 1}, {1, 1},  {1, 1},  {1, 1},   {1, 1},  {1, 1},   {0, 1},  {0, 1},  {1, 1},
      {-1, 1}, {2, 1}, {-2, 1}, {1, 2},  {-1, 2},  {0, 1},  {0, 1},   {1, 1},  {1, 1},  {4, 1},
      {4, 1},  {1, 4}, {1, 4},  {0, 1},  {0, 1},   {1, 1},  {-1, 1},  {8, 1},  {-8, 1}, {1, 8},
      {-1, 8}, {0, 1}, {0, 1},  {1, 1},  {1, 1},   {16, 1}, {16, 1},  {1, 16}, {1, 16}, {0, 1},
      {0, 1},  {1, 1}, {-1, 1}, {32, 1}, {-32, 1}, {1, 32}, {-1, 32}, {1, 1}},
     {{-1, 1},  {0, 1},  {0, 1},   {-2, 9},   {-2, 9}, {-2, 9},  {-2, 9}, {2, 9},
      {-2, 9},  {1, 90}, {1, 45},  {2, 45},   {1, 90}, {-1, 45}, {2, 45}, {32, 45},
      {16, 45}, {8, 45}, {32, 45}, {-16, 45}, {8, 45}, {0, 1},   {0, 1},  {1, 1}},
     {{-1, 1},  {0, 1},   {21, 4},  {0, 1},  {-21, 4}, {0, 1}, {1, 1},  {0, 1},  {0, 1},  {1, 1},
      {1, 1},   {-17, 4}, {-17, 4}, {1, 1},  {1, 1},   {0, 1}, {0, 1},  {-1, 1}, {1, 1},  {17, 4},
      {-17, 4}, {-1, 1},  {1, 1},   {0, 1},  {0, 1},   {1, 2}, {1, 4},  {-5, 2}, {-5, 4}, {2, 1},
      {1, 1},   {0, 1},   {0, 1},   {-1, 2}, {1, 4},   {5, 2}, {-5, 4}, {-2, 1}, {1, 1},  {0, 1},
      {0, 1},   {2, 1},   {4, 1},   {-5, 2}, {-5, 1},  {1, 2}, {1, 1},  {0, 1},  {0, 1},  {-2, 1},
      {4, 1},   {5, 2},   {-5, 1},  {-1, 2}, {1, 1},   {0, 1}, {0, 1},  {-1, 1}, {0, 1},  {21, 4},
      {0, 1},   {-21, 4}, {0, 1},   {1, 1}}},

};

} // namespace

const ToomCook* toom_cook(int m, int r)
{
    for(const ToomCook& t : kTables)
    {
        if(t.m == m && t.r == r)
            return &t;
    }
    return nullptr;
}

int pow2_vec(int n, int cap)
{
    int v = cap;
    while(v > 1 && n % v)
        v /= 2;
    return v;
}

int pow2_ceil(int n)
{
    int p = 1;
    while(p < n)
        p *= 2;
    return p;
}

const rocke_type_t* ir_dtype(const char* dtype)
{
    return (dtype && strcmp(dtype, "fp16") == 0) ? rocke_f16() : rocke_bf16();
}

rocke_value_t* load_raw(
    rocke_ir_builder_t* b, rocke_value_t* rsrc, rocke_value_t* off, const char* dtype, int n)
{
    const rocke_type_t* ty = ir_dtype(dtype);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    if(n == 1)
        return rocke_b_buffer_load(b, rsrc, off, zero, ty);
    return rocke_b_buffer_load_vN(b, rsrc, off, zero, ty, n);
}

std::vector<rocke_value_t*> raw_to_f32(rocke_ir_builder_t* b, rocke_value_t* raw, int n)
{
    std::vector<rocke_value_t*> out;
    if(n == 1)
    {
        out.push_back(rocke_b_cast_to_f32(b, raw));
        return out;
    }
    for(int i = 0; i < n; ++i)
    {
        rocke_value_t* e = rocke_b_vec_extract(b, raw, i);
        out.push_back(rocke_b_cast_to_f32(b, e));
    }
    return out;
}

rocke_value_t*
    pack_vec_f32(rocke_ir_builder_t* b, const std::vector<rocke_value_t*>& vals, const char* dtype)
{
    const rocke_type_t* ty = ir_dtype(dtype);
    std::vector<rocke_value_t*> casts;
    for(rocke_value_t* v : vals)
        casts.push_back(rocke_b_cast_f32_to(b, v, ty));
    if(casts.size() == 1)
        return casts[0];
    return rocke_b_vec_pack(b, casts.data(), (int)casts.size(), ty);
}

void store_packed(rocke_ir_builder_t* b,
                  rocke_value_t* rsrc,
                  rocke_value_t* off,
                  rocke_value_t* packed,
                  const char* dtype,
                  int cpol)
{
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    bool fp16 = dtype && strcmp(dtype, "fp16") == 0;
    if(packed == nullptr || packed->type == nullptr)
        return;
    if(packed->type->kind != ROCKE_TYPE_VECTOR)
    {
        if(fp16)
            rocke_b_buffer_store_f16_ex(b, rsrc, off, zero, packed, cpol);
        else
            rocke_b_buffer_store_bf16_ex(b, rsrc, off, zero, packed, cpol);
        return;
    }
    int n = packed->type->count;
    if(fp16)
        rocke_b_buffer_store_vN_f16_ex(b, rsrc, off, zero, packed, n / 2, cpol);
    else
        rocke_b_buffer_store_vN_bf16_ex(b, rsrc, off, zero, packed, n / 2, cpol);
}

void store_vec_f32(rocke_ir_builder_t* b,
                   rocke_value_t* rsrc,
                   rocke_value_t* off,
                   const std::vector<rocke_value_t*>& vals,
                   const char* dtype)
{
    rocke_value_t* packed = pack_vec_f32(b, vals, dtype);
    store_packed(b, rsrc, off, packed, dtype, 0);
}

rocke_value_t* oob_sel(rocke_ir_builder_t* b, rocke_value_t* ok, rocke_value_t* off)
{
    rocke_value_t* oob = rocke_b_const_i32(b, kOob);
    return rocke_b_select(b, ok, off, oob);
}

rocke_value_t* lincomb(rocke_ir_builder_t* b,
                       const std::vector<std::pair<int64_t, rocke_value_t*>>& terms)
{
    rocke_value_t* acc = nullptr;
    for(const auto& t : terms)
    {
        int64_t c = t.first;
        rocke_value_t* v = t.second;
        if(c == 0)
            continue;
        if(acc == nullptr)
        {
            if(c == 1)
                acc = v;
            else if(c == -1)
                acc = rocke_b_fneg(b, v);
            else
            {
                rocke_value_t* k = rocke_b_const_f32(b, (double)c);
                acc = rocke_b_fmul(b, k, v);
            }
        }
        else if(c == 1)
            acc = rocke_b_fadd(b, acc, v);
        else if(c == -1)
            acc = rocke_b_fsub(b, acc, v);
        else
        {
            rocke_value_t* k = rocke_b_const_f32(b, (double)c);
            acc = rocke_b_fma(b, k, v, acc);
        }
    }
    return acc != nullptr ? acc : rocke_b_const_f32(b, 0.0);
}

} // namespace rocke_wino
