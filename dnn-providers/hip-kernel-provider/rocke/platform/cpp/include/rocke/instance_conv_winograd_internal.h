/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_winograd_internal.h -- pieces the C++ ports of the
 * Winograd forward kernels share (library/kernels/common/conv_winograd_fwd.py):
 *
 *   * the exact Toom-Cook matrices (A^T, G, B^T) of the F(m, 3) point sets of
 *     conv_winograd_fwd._POINTS, as rationals;
 *   * the small IR helpers of conv_winograd_fwd (_load_raw, _raw_to_f32,
 *     _pack_vec_f32, _store_packed, _store_vec_f32, _oob_sel, _lincomb).
 *
 * C++ only (not part of the C ABI).
 */
#ifndef ROCKE_INSTANCE_CONV_WINOGRAD_INTERNAL_H
#define ROCKE_INSTANCE_CONV_WINOGRAD_INTERNAL_H

#include <stdint.h>

#include <utility>
#include <vector>

#include "rocke/ir.h"

namespace rocke_wino
{

/* conv_winograd_fwd._OOB: a byte offset out of range of every buffer. */
constexpr int64_t kOob = 0x7FFF0000;
/* conv_winograd_fwd._CPOL_NT: th:TH_STORE_NT. */
constexpr int kCpolNt = 1;

struct Rational
{
    int64_t num;
    int64_t den;
    /* float(Fraction): both parts are exact doubles, so this is the
     * correctly rounded quotient, as in Python. */
    double value() const
    {
        return (double)num / (double)den;
    }
    bool is_zero() const
    {
        return num == 0;
    }
};

constexpr int kMaxTc = 144;

/* Row-major: at is m x n, g is n x r, bt is n x n (n = m + r - 1). */
struct ToomCook
{
    int m;
    int r;
    int n;
    Rational at[kMaxTc];
    Rational g[kMaxTc];
    Rational bt[kMaxTc];
    const Rational& AT(int i, int j) const
    {
        return at[i * n + j];
    }
    const Rational& G(int i, int j) const
    {
        return g[i * r + j];
    }
    const Rational& BT(int i, int j) const
    {
        return bt[i * n + j];
    }
};

/* toom_cook(m) of conv_winograd_fwd (r = 3); NULL when there is none. */
const ToomCook* toom_cook(int m, int r);

/* _pow2_vec: the largest power of two <= cap that divides n. */
int pow2_vec(int n, int cap);
/* _pow2_ceil. */
int pow2_ceil(int n);
/* _ir_dtype: "fp16" -> f16, anything else -> bf16. */
const rocke_type_t* ir_dtype(const char* dtype);

/* _load_raw: buffer load of n 16-bit elements, kept packed (OOB -> 0). */
rocke_value_t* load_raw(
    rocke_ir_builder_t* b, rocke_value_t* rsrc, rocke_value_t* off, const char* dtype, int n);
/* _raw_to_f32. */
std::vector<rocke_value_t*> raw_to_f32(rocke_ir_builder_t* b, rocke_value_t* raw, int n);
/* _pack_vec_f32: a scalar for one value, else a vector. */
rocke_value_t*
    pack_vec_f32(rocke_ir_builder_t* b, const std::vector<rocke_value_t*>& vals, const char* dtype);
/* _store_packed (cpol: the cache-policy immediate). */
void store_packed(rocke_ir_builder_t* b,
                  rocke_value_t* rsrc,
                  rocke_value_t* off,
                  rocke_value_t* packed,
                  const char* dtype,
                  int cpol);
/* _store_vec_f32 (cpol 0). */
void store_vec_f32(rocke_ir_builder_t* b,
                   rocke_value_t* rsrc,
                   rocke_value_t* off,
                   const std::vector<rocke_value_t*>& vals,
                   const char* dtype);
/* _oob_sel: select(ok, off, _OOB). */
rocke_value_t* oob_sel(rocke_ir_builder_t* b, rocke_value_t* ok, rocke_value_t* off);
/* conv_winograd_fwd._lincomb: sum(c * v) over the non-zero integer c, fp32. */
rocke_value_t* lincomb(rocke_ir_builder_t* b,
                       const std::vector<std::pair<int64_t, rocke_value_t*>>& terms);

} // namespace rocke_wino

#endif /* ROCKE_INSTANCE_CONV_WINOGRAD_INTERNAL_H */
