/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// The data type rows of the rocKE backend's generic elementwise kernel (rocKE's
// library/kernels/common/hiptensor_aot/ew.py): the rows of the API's elementwise operations and
// of the contractions' operand packing and epilogues. The kernel pack builder
// (pack/rocke_pack_builder.cpp) adds the rows of mixed-type contractions to them; the JIT
// (rocke_jit.cpp) specializes the launches of these rows only, the mixed-type rows keep their
// kernel-pack kernels. jitClassAdmitted() is the JIT's class rule; the pack builder lists the
// classes it admits in its manifest (--list), which rocKE's manifest pins to the classes of
// rocKE's jit_ew.jit_classes().

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP

#include <algorithm>
#include <array>
#include <cstring>

#include "rocke/instance_hiptensor_aot.h"

namespace hiptensor
{
    namespace rockebackend
    {
        using EwRow2 = std::array<char const*, 2>;
        using EwRow3 = std::array<char const*, 3>;
        using EwRow4 = std::array<char const*, 4>;

        // permute / pack: (A, D)
        inline constexpr std::array<EwRow2, 11> kEwRows1 = {{{"f16", "f16"},
                                                             {"f16", "f32"},
                                                             {"bf16", "bf16"},
                                                             {"bf16", "f32"},
                                                             {"f32", "f32"},
                                                             {"f32", "f16"},
                                                             {"f32", "bf16"},
                                                             {"f64", "f64"},
                                                             // operands of the f64 route
                                                             {"f16", "f64"},
                                                             {"bf16", "f64"},
                                                             {"f32", "f64"}}};
        // binary: (A, C, D); epilogues read an f32 / f64 A
        inline constexpr std::array<EwRow3, 11> kEwRows2 = {{{"f16", "f16", "f16"},
                                                             {"f16", "f16", "f32"},
                                                             {"bf16", "bf16", "bf16"},
                                                             {"bf16", "bf16", "f32"},
                                                             {"f32", "f32", "f32"},
                                                             {"f32", "f16", "f16"},
                                                             {"f32", "bf16", "bf16"},
                                                             {"f64", "f64", "f64"},
                                                             // epilogues of the f64 route
                                                             {"f64", "f16", "f16"},
                                                             {"f64", "bf16", "bf16"},
                                                             {"f64", "f32", "f32"}}};
        // trinary: (A, B, C, D)
        inline constexpr std::array<EwRow4, 6> kEwRows3 = {{{"f16", "f16", "f16", "f16"},
                                                            {"f16", "f16", "f16", "f32"},
                                                            {"bf16", "bf16", "bf16", "bf16"},
                                                            {"bf16", "bf16", "bf16", "f32"},
                                                            {"f32", "f32", "f32", "f32"},
                                                            {"f64", "f64", "f64", "f64"}}};
        // complex (component types): pack (A, D) and epilogue (A, C, D)
        inline constexpr std::array<EwRow2, 3> kEwRowsC1
            = {{{"f32", "f32"}, {"f64", "f64"}, {"f32", "f64"}}};
        inline constexpr std::array<EwRow3, 3> kEwRowsC2
            = {{{"f32", "f32", "f32"}, {"f64", "f64", "f64"}, {"f64", "f32", "f32"}}};

        // Whether `rows` holds `row` (types compared by value).
        template <typename Rows, typename Row>
        bool hasEwRow(Rows const& rows, Row const& row)
        {
            for(auto const& r : rows)
            {
                bool same = true;
                for(size_t i = 0; i < row.size() && same; ++i)
                {
                    same = std::strcmp(r[i], row[i]) == 0;
                }
                if(same)
                {
                    return true;
                }
            }
            return false;
        }

        // The rows the JIT specializes: the rows above with the default block size, the math
        // rows with 32-bit offsets only. The rows of mixed-type contractions and the
        // 64-bit-offset math rows keep their pack kernels.
        inline bool jitRow(rocke_htaot_ew_spec_t const& s)
        {
            auto const name = [](char const* dt) { return dt != nullptr ? dt : ""; };
            if((s.math && s.idx64) || s.block != rocke_htaot_ew_spec_default().block)
            {
                return false;
            }
            if(s.cplx)
            {
                return s.arity == 1
                           ? hasEwRow(kEwRowsC1, EwRow2{name(s.a), name(s.d)})
                           : s.arity == 2
                                 && hasEwRow(kEwRowsC2, EwRow3{name(s.a), name(s.c), name(s.d)});
            }
            switch(s.arity)
            {
            case 1:
                return hasEwRow(kEwRows1, EwRow2{name(s.a), name(s.d)});
            case 2:
                return hasEwRow(kEwRows2, EwRow3{name(s.a), name(s.c), name(s.d)});
            case 3:
                return hasEwRow(kEwRows3, EwRow4{name(s.a), name(s.b), name(s.c), name(s.d)});
            default:
                return false;
            }
        }

        // The JIT's kernel classes: a class of a row of jitRow() that rocKE's builder accepts;
        // vec 1 (no vector access) except for the math rows, or the output's vector width
        // (16-byte accesses), every input a vector for the 64-bit-offset and the math rows.
        inline bool jitClassAdmitted(rocke_htaot_ew_jit_spec_t const& s)
        {
            char why[160] = {0};
            if(!jitRow(s.base) || !rocke_htaot_ew_jit_check(&s, why, sizeof(why)))
            {
                return false;
            }
            if(s.vec == 1)
            {
                return !s.base.math;
            }
            if(s.vec != rocke_htaot_ew_jit_vec(s.base.d))
            {
                return false;
            }
            return !(s.base.idx64 || s.base.math)
                   || std::all_of(s.unit, s.unit + s.n_unit, [](bool u) { return u; });
        }
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP
