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
// of the contractions' operand packing and epilogues, and the rows of mixed-type contractions
// (ewRows1Mixed / ewRows2Mixed), each also with the math operators and with 64-bit offsets. The
// kernel pack builder (pack/rocke_pack_builder.cpp) builds every row; the JIT (rocke_jit.cpp)
// specializes the launches of every row. jitClassAdmitted() is the JIT's class rule; the pack
// builder lists the classes it admits in its manifest (--list), which rocKE's manifest pins to
// the classes of rocKE's jit_ew.jit_classes() of every pack row.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

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

        // The rows of mixed-type contractions (rocKE's make_rocke_pack.py EW_ROWS_1_MIXED /
        // EW_ROWS_2_MIXED / EW_ROWS_C2_MIXED, in its order): packs of any real type to f16 /
        // bf16, epilogues of an f32 / f64 result with any C and D type (complex: f32 / f64
        // components).
        inline std::vector<EwRow2> ewRows1Mixed()
        {
            std::vector<EwRow2> out;
            for(char const* d : {"f16", "bf16"})
            {
                for(char const* a : {"f16", "bf16", "f32", "f64"})
                {
                    if(!hasEwRow(kEwRows1, EwRow2{a, d}))
                    {
                        out.push_back({a, d});
                    }
                }
            }
            return out;
        }

        inline std::vector<EwRow3> ewRows2Mixed(bool cplx)
        {
            std::vector<EwRow3>      out;
            std::vector<char const*> types
                = cplx ? std::vector<char const*>{"f32", "f64"}
                       : std::vector<char const*>{"f16", "bf16", "f32", "f64"};
            for(char const* t : {"f32", "f64"})
            {
                for(char const* c : types)
                {
                    for(char const* d : types)
                    {
                        if(!(cplx ? hasEwRow(kEwRowsC2, EwRow3{t, c, d})
                                  : hasEwRow(kEwRows2, EwRow3{t, c, d})))
                        {
                            out.push_back({t, c, d});
                        }
                    }
                }
            }
            return out;
        }

        // The rows the JIT specializes: every row of the pack's generic elementwise kernel (the
        // rows above and the mixed-type rows, 32- and 64-bit offsets, the real ones also with the
        // math operators) with the default block size.
        inline bool jitRow(rocke_htaot_ew_spec_t const& s)
        {
            auto const name = [](char const* dt) { return dt != nullptr ? dt : ""; };
            if(s.block != rocke_htaot_ew_spec_default().block || (s.math && s.cplx))
            {
                return false;
            }
            if(s.cplx)
            {
                EwRow3 const r{name(s.a), name(s.c), name(s.d)};
                return s.arity == 1
                           ? hasEwRow(kEwRowsC1, EwRow2{name(s.a), name(s.d)})
                           : s.arity == 2
                                 && (hasEwRow(kEwRowsC2, r) || hasEwRow(ewRows2Mixed(true), r));
            }
            switch(s.arity)
            {
            case 1:
            {
                EwRow2 const r{name(s.a), name(s.d)};
                return hasEwRow(kEwRows1, r) || hasEwRow(ewRows1Mixed(), r);
            }
            case 2:
            {
                EwRow3 const r{name(s.a), name(s.c), name(s.d)};
                return hasEwRow(kEwRows2, r) || hasEwRow(ewRows2Mixed(false), r);
            }
            case 3:
                return hasEwRow(kEwRows3, EwRow4{name(s.a), name(s.b), name(s.c), name(s.d)});
            default:
                return false;
            }
        }

        // The JIT's kernel classes (rocKE's jit_ew.jit_classes): a class of a row of jitRow()
        // that rocKE's builder accepts; vec 1 (no vector access), or the output's vector width
        // (16-byte accesses) with every combination of vector / gathered inputs for the real
        // 32-bit-offset rows and every input a vector for the 64-bit-offset and the math rows.
        // Every launch of the generic kernel has a class: classify() takes vec 1 when the
        // layout has no admitted vector class.
        inline bool jitClassAdmitted(rocke_htaot_ew_jit_spec_t const& s)
        {
            char why[160] = {0};
            if(!jitRow(s.base) || !rocke_htaot_ew_jit_check(&s, why, sizeof(why)))
            {
                return false;
            }
            if(s.vec == 1)
            {
                return true;
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
