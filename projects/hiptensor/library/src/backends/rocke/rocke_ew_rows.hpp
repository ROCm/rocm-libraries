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
// kernel-pack kernels.

#ifndef HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP
#define HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP

#include <array>
#include <cstring>

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
    } // namespace rockebackend
} // namespace hiptensor

#endif // HIPTENSOR_BACKENDS_ROCKE_ROCKE_EW_ROWS_HPP
