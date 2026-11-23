/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

// rocsparse_extract: the lower or upper part of a CSR or CSC matrix. The extraction is
// checked against a host reference for every value type, for the index types, the CSR
// and CSC formats, and both fill modes and diagonal types. The matrices that the routine
// cannot handle are rejected; the status is not documented for them.

#include "unit_test_device_conv_common.hpp"

#include <limits>

using namespace convut;

namespace
{
    using ConvExtract = ConvTest;

    // A 3x3 matrix with entries on and off the diagonal, so that the lower and the upper
    // part are different. The values are the same bytes for every datatype.
    const std::vector<int32_t> s_ptr{0, 2, 3, 5};
    const std::vector<int32_t> s_ind{0, 1, 1, 0, 2};

    TEST_F(ConvExtract, ExtractDatatypesFillAndDiag)
    {
        for(auto dt : {rocsparse_datatype_f16_r,
                       rocsparse_datatype_bf16_r,
                       f32,
                       rocsparse_datatype_f64_r,
                       rocsparse_datatype_f32_c,
                       rocsparse_datatype_f64_c,
                       rocsparse_datatype_i8_r,
                       rocsparse_datatype_u8_r,
                       rocsparse_datatype_i32_r,
                       rocsparse_datatype_u32_r})
        {
            run_extract(handle,
                        false,
                        3,
                        3,
                        0,
                        dt,
                        s_ptr,
                        s_ind,
                        rocsparse_fill_mode_lower,
                        rocsparse_diag_type_non_unit);
        }
        for(auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            for(auto diag : {rocsparse_diag_type_non_unit, rocsparse_diag_type_unit})
                run_extract(handle, false, 3, 3, 0, f32, s_ptr, s_ind, fill, diag);
    }

    // Index types and formats: 64-bit row pointers with 32-bit columns, 64-bit indices with
    // the index base one, the column direction (CSC) with 32-bit column pointers and 64-bit
    // row indices, and an empty matrix.
    TEST_F(ConvExtract, ExtractIndexTypesAndFormats)
    {
        const std::vector<int64_t> ptr64{0, 2, 3, 5}, ptr64_one{1, 3, 4, 6},
            ind64_one{1, 2, 2, 1, 3};
        const std::vector<int32_t> col32(s_ind);
        for(auto diag : {rocsparse_diag_type_unit, rocsparse_diag_type_non_unit})
            run_extract(handle, false, 3, 3, 0, f32, ptr64, col32, rocsparse_fill_mode_upper, diag);
        run_extract(handle,
                    false,
                    3,
                    3,
                    1,
                    rocsparse_datatype_f64_r,
                    ptr64_one,
                    ind64_one,
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_unit);
        run_extract(handle,
                    false,
                    3,
                    3,
                    1,
                    rocsparse_datatype_f64_r,
                    ptr64_one,
                    ind64_one,
                    rocsparse_fill_mode_upper,
                    rocsparse_diag_type_non_unit);
        run_extract(handle,
                    true,
                    3,
                    3,
                    0,
                    rocsparse_datatype_f64_c,
                    std::vector<int32_t>{0, 2, 3, 5},
                    std::vector<int64_t>{0, 2, 1, 0, 2},
                    rocsparse_fill_mode_upper,
                    rocsparse_diag_type_non_unit);
        run_extract(handle,
                    true,
                    3,
                    3,
                    0,
                    f32,
                    std::vector<int32_t>{0, 2, 3, 5},
                    std::vector<int32_t>{0, 2, 1, 0, 2},
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_non_unit);
        run_extract(handle,
                    false,
                    0,
                    4,
                    0,
                    f32,
                    std::vector<int32_t>{0},
                    std::vector<int32_t>{},
                    rocsparse_fill_mode_lower,
                    rocsparse_diag_type_non_unit);
    }

    TEST_F(ConvExtract, ExtractRejected)
    {
        DevBuf<int32_t> idx(std::vector<int32_t>{0, 1, 2, 3}), off(std::vector<int32_t>{0, 1});
        DevBuf<float>   val(std::vector<float>(4, 1.f));

        // Creates the descriptor of the matrix `m`, and a copy of it.
        const auto rejected_at_creation = [&](const auto& make) {
            Spmat source, target;
            ASSERT_EQ(make(source), st_ok);
            ASSERT_EQ(make(target), st_ok);
            ExtractDescr descr;
            EXPECT_REJECTED(create_extract(descr, source, target));
        };
        // Creation works, the operations fail.
        const auto rejected_operations = [&](rocsparse_spmat_descr source,
                                             rocsparse_spmat_descr target) {
            ExtractDescr descr;
            ASSERT_EQ(create_extract(descr, source, target), st_ok);
            size_t size = 0;
            // The query may already fail; the analysis is rejected in any case.
            (void)rocsparse_extract_buffer_size(
                handle, descr, source, target, rocsparse_extract_stage_analysis, &size);
            DevBuf<char> buffer(size);
            EXPECT_REJECTED(rocsparse_extract(
                handle, descr, source, target, rocsparse_extract_stage_analysis, size, buffer));
            EXPECT_REJECTED(rocsparse_extract(
                handle, descr, source, target, rocsparse_extract_stage_compute, 0, buffer));
        };

        // The formats other than CSR and CSC are not supported.
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_ell_descr(&s, 2, 2, idx, val, 1, i32, base0, f32);
        });
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_coo_aos_descr(&s, 2, 2, 2, idx, val, i32, base0, f32);
        });
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_coo_descr(&s, 2, 2, 2, idx, idx, val, i32, base0, f32);
        });
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_bsr_descr(
                &s, 2, 2, 2, dir_r, 1, idx, idx, val, i32, i32, base0, f32);
        });
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_bell_descr(&s, 2, 2, dir_r, 1, 1, idx, val, i32, base0, f32);
        });
        rejected_at_creation([&](Spmat& s) {
            return rocsparse_create_sell_descr(
                &s, 2, 2, 1, 1, 1, off, idx, val, i32, i32, base0, f32);
        });

        // The source and the target must have the same format and the same storage mode.
        DevBuf<int32_t> row(std::vector<int32_t>{0, 1, 2}), col(std::vector<int32_t>{0, 1}),
            t_row(3);
        {
            Spmat csr, csc;
            ASSERT_EQ(create_csx(csr, false, 2, 2, 2, row, col, val, i32, i32, 0, f32), st_ok);
            ASSERT_EQ(create_csx(csc, true, 2, 2, 2, row, col, val, i32, i32, 0, f32), st_ok);
            ExtractDescr descr;
            EXPECT_REJECTED(create_extract(descr, csr, csc));
        }
        {
            Spmat source, target;
            ASSERT_EQ(create_csx(source, false, 2, 2, 2, row, col, val, i32, i32, 0, f32), st_ok);
            ASSERT_EQ(create_csx(target, false, 2, 2, 0, t_row, nullptr, nullptr, i32, i32, 0, f32),
                      st_ok);
            const rocsparse_storage_mode unsorted = rocsparse_storage_mode_unsorted;
            ASSERT_EQ(rocsparse_spmat_set_attribute(
                          target, rocsparse_spmat_storage_mode, &unsorted, sizeof(unsorted)),
                      st_ok);
            ExtractDescr descr;
            EXPECT_REJECTED(create_extract(descr, source, target));
        }

        // The deprecated 16-bit index type, for the row pointer and for the column indices.
        const auto u16 = static_cast<rocsparse_indextype>(1);
        for(int which = 0; which < 2; ++which)
        {
            const auto pt = which ? i32 : u16, it = which ? u16 : i32;
            Spmat      source, target;
            ASSERT_EQ(create_csx(source, false, 2, 2, 2, row, col, val, pt, it, 0, f32), st_ok);
            ASSERT_EQ(create_csx(target, false, 2, 2, 0, t_row, nullptr, nullptr, pt, it, 0, f32),
                      st_ok);
            rejected_operations(source, target);
        }

        // Dimensions that do not fit into 32 bits.
        const int64_t huge = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
        for(int which = 0; which < 2; ++which)
        {
            DevBuf<int32_t> dummy(4);
            Spmat           source, target;
            const int64_t   rows = which ? 1 : huge, cols = which ? huge : 1;
            ASSERT_EQ(
                create_csx(source, false, rows, cols, 0, dummy, nullptr, nullptr, i32, i32, 0, f32),
                st_ok);
            ASSERT_EQ(
                create_csx(target, false, rows, cols, 0, dummy, nullptr, nullptr, i32, i32, 0, f32),
                st_ok);
            rejected_operations(source, target);
        }
    }
} // namespace
