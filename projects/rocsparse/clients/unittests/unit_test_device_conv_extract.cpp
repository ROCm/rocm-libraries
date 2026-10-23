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

    using ExtractDescr = Owned<rocsparse_extract_descr, rocsparse_destroy_extract_descr>;

    size_t value_size(rocsparse_datatype dt)
    {
        switch(dt)
        {
        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
            return 1;
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
            return 2;
        case rocsparse_datatype_f64_r:
        case rocsparse_datatype_f32_c:
            return 8;
        case rocsparse_datatype_f64_c:
            return 16;
        default:
            return 4;
        }
    }

    // Library bug (AISPARSE-925): for a matrix it does not support,
    // rocsparse_create_extract_descr lets a rocsparse_status thrown by the descriptor
    // constructor escape the C API instead of returning it. Both behaviours are accepted.
    rocsparse_status create_extract(ExtractDescr&               descr,
                                    rocsparse_const_spmat_descr source,
                                    rocsparse_spmat_descr       target)
    {
        try
        {
            return rocsparse_create_extract_descr(
                &descr, source, target, rocsparse_extract_alg_default);
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
    }

    rocsparse_status create_csx(Spmat&              m,
                                bool                csc,
                                int64_t             rows,
                                int64_t             cols,
                                int64_t             nnz,
                                void*               ptr,
                                void*               ind,
                                void*               val,
                                rocsparse_indextype ptr_type,
                                rocsparse_indextype ind_type,
                                int                 base,
                                rocsparse_datatype  dt)
    {
        const auto create = csc ? rocsparse_create_csc_descr : rocsparse_create_csr_descr;
        return create(&m, rows, cols, nnz, ptr, ind, val, ptr_type, ind_type, base_of(base), dt);
    }

    // Extracts the part of the matrix given by (ptr, ind) and checks it. For a unit
    // diagonal the result may keep or drop the diagonal (the header does not say which);
    // it must match one of the two references exactly.
    template <typename P, typename J>
    void run_extract(rocsparse_handle      handle,
                     bool                  csc,
                     int64_t               rows,
                     int64_t               cols,
                     int                   base,
                     rocsparse_datatype    dt,
                     const std::vector<P>& ptr,
                     const std::vector<J>& ind,
                     rocsparse_fill_mode   fill,
                     rocsparse_diag_type   diag)
    {
        SCOPED_TRACE("csc " + std::to_string(csc) + " datatype " + std::to_string(dt) + " fill "
                     + std::to_string(fill) + " diag " + std::to_string(diag));
        const int64_t nnz = ind.size();
        const size_t  vs  = value_size(dt);
        // Entry k holds the byte 0x3f + k in every byte, a finite value of every datatype.
        std::vector<char> val;
        for(int64_t k = 0; k < nnz; ++k)
            val.insert(val.end(), vs, char(0x3f + k));
        DevBuf<P>    d_ptr(ptr), t_ptr(std::vector<P>(ptr.size(), P(-7)));
        DevBuf<J>    d_ind(ind), t_ind(ind.size());
        DevBuf<char> d_val(val), t_val(nnz * vs);
        Spmat        source, target;
        ASSERT_EQ(create_csx(source,
                             csc,
                             rows,
                             cols,
                             nnz,
                             d_ptr,
                             d_ind,
                             d_val,
                             it_of<P>(),
                             it_of<J>(),
                             base,
                             dt),
                  st_ok);
        ASSERT_EQ(create_csx(target,
                             csc,
                             rows,
                             cols,
                             0,
                             t_ptr,
                             nullptr,
                             nullptr,
                             it_of<P>(),
                             it_of<J>(),
                             base,
                             dt),
                  st_ok);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(target, rocsparse_spmat_fill_mode, &fill, sizeof(fill)),
            st_ok);
        ASSERT_EQ(
            rocsparse_spmat_set_attribute(target, rocsparse_spmat_diag_type, &diag, sizeof(diag)),
            st_ok);

        ExtractDescr descr;
        ASSERT_EQ(create_extract(descr, source, target), st_ok);
        const auto stage = [&](rocsparse_extract_stage s) {
            size_t size = 0;
            ASSERT_EQ(rocsparse_extract_buffer_size(handle, descr, source, target, s, &size),
                      st_ok);
            DevBuf<char> buffer(size);
            ASSERT_EQ(rocsparse_extract(handle, descr, source, target, s, size, buffer), st_ok);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        };
        stage(rocsparse_extract_stage_analysis);
        int64_t target_nnz = -1;
        ASSERT_EQ(rocsparse_extract_nnz(handle, descr, &target_nnz), st_ok);
        ASSERT_EQ((csc ? rocsparse_csc_set_pointers
                       : rocsparse_csr_set_pointers)(target, t_ptr, t_ind, t_val),
                  st_ok);
        stage(rocsparse_extract_stage_compute);

        // Host references: the entries below (above) the diagonal, with the diagonal
        // (strict = 0) and without it (strict = 1).
        std::vector<P>    want_ptr[2];
        std::vector<J>    want_ind[2];
        std::vector<char> want_val[2];
        for(int strict = 0; strict < 2; ++strict)
        {
            want_ptr[strict] = {static_cast<P>(base)};
            for(int64_t outer = 0; outer + 1 < static_cast<int64_t>(ptr.size()); ++outer)
            {
                for(int64_t k = ptr[outer] - base; k < ptr[outer + 1] - base; ++k)
                {
                    const int64_t i = csc ? ind[k] - base : outer, j = csc ? outer : ind[k] - base;
                    if(i == j ? !strict : (fill == rocsparse_fill_mode_lower) == (j < i))
                    {
                        want_ind[strict].push_back(ind[k]);
                        want_val[strict].insert(want_val[strict].end(), vs, char(0x3f + k));
                    }
                }
                want_ptr[strict].push_back(base + want_ind[strict].size());
            }
        }
        const int strict = diag == rocsparse_diag_type_unit
                           && target_nnz != static_cast<int64_t>(want_ind[0].size());
        ASSERT_EQ(target_nnz, static_cast<int64_t>(want_ind[strict].size()));
        EXPECT_EQ(t_ptr.read(ptr.size()), want_ptr[strict]);
        EXPECT_EQ(t_ind.read(target_nnz), want_ind[strict]);
        EXPECT_EQ(t_val.read(target_nnz * vs), want_val[strict]);
    }

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
