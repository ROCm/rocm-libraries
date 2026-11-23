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

// bsrsv analysis reuse and zero-pivot ownership, on the block bidiagonal
// system of unit_test_internal_level2_bsrsv_test_matrix.hpp: reuse borrowing a
// bsric0, bsrilu0 or bsrsm analysis, a force after such a borrow, analyses
// written into a separate bsrsv container, and analyses of different index
// widths sharing one zero pivot. bsrsv_analysis_template and bsrsv_solve_core
// run at (i32, i32), (i64, i32) and (i64, i64).

#include "unit_test_device_level2_common.hpp"
#include "unit_test_internal_level2_bsrsv_test_matrix.hpp"

#include "rocsparse_bsrsv.hpp"
#include "rocsparse_mat_info.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    using namespace bsrsv_test;

    // Drops the diagonal block of block row missing_row from a zero-based matrix.
    // The analysis sees a structural zero there, which the solve does not.
    template <typename I, typename J>
    void remove_diagonal_block(test_matrix<I, J>& a, int32_t missing_row)
    {
        constexpr size_t  blk = static_cast<size_t>(kBlockDim) * kBlockDim;
        test_matrix<I, J> b   = a;
        b.row_ptr.assign(1, static_cast<I>(0));
        b.col_ind.clear();
        b.val.clear();
        for(int32_t row = 0; row < kMb; ++row)
        {
            for(I j = a.row_ptr[row]; j < a.row_ptr[row + 1]; ++j)
            {
                if(row == missing_row && a.col_ind[j] == static_cast<J>(row))
                {
                    continue;
                }
                b.col_ind.push_back(a.col_ind[j]);
                b.val.insert(b.val.end(),
                             a.val.begin() + blk * static_cast<size_t>(j),
                             a.val.begin() + blk * static_cast<size_t>(j + 1));
            }
            b.row_ptr.push_back(static_cast<I>(b.col_ind.size()));
        }
        a = b;
    }

    // Adds a kOffAsym block at column row - 3 to every block row of a zero-based
    // lower matrix that has one, so the transposed permutation is not a shift.
    template <typename I, typename J>
    void add_far_lower_block(test_matrix<I, J>& a)
    {
        constexpr int32_t offset = 3;
        constexpr size_t  blk    = static_cast<size_t>(kBlockDim) * kBlockDim;
        test_matrix<I, J> b      = a;
        b.row_ptr.assign(1, static_cast<I>(0));
        b.col_ind.clear();
        b.val.clear();
        for(int32_t row = 0; row < kMb; ++row)
        {
            if(row >= offset)
            {
                b.col_ind.push_back(static_cast<J>(row - offset));
                append_block(b.val, kOffAsym, a.dir);
            }
            for(I j = a.row_ptr[row]; j < a.row_ptr[row + 1]; ++j)
            {
                b.col_ind.push_back(a.col_ind[j]);
                b.val.insert(b.val.end(),
                             a.val.begin() + blk * static_cast<size_t>(j),
                             a.val.begin() + blk * static_cast<size_t>(j + 1));
            }
            b.row_ptr.push_back(static_cast<I>(b.col_ind.size()));
        }
        a = b;
    }

    test_matrix<int64_t, int64_t> widen(const test_matrix<int32_t, int32_t>& a)
    {
        test_matrix<int64_t, int64_t> b;
        b.lower = a.lower;
        b.base  = a.base;
        b.dir   = a.dir;
        b.row_ptr.assign(a.row_ptr.begin(), a.row_ptr.end());
        b.col_ind.assign(a.col_ind.begin(), a.col_ind.end());
        b.val = a.val;
        b.map.assign(a.map.begin(), a.map.end());
        return b;
    }

    // The routines whose lower analysis bsrsv reuse borrows.
    enum class donor_t
    {
        bsric0,
        bsrilu0,
        bsrsm
    };

    // One block-lower matrix with 32-bit and 64-bit indices on one handle, matrix
    // descriptor and mat_info, and a buffer large enough for every donor and for
    // bsrsv at either width, fill mode and operation. The bsrsv analyses of both
    // fill modes share one zero pivot, which is how an analysis of one width
    // retypes the pivot that another analysis left.
    class internal_level2_bsrsv_analysis_reuse : public rocsparse_ut::HandleTest
    {
    protected:
        test_matrix<int32_t, int32_t> a32;
        test_matrix<int64_t, int64_t> a64;
        std::vector<test_T>           x = make_x();

        level2::mat_descr descr;
        level2::mat_info  info;

        std::unique_ptr<rocsparse_ut::device_vector<int32_t>> d_row32;
        std::unique_ptr<rocsparse_ut::device_vector<int32_t>> d_col32;
        std::unique_ptr<rocsparse_ut::device_vector<int64_t>> d_row64;
        std::unique_ptr<rocsparse_ut::device_vector<int64_t>> d_col64;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_val;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_x;
        std::unique_ptr<rocsparse_ut::device_vector<test_T>>  d_y;
        std::unique_ptr<rocsparse_ut::device_vector<char>>    d_buffer;

        // With missing_row >= 0 that block row has no diagonal block, which the
        // analyses report as a structural zero. far_block adds the blocks of
        // add_far_lower_block.
        void build(int32_t missing_row = -1, bool far_block = false)
        {
            a32 = build_matrix<int32_t, int32_t>(true);
            if(missing_row >= 0)
            {
                remove_diagonal_block(a32, missing_row);
            }
            if(far_block)
            {
                add_far_lower_block(a32);
            }
            a64 = widen(a32);

            d_row32 = std::make_unique<rocsparse_ut::device_vector<int32_t>>(a32.row_ptr);
            d_col32 = std::make_unique<rocsparse_ut::device_vector<int32_t>>(a32.col_ind);
            d_row64 = std::make_unique<rocsparse_ut::device_vector<int64_t>>(a64.row_ptr);
            d_col64 = std::make_unique<rocsparse_ut::device_vector<int64_t>>(a64.col_ind);
            d_val   = std::make_unique<rocsparse_ut::device_vector<test_T>>(a32.val);
            d_x     = std::make_unique<rocsparse_ut::device_vector<test_T>>(x);
            d_y     = std::make_unique<rocsparse_ut::device_vector<test_T>>(
                std::vector<test_T>(static_cast<size_t>(kRows), static_cast<test_T>(-7)));
            ASSERT_NE(d_row32->ptr, nullptr);
            ASSERT_NE(d_col32->ptr, nullptr);
            ASSERT_NE(d_row64->ptr, nullptr);
            ASSERT_NE(d_col64->ptr, nullptr);
            ASSERT_NE(d_val->ptr, nullptr);
            ASSERT_NE(d_x->ptr, nullptr);
            ASSERT_NE(d_y->ptr, nullptr);

            ASSERT_NE(descr.d, nullptr);
            ASSERT_NE(info.d, nullptr);
            ASSERT_EQ(rocsparse_set_mat_diag_type(descr.d, rocsparse_diag_type_non_unit),
                      rocsparse_status_success);

            size_t bytes = 0;
            for(const auto fill : {rocsparse_fill_mode_lower, rocsparse_fill_mode_upper})
            {
                ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, fill), rocsparse_status_success);
                for(const auto trans : {rocsparse_operation_none, rocsparse_operation_transpose})
                {
                    size_t b32    = 0;
                    size_t b64_32 = 0;
                    size_t b64    = 0;
                    ASSERT_EQ((buffer_size<int32_t, int32_t>(&b32, trans)),
                              rocsparse_status_success);
                    ASSERT_EQ((buffer_size<int64_t, int32_t>(&b64_32, trans)),
                              rocsparse_status_success);
                    ASSERT_EQ((buffer_size<int64_t, int64_t>(&b64, trans)),
                              rocsparse_status_success);
                    bytes = std::max({bytes, b32, b64_32, b64});
                }
            }
            ASSERT_EQ(rocsparse_set_mat_fill_mode(descr.d, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
            size_t ic0_bytes  = 0;
            size_t ilu0_bytes = 0;
            size_t sm_bytes   = 0;
            ASSERT_EQ(rocsparse_sbsric0_buffer_size(handle,
                                                    rocsparse_direction_row,
                                                    kMb,
                                                    a32.nnzb(),
                                                    descr.d,
                                                    d_val->ptr,
                                                    d_row32->ptr,
                                                    d_col32->ptr,
                                                    kBlockDim,
                                                    info.d,
                                                    &ic0_bytes),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_sbsrilu0_buffer_size(handle,
                                                     rocsparse_direction_row,
                                                     kMb,
                                                     a32.nnzb(),
                                                     descr.d,
                                                     d_val->ptr,
                                                     d_row32->ptr,
                                                     d_col32->ptr,
                                                     kBlockDim,
                                                     info.d,
                                                     &ilu0_bytes),
                      rocsparse_status_success);
            ASSERT_EQ(rocsparse_sbsrsm_buffer_size(handle,
                                                   rocsparse_direction_row,
                                                   rocsparse_operation_none,
                                                   rocsparse_operation_none,
                                                   kMb,
                                                   1,
                                                   a32.nnzb(),
                                                   descr.d,
                                                   d_val->ptr,
                                                   d_row32->ptr,
                                                   d_col32->ptr,
                                                   kBlockDim,
                                                   info.d,
                                                   &sm_bytes),
                      rocsparse_status_success);
            bytes = std::max({bytes, ic0_bytes, ilu0_bytes, sm_bytes});

            d_buffer = std::make_unique<rocsparse_ut::device_vector<char>>(
                std::vector<char>(bytes + 1, 0));
            ASSERT_NE(d_buffer->ptr, nullptr);
        }

        template <typename I>
        const I* row_ptr() const
        {
            if constexpr(std::is_same<I, int64_t>::value)
            {
                return d_row64->ptr;
            }
            else
            {
                return d_row32->ptr;
            }
        }

        template <typename J>
        const J* col_ind() const
        {
            if constexpr(std::is_same<J, int64_t>::value)
            {
                return d_col64->ptr;
            }
            else
            {
                return d_col32->ptr;
            }
        }

        template <typename I, typename J>
        rocsparse_status buffer_size(size_t* bytes, rocsparse_operation trans)
        {
            return rocsparse::bsrsv_buffer_size_template<test_T, I, J>(handle,
                                                                       rocsparse_direction_row,
                                                                       trans,
                                                                       static_cast<J>(kMb),
                                                                       static_cast<I>(a32.nnzb()),
                                                                       descr.d,
                                                                       d_val->ptr,
                                                                       row_ptr<I>(),
                                                                       col_ind<J>(),
                                                                       kBlockDim,
                                                                       info.d,
                                                                       bytes);
        }

        // Writes into dest, or into the bsrsv container of info.d when dest
        // is null. The donors always come from info.d.
        template <typename I, typename J>
        rocsparse_status analysis(rocsparse_fill_mode       fill,
                                  rocsparse_analysis_policy policy,
                                  rocsparse_operation       trans = rocsparse_operation_none,
                                  rocsparse_bsrsv_info      dest  = nullptr)
        {
            const rocsparse_status status = rocsparse_set_mat_fill_mode(descr.d, fill);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            return rocsparse::bsrsv_analysis_template<test_T, I, J>(
                handle,
                rocsparse_direction_row,
                trans,
                static_cast<J>(kMb),
                static_cast<I>(a32.nnzb()),
                descr.d,
                d_val->ptr,
                row_ptr<I>(),
                col_ind<J>(),
                kBlockDim,
                info.d,
                (dest != nullptr) ? dest : info.d->get_bsrsv_info(),
                policy,
                rocsparse_solve_policy_auto,
                d_buffer->ptr);
        }

        // Lower analysis of donor, which bsrsv reuse may borrow.
        rocsparse_status donor_analysis(donor_t donor)
        {
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(descr.d, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            switch(donor)
            {
            case donor_t::bsric0:
                return rocsparse_sbsric0_analysis(handle,
                                                  rocsparse_direction_row,
                                                  kMb,
                                                  a32.nnzb(),
                                                  descr.d,
                                                  d_val->ptr,
                                                  d_row32->ptr,
                                                  d_col32->ptr,
                                                  kBlockDim,
                                                  info.d,
                                                  rocsparse_analysis_policy_force,
                                                  rocsparse_solve_policy_auto,
                                                  d_buffer->ptr);
            case donor_t::bsrilu0:
                return rocsparse_sbsrilu0_analysis(handle,
                                                   rocsparse_direction_row,
                                                   kMb,
                                                   a32.nnzb(),
                                                   descr.d,
                                                   d_val->ptr,
                                                   d_row32->ptr,
                                                   d_col32->ptr,
                                                   kBlockDim,
                                                   info.d,
                                                   rocsparse_analysis_policy_force,
                                                   rocsparse_solve_policy_auto,
                                                   d_buffer->ptr);
            case donor_t::bsrsm:
                return rocsparse_sbsrsm_analysis(handle,
                                                 rocsparse_direction_row,
                                                 rocsparse_operation_none,
                                                 rocsparse_operation_none,
                                                 kMb,
                                                 1,
                                                 a32.nnzb(),
                                                 descr.d,
                                                 d_val->ptr,
                                                 d_row32->ptr,
                                                 d_col32->ptr,
                                                 kBlockDim,
                                                 info.d,
                                                 rocsparse_analysis_policy_force,
                                                 rocsparse_solve_policy_auto,
                                                 d_buffer->ptr);
            }
            return rocsparse_status_internal_error;
        }

        rocsparse::trm_info_t* donor_slot(donor_t donor)
        {
            switch(donor)
            {
            case donor_t::bsric0:
                return info.d->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            case donor_t::bsrilu0:
                return info.d->get_bsrilu0_info(rocsparse_operation_none,
                                                rocsparse_fill_mode_lower);
            case donor_t::bsrsm:
                return info.d->get_bsrsm_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            }
            return nullptr;
        }

        rocsparse_status donor_clear(donor_t donor)
        {
            switch(donor)
            {
            case donor_t::bsric0:
                return rocsparse_bsric0_clear(handle, info.d);
            case donor_t::bsrilu0:
                return rocsparse_bsrilu0_clear(handle, info.d);
            case donor_t::bsrsm:
                return rocsparse_bsrsm_clear(handle, info.d);
            }
            return rocsparse_status_internal_error;
        }

        // bsrsv reuse borrows the donor's analysis, and a later bsrsv force must
        // build its own analysis instead of deleting the borrowed one. The force
        // would otherwise free the borrowed analysis and allocate the new one,
        // often at the same address, so the check is that both analyses are alive
        // at once.
        void expect_force_keeps_borrowed(donor_t donor)
        {
            ASSERT_NO_FATAL_FAILURE(build());
            ASSERT_EQ(donor_analysis(donor), rocsparse_status_success);
            ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                                  rocsparse_analysis_policy_reuse)),
                      rocsparse_status_success);

            rocsparse::trm_info_t* const borrowed = donor_slot(donor);
            ASSERT_NE(borrowed, nullptr);
            EXPECT_EQ(info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                      borrowed);

            ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                                  rocsparse_analysis_policy_force)),
                      rocsparse_status_success);
            rocsparse::trm_info_t* const own
                = info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            EXPECT_NE(own, nullptr);
            EXPECT_NE(own, borrowed);
            EXPECT_EQ(donor_slot(donor), borrowed);

            ASSERT_EQ(donor_clear(donor), rocsparse_status_success);
        }

        // Block row 3 has no diagonal block. The donor's analysis finds it, and
        // the solve on the borrowed analysis must report row 3, not success. A
        // reuse that missed the donor would rebuild and report the same row, so
        // the slot must hold the donor's analysis.
        void expect_reuse_reports_structural_zero(donor_t donor)
        {
            ASSERT_NO_FATAL_FAILURE(build(3));
            ASSERT_EQ(donor_analysis(donor), rocsparse_status_success);
            ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                                  rocsparse_analysis_policy_reuse)),
                      rocsparse_status_success);
            ASSERT_NE(donor_slot(donor), nullptr);
            EXPECT_EQ(info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                      donor_slot(donor));
            ASSERT_EQ(solve_public(), rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_public_zero_pivot(3);
            ASSERT_EQ(donor_clear(donor), rocsparse_status_success);
        }

        // The zero pivot of a bsrsv container with 32-bit column indices.
        static int32_t pivot_of(rocsparse_bsrsv_info container)
        {
            int32_t value = -2;
            EXPECT_EQ(container->get_indextype(), rocsparse_indextype_i32);
            EXPECT_NE(container->get_position(), nullptr);
            if(container->get_position() != nullptr)
            {
                EXPECT_EQ(
                    hipMemcpy(
                        &value, container->get_position(), sizeof(value), hipMemcpyDeviceToHost),
                    hipSuccess);
            }
            return value;
        }

        // Lower solve through the analysis and pivot that container holds.
        rocsparse_status solve_container(rocsparse_bsrsv_info container)
        {
            return solve_core<int32_t, int32_t>(
                container->get(rocsparse_operation_none, rocsparse_fill_mode_lower),
                container->get_indextype(),
                static_cast<int32_t*>(container->get_position()));
        }

        // Reuse borrows the bsric0 analysis into the bsrsv container of
        // info.d, then a force writes into a separate destination, which
        // builds an analysis of its own and leaves the alias alone. Either side
        // can then go first. Clearing the donor hands the borrowed analysis to
        // the bsrsv container of info.d and leaves the destination's
        // analysis alone, and destroying the destination frees only its own.
        void expect_force_into_separate_destination_after_borrow(bool donor_first)
        {
            ASSERT_NO_FATAL_FAILURE(build());
            ASSERT_EQ(donor_analysis(donor_t::bsric0), rocsparse_status_success);
            ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                                  rocsparse_analysis_policy_reuse)),
                      rocsparse_status_success);
            rocsparse::trm_info_t* const borrowed = donor_slot(donor_t::bsric0);
            ASSERT_NE(borrowed, nullptr);
            ASSERT_EQ(info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                      borrowed);

            level2::mat_info dest_info;
            ASSERT_NE(dest_info.d, nullptr);
            rocsparse_bsrsv_info dest = dest_info.d->get_bsrsv_info();
            ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                                  rocsparse_analysis_policy_force,
                                                  rocsparse_operation_none,
                                                  dest)),
                      rocsparse_status_success);

            rocsparse::trm_info_t* const own
                = dest->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
            ASSERT_NE(own, nullptr);
            EXPECT_NE(own, borrowed);
            EXPECT_EQ(info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                      borrowed);
            EXPECT_EQ(donor_slot(donor_t::bsric0), borrowed);

            if(donor_first)
            {
                ASSERT_EQ(donor_clear(donor_t::bsric0), rocsparse_status_success);
                EXPECT_EQ(
                    info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                    borrowed);
                EXPECT_EQ(dest->get(rocsparse_operation_none, rocsparse_fill_mode_lower), own);
                ASSERT_EQ(solve_container(dest), rocsparse_status_success);
                expect_lower_solution();

                ASSERT_EQ(rocsparse_destroy_mat_info(std::exchange(dest_info.d, nullptr)),
                          rocsparse_status_success);
                ASSERT_EQ((solve_stored<int32_t, int32_t>()), rocsparse_status_success);
                expect_lower_solution();
            }
            else
            {
                ASSERT_EQ(rocsparse_destroy_mat_info(std::exchange(dest_info.d, nullptr)),
                          rocsparse_status_success);
                EXPECT_EQ(donor_slot(donor_t::bsric0), borrowed);
                EXPECT_EQ(
                    info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                    borrowed);
                ASSERT_EQ((solve_stored<int32_t, int32_t>()), rocsparse_status_success);
                expect_lower_solution();

                ASSERT_EQ(donor_clear(donor_t::bsric0), rocsparse_status_success);
                ASSERT_EQ((solve_stored<int32_t, int32_t>()), rocsparse_status_success);
                expect_lower_solution();
            }
        }

        // Lower solve through bsrsv_solve_core, with the analysis and pivot given
        // explicitly. y is reset first.
        template <typename I, typename J>
        rocsparse_status solve_core(rocsparse::trm_info_t* trm,
                                    rocsparse_indextype    pivot_type,
                                    J*                     pivot,
                                    rocsparse_operation    trans = rocsparse_operation_none)
        {
            if(hipMemcpy(
                   d_y->ptr,
                   std::vector<test_T>(static_cast<size_t>(kRows), static_cast<test_T>(-7)).data(),
                   sizeof(test_T) * kRows,
                   hipMemcpyHostToDevice)
               != hipSuccess)
            {
                return rocsparse_status_internal_error;
            }
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(descr.d, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            const test_T alpha = kAlpha;
            return rocsparse::bsrsv_solve_core<test_T, I, J>(handle,
                                                             rocsparse_direction_row,
                                                             trans,
                                                             static_cast<J>(kMb),
                                                             static_cast<I>(a32.nnzb()),
                                                             &alpha,
                                                             descr.d,
                                                             d_val->ptr,
                                                             row_ptr<I>(),
                                                             col_ind<J>(),
                                                             kBlockDim,
                                                             trm,
                                                             pivot_type,
                                                             pivot,
                                                             d_x->ptr,
                                                             d_y->ptr,
                                                             rocsparse_solve_policy_auto,
                                                             d_buffer->ptr);
        }

        // The same solve with the lower analysis and the pivot that mat_info holds.
        template <typename I, typename J>
        rocsparse_status solve_stored()
        {
            auto* container = info.d->get_bsrsv_info();
            return solve_core<I, J>(
                info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                container->get_indextype(),
                static_cast<J*>(container->get_position()));
        }

        rocsparse_status solve_public()
        {
            const rocsparse_status status
                = rocsparse_set_mat_fill_mode(descr.d, rocsparse_fill_mode_lower);
            if(status != rocsparse_status_success)
            {
                return status;
            }
            const test_T alpha = kAlpha;
            return rocsparse_sbsrsv_solve(handle,
                                          rocsparse_direction_row,
                                          rocsparse_operation_none,
                                          kMb,
                                          a32.nnzb(),
                                          &alpha,
                                          descr.d,
                                          d_val->ptr,
                                          d_row32->ptr,
                                          d_col32->ptr,
                                          kBlockDim,
                                          info.d,
                                          d_x->ptr,
                                          d_y->ptr,
                                          rocsparse_solve_policy_auto,
                                          d_buffer->ptr);
        }

        void expect_lower_solution()
        {
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_solution(
                rocsparse_ut::to_host(*d_y),
                reference(a32, true, rocsparse_operation_none, rocsparse_diag_type_non_unit));
        }

        rocsparse_indextype pivot_type()
        {
            return info.d->get_bsrsv_info()->get_indextype();
        }

        // The shared pivot, read with type J, which it must have.
        template <typename J>
        J stored_pivot()
        {
            auto* container = info.d->get_bsrsv_info();
            EXPECT_EQ(container->get_indextype(), rocsparse::get_indextype<J>());
            J value = 0;
            EXPECT_NE(container->get_position(), nullptr);
            if(container->get_position() != nullptr)
            {
                EXPECT_EQ(
                    hipMemcpy(&value, container->get_position(), sizeof(J), hipMemcpyDeviceToHost),
                    hipSuccess);
            }
            return value;
        }

        void expect_public_zero_pivot(rocsparse_int row)
        {
            rocsparse_int position = -2;
            EXPECT_EQ(rocsparse_bsrsv_zero_pivot(handle, info.d, &position),
                      rocsparse_status_zero_pivot);
            EXPECT_EQ(position, row);
        }

        // Transposed analysis and solve of the lower matrix through the core
        // entry, checked against the host solve of A^T y = kAlpha x.
        template <typename I, typename J>
        void expect_core_transpose()
        {
            SCOPED_TRACE(::testing::Message() << "I " << sizeof(I) * 8 << " J " << sizeof(J) * 8);
            ASSERT_EQ((analysis<I, J>(rocsparse_fill_mode_lower,
                                      rocsparse_analysis_policy_force,
                                      rocsparse_operation_transpose)),
                      rocsparse_status_success);
            auto* container = info.d->get_bsrsv_info();
            ASSERT_EQ((solve_core<I, J>(info.d->get_bsrsv_info(rocsparse_operation_transpose,
                                                               rocsparse_fill_mode_lower),
                                        container->get_indextype(),
                                        static_cast<J*>(container->get_position()),
                                        rocsparse_operation_transpose)),
                      rocsparse_status_success);
            ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
            expect_solution(
                rocsparse_ut::to_host(*d_y),
                reference(a32, true, rocsparse_operation_transpose, rocsparse_diag_type_non_unit));
            EXPECT_EQ(stored_pivot<J>(), std::numeric_limits<J>::max());
        }
    };
}

TEST_F(internal_level2_bsrsv_analysis_reuse, force_keeps_borrowed_bsric0_analysis)
{
    expect_force_keeps_borrowed(donor_t::bsric0);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, force_keeps_borrowed_bsrilu0_analysis)
{
    expect_force_keeps_borrowed(donor_t::bsrilu0);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, force_keeps_borrowed_bsrsm_analysis)
{
    expect_force_keeps_borrowed(donor_t::bsrsm);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_borrowed_bsric0_solves)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(donor_analysis(donor_t::bsric0), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    ASSERT_EQ(solve_public(), rocsparse_status_success);
    expect_lower_solution();

    rocsparse_int position = -2;
    EXPECT_EQ(rocsparse_bsrsv_zero_pivot(handle, info.d, &position), rocsparse_status_success);
    ASSERT_EQ(rocsparse_bsric0_clear(handle, info.d), rocsparse_status_success);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_borrowed_bsric0_structural_zero_reported)
{
    expect_reuse_reports_structural_zero(donor_t::bsric0);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_borrowed_bsrilu0_structural_zero_reported)
{
    expect_reuse_reports_structural_zero(donor_t::bsrilu0);
}

TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_borrowed_bsrsm_structural_zero_reported)
{
    expect_reuse_reports_structural_zero(donor_t::bsrsm);
}

// The analysis goes to a container other than the one in the mat_info that
// holds the donor. trm_t::uncouple would not see an alias stored there, so
// reuse must not borrow the bsric0 analysis. The destination builds its own,
// which finds the structural zero itself, and the bsrsv container of the
// mat_info stays empty. The destination's analysis survives the donor's clear
// and goes with the destination.
TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_into_separate_destination_builds_own_analysis)
{
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(donor_analysis(donor_t::bsric0), rocsparse_status_success);

    level2::mat_info dest_info;
    ASSERT_NE(dest_info.d, nullptr);
    rocsparse_bsrsv_info dest = dest_info.d->get_bsrsv_info();

    ASSERT_EQ((analysis<int32_t, int32_t>(rocsparse_fill_mode_lower,
                                          rocsparse_analysis_policy_reuse,
                                          rocsparse_operation_none,
                                          dest)),
              rocsparse_status_success);
    rocsparse::trm_info_t* const own
        = dest->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ASSERT_NE(own, nullptr);
    EXPECT_NE(own, donor_slot(donor_t::bsric0));
    EXPECT_EQ(info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower), nullptr);
    EXPECT_EQ(pivot_of(dest), 3);

    ASSERT_EQ(donor_clear(donor_t::bsric0), rocsparse_status_success);
    EXPECT_EQ(dest->get(rocsparse_operation_none, rocsparse_fill_mode_lower), own);
    ASSERT_EQ(solve_container(dest), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(pivot_of(dest), 3);

    EXPECT_EQ(rocsparse_destroy_mat_info(std::exchange(dest_info.d, nullptr)),
              rocsparse_status_success);
}

TEST_F(internal_level2_bsrsv_analysis_reuse,
       force_into_separate_destination_after_borrow_donor_first)
{
    expect_force_into_separate_destination_after_borrow(true);
}

TEST_F(internal_level2_bsrsv_analysis_reuse,
       force_into_separate_destination_after_borrow_destination_first)
{
    expect_force_into_separate_destination_after_borrow(false);
}

// The extra block at column r - 3 makes the transposed permutation more than a
// shift, so a gather launched with the narrow index types at a 64-bit width
// reads the wrong values.
TEST_F(internal_level2_bsrsv_analysis_reuse, solve_core_transpose_all_widths)
{
    ASSERT_NO_FATAL_FAILURE(build(-1, true));
    ASSERT_NO_FATAL_FAILURE((expect_core_transpose<int32_t, int32_t>()));
    ASSERT_NO_FATAL_FAILURE((expect_core_transpose<int64_t, int32_t>()));
    ASSERT_NO_FATAL_FAILURE((expect_core_transpose<int64_t, int64_t>()));
}

// An analysis at one width, then at another, on the same mat_info. The pivot
// type follows the latest analysis, and a solve of that width succeeds.
TEST_F(internal_level2_bsrsv_analysis_reuse, mixed_width_retypes_pivot_and_solves)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);
    // The public entry reads the pivot as rocsparse_int, so an i64 pivot is rejected.
    EXPECT_EQ(solve_public(), rocsparse_status_invalid_value);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
    EXPECT_EQ(stored_pivot<int64_t>(), std::numeric_limits<int64_t>::max());

    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);
    ASSERT_EQ(solve_public(), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
}

// An upper i32 analysis retypes the shared pivot. Until the lower i64 analysis
// is reused, an i64 solve rejects that i32 pivot, and the reuse gives it an i64
// pivot again.
TEST_F(internal_level2_bsrsv_analysis_reuse, mixed_width_own_reuse_retypes_pivot)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);
    EXPECT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_invalid_value);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
    EXPECT_EQ(stored_pivot<int64_t>(), std::numeric_limits<int64_t>::max());
}

// Block row 3 has no diagonal block, a structural zero that only the analysis
// records, in the shared pivot. The upper i32 analysis replaces that pivot with
// an i32 one, so the reuse of the lower i64 analysis has to run the analysis
// again to record the zero in an i64 pivot.
TEST_F(internal_level2_bsrsv_analysis_reuse, mixed_width_own_reuse_keeps_structural_zero)
{
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int32_t>(), 3);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    EXPECT_EQ(stored_pivot<int64_t>(), 3);
}

// bsrsv_solve_core must reject an analysis of another width even when the
// pivot it is given has the right type. The i64 analysis comes first: read as
// i32, its row map still holds valid block rows, so a solve that skipped the
// check would return success instead of reading out of bounds.
TEST_F(internal_level2_bsrsv_analysis_reuse, solve_core_rejects_analysis_of_other_width)
{
    ASSERT_NO_FATAL_FAILURE(build());
    rocsparse_ut::device_vector<int32_t> d_pivot32(
        std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
    rocsparse_ut::device_vector<int64_t> d_pivot64(
        std::vector<int64_t>{std::numeric_limits<int64_t>::max()});
    ASSERT_NE(d_pivot32.ptr, nullptr);
    ASSERT_NE(d_pivot64.ptr, nullptr);

    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ((solve_core<int32_t, int32_t>(
                  info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                  rocsparse_indextype_i32,
                  d_pivot32.ptr)),
              rocsparse_status_invalid_value);

    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ((solve_core<int64_t, int64_t>(
                  info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower),
                  rocsparse_indextype_i64,
                  d_pivot64.ptr)),
              rocsparse_status_invalid_value);
}

// Reuse of an analysis of another width rebuilds it. The upper i64 analysis
// leaves an i64 pivot, so the pivot type alone does not force the rebuild.
TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_rebuilds_analysis_of_other_width)
{
    ASSERT_NO_FATAL_FAILURE(build());
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);

    rocsparse::trm_info_t* const trm
        = info.d->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
    ASSERT_NE(trm, nullptr);
    EXPECT_TRUE(trm->has_indextypes(rocsparse_indextype_i64, rocsparse_indextype_i64));
    ASSERT_EQ((solve_stored<int64_t, int64_t>()), rocsparse_status_success);
    ASSERT_NO_FATAL_FAILURE(expect_lower_solution());
}

// An i64 analysis of the other fill mode leaves an i64 pivot. Reuse of the
// lower i32 bsric0 analysis must drop it and copy the donor pivot, which holds
// the structural zero at block row 3.
TEST_F(internal_level2_bsrsv_analysis_reuse, reuse_drops_mismatched_pivot)
{
    ASSERT_NO_FATAL_FAILURE(build(3));
    ASSERT_EQ(
        (analysis<int64_t, int64_t>(rocsparse_fill_mode_upper, rocsparse_analysis_policy_force)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i64);

    ASSERT_EQ(donor_analysis(donor_t::bsric0), rocsparse_status_success);
    ASSERT_EQ(
        (analysis<int32_t, int32_t>(rocsparse_fill_mode_lower, rocsparse_analysis_policy_reuse)),
        rocsparse_status_success);
    EXPECT_EQ(pivot_type(), rocsparse_indextype_i32);

    ASSERT_EQ(solve_public(), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    expect_public_zero_pivot(3);
    ASSERT_EQ(rocsparse_bsric0_clear(handle, info.d), rocsparse_status_success);
}
