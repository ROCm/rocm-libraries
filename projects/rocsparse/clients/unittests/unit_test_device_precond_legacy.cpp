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

// The mat_info based preconditioner entry points that the generic descriptor API does
// not use: the numerical tolerance and singular pivot of csrilu0 / csric0, and the
// numeric boost of csrilu0 / bsrilu0 for the four value types, in both pointer modes and
// both index bases. The factors are compared with the host references on diagonally
// dominant matrices, so every pivot is finite and bounded away from zero.

#include "unit_test_device_precond_common.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace precond_ut;

namespace
{
    // The boost value types of the numeric boost entry points: the standard ones
    // (tolerance and value of the precision of T) and the mixed ones (a double
    // tolerance with single precision values).
    enum class boost_api
    {
        standard,
        mixed
    };

    struct boost_request
    {
        bool      enable = false;
        double    tol    = 0.0;
        double    val    = 1.0;
        bool      device = false; // pointer mode of the boost parameters
        boost_api api    = boost_api::standard;
    };

    // The boost parameters in host and in device memory. The library keeps the pointers
    // it is given, so the object must outlive the factorization.
    template <typename T>
    struct boost_params
    {
        real_of<T>                tol_s;
        double                    tol_d;
        T                         val;
        device_vector<real_of<T>> dtol_s;
        device_vector<double>     dtol_d;
        device_vector<T>          dval;

        explicit boost_params(const boost_request& r)
            : tol_s(static_cast<real_of<T>>(r.tol))
            , tol_d(r.tol)
            , val(make_value<T>(r.val, 0.0))
            , dtol_s(std::vector<real_of<T>>{tol_s})
            , dtol_d(std::vector<double>{tol_d})
            , dval(std::vector<T>{val})
        {
        }

        template <bool BSR>
        rocsparse_status call(rocsparse_handle   handle,
                              rocsparse_mat_info info,
                              int                enable,
                              bool               device,
                              boost_api          api) const
        {
            const void* tol
                = device ? static_cast<const void*>(dtol_s.ptr) : static_cast<const void*>(&tol_s);
            const void* v
                = device ? static_cast<const void*>(dval.ptr) : static_cast<const void*>(&val);
            if(enable == 0)
                tol = v = nullptr;
            if(api == boost_api::mixed)
            {
                const void* tol_double = device ? static_cast<const void*>(dtol_d.ptr)
                                                : static_cast<const void*>(&tol_d);
                const auto  x          = static_cast<const T*>(v);
                if constexpr(std::is_same_v<T, float>)
                    return rocsparse_dscsrilu0_numeric_boost(
                        handle, info, enable, static_cast<const double*>(tol_double), x);
                else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
                    return rocsparse_dccsrilu0_numeric_boost(
                        handle, info, enable, static_cast<const double*>(tol_double), x);
                else
                    return rocsparse_status_not_implemented;
            }
            const auto t = static_cast<const real_of<T>*>(tol);
            const auto x = static_cast<const T*>(v);
            if constexpr(BSR)
                return PRECOND_CALL(T, bsrilu0_numeric_boost, handle, info, enable, t, x);
            else
                return PRECOND_CALL(T, csrilu0_numeric_boost, handle, info, enable, t, x);
        }

        // Defines the boost the way `r` asks for: enabled in the pointer mode of the
        // request, or, when `r.tol` is not zero, enabled and then disabled again.
        template <bool BSR>
        rocsparse_status
            apply(rocsparse_handle handle, rocsparse_mat_info info, const boost_request& r) const
        {
            if(!r.enable)
            {
                if(r.tol == 0.0 || r.api != boost_api::standard)
                    return rocsparse_status_success;
                const auto status = call<BSR>(handle, info, 1, false, r.api);
                return (status != rocsparse_status_success)
                           ? status
                           : call<BSR>(handle, info, 0, false, r.api);
            }
            if(r.device)
            {
                const auto mode = rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device);
                if(mode != rocsparse_status_success)
                    return mode;
            }
            const auto status = call<BSR>(handle, info, 1, r.device, r.api);
            // The pointer mode that counts is the one of the call that defined the boost;
            // the compute runs in host mode.
            if(r.device)
                (void)rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host);
            return status;
        }
    };

    // Index arrays of `ptr` / `ind` shifted for `base`.
    void upload_indices(const std::vector<int32_t>&   ptr,
                        const std::vector<int32_t>&   ind,
                        rocsparse_index_base          base,
                        device_vector<rocsparse_int>& drow,
                        device_vector<rocsparse_int>& dcol)
    {
        const rocsparse_int        shift = (base == rocsparse_index_base_one) ? 1 : 0;
        std::vector<rocsparse_int> hptr  = as_rocsparse_int(ptr);
        std::vector<rocsparse_int> hind  = as_rocsparse_int(ind);
        for(auto& v : hptr)
            v += shift;
        for(auto& v : hind)
            v += shift;
        upload(drow, hptr);
        upload(dcol, hind);
    }

    // ------------------------------------------------------------------
    // csrilu0 / csric0
    // ------------------------------------------------------------------
    template <Kind K, typename T>
    rocsparse_status csr_buffer_size(rocsparse_handle     handle,
                                     const csr_matrix<T>& a,
                                     rocsparse_mat_descr  descr,
                                     const T*             val,
                                     const rocsparse_int* row,
                                     const rocsparse_int* col,
                                     rocsparse_mat_info   info,
                                     size_t*              size)
    {
        if constexpr(K == Kind::ilu)
            return PRECOND_CALL(
                T, csrilu0_buffer_size, handle, a.n, a.nnz(), descr, val, row, col, info, size);
        else
            return PRECOND_CALL(
                T, csric0_buffer_size, handle, a.n, a.nnz(), descr, val, row, col, info, size);
    }

    template <Kind K, typename T>
    rocsparse_status csr_analysis(rocsparse_handle     handle,
                                  const csr_matrix<T>& a,
                                  rocsparse_mat_descr  descr,
                                  const T*             val,
                                  const rocsparse_int* row,
                                  const rocsparse_int* col,
                                  rocsparse_mat_info   info,
                                  void*                buffer)
    {
        const auto reuse = rocsparse_analysis_policy_reuse;
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(K == Kind::ilu)
            return PRECOND_CALL(T,
                                csrilu0_analysis,
                                handle,
                                a.n,
                                a.nnz(),
                                descr,
                                val,
                                row,
                                col,
                                info,
                                reuse,
                                solve,
                                buffer);
        else
            return PRECOND_CALL(T,
                                csric0_analysis,
                                handle,
                                a.n,
                                a.nnz(),
                                descr,
                                val,
                                row,
                                col,
                                info,
                                reuse,
                                solve,
                                buffer);
    }

    template <Kind K, typename T>
    rocsparse_status csr_compute(rocsparse_handle     handle,
                                 const csr_matrix<T>& a,
                                 rocsparse_mat_descr  descr,
                                 T*                   val,
                                 const rocsparse_int* row,
                                 const rocsparse_int* col,
                                 rocsparse_mat_info   info,
                                 void*                buffer)
    {
        const auto solve = rocsparse_solve_policy_auto;
        if constexpr(K == Kind::ilu)
            return PRECOND_CALL(
                T, csrilu0, handle, a.n, a.nnz(), descr, val, row, col, info, solve, buffer);
        else
            return PRECOND_CALL(
                T, csric0, handle, a.n, a.nnz(), descr, val, row, col, info, solve, buffer);
    }

    template <Kind K>
    rocsparse_status get_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double* tol)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_get_tolerance(handle, info, tol)
                                : rocsparse_csric0_get_tolerance(handle, info, tol);
    }

    template <Kind K>
    rocsparse_status set_tolerance(rocsparse_handle handle, rocsparse_mat_info info, double tol)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_set_tolerance(handle, info, tol)
                                : rocsparse_csric0_set_tolerance(handle, info, tol);
    }

    template <Kind K>
    rocsparse_status
        singular_pivot(rocsparse_handle handle, rocsparse_mat_info info, rocsparse_int* position)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_singular_pivot(handle, info, position)
                                : rocsparse_csric0_singular_pivot(handle, info, position);
    }

    template <Kind K>
    rocsparse_status clear(rocsparse_handle handle, rocsparse_mat_info info)
    {
        return (K == Kind::ilu) ? rocsparse_csrilu0_clear(handle, info)
                                : rocsparse_csric0_clear(handle, info);
    }

    template <typename T>
    struct csr_out
    {
        rocsparse_status compute_status = rocsparse_status_internal_error;
        std::vector<T>   values;
        rocsparse_int    singular_pivot = -7;
        double           tolerance_back = -1.0;
    };

    // Analysis and compute of `a` (kept zero based, the device copy is shifted for
    // `base`). `tolerance` < 0 keeps the default; it is set after the analysis, once the
    // data of the mat_info exists. The singular pivot is read back in the pointer mode
    // selected by `device_position` and is expressed in `base`.
    template <Kind K, typename T>
    void run_csr(rocsparse_handle     handle,
                 const csr_matrix<T>& a,
                 rocsparse_index_base base,
                 double               tolerance,
                 const boost_request& boost,
                 bool                 device_position,
                 csr_out<T>&          out)
    {
        descr_info di;
        ASSERT_NE(di.info, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(di.descr, base), rocsparse_status_success);
        if(K == Kind::ic)
        {
            ASSERT_EQ(rocsparse_set_mat_fill_mode(di.descr, rocsparse_fill_mode_lower),
                      rocsparse_status_success);
        }
        device_vector<rocsparse_int> drow, dcol;
        upload_indices(a.ptr, a.ind, base, drow, dcol);
        device_vector<T> dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        size_t size = 0;
        ASSERT_EQ((csr_buffer_size<K, T>(handle, a, di.descr, dval, drow, dcol, di.info, &size)),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ((csr_analysis<K, T>(handle, a, di.descr, dval, drow, dcol, di.info, buffer)),
                  rocsparse_status_success);
        if(tolerance >= 0.0)
        {
            ASSERT_EQ(set_tolerance<K>(handle, di.info, tolerance), rocsparse_status_success);
        }
        ASSERT_EQ(get_tolerance<K>(handle, di.info, &out.tolerance_back), rocsparse_status_success);

        const boost_params<T> params(boost);
        if constexpr(K == Kind::ilu)
        {
            ASSERT_EQ(params.template apply<false>(handle, di.info, boost),
                      rocsparse_status_success);
        }

        out.compute_status
            = csr_compute<K, T>(handle, a, di.descr, dval, drow, dcol, di.info, buffer);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values = to_host(dval);

        out.singular_pivot = -7;
        if(device_position)
        {
            device_vector<rocsparse_int> dpos{std::vector<rocsparse_int>{-7}};
            ASSERT_TRUE(dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device),
                      rocsparse_status_success);
            const auto st = singular_pivot<K>(handle, di.info, dpos.ptr);
            ASSERT_EQ(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host),
                      rocsparse_status_success);
            ASSERT_EQ(st, rocsparse_status_success);
            out.singular_pivot = to_host(dpos)[0];
        }
        else
        {
            ASSERT_EQ(singular_pivot<K>(handle, di.info, &out.singular_pivot),
                      rocsparse_status_success);
        }
        ASSERT_EQ(clear<K>(handle, di.info), rocsparse_status_success);
    }

    // ------------------------------------------------------------------
    // bsrilu0
    // ------------------------------------------------------------------
    template <typename T>
    struct bsr_out
    {
        rocsparse_status compute_status = rocsparse_status_internal_error;
        std::vector<T>   values;
        rocsparse_status zero_pivot_status = rocsparse_status_internal_error;
        rocsparse_int    zero_pivot        = -7;
    };

    template <typename T>
    void run_bsrilu0(rocsparse_handle     handle,
                     const bsr_matrix<T>& bsr,
                     rocsparse_index_base base,
                     const boost_request& boost,
                     bsr_out<T>&          out)
    {
        descr_info di;
        ASSERT_NE(di.info, nullptr);
        ASSERT_EQ(rocsparse_set_mat_index_base(di.descr, base), rocsparse_status_success);
        device_vector<rocsparse_int> drow, dcol;
        upload_indices(bsr.ptr, bsr.ind, base, drow, dcol);
        device_vector<T> dval{bsr.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        const rocsparse_int mb = bsr.mb, nnzb = bsr.nnzb(), bd = bsr.block_dim;
        size_t              size = 0;
        ASSERT_EQ(PRECOND_CALL(T,
                               bsrilu0_buffer_size,
                               handle,
                               bsr.dir,
                               mb,
                               nnzb,
                               di.descr,
                               dval.ptr,
                               drow.ptr,
                               dcol.ptr,
                               bd,
                               di.info,
                               &size),
                  rocsparse_status_success);
        device_vector<char> buffer{size ? size : size_t{1}};
        ASSERT_TRUE(buffer.ptr);
        ASSERT_EQ(PRECOND_CALL(T,
                               bsrilu0_analysis,
                               handle,
                               bsr.dir,
                               mb,
                               nnzb,
                               di.descr,
                               dval.ptr,
                               drow.ptr,
                               dcol.ptr,
                               bd,
                               di.info,
                               rocsparse_analysis_policy_reuse,
                               rocsparse_solve_policy_auto,
                               buffer.ptr),
                  rocsparse_status_success);

        const boost_params<T> params(boost);
        ASSERT_EQ(params.template apply<true>(handle, di.info, boost), rocsparse_status_success);
        out.compute_status = PRECOND_CALL(T,
                                          bsrilu0,
                                          handle,
                                          bsr.dir,
                                          mb,
                                          nnzb,
                                          di.descr,
                                          dval.ptr,
                                          drow.ptr,
                                          dcol.ptr,
                                          bd,
                                          di.info,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        out.values            = to_host(dval);
        out.zero_pivot_status = rocsparse_bsrilu0_zero_pivot(handle, di.info, &out.zero_pivot);
        ASSERT_EQ(rocsparse_bsrilu0_clear(handle, di.info), rocsparse_status_success);
    }

    // The factor of the block matrix `bsr` is the exact scalar ILU(0) of its expansion
    // (with the boost applied on the host).
    template <typename T>
    void expect_bsr_factor(const bsr_matrix<T>& bsr, const bsr_out<T>& out, const boost_request& b)
    {
        bsr_matrix<T> got = bsr;
        got.val           = out.values;
        expect_close(expand_bsr(got).val,
                     host_ilu0(expand_bsr(bsr), b.enable ? b.tol : -1.0, b.val));
    }
}

class PrecondLegacy : public HandleTest
{
};

template <typename KindTag>
class PrecondLegacyCsr : public KindFixture<KindTag>
{
};
TYPED_TEST_SUITE(PrecondLegacyCsr, IluIc);

// ---------------------------------------------------------------------------
// csrilu0 / csric0 tolerance and singular pivot
// ---------------------------------------------------------------------------

// The singular pivot is the first row whose pivot (|u_ii| for ILU(0), l_ii for IC(0)) is
// at most the tolerance, reported in the index base of the matrix (-1: none), and the
// factor is computed whether or not a pivot is flagged. The tolerances sit away from
// every pivot: one below all pivots, one between the smallest and the largest, one
// above all of them.
TYPED_TEST(PrecondLegacyCsr, tolerance_and_singular_pivot)
{
    constexpr Kind K        = TestFixture::K;
    const auto     handle   = this->handle;
    const auto     grid     = make_grid_matrix<double>(4, false);
    const auto     pivots   = pivots_of<K>(grid);
    const double   smallest = *std::min_element(pivots.begin(), pivots.end());
    const double   largest  = *std::max_element(pivots.begin(), pivots.end());
    ASSERT_LT(smallest, largest);
    const double middle = 0.5 * (smallest + largest);
    ASSERT_TRUE(clear_of_pivots(pivots, middle));
    ASSERT_GT(first_pivot_at_most(pivots, middle), 0);
    const std::vector<double> want
        = (K == Kind::ic) ? keep_lower_only(grid, host_ic0(grid)) : host_ilu0(grid);

    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(const bool device_position : {false, true})
        {
            for(const double pivot_tol : {-1.0, 0.5 * smallest, middle, 2.0 * largest})
            {
                SCOPED_TRACE(::testing::Message()
                             << "tolerance " << pivot_tol << " device_position " << device_position
                             << " base " << static_cast<int>(base));
                csr_out<double> out;
                run_csr<K, double>(
                    handle, grid, base, pivot_tol, boost_request{}, device_position, out);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                if(pivot_tol >= 0.0)
                {
                    EXPECT_EQ(out.tolerance_back, pivot_tol);
                    const int64_t row = first_pivot_at_most(pivots, pivot_tol);
                    EXPECT_EQ(out.singular_pivot, position_in_base(row, base));
                }
                else
                {
                    EXPECT_GE(out.tolerance_back, 0.0); // the default
                }
                expect_close(K == Kind::ic ? keep_lower_only(grid, out.values) : out.values,
                             want,
                             tolerance<double>(max_row_length(grid)));
            }
        }
    }
}

TYPED_TEST(PrecondLegacyCsr, tolerance_argument_checks)
{
    constexpr Kind K      = TestFixture::K;
    const auto     handle = this->handle;
    descr_info     di;
    double         tol      = 0.0;
    rocsparse_int  position = 0;
    EXPECT_EQ(get_tolerance<K>(nullptr, di.info, &tol), rocsparse_status_invalid_handle);
    EXPECT_EQ(get_tolerance<K>(handle, nullptr, &tol), rocsparse_status_invalid_pointer);
    EXPECT_EQ(get_tolerance<K>(handle, di.info, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(set_tolerance<K>(nullptr, di.info, 1.0), rocsparse_status_invalid_handle);
    // A negative tolerance is rejected (the status is not documented).
    EXPECT_NE(set_tolerance<K>(handle, di.info, -1.0), rocsparse_status_success);
    EXPECT_EQ(singular_pivot<K>(nullptr, di.info, &position), rocsparse_status_invalid_handle);
    EXPECT_EQ(singular_pivot<K>(handle, nullptr, &position), rocsparse_status_invalid_pointer);
    EXPECT_EQ(singular_pivot<K>(handle, di.info, nullptr), rocsparse_status_invalid_pointer);
    // Without an analysis there is no singular pivot.
    EXPECT_EQ(singular_pivot<K>(handle, di.info, &position), rocsparse_status_success);
    EXPECT_EQ(position, -1);
}

// ---------------------------------------------------------------------------
// csrilu0 numeric boost
// ---------------------------------------------------------------------------
TEST_F(PrecondLegacy, csrilu0_numeric_boost_types_and_pointer_modes)
{
    for_all_types([&](auto tag) {
        using T               = decltype(tag);
        const auto   grid     = make_grid_matrix<T>(4, true);
        const auto   pivots   = pivots_of<Kind::ilu>(grid);
        const double smallest = *std::min_element(pivots.begin(), pivots.end());
        const double largest  = *std::max_element(pivots.begin(), pivots.end());
        ASSERT_LT(smallest, largest);
        // Every pivot is at most 2 * largest (boosted to 7); only the first pivots of the
        // sweep are at most the middle value, later pivots follow from the boosted rows.
        const double                     middle   = 0.5 * (smallest + largest);
        const std::vector<boost_request> requests = {
            // enabled with a tolerance that catches every pivot
            {true, 2.0 * largest, 7.0, false, boost_api::standard},
            {true, 2.0 * largest, 7.0, true, boost_api::standard},
            // a tolerance in the middle of the pivot range
            {true, middle, 7.0, false, boost_api::standard},
            {true, middle, 7.0, true, boost_api::standard},
            // a tolerance below every pivot: nothing is replaced
            {true, 0.5 * smallest, 7.0, false, boost_api::standard},
            // disabled again after having been enabled
            {false, 2.0 * largest, 7.0, false, boost_api::standard},
        };
        for(const auto& req : requests)
        {
            SCOPED_TRACE(::testing::Message() << "enable " << req.enable << " tol " << req.tol
                                              << " device " << req.device);
            csr_out<T> out;
            run_csr<Kind::ilu, T>(handle, grid, rocsparse_index_base_zero, -1.0, req, false, out);
            ASSERT_EQ(out.compute_status, rocsparse_status_success);
            expect_close(out.values,
                         host_ilu0(grid, req.enable ? req.tol : -1.0, req.val),
                         tolerance<T>(max_row_length(grid)));
        }
        // Mixed precision entry points: a double tolerance with single precision values.
        if constexpr(is_single_precision<T>)
        {
            for(const bool device : {false, true})
            {
                csr_out<T>          out;
                const boost_request req{true, 2.0 * largest, 7.0, device, boost_api::mixed};
                run_csr<Kind::ilu, T>(
                    handle, grid, rocsparse_index_base_zero, -1.0, req, false, out);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                expect_close(out.values,
                             host_ilu0(grid, req.tol, req.val),
                             tolerance<T>(max_row_length(grid)));
            }
        }
    });
}

// One dense 3x3 block (a single block row) whose first pivot is `pivot0`, the rest of
// the block being well conditioned. Without the boost an exactly zero first pivot is
// reported as a zero pivot at the block row (in the index base of the matrix); with a
// boost tolerance of 1e-2 (boost value 2) the pivots that are at most 1e-2 in magnitude
// (here only the first, of either sign) are replaced by 2 with their sign and the
// factor matches the host reference, with no zero pivot reported.
TEST_F(PrecondLegacy, bsrilu0_numeric_boost_first_pivot)
{
    for(const auto base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for_all_types([&](auto tag) {
            using T = decltype(tag);
            for(const auto& [pivot0, boost] : {std::pair{0.0, false},
                                               std::pair{0.0, true},
                                               std::pair{1.0e-3, true},
                                               std::pair{-1.0e-3, true}})
            {
                SCOPED_TRACE(::testing::Message() << "pivot0 " << pivot0 << " boost " << boost);
                bsr_matrix<T> bsr;
                bsr.mb        = 1;
                bsr.block_dim = 3;
                bsr.dir       = rocsparse_direction_row;
                bsr.ptr       = {0, 1};
                bsr.ind       = {0};
                for(const double x : {pivot0, 1.0, 0.5, 1.0, 4.0, 1.0, 0.5, 1.0, 5.0})
                    bsr.val.push_back(make_value<T>(x));
                const boost_request req{boost, 1.0e-2, 2.0};
                bsr_out<T>          out;
                run_bsrilu0<T>(handle, bsr, base, req, out);
                ASSERT_EQ(out.compute_status, rocsparse_status_success);
                if(!boost && pivot0 == 0.0)
                {
                    EXPECT_EQ(out.zero_pivot_status, rocsparse_status_zero_pivot);
                    EXPECT_EQ(out.zero_pivot, base == rocsparse_index_base_one ? 1 : 0);
                    continue;
                }
                EXPECT_EQ(out.zero_pivot_status, rocsparse_status_success);
                EXPECT_EQ(out.zero_pivot, -1);
                // The boosted pivot (block (0,0), entry (0,0)) is exactly +-2.
                if(boost)
                {
                    const double sign = (pivot0 < 0.0) ? -1.0 : 1.0;
                    EXPECT_EQ(to_cd(out.values[0]), std::complex<double>(2.0 * sign, 0.0));
                }
                expect_bsr_factor(bsr, out, req);
            }
        });
    }
}

// ---------------------------------------------------------------------------
// numeric boost argument checks
// ---------------------------------------------------------------------------
namespace
{
    template <bool BSR, typename T>
    void check_boost_arguments(rocsparse_handle handle)
    {
        descr_info       di;
        const real_of<T> tol   = 1.0;
        const T          val   = make_value<T>(2.0);
        const auto       boost = [&](rocsparse_handle   h,
                               rocsparse_mat_info i,
                               int                enable,
                               const real_of<T>*  t,
                               const T*           v) {
            return BSR ? PRECOND_CALL(T, bsrilu0_numeric_boost, h, i, enable, t, v)
                             : PRECOND_CALL(T, csrilu0_numeric_boost, h, i, enable, t, v);
        };
        EXPECT_EQ(boost(nullptr, di.info, 1, &tol, &val), rocsparse_status_invalid_handle);
        EXPECT_EQ(boost(handle, nullptr, 1, &tol, &val), rocsparse_status_invalid_pointer);
        EXPECT_EQ(boost(handle, di.info, 1, nullptr, &val), rocsparse_status_invalid_pointer);
        EXPECT_EQ(boost(handle, di.info, 1, &tol, nullptr), rocsparse_status_invalid_pointer);
        EXPECT_EQ(boost(handle, di.info, 0, &tol, &val), rocsparse_status_success);
        EXPECT_EQ(boost(handle, di.info, 1, &tol, &val), rocsparse_status_success);
    }
}

TEST_F(PrecondLegacy, numeric_boost_argument_checks)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        check_boost_arguments<false, T>(handle);
        check_boost_arguments<true, T>(handle);
    });
}
