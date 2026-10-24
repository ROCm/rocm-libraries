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

// Small auxiliary pieces of the preconditioner area:
//  * the copy of a matrix info structure that holds the analysis of an iterative
//    triangular solve (the copy must solve like the original);
//  * rocsparse_copy_hyb_mat of every value type: into an empty destination, into a
//    destination that already holds a matrix of the same shape and into one of another
//    shape, each copy validated by the product with a vector against the host product;
//  * the internal dense transpose and back transpose helpers and
//    rocsparse_dnmat_transpose, every data movement checked against a host reference;
//  * the consistency checks of the singularity read-back: the batch counts and the index
//    types recorded by the exact and near singularity trackers must agree with the
//    requested batch.

#include "unit_test_device_precond_common.hpp"

#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_dense_transpose.hpp"
#include "rocsparse_dense_transpose_back.hpp"
#include "rocsparse_singular_info_t.hpp"
#include "rocsparse_singularity.hpp"

#include <algorithm>
#include <complex>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

using namespace precond_ut;

class PrecondAuxiliary : public HandleTest
{
};

TEST_F(PrecondAuxiliary, copy_mat_info_keeps_the_iterative_triangular_solve_analysis)
{
    // Lower triangular matrix, strictly diagonally dominant.
    const rocsparse_int        m = 24;
    std::vector<rocsparse_int> ptr{0};
    std::vector<rocsparse_int> ind;
    std::vector<float>         val;
    for(rocsparse_int i = 0; i < m; ++i)
    {
        for(rocsparse_int j = std::max<rocsparse_int>(0, i - 2); j <= i; ++j)
        {
            ind.push_back(j);
            val.push_back(j == i ? 4.0f : (j == i - 1 ? -1.0f : 0.5f));
        }
        ptr.push_back(static_cast<rocsparse_int>(ind.size()));
    }
    const rocsparse_int nnz = static_cast<rocsparse_int>(ind.size());

    device_vector<rocsparse_int> dptr{ptr};
    device_vector<rocsparse_int> dind{ind};
    device_vector<float>         dval{val};
    ASSERT_TRUE(dptr.ptr && dind.ptr && dval.ptr);

    // `original` owns the descriptor and the analysed info, `copy` the info copied into.
    descr_info original, copy;
    ASSERT_TRUE(original.descr && original.info && copy.info);
    const rocsparse_mat_descr descr = original.descr;
    const rocsparse_mat_info  src = original.info, dest = copy.info;
    ASSERT_EQ(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower),
              rocsparse_status_success);

    const auto trans       = rocsparse_operation_none;
    size_t     buffer_size = 0;
    ASSERT_EQ(rocsparse_scsritsv_buffer_size(
                  handle, trans, m, nnz, descr, dval.ptr, dptr.ptr, dind.ptr, src, &buffer_size),
              rocsparse_status_success);
    device_vector<char> buffer{buffer_size ? buffer_size : size_t{1}};
    ASSERT_TRUE(buffer.ptr);
    ASSERT_EQ(rocsparse_scsritsv_analysis(handle,
                                          trans,
                                          m,
                                          nnz,
                                          descr,
                                          dval.ptr,
                                          dptr.ptr,
                                          dind.ptr,
                                          src,
                                          rocsparse_analysis_policy_force,
                                          rocsparse_solve_policy_auto,
                                          buffer.ptr),
              rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // The first copy creates what the destination lacks, the second one overwrites it.
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    EXPECT_EQ(rocsparse_copy_mat_info(dest, src), rocsparse_status_success);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Host forward substitution of L y = alpha x.
    const float        alpha = 2.0f;
    std::vector<float> x(static_cast<size_t>(m));
    for(rocsparse_int i = 0; i < m; ++i)
        x[static_cast<size_t>(i)] = 1.0f + 0.25f * static_cast<float>(i % 5);
    std::vector<double> want(static_cast<size_t>(m), 0.0);
    for(rocsparse_int i = 0; i < m; ++i)
    {
        double sum  = alpha * static_cast<double>(x[static_cast<size_t>(i)]);
        double diag = 1.0;
        for(rocsparse_int k = ptr[static_cast<size_t>(i)]; k < ptr[static_cast<size_t>(i) + 1]; ++k)
        {
            const auto j = ind[static_cast<size_t>(k)];
            if(j == i)
                diag = val[static_cast<size_t>(k)];
            else
                sum -= static_cast<double>(val[static_cast<size_t>(k)])
                       * want[static_cast<size_t>(j)];
        }
        want[static_cast<size_t>(i)] = sum / diag;
    }

    for(const auto info : {src, dest})
    {
        device_vector<float> dx{x};
        device_vector<float> dy{std::vector<float>(static_cast<size_t>(m), 0.0f)};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        rocsparse_int      nmaxiter = 200;
        const float        tol      = 1.0e-6f;
        std::vector<float> history(200, 0.0f);
        ASSERT_EQ(rocsparse_scsritsv_solve(handle,
                                           &nmaxiter,
                                           &tol,
                                           history.data(),
                                           trans,
                                           m,
                                           nnz,
                                           &alpha,
                                           descr,
                                           dval.ptr,
                                           dptr.ptr,
                                           dind.ptr,
                                           info,
                                           dx.ptr,
                                           dy.ptr,
                                           rocsparse_solve_policy_auto,
                                           buffer.ptr),
                  rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        const auto got = to_host(dy);
        for(rocsparse_int i = 0; i < m; ++i)
        {
            EXPECT_NEAR(got[static_cast<size_t>(i)],
                        static_cast<float>(want[static_cast<size_t>(i)]),
                        1.0e-4f
                            * (1.0f + std::abs(static_cast<float>(want[static_cast<size_t>(i)]))))
                << "row " << i << (info == dest ? " (copy)" : " (original)");
        }
    }
}

namespace
{
    struct hyb_guard
    {
        rocsparse_hyb_mat hyb = nullptr;

        hyb_guard()
        {
            EXPECT_EQ(rocsparse_create_hyb_mat(&hyb), rocsparse_status_success);
        }

        ~hyb_guard()
        {
            if(hyb)
                (void)rocsparse_destroy_hyb_mat(hyb);
        }

        hyb_guard(const hyb_guard&) = delete;
        hyb_guard& operator=(const hyb_guard&) = delete;
    };

    template <typename T>
    rocsparse_status csr2hyb(rocsparse_handle        handle,
                             rocsparse_int           m,
                             rocsparse_mat_descr     descr,
                             const T*                val,
                             const rocsparse_int*    ptr,
                             const rocsparse_int*    ind,
                             rocsparse_hyb_mat       hyb,
                             rocsparse_int           width,
                             rocsparse_hyb_partition partition)
    {
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_scsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dcsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_ccsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
        else
            return rocsparse_zcsr2hyb(handle, m, m, descr, val, ptr, ind, hyb, width, partition);
    }

    template <typename T>
    rocsparse_status hybmv(
        rocsparse_handle handle, rocsparse_mat_descr descr, rocsparse_hyb_mat hyb, const T* x, T* y)
    {
        const T    alpha = make_value<T>(1.0);
        const T    beta  = make_value<T>(0.0);
        const auto op    = rocsparse_operation_none;
        if constexpr(std::is_same_v<T, float>)
            return rocsparse_shybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else if constexpr(std::is_same_v<T, double>)
            return rocsparse_dhybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else if constexpr(std::is_same_v<T, rocsparse_float_complex>)
            return rocsparse_chybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
        else
            return rocsparse_zhybmv(handle, op, &alpha, descr, hyb, x, &beta, y);
    }

    // y = A x on the host.
    template <typename T>
    std::vector<T> host_product(const csr_matrix<T>& a, const std::vector<T>& x)
    {
        std::vector<T> y(static_cast<size_t>(a.n));
        for(int32_t i = 0; i < a.n; ++i)
        {
            std::complex<double> sum = 0.0;
            for(int32_t p = a.ptr[i]; p < a.ptr[i + 1]; ++p)
                sum += to_cd(a.val[p]) * to_cd(x[a.ind[p]]);
            y[i] = make_value<T>(sum.real(), sum.imag());
        }
        return y;
    }

    // Checks the product of the HYB matrix with a vector.
    template <typename T>
    void expect_product(rocsparse_handle     handle,
                        rocsparse_mat_descr  descr,
                        rocsparse_hyb_mat    hyb,
                        const csr_matrix<T>& a,
                        const std::string&   what)
    {
        SCOPED_TRACE(what);
        std::vector<T> x(static_cast<size_t>(a.n));
        for(int32_t i = 0; i < a.n; ++i)
            x[i] = make_value<T>(1.0 + 0.125 * (i % 7), is_cplx<T> ? 0.25 * (i % 3) : 0.0);
        device_vector<T> dx{x};
        device_vector<T> dy{std::vector<T>(static_cast<size_t>(a.n), make_value<T>(-1.0))};
        ASSERT_TRUE(dx.ptr && dy.ptr);
        ASSERT_EQ(hybmv<T>(handle, descr, hyb, dx, dy), rocsparse_status_success);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        expect_close(to_host(dy), host_product(a, x), tolerance<T>(max_row_length(a)));
    }
}

TEST_F(PrecondAuxiliary, copy_hyb_mat_of_every_type_and_partition)
{
    for_all_types([&](auto tag) {
        using T = decltype(tag);
        descr_info d;
        ASSERT_NE(d.descr, nullptr);
        const csr_matrix<T>          a = make_grid_matrix<T>(4, true);
        device_vector<rocsparse_int> drow{as_rocsparse_int(a.ptr)};
        device_vector<rocsparse_int> dcol{as_rocsparse_int(a.ind)};
        device_vector<T>             dval{a.val};
        ASSERT_TRUE(drow.ptr && dcol.ptr && dval.ptr);

        struct partition_case
        {
            rocsparse_hyb_partition partition;
            rocsparse_int           width;
            const char*             name;
        };
        // A narrow user width leaves a COO part, auto picks its own width and max keeps
        // every entry in the ELL part.
        const partition_case cases[] = {{rocsparse_hyb_partition_user, 2, "user width 2"},
                                        {rocsparse_hyb_partition_auto, 0, "auto"},
                                        {rocsparse_hyb_partition_max, 0, "max"}};
        for(const auto& c : cases)
        {
            SCOPED_TRACE(c.name);
            hyb_guard src;
            ASSERT_NE(src.hyb, nullptr);
            ASSERT_EQ(
                csr2hyb<T>(handle, a.n, d.descr, dval, drow, dcol, src.hyb, c.width, c.partition),
                rocsparse_status_success);
            expect_product<T>(handle, d.descr, src.hyb, a, "source");

            // Into an empty destination.
            hyb_guard dest;
            ASSERT_NE(dest.hyb, nullptr);
            ASSERT_EQ(rocsparse_copy_hyb_mat(dest.hyb, src.hyb), rocsparse_status_success);
            expect_product<T>(handle, d.descr, dest.hyb, a, "copy");

            // Into a destination that already holds a matrix of the same shape.
            ASSERT_EQ(rocsparse_copy_hyb_mat(dest.hyb, src.hyb), rocsparse_status_success);
            expect_product<T>(handle, d.descr, dest.hyb, a, "second copy");
        }
    });
}

TEST_F(PrecondAuxiliary, copy_hyb_mat_rejects_a_destination_of_another_shape)
{
    descr_info d;
    ASSERT_NE(d.descr, nullptr);
    const csr_matrix<double>     small = make_grid_matrix<double>(3, true);
    const csr_matrix<double>     large = make_grid_matrix<double>(4, true);
    device_vector<rocsparse_int> srow{as_rocsparse_int(small.ptr)};
    device_vector<rocsparse_int> scol{as_rocsparse_int(small.ind)};
    device_vector<double>        sval{small.val};
    device_vector<rocsparse_int> lrow{as_rocsparse_int(large.ptr)};
    device_vector<rocsparse_int> lcol{as_rocsparse_int(large.ind)};
    device_vector<double>        lval{large.val};
    ASSERT_TRUE(srow.ptr && scol.ptr && sval.ptr && lrow.ptr && lcol.ptr && lval.ptr);

    hyb_guard hs;
    hyb_guard hl;
    ASSERT_EQ(
        csr2hyb<double>(
            handle, small.n, d.descr, sval, srow, scol, hs.hyb, 2, rocsparse_hyb_partition_user),
        rocsparse_status_success);
    ASSERT_EQ(
        csr2hyb<double>(
            handle, large.n, d.descr, lval, lrow, lcol, hl.hyb, 2, rocsparse_hyb_partition_user),
        rocsparse_status_success);
    // The destination holds a matrix of another order: the shapes must match.
    EXPECT_NE(rocsparse_copy_hyb_mat(hl.hyb, hs.hyb), rocsparse_status_success);
    // A failed copy leaves the destination usable.
    expect_product<double>(handle, d.descr, hl.hyb, large, "destination after the failed copy");
}

namespace
{
    // Small integer valued entry (the imaginary part only for complex types).
    template <typename T>
    T entry(int64_t i, int64_t j, int64_t b)
    {
        return make_value<T>(static_cast<double>((i * 7 + j * 3 + b * 5) % 11 - 5),
                             static_cast<double>((i * 5 + j * 2 + b) % 7 - 3));
    }

    template <typename T>
    T product(T a, T b)
    {
        return from_cd<T>(to_cd(a) * to_cd(b));
    }

    // The scalar of batch b: 1 or a small integer.
    template <typename T>
    T scalar_of(int64_t b)
    {
        return make_value<T>(static_cast<double>(b + 2), is_cplx<T> ? -1.0 : 0.0);
    }

    constexpr double exact = 1.0e-6;

    struct dim_pair
    {
        int64_t m, n;
    };

    constexpr dim_pair shapes[] = {{37, 53}, {32, 32}, {65, 1}, {1, 40}, {100, 33}, {9, 70}};
}

// The datatype utilities used by the dense helpers below that develop's datatype_utils
// tests do not reach: the 16-bit types and datatype_is_complex.
TEST_F(PrecondAuxiliary, datatype_utilities)
{
    EXPECT_EQ(rocsparse::get_datatype<_Float16>(), rocsparse_datatype_f16_r);
    EXPECT_EQ(rocsparse::get_datatype<rocsparse_bfloat16>(), rocsparse_datatype_bf16_r);
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_f16_r), 2u);
    EXPECT_EQ(rocsparse::datatype_sizeof(rocsparse_datatype_bf16_r), 2u);
    for(const auto t : {rocsparse_datatype_i8_r,
                        rocsparse_datatype_u8_r,
                        rocsparse_datatype_i32_r,
                        rocsparse_datatype_u32_r,
                        rocsparse_datatype_f16_r,
                        rocsparse_datatype_bf16_r,
                        rocsparse_datatype_f32_r,
                        rocsparse_datatype_f64_r})
        EXPECT_FALSE(rocsparse::datatype_is_complex(t)) << t;
    EXPECT_TRUE(rocsparse::datatype_is_complex(rocsparse_datatype_f32_c));
    EXPECT_TRUE(rocsparse::datatype_is_complex(rocsparse_datatype_f64_c));
}

TEST_F(PrecondAuxiliary, dense_transpose_single_matrix_entry_points)
{
    for_all_types([&](auto tag) {
        using T              = decltype(tag);
        const auto     dtype = rocsparse::get_datatype<T>();
        const int64_t  m = 41, n = 29, lda = m + 1, ldb = n + 4;
        std::vector<T> a(static_cast<size_t>(lda * n));
        for(int64_t j = 0; j < n; ++j)
            for(int64_t i = 0; i < lda; ++i)
                a[static_cast<size_t>(i + lda * j)] = entry<T>(i, j, 2);
        const std::vector<T> sentinel(static_cast<size_t>(ldb * m),
                                      make_value<T>(-99.0, is_cplx<T> ? 98.0 : 0.0));
        const T              alpha = scalar_of<T>(3);

        auto expected = [&](T scale) {
            std::vector<T> want = sentinel;
            for(int64_t i = 0; i < m; ++i)
                for(int64_t j = 0; j < n; ++j)
                    want[static_cast<size_t>(j + ldb * i)]
                        = product(scale, a[static_cast<size_t>(i + lda * j)]);
            return want;
        };

        device_vector<T> da{a};
        ASSERT_TRUE(da.ptr);

        // No scalar: a plain transposition.
        {
            device_vector<T> db{sentinel};
            ASSERT_TRUE(db.ptr);
            ASSERT_EQ(
                rocsparse::dense_transpose(handle, m, n, dtype, da.ptr, lda, dtype, db.ptr, ldb),
                rocsparse_status_success);
            expect_close(to_host(db), expected(make_value<T>(1.0)), exact);
        }

        // Scalar in host or device memory.
        for(const bool host_mode : {true, false})
        {
            SCOPED_TRACE(host_mode ? "host scalar" : "device scalar");
            device_vector<T> db{sentinel};
            device_vector<T> dalpha{std::vector<T>{alpha}};
            ASSERT_TRUE(db.ptr && dalpha.ptr);
            ASSERT_EQ(rocsparse::dense_transpose(handle,
                                                 host_mode ? rocsparse_pointer_mode_host
                                                           : rocsparse_pointer_mode_device,
                                                 m,
                                                 n,
                                                 dtype,
                                                 host_mode ? static_cast<const void*>(&alpha)
                                                           : static_cast<const void*>(dalpha.ptr),
                                                 dtype,
                                                 da.ptr,
                                                 lda,
                                                 dtype,
                                                 db.ptr,
                                                 ldb),
                      rocsparse_status_success);
            expect_close(to_host(db), expected(alpha), exact);
        }

        // Typed entry point (the scalar lives on the device).
        {
            device_vector<T> db{sentinel};
            device_vector<T> dalpha{std::vector<T>{alpha}};
            ASSERT_TRUE(db.ptr && dalpha.ptr);
            ASSERT_EQ(
                rocsparse::dense_transpose(handle, m, n, dalpha.ptr, da.ptr, lda, db.ptr, ldb),
                rocsparse_status_success);
            expect_close(to_host(db), expected(alpha), exact);
        }
    });

    // Types the kernels do not handle are rejected before any launch.
    device_vector<float> x{std::vector<float>(64, 1.0f)};
    device_vector<float> y{std::vector<float>(64, 0.0f)};
    ASSERT_TRUE(x.ptr && y.ptr);
    for(const auto bad : {rocsparse_datatype_i8_r,
                          rocsparse_datatype_u8_r,
                          rocsparse_datatype_i32_r,
                          rocsparse_datatype_f16_r,
                          rocsparse_datatype_bf16_r})
    {
        EXPECT_NE(rocsparse::dense_transpose(handle, 8, 8, bad, x.ptr, 8, bad, y.ptr, 8),
                  rocsparse_status_success);
        EXPECT_NE(rocsparse::dense_transpose_back(handle, 8, 8, bad, x.ptr, 8, bad, y.ptr, 8),
                  rocsparse_status_success);
    }
}

TEST_F(PrecondAuxiliary, dense_transpose_back_of_batches)
{
    for_all_types([&](auto tag) {
        using T          = decltype(tag);
        const auto dtype = rocsparse::get_datatype<T>();
        for(const auto& s : shapes)
        {
            for(const int64_t batch : {int64_t{1}, int64_t{3}})
            {
                SCOPED_TRACE("m " + std::to_string(s.m) + " n " + std::to_string(s.n) + " batch "
                             + std::to_string(batch));
                // A holds the transposed matrix (n x m), B receives the matrix (m x n).
                const int64_t  lda = s.n + 2, ldb = s.m + 3;
                const int64_t  a_stride = lda * s.m, b_stride = ldb * s.n;
                std::vector<T> a(static_cast<size_t>(batch * a_stride));
                for(int64_t b = 0; b < batch; ++b)
                    for(int64_t j = 0; j < s.m; ++j)
                        for(int64_t i = 0; i < lda; ++i)
                            a[static_cast<size_t>(b * a_stride + i + lda * j)] = entry<T>(i, j, b);
                const std::vector<T> sentinel(static_cast<size_t>(batch * b_stride),
                                              make_value<T>(-99.0, is_cplx<T> ? 98.0 : 0.0));
                device_vector<T>     da{a};
                device_vector<T>     db{sentinel};
                ASSERT_TRUE(da.ptr && db.ptr);

                if(batch == 1)
                {
                    ASSERT_EQ(rocsparse::dense_transpose_back(
                                  handle, s.m, s.n, dtype, da.ptr, lda, dtype, db.ptr, ldb),
                              rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(rocsparse::dense_transpose_back_strided_batched(handle,
                                                                              batch,
                                                                              s.m,
                                                                              s.n,
                                                                              dtype,
                                                                              da.ptr,
                                                                              lda,
                                                                              a_stride,
                                                                              dtype,
                                                                              db.ptr,
                                                                              ldb,
                                                                              b_stride),
                              rocsparse_status_success);
                }
                ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                std::vector<T> want = sentinel;
                for(int64_t b = 0; b < batch; ++b)
                    for(int64_t r = 0; r < s.m; ++r)
                        for(int64_t c = 0; c < s.n; ++c)
                            want[static_cast<size_t>(b * b_stride + r + ldb * c)]
                                = a[static_cast<size_t>(b * a_stride + c + lda * r)];
                expect_close(to_host(db), want, exact);
            }
        }

        // The typed entry point.
        const int64_t  m = 21, n = 34;
        std::vector<T> a(static_cast<size_t>(n * m));
        for(int64_t r = 0; r < m; ++r)
            for(int64_t c = 0; c < n; ++c)
                a[static_cast<size_t>(c + n * r)] = entry<T>(c, r, 4);
        device_vector<T> da{a};
        device_vector<T> db{std::vector<T>(static_cast<size_t>(m * n))};
        ASSERT_TRUE(da.ptr && db.ptr);
        ASSERT_EQ(rocsparse::dense_transpose_back(handle, m, n, da.ptr, n, db.ptr, m),
                  rocsparse_status_success);
        std::vector<T> want(static_cast<size_t>(m * n));
        for(int64_t r = 0; r < m; ++r)
            for(int64_t c = 0; c < n; ++c)
                want[static_cast<size_t>(r + m * c)] = a[static_cast<size_t>(c + n * r)];
        expect_close(to_host(db), want, exact);
    });
}

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
namespace
{
    // A dense matrix of the public interface with its device storage.
    template <typename T>
    struct dn_matrix
    {
        int64_t               rows, cols, ld, batch_stride, batch_count;
        rocsparse_order       order;
        std::vector<T>        host;
        device_vector<T>      device;
        rocsparse_dnmat_descr descr = nullptr;

        dn_matrix(int64_t               r,
                  int64_t               c,
                  rocsparse_order       o,
                  int64_t               batch,
                  const std::vector<T>* values = nullptr)
            : rows(r)
            , cols(c)
            , batch_count(batch)
            , order(o)
        {
            ld           = (o == rocsparse_order_column ? r : c) + 2;
            batch_stride = ld * (o == rocsparse_order_column ? c : r);
            host.assign(static_cast<size_t>(batch * batch_stride), make_value<T>(-99.0));
            if(values)
                host = *values;
            upload(device, host);
            EXPECT_EQ(rocsparse_create_dnmat_descr(
                          &descr, rows, cols, ld, device.ptr, rocsparse::get_datatype<T>(), order),
                      rocsparse_status_success);
            if(batch > 1)
                EXPECT_EQ(rocsparse_dnmat_set_strided_batch(
                              descr, static_cast<rocsparse_int>(batch), batch_stride),
                          rocsparse_status_success);
        }

        ~dn_matrix()
        {
            if(descr)
                (void)rocsparse_destroy_dnmat_descr(descr);
        }

        dn_matrix(const dn_matrix&) = delete;
        dn_matrix& operator=(const dn_matrix&) = delete;

        size_t position(int64_t b, int64_t i, int64_t j) const
        {
            return static_cast<size_t>(
                b * batch_stride + (order == rocsparse_order_column ? i + ld * j : i * ld + j));
        }
    };

    // The vector of the scalars of a batch.
    template <typename T>
    struct dn_scalars
    {
        std::vector<T>        host;
        device_vector<T>      device;
        rocsparse_dnvec_descr descr = nullptr;

        dn_scalars(int64_t batch, rocsparse_datatype type)
        {
            host.resize(static_cast<size_t>(batch));
            for(int64_t b = 0; b < batch; ++b)
                host[static_cast<size_t>(b)] = scalar_of<T>(b);
            upload(device, host);
            EXPECT_EQ(rocsparse_create_dnvec_descr(&descr, batch, device.ptr, type),
                      rocsparse_status_success);
            if(batch > 1)
                EXPECT_EQ(
                    rocsparse_dnvec_set_strided_batch(descr, static_cast<rocsparse_int>(batch), 1),
                    rocsparse_status_success);
        }

        ~dn_scalars()
        {
            if(descr)
                (void)rocsparse_destroy_dnvec_descr(descr);
        }

        dn_scalars(const dn_scalars&) = delete;
        dn_scalars& operator=(const dn_scalars&) = delete;
    };

    template <typename T>
    std::vector<T> fill_matrix(const dn_matrix<T>& m)
    {
        std::vector<T> values = m.host;
        for(int64_t b = 0; b < m.batch_count; ++b)
            for(int64_t i = 0; i < m.rows; ++i)
                for(int64_t j = 0; j < m.cols; ++j)
                    values[m.position(b, i, j)] = entry<T>(i, j, b);
        return values;
    }
}

TEST_F(PrecondAuxiliary, dnmat_transpose_of_every_order_and_type)
{
    for_all_types([&](auto tag) {
        using T          = decltype(tag);
        const auto dtype = rocsparse::get_datatype<T>();
        for(const auto source_order : {rocsparse_order_column, rocsparse_order_row})
        {
            for(const auto target_order : {rocsparse_order_column, rocsparse_order_row})
            {
                for(const int64_t batch : {int64_t{1}, int64_t{3}})
                {
                    for(const bool with_alpha : {false, true})
                    {
                        SCOPED_TRACE(
                            std::string(source_order == rocsparse_order_row ? "row" : "col")
                            + " -> " + (target_order == rocsparse_order_row ? "row" : "col")
                            + " batch " + std::to_string(batch)
                            + (with_alpha ? " alpha" : " no alpha"));
                        const int64_t m = 37, n = 53;

                        dn_matrix<T>         probe(m, n, source_order, batch);
                        const std::vector<T> filled = fill_matrix(probe);
                        dn_matrix<T>         src(m, n, source_order, batch, &filled);
                        dn_matrix<T>         target(n, m, target_order, batch);
                        dn_scalars<T>        alpha(batch, dtype);
                        rocsparse_error*     p_error = nullptr;

                        ASSERT_EQ(rocsparse_dnmat_transpose(handle,
                                                            with_alpha ? alpha.descr : nullptr,
                                                            src.descr,
                                                            target.descr,
                                                            p_error),
                                  rocsparse_status_success);
                        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

                        std::vector<T> got  = to_host(target.device);
                        std::vector<T> want = target.host;
                        for(int64_t b = 0; b < batch; ++b)
                            for(int64_t i = 0; i < m; ++i)
                                for(int64_t j = 0; j < n; ++j)
                                    want[target.position(b, j, i)]
                                        = with_alpha ? product(alpha.host[static_cast<size_t>(b)],
                                                               filled[src.position(b, i, j)])
                                                     : filled[src.position(b, i, j)];
                        expect_close(got, want, exact);
                    }
                }
            }
        }
    });
}
#endif

namespace
{
    struct trackers
    {
        rocsparse::singular_info_t near4, near3, exact4, exact3;
        rocsparse::singular_info_t exact4_wide, near4_wide;

        explicit trackers(hipStream_t stream)
        {
            EXPECT_EQ(near4.create_singular_pivot_async(4, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(near3.create_singular_pivot_async(3, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact4.create_singular_pivot_async(4, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact3.create_singular_pivot_async(3, rocsparse_indextype_i32, stream),
                      rocsparse_status_success);
            EXPECT_EQ(exact4_wide.create_singular_pivot_async(4, rocsparse_indextype_i64, stream),
                      rocsparse_status_success);
            EXPECT_EQ(near4_wide.create_singular_pivot_async(4, rocsparse_indextype_i64, stream),
                      rocsparse_status_success);
        }

        // Writes the markers of the 4-entry trackers. Entry 0: exact 6 before near 11,
        // entry 1: near 3 before exact 9, entry 2: exact 8 only, entry 3: nothing.
        void plant(hipStream_t stream)
        {
            constexpr int32_t n32 = std::numeric_limits<int32_t>::max();
            constexpr int64_t n64 = std::numeric_limits<int64_t>::max();
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
            set(exact4, std::vector<int32_t>{6, 9, 8, n32});
            set(near4, std::vector<int32_t>{11, 3, n32, n32});
            set(exact4_wide, std::vector<int64_t>{6, 9, 8, n64});
            set(near4_wide, std::vector<int64_t>{11, 3, n64, n64});
        }

        template <typename I>
        static void set(rocsparse::singular_info_t& t, const std::vector<I>& markers)
        {
            EXPECT_EQ(hipMemcpy(t.get_position(),
                                markers.data(),
                                markers.size() * sizeof(I),
                                hipMemcpyHostToDevice),
                      hipSuccess);
        }
    };
}

TEST_F(PrecondAuxiliary, position_read_back_checks_the_trackers)
{
    trackers t(handle->stream);
    int32_t  host_position[4]   = {-5, -5, -5, -5};
    int64_t  host_position64[4] = {-5, -5, -5, -5};
    auto     read               = [&](int64_t                           batch,
                    const rocsparse::singular_info_t* exact,
                    const rocsparse::singular_info_t* near,
                    bool                              wide) {
        return rocsparse::singularity_get_position_async(
            handle,
            batch,
            nullptr,
            exact,
            near,
            rocsparse_pointer_mode_host,
            wide ? rocsparse_indextype_i64 : rocsparse_indextype_i32,
            wide ? static_cast<void*>(host_position64) : static_cast<void*>(host_position));
    };

    // A tracker of another batch than the requested one.
    EXPECT_NE(read(4, nullptr, &t.near3, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, nullptr, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, &t.near4, false), rocsparse_status_success);

    // Trackers that disagree on the type of the positions.
    EXPECT_NE(read(4, &t.exact4_wide, &t.near4, false), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4, &t.near4_wide, true), rocsparse_status_success);

    // Consistent trackers are read back: fresh ones hold no position, planted ones the
    // smaller of the exact and the near marker.
    const auto read_both = [&](const std::vector<int64_t>& want) {
        ASSERT_EQ(read(4, &t.exact4, &t.near4, false), rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
        ASSERT_EQ(read(4, &t.exact4_wide, &t.near4_wide, true), rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
        for(size_t i = 0; i < want.size(); ++i)
        {
            EXPECT_EQ(host_position[i], want[i]) << "entry " << i;
            EXPECT_EQ(host_position64[i], want[i]) << "entry " << i;
        }
    };
    read_both({-1, -1, -1, -1});
    t.plant(handle->stream);
    read_both({6, 3, 8, -1});
}

TEST_F(PrecondAuxiliary, singularity_read_back_checks_the_trackers)
{
    trackers              t(handle->stream);
    const auto            sentinel            = static_cast<rocsparse_singularity>(-123);
    rocsparse_singularity host_singularity[4] = {sentinel, sentinel, sentinel, sentinel};
    auto                  read                = [&](int64_t                           batch,
                    const rocsparse::singular_info_t* exact,
                    const rocsparse::singular_info_t* near) {
        return rocsparse::singularity_get_async(
            handle, batch, nullptr, exact, near, rocsparse_pointer_mode_host, host_singularity);
    };

    EXPECT_NE(read(4, nullptr, &t.near3), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, nullptr), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact3, &t.near4), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4_wide, &t.near4), rocsparse_status_success);
    EXPECT_NE(read(4, &t.exact4, &t.near4_wide), rocsparse_status_success);

    const auto check = [&](const rocsparse::singular_info_t&         exact,
                           const rocsparse::singular_info_t&         near,
                           const std::vector<rocsparse_singularity>& want) {
        std::fill(std::begin(host_singularity), std::end(host_singularity), sentinel);
        ASSERT_EQ(read(4, &exact, &near), rocsparse_status_success);
        ASSERT_EQ(hipStreamSynchronize(handle->stream), hipSuccess);
        for(size_t i = 0; i < want.size(); ++i)
            EXPECT_EQ(host_singularity[i], want[i]) << "entry " << i;
    };
    const auto none  = rocsparse_singularity_none;
    const auto exact = rocsparse_singularity_numeric_exact;
    const auto near  = rocsparse_singularity_numeric_near;
    check(t.exact4, t.near4, {none, none, none, none});
    t.plant(handle->stream);
    check(t.exact4, t.near4, {exact, near, exact, none});
    check(t.exact4_wide, t.near4_wide, {exact, near, exact, none});
}
