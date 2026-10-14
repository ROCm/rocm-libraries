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

//
// Regression tests for the grid-stride loops of the csrmm row_split and coomm
// kernels (AISPARSE-668, AISPARSE-696).
//
// FOCUS. These kernels put the dense column panel on grid.y, the batch on grid.z
// (or on grid.y for the csrmm nt and coomm atomic nn kernels) and, for the
// coo_segmented final reduction, the dense column on grid.x. The launchers clamp
// each extent with rocsparse::get_grid_size_{x,y,z}, which reads
// handle->properties.maxGridSize[0..2], and every kernel grid-strides over its
// axis, so a launch with more panels, batches or columns than the limit still
// covers all of them. The real limits are far above these problem sizes, so the
// tests shrink maxGridSize to 1, 2 and 3 blocks and run a small problem through
// the same code path. Every case runs the unclamped control and a list of
// (x, y, z) limits that shrink one or several axes at once, so the batch loop
// nested in the panel loop takes more than one step on both axes:
//
//   kernel                                  algorithm, operations        panels
//   csrmmnn_row_split_shared_kernel         csr_row_split, A n, n <= 32  4 at n = 30
//   csrmmnn_row_split_kernel                csr_row_split, A n, n > 32   10 + 3 at n = 83
//   csrmmnt_row_split_* (batch on grid.y)   csr_row_split, A n, B t      n = 16, 30, 70
//   csrmmtn_row_split_kernel                csr_row_split, A t, B n      8 at n = 30
//   csrmmtt_row_split_kernel                csr_row_split, A t, B t      8 at n = 30
//   coommnn_atomic_* (batch on grid.y)      coo_atomic, A n              n = 43
//   coommtn_atomic_main                     coo_atomic, A t              13 at n = 13
//   coommnn_segmented_main_kernel           coo_segmented, A n           5 at n = 43
//   coommnn_general_block_reduce (grid.x)   coo_segmented, A n           43 columns
//   coommnn_segmented_atomic                coo_segmented_atomic, A n    5 at n = 43
//
// Every panel count exceeds the limits 1 and 2, and at least one of the three
// limits does not divide it, so the last stride step is a partial one. A kernel
// whose loop does not stride leaves the columns or batches past the first
// `limit` ones unwritten.
//
// ORDER, OPERATIONS, TYPES. B and C run in column and in row order, with A, B
// each non-transposed, transposed and (for complex values) conjugate-transposed
// in the combinations the algorithms support. Real values run in float and
// double, complex values in float and double complex, so conjugation of A and of
// B is checked against a host reference.
//
// BATCHES. Every case also runs with A, B and C strided-batched over three
// batches with different values, so the batch loop has to pick the right
// per-batch pointers for every panel.
//
// INDEX TYPES. CSR runs with (row offset, column index) types (int32, int32),
// (int64, int32) and (int64, int64); COO runs with int32 and int64 indices. The
// kernels are instantiated per index type, so each one has its own copy of the
// loops.
//
// 16-BIT OUTPUT. f16 and bf16 C use a 16-bit atomic add whose pairing of halves
// into 32-bit words depends on the alignment of the C pointer. With an odd
// batch_stride_C every odd batch starts only 2-byte aligned, so the tests that
// run f16/bf16 output through the atomic kernels (csrmm A^T, coomm atomic A^T and
// A, coomm segmented atomic) use an odd batch stride and compare every entry of
// every batch exactly.
//
// SEGMENTED BLOCKS. The coo_segmented cases use A with 300 rows (about 2,500
// nonzeros), which spans three blocks of 1,024 nonzeros. With a single block
// every row completes in the main kernel and coommnn_general_block_reduce has
// nothing to add, so a defect in it, in its grid.x loop or in its reads of the
// reduction buffers would go unnoticed.
//
// ARITHMETIC. All values are small integers, so the host reference is exact in
// either precision and does not depend on the order in which atomics land. The
// 16-bit tests use values small enough that every partial sum is exactly
// representable in bf16.
//
// TARGET: rocsparse-unit-test-device. The tests drive the public rocsparse_spmm
// entry point and need a real device and the complete handle type. The largest
// allocation is a few kilobytes, so there is no device-memory guard.
//
#include "unit_test_utils.hpp"

// Internal handle definition: the complete _rocsparse_handle type is required to
// reach handle->properties.
#include "rocsparse_handle.hpp"

#include "rocsparse.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace rocsparse_ut
{
    template <>
    inline rocsparse_datatype dt_of<_Float16>()
    {
        return rocsparse_datatype_f16_r;
    }

    template <>
    inline rocsparse_datatype dt_of<rocsparse_bfloat16>()
    {
        return rocsparse_datatype_bf16_r;
    }
}

using namespace rocsparse_ut;

namespace
{
    // Shrinks handle->properties.maxGridSize[axis] for the lifetime of the
    // object; a limit of 0 leaves the axis alone. Generalises
    // rocsparse_ut::ScopedMaxGridSizeX (unit_test_grid_clamp.hpp) to all axes.
    struct ScopedMaxGridSize
    {
        rocsparse_handle handle;
        int              saved[3];

        ScopedMaxGridSize(rocsparse_handle h, const int (&limit)[3])
            : handle(h)
        {
            for(int axis = 0; axis < 3; ++axis)
            {
                saved[axis] = h->properties.maxGridSize[axis];
                if(limit[axis] > 0)
                {
                    h->properties.maxGridSize[axis] = limit[axis];
                }
            }
        }

        ~ScopedMaxGridSize()
        {
            for(int axis = 0; axis < 3; ++axis)
            {
                handle->properties.maxGridSize[axis] = saved[axis];
            }
        }

        ScopedMaxGridSize(const ScopedMaxGridSize&) = delete;

        ScopedMaxGridSize& operator=(const ScopedMaxGridSize&) = delete;
    };

    // maxGridSize[0], [1], [2]; 0 keeps the real limit, so {0, 0, 0} is the
    // unclamped control.
    struct Limits
    {
        int v[3];
    };

    // Control, grid.y alone, grid.z alone, and both together. Both together is
    // what makes the batch loop nested in the panel loop take several steps on
    // each axis.
    const Limits yz_limits[] = {{{0, 0, 0}},
                                {{0, 1, 0}},
                                {{0, 2, 0}},
                                {{0, 3, 0}},
                                {{0, 0, 1}},
                                {{0, 0, 2}},
                                {{0, 1, 1}},
                                {{0, 2, 2}},
                                {{0, 3, 1}}};

    // The same plus grid.x, which only the coo_segmented final reduction reads.
    const Limits xyz_limits[] = {{{0, 0, 0}},
                                 {{0, 1, 0}},
                                 {{0, 2, 0}},
                                 {{0, 3, 0}},
                                 {{0, 0, 1}},
                                 {{0, 0, 2}},
                                 {{0, 1, 1}},
                                 {{0, 2, 2}},
                                 {{0, 3, 1}},
                                 {{1, 0, 0}},
                                 {{2, 0, 0}},
                                 {{3, 0, 0}},
                                 {{1, 1, 1}},
                                 {{2, 2, 2}},
                                 {{3, 3, 1}}};

    // The 16-bit tests are about alignment, not about the clamps; run the
    // control and one clamped configuration.
    const Limits narrow_limits[] = {{{0, 0, 0}}, {{0, 1, 1}}};

    constexpr int32_t batch_counts[] = {1, 3};

    enum class Format
    {
        csr,
        coo
    };

    struct Case
    {
        Format              format;
        rocsparse_spmm_alg  alg;
        rocsparse_operation trans_A;
        rocsparse_operation trans_B;
        int32_t             n;
        // Storage order of B and C.
        rocsparse_order order = rocsparse_order_column;
        // Also clamp grid.x; only coo_segmented reads it.
        bool clamp_x = false;
        // Size of A. The coo_segmented cases need more than 1024 nonzeros, so
        // that A spans several blocks and the final reduction has work to do.
        int32_t m = 37;
        int32_t k = 29;
    };

    // ----- value helpers: one code path for real, complex and 16-bit types ---
    template <typename X>
    struct is_complex_value : std::false_type
    {
    };
    template <>
    struct is_complex_value<rocsparse_float_complex> : std::true_type
    {
    };
    template <>
    struct is_complex_value<rocsparse_double_complex> : std::true_type
    {
    };

    template <typename X>
    X make_val(int re, int im)
    {
        if constexpr(is_complex_value<X>::value)
        {
            return X(re, im);
        }
        else
        {
            return static_cast<X>(re);
        }
    }

    template <>
    rocsparse_bfloat16 make_val<rocsparse_bfloat16>(int re, int)
    {
        return rocsparse_bfloat16(static_cast<float>(re));
    }

    template <typename X>
    std::complex<double> to_ref(X v)
    {
        if constexpr(is_complex_value<X>::value)
        {
            return {static_cast<double>(std::real(v)), static_cast<double>(std::imag(v))};
        }
        else
        {
            return {static_cast<double>(static_cast<float>(v)), 0.0};
        }
    }

    // Fixed sparsity pattern, sorted by row and then column, about 2 in 7 dense.
    struct Pattern
    {
        std::vector<int32_t> row_ptr;
        std::vector<int32_t> row_ind;
        std::vector<int32_t> col_ind;
    };

    Pattern make_pattern(int32_t mat_m, int32_t mat_k)
    {
        Pattern p;
        p.row_ptr.push_back(0);
        for(int32_t i = 0; i < mat_m; ++i)
        {
            for(int32_t j = 0; j < mat_k; ++j)
            {
                if((3 * i + 5 * j) % 7 < 2)
                {
                    p.row_ind.push_back(i);
                    p.col_ind.push_back(j);
                }
            }
            p.row_ptr.push_back(static_cast<int32_t>(p.col_ind.size()));
        }
        return p;
    }

    struct SpMatDescr
    {
        rocsparse_spmat_descr d = nullptr;
        ~SpMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_spmat_descr(d);
        }
    };

    struct DnMatDescr
    {
        rocsparse_dnmat_descr d = nullptr;
        ~DnMatDescr()
        {
            if(d)
                (void)rocsparse_destroy_dnmat_descr(d);
        }
    };

    // T is the compute type (alpha, beta), AB the type of A and B, TC the type
    // of C. odd_stride_C makes batch_stride_C odd (one element of padding when
    // the dense C is an even number of elements).
    template <typename T, typename AB, typename TC, typename I, typename J>
    void check_case(rocsparse_handle handle,
                    const Case&      c,
                    const Limits*    limits,
                    size_t           num_limits,
                    bool             odd_stride_C)
    {
        const Pattern p     = make_pattern(c.m, c.k);
        const int32_t mat_m = c.m;
        const int32_t mat_k = c.k;
        const int64_t nnz   = static_cast<int64_t>(p.col_ind.size());

        // 16-bit output: keep every partial sum an exactly representable integer.
        constexpr bool narrow = (sizeof(TC) == 2);

        // op(A) is mo x ko, op(B) is ko x n and C is mo x n. B is stored
        // transposed when trans_B is not none.
        const bool    trans_A  = (c.trans_A != rocsparse_operation_none);
        const bool    trans_B  = (c.trans_B != rocsparse_operation_none);
        const bool    conj_A   = (c.trans_A == rocsparse_operation_conjugate_transpose);
        const bool    conj_B   = (c.trans_B == rocsparse_operation_conjugate_transpose);
        const bool    row_ord  = (c.order == rocsparse_order_row);
        const int32_t n        = c.n;
        const int32_t mo       = trans_A ? mat_k : mat_m;
        const int32_t ko       = trans_A ? mat_m : mat_k;
        const int64_t rows_B   = trans_B ? n : ko;
        const int64_t cols_B   = trans_B ? ko : n;
        const int64_t ldb      = row_ord ? cols_B : rows_B;
        const int64_t ldc      = row_ord ? n : mo;
        const int64_t stride_B = rows_B * cols_B;
        int64_t       stride_C = static_cast<int64_t>(mo) * n;
        if(odd_stride_C && (stride_C % 2 == 0))
        {
            ++stride_C;
        }

        const T alpha = narrow ? make_val<T>(1, 0) : make_val<T>(2, 1);
        const T beta  = narrow ? make_val<T>(1, 0) : make_val<T>(-1, 0);

        // Element (r, col) of B and (row, col) of C in the chosen order.
        auto b_index = [&](int64_t stored_row, int64_t stored_col) {
            return row_ord ? stored_row * ldb + stored_col : stored_row + stored_col * ldb;
        };
        auto c_index
            = [&](int64_t row, int64_t col) { return row_ord ? row * ldc + col : row + col * ldc; };

        for(int32_t batch_count : batch_counts)
        {
            std::vector<I>  row_ptr;
            std::vector<J>  row_ind;
            std::vector<J>  col_ind;
            std::vector<AB> val;
            for(int32_t b = 0; b < batch_count; ++b)
            {
                row_ptr.insert(row_ptr.end(), p.row_ptr.begin(), p.row_ptr.end());
                row_ind.insert(row_ind.end(), p.row_ind.begin(), p.row_ind.end());
                col_ind.insert(col_ind.end(), p.col_ind.begin(), p.col_ind.end());
                for(int64_t q = 0; q < nnz; ++q)
                {
                    val.push_back(make_val<AB>(narrow ? 1 + (q + 2 * b) % 2 : 1 + (q + 2 * b) % 5,
                                               static_cast<int>((q + b) % 3) - 1));
                }
            }

            std::vector<AB> dense_B(stride_B * batch_count);
            for(size_t i = 0; i < dense_B.size(); ++i)
            {
                dense_B[i] = make_val<AB>(narrow ? static_cast<int>(i % 3) - 1
                                                 : static_cast<int>(i % 7) - 3,
                                          static_cast<int>(i % 5) - 2);
            }

            std::vector<TC> dense_C(stride_C * batch_count);
            for(size_t i = 0; i < dense_C.size(); ++i)
            {
                dense_C[i] = make_val<TC>(static_cast<int>((i * 3) % 5) - 2,
                                          static_cast<int>((i * 7) % 3) - 1);
            }

            // Host reference, C = alpha * op(A) * op(B) + beta * C per batch.
            // Entries outside the logical C (batch padding) must stay untouched.
            std::vector<std::complex<double>> want(dense_C.size());
            for(size_t i = 0; i < want.size(); ++i)
            {
                want[i] = to_ref(dense_C[i]);
            }
            for(int32_t b = 0; b < batch_count; ++b)
            {
                std::vector<std::complex<double>> op_A(static_cast<size_t>(mo) * ko, 0.0);
                for(int32_t i = 0; i < mat_m; ++i)
                {
                    for(int32_t q = p.row_ptr[i]; q < p.row_ptr[i + 1]; ++q)
                    {
                        const int32_t        j = p.col_ind[q];
                        std::complex<double> v = to_ref(val[b * nnz + q]);
                        if(conj_A)
                        {
                            v = std::conj(v);
                        }
                        if(trans_A)
                        {
                            op_A[static_cast<size_t>(j) * ko + i] = v;
                        }
                        else
                        {
                            op_A[static_cast<size_t>(i) * ko + j] = v;
                        }
                    }
                }

                for(int32_t col = 0; col < n; ++col)
                {
                    for(int32_t row = 0; row < mo; ++row)
                    {
                        std::complex<double> sum = 0.0;
                        for(int32_t r = 0; r < ko; ++r)
                        {
                            const int64_t        bi = trans_B ? b_index(col, r) : b_index(r, col);
                            std::complex<double> bv = to_ref(dense_B[b * stride_B + bi]);
                            if(conj_B)
                            {
                                bv = std::conj(bv);
                            }
                            sum += op_A[static_cast<size_t>(row) * ko + r] * bv;
                        }
                        const int64_t ci = b * stride_C + c_index(row, col);
                        want[ci]         = to_ref(alpha) * sum + to_ref(beta) * to_ref(dense_C[ci]);
                    }
                }
            }

            // When op(A) has more rows than A, the rows of C beyond A's row count must
            // be updated by the operation, otherwise a dropped update there would not
            // show up as a mismatch.
            if(mo > mat_m)
            {
                size_t changed = 0;
                for(int32_t b = 0; b < batch_count; ++b)
                {
                    for(int32_t col = 0; col < n; ++col)
                    {
                        for(int32_t row = mat_m; row < mo; ++row)
                        {
                            const int64_t ci = b * stride_C + c_index(row, col);
                            changed += (want[ci] != to_ref(dense_C[ci])) ? 1 : 0;
                        }
                    }
                }
                ASSERT_GT(changed, size_t(0));
            }

            for(size_t il = 0; il < num_limits; ++il)
            {
                const Limits& limit = limits[il];

                SCOPED_TRACE(testing::Message()
                             << "I = int" << 8 * sizeof(I) << ", J = int" << 8 * sizeof(J)
                             << ", C = "
                             << (std::is_same<TC, _Float16>::value             ? "f16"
                                 : std::is_same<TC, rocsparse_bfloat16>::value ? "bf16"
                                                                               : "other")
                             << ", n = " << n << ", batch_count = " << batch_count
                             << ", batch_stride_C = " << stride_C
                             << ", order = " << (row_ord ? "row" : "column")
                             << ", maxGridSize[0..2] = " << limit.v[0] << "," << limit.v[1] << ","
                             << limit.v[2] << " (0 = unclamped)");

                device_vector<I>  d_row_ptr(row_ptr);
                device_vector<J>  d_row_ind(row_ind);
                device_vector<J>  d_col_ind(col_ind);
                device_vector<AB> d_val(val);
                device_vector<AB> d_B(dense_B);
                device_vector<TC> d_C(dense_C);
                ASSERT_TRUE(d_row_ptr.ptr && d_row_ind.ptr && d_col_ind.ptr && d_val.ptr && d_B.ptr
                            && d_C.ptr);

                SpMatDescr mat_A;
                DnMatDescr mat_B;
                DnMatDescr mat_C;
                if(c.format == Format::csr)
                {
                    ASSERT_EQ(rocsparse_create_csr_descr(&mat_A.d,
                                                         mat_m,
                                                         mat_k,
                                                         nnz,
                                                         d_row_ptr.ptr,
                                                         d_col_ind.ptr,
                                                         d_val.ptr,
                                                         it_of<I>(),
                                                         it_of<J>(),
                                                         rocsparse_index_base_zero,
                                                         dt_of<AB>()),
                              rocsparse_status_success);
                }
                else
                {
                    ASSERT_EQ(rocsparse_create_coo_descr(&mat_A.d,
                                                         mat_m,
                                                         mat_k,
                                                         nnz,
                                                         d_row_ind.ptr,
                                                         d_col_ind.ptr,
                                                         d_val.ptr,
                                                         it_of<J>(),
                                                         rocsparse_index_base_zero,
                                                         dt_of<AB>()),
                              rocsparse_status_success);
                }
                ASSERT_EQ(rocsparse_create_dnmat_descr(
                              &mat_B.d, rows_B, cols_B, ldb, d_B.ptr, dt_of<AB>(), c.order),
                          rocsparse_status_success);
                ASSERT_EQ(rocsparse_create_dnmat_descr(
                              &mat_C.d, mo, n, ldc, d_C.ptr, dt_of<TC>(), c.order),
                          rocsparse_status_success);

                if(batch_count > 1)
                {
                    if(c.format == Format::csr)
                    {
                        ASSERT_EQ(
                            rocsparse_csr_set_strided_batch(mat_A.d, batch_count, mat_m + 1, nnz),
                            rocsparse_status_success);
                    }
                    else
                    {
                        ASSERT_EQ(rocsparse_coo_set_strided_batch(mat_A.d, batch_count, nnz),
                                  rocsparse_status_success);
                    }
                    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mat_B.d, batch_count, stride_B),
                              rocsparse_status_success);
                    ASSERT_EQ(rocsparse_dnmat_set_strided_batch(mat_C.d, batch_count, stride_C),
                              rocsparse_status_success);
                }

                {
                    ScopedMaxGridSize clamp(handle, limit.v);

                    size_t buffer_size = 0;
                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_buffer_size,
                                             &buffer_size,
                                             nullptr),
                              rocsparse_status_success);

                    device_vector<char> d_buffer(std::max<size_t>(buffer_size, 1));
                    ASSERT_NE(d_buffer.ptr, nullptr);

                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_preprocess,
                                             &buffer_size,
                                             d_buffer.ptr),
                              rocsparse_status_success);

                    ASSERT_EQ(rocsparse_spmm(handle,
                                             c.trans_A,
                                             c.trans_B,
                                             &alpha,
                                             mat_A.d,
                                             mat_B.d,
                                             &beta,
                                             mat_C.d,
                                             dt_of<T>(),
                                             c.alg,
                                             rocsparse_spmm_stage_compute,
                                             &buffer_size,
                                             d_buffer.ptr),
                              rocsparse_status_success);
                    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
                }

                const std::vector<TC> got = to_host(d_C);
                ASSERT_EQ(got.size(), want.size());

                const double tol = narrow ? 0.0 : (sizeof(T) == sizeof(float) ? 1e-5 : 1e-12);
                for(size_t i = 0; i < got.size(); ++i)
                {
                    const double diff = std::abs(to_ref(got[i]) - want[i]);
                    if(diff > tol * std::max(1.0, std::abs(want[i])))
                    {
                        const int64_t batch = static_cast<int64_t>(i) / stride_C;
                        const int64_t local = static_cast<int64_t>(i) % stride_C;
                        ADD_FAILURE() << "first mismatch at batch " << batch << ", flat index "
                                      << local << " (row " << (row_ord ? local / ldc : local % ldc)
                                      << ", column " << (row_ord ? local % ldc : local / ldc)
                                      << "): got " << to_ref(got[i]) << ", want " << want[i];
                        break;
                    }
                }
            }
        }
    }

    const Limits* limits_of(const Case& c, size_t& count)
    {
        if(c.clamp_x)
        {
            count = sizeof(xyz_limits) / sizeof(xyz_limits[0]);
            return xyz_limits;
        }
        count = sizeof(yz_limits) / sizeof(yz_limits[0]);
        return yz_limits;
    }

    template <typename T, typename I, typename J>
    void check_real(rocsparse_handle handle, const Case& c)
    {
        size_t        count;
        const Limits* limits = limits_of(c, count);
        check_case<T, T, T, I, J>(handle, c, limits, count, false);
    }

    // COO has a single index type, so it only runs the I == J combinations.
    void check_all(rocsparse_handle handle, const Case& c)
    {
        check_real<float, int32_t, int32_t>(handle, c);
        check_real<double, int32_t, int32_t>(handle, c);
        if(c.format == Format::csr)
        {
            check_real<float, int64_t, int32_t>(handle, c);
            check_real<double, int64_t, int32_t>(handle, c);
        }
        check_real<float, int64_t, int64_t>(handle, c);
        check_real<double, int64_t, int64_t>(handle, c);
    }

    // Complex values, so that conjugate_transpose of A and of B is observable.
    void check_all_complex(rocsparse_handle handle, const Case& c)
    {
        check_real<rocsparse_float_complex, int32_t, int32_t>(handle, c);
        check_real<rocsparse_double_complex, int32_t, int32_t>(handle, c);
        check_real<rocsparse_float_complex, int64_t, int64_t>(handle, c);
    }

    // f16 and bf16 output with float compute and an odd batch stride, plus an
    // f32 output control with the same odd stride.
    void check_all_16bit(rocsparse_handle handle, const Case& c)
    {
        const size_t count = sizeof(narrow_limits) / sizeof(narrow_limits[0]);

        check_case<float, float, float, int32_t, int32_t>(handle, c, narrow_limits, count, true);
        check_case<float, _Float16, _Float16, int32_t, int32_t>(
            handle, c, narrow_limits, count, true);
        check_case<float, rocsparse_bfloat16, rocsparse_bfloat16, int32_t, int32_t>(
            handle, c, narrow_limits, count, true);
        if(c.format == Format::csr)
        {
            check_case<float, _Float16, _Float16, int64_t, int64_t>(
                handle, c, narrow_limits, count, true);
            check_case<float, rocsparse_bfloat16, rocsparse_bfloat16, int64_t, int64_t>(
                handle, c, narrow_limits, count, true);
        }
    }
}

class MmGridY : public HandleTest
{
};

// Shorthands.
namespace
{
    constexpr rocsparse_operation op_n = rocsparse_operation_none;
    constexpr rocsparse_operation op_t = rocsparse_operation_transpose;
    constexpr rocsparse_operation op_c = rocsparse_operation_conjugate_transpose;

    constexpr rocsparse_order col_ord = rocsparse_order_column;
    constexpr rocsparse_order row_ord = rocsparse_order_row;

    constexpr rocsparse_spmm_alg csr_rs = rocsparse_spmm_alg_csr_row_split;
    constexpr rocsparse_spmm_alg coo_at = rocsparse_spmm_alg_coo_atomic;
    constexpr rocsparse_spmm_alg coo_sg = rocsparse_spmm_alg_coo_segmented;
    constexpr rocsparse_spmm_alg coo_sa = rocsparse_spmm_alg_coo_segmented_atomic;
}

// ---------------------------------------------------------------------------
// csrmm row_split
// ---------------------------------------------------------------------------

// csrmmnn_row_split_shared_kernel: n <= 32, ceil(30 / 8) = 4 panels of 8 columns.
TEST_F(MmGridY, csrmm_row_split_nn_shared)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_n, 30});
}

TEST_F(MmGridY, csrmm_row_split_nn_shared_row_order)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_n, 30, row_ord});
}

// csrmmnn_row_split_kernel: n > 32, 80 / 8 = 10 panels of 8 columns in the main
// launch and 3 one-column panels in the remainder launch.
TEST_F(MmGridY, csrmm_row_split_nn_main_and_remainder)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_n, 83});
}

TEST_F(MmGridY, csrmm_row_split_nn_main_and_remainder_row_order)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_n, 83, row_ord});
}

// csrmmnt_row_split_*: A not transposed, B transposed. The batch is on grid.y
// there. n = 16 runs the shared subwf kernel, n = 30 the "loop plus subwf"
// kernel and n = 70 the 64-column loop kernel with a remainder launch.
TEST_F(MmGridY, csrmm_row_split_nt_shared_loop)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_t, 16});
}

TEST_F(MmGridY, csrmm_row_split_nt_loop_plus_subwf)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_t, 30});
}

TEST_F(MmGridY, csrmm_row_split_nt_main_and_remainder)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_t, 70});
}

TEST_F(MmGridY, csrmm_row_split_nt_row_order)
{
    check_all(handle, {Format::csr, csr_rs, op_n, op_t, 30, row_ord});
    check_all(handle, {Format::csr, csr_rs, op_n, op_t, 70, row_ord});
}

// csrmmtn_row_split_kernel: ceil(30 / 4) = 8 panels of 4 columns.
TEST_F(MmGridY, csrmm_row_split_tn)
{
    check_all(handle, {Format::csr, csr_rs, op_t, op_n, 30});
}

TEST_F(MmGridY, csrmm_row_split_tn_row_order)
{
    check_all(handle, {Format::csr, csr_rs, op_t, op_n, 30, row_ord});
}

// csrmmtt_row_split_kernel: ceil(30 / 4) = 8 panels of 4 columns.
TEST_F(MmGridY, csrmm_row_split_tt)
{
    check_all(handle, {Format::csr, csr_rs, op_t, op_t, 30});
}

TEST_F(MmGridY, csrmm_row_split_tt_row_order)
{
    check_all(handle, {Format::csr, csr_rs, op_t, op_t, 30, row_ord});
}

// Complex values with conjugate_transpose of A and of B.
TEST_F(MmGridY, csrmm_row_split_complex_conjugate)
{
    check_all_complex(handle, {Format::csr, csr_rs, op_c, op_n, 30});
    check_all_complex(handle, {Format::csr, csr_rs, op_c, op_c, 30, row_ord});
    check_all_complex(handle, {Format::csr, csr_rs, op_n, op_c, 30});
    check_all_complex(handle, {Format::csr, csr_rs, op_n, op_n, 83, row_ord});
}

// ---------------------------------------------------------------------------
// coomm
// ---------------------------------------------------------------------------

// coommnn_atomic_*: A not transposed. The batch is on grid.y there and the
// columns are walked by the kernel, so only the batch clamp is observable.
TEST_F(MmGridY, coomm_atomic_nn)
{
    check_all(handle, {Format::coo, coo_at, op_n, op_n, 43});
    check_all(handle, {Format::coo, coo_at, op_n, op_t, 43, row_ord});
}

// coommtn_atomic_main<.., false, ..>: one column per grid.y block, 13 panels. The
// atomic algorithm only puts columns on grid.y for a transposed A.
TEST_F(MmGridY, coomm_atomic_tn)
{
    check_all(handle, {Format::coo, coo_at, op_t, op_n, 13});
}

TEST_F(MmGridY, coomm_atomic_tn_row_order)
{
    check_all(handle, {Format::coo, coo_at, op_t, op_n, 13, row_ord});
}

// coommtn_atomic_main<.., true, ..>.
TEST_F(MmGridY, coomm_atomic_tt)
{
    check_all(handle, {Format::coo, coo_at, op_t, op_t, 13});
}

TEST_F(MmGridY, coomm_atomic_tt_row_order)
{
    check_all(handle, {Format::coo, coo_at, op_t, op_t, 13, row_ord});
}

TEST_F(MmGridY, coomm_atomic_complex_conjugate)
{
    check_all_complex(handle, {Format::coo, coo_at, op_c, op_n, 13});
    check_all_complex(handle, {Format::coo, coo_at, op_c, op_c, 13, row_ord});
}

// coommnn_segmented_main_kernel<.., 8, .., false>: 40 / 8 = 5 panels, plus the
// 3 remainder columns, which run on a grid.y of 1. The final reduction
// coommnn_general_block_reduce puts one dense column on each grid.x block, so
// the grid.x limits shrink it to fewer blocks than the 43 columns.
TEST_F(MmGridY, coomm_segmented_nn)
{
    check_all(handle, {Format::coo, coo_sg, op_n, op_n, 43, col_ord, true, 300, 29});
}

TEST_F(MmGridY, coomm_segmented_nn_row_order)
{
    check_all(handle, {Format::coo, coo_sg, op_n, op_n, 43, row_ord, true, 300, 29});
}

// coommnn_segmented_main_kernel<.., 8, .., true>.
TEST_F(MmGridY, coomm_segmented_nt)
{
    check_all(handle, {Format::coo, coo_sg, op_n, op_t, 43, col_ord, true, 300, 29});
}

TEST_F(MmGridY, coomm_segmented_complex_conjugate)
{
    check_all_complex(handle, {Format::coo, coo_sg, op_n, op_c, 43, col_ord, true, 300, 29});
    check_all_complex(handle, {Format::coo, coo_sg, op_n, op_n, 43, row_ord, true, 300, 29});
}

// coommnn_segmented_atomic<.., COLS = 8, ..>: 40 / 8 = 5 panels, plus the 3
// remainder columns, which run on a grid.y of 1.
TEST_F(MmGridY, coomm_segmented_atomic_nn)
{
    check_all(handle, {Format::coo, coo_sa, op_n, op_n, 43});
}

TEST_F(MmGridY, coomm_segmented_atomic_nn_row_order)
{
    check_all(handle, {Format::coo, coo_sa, op_n, op_n, 43, row_ord});
}

// coommnn_segmented_atomic<.., COLS = 8, NT = true, ..>.
TEST_F(MmGridY, coomm_segmented_atomic_nt)
{
    check_all(handle, {Format::coo, coo_sa, op_n, op_t, 43});
}

TEST_F(MmGridY, coomm_segmented_atomic_complex_conjugate)
{
    check_all_complex(handle, {Format::coo, coo_sa, op_n, op_c, 43});
}

// ---------------------------------------------------------------------------
// f16 / bf16 output and odd batch strides
// ---------------------------------------------------------------------------

// csrmmtn_row_split_kernel accumulates with the 16-bit atomic add.
TEST_F(MmGridY, csrmm_row_split_tn_16bit_odd_batch_stride)
{
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_n, 30});
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_n, 30, row_ord});
}

TEST_F(MmGridY, csrmm_row_split_tt_16bit_odd_batch_stride)
{
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_t, 30});
}

// coommtn_atomic_main and coommnn_atomic_* accumulate with the 16-bit atomic add.
TEST_F(MmGridY, coomm_atomic_tn_16bit_odd_batch_stride)
{
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_n, 13});
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_n, 13, row_ord});
}

TEST_F(MmGridY, coomm_atomic_nn_16bit_odd_batch_stride)
{
    check_all_16bit(handle, {Format::coo, coo_at, op_n, op_n, 43});
}

TEST_F(MmGridY, coomm_segmented_atomic_16bit_odd_batch_stride)
{
    check_all_16bit(handle, {Format::coo, coo_sa, op_n, op_n, 43});
}

// ---------------------------------------------------------------------------
// f16 / bf16 output with k > m
// ---------------------------------------------------------------------------

// With op(A) = A^T, C is k x n. The 16-bit atomic add bounds-checks against the
// size of C, so with row-order C and k > m the rows of C beyond m must still be
// updated. Every entry of C, including rows m..k-1, is compared exactly.
TEST_F(MmGridY, csrmm_row_split_tn_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_n, 30, row_ord, false, 29, 37});
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_n, 30, col_ord, false, 29, 37});
}

TEST_F(MmGridY, csrmm_row_split_tt_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_t, 30, row_ord, false, 29, 37});
    check_all_16bit(handle, {Format::csr, csr_rs, op_t, op_t, 30, col_ord, false, 29, 37});
}

TEST_F(MmGridY, coomm_atomic_tn_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_n, 13, row_ord, false, 29, 37});
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_n, 13, col_ord, false, 29, 37});
}

TEST_F(MmGridY, coomm_atomic_tt_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_t, 13, row_ord, false, 29, 37});
    check_all_16bit(handle, {Format::coo, coo_at, op_t, op_t, 13, col_ord, false, 29, 37});
}

// Controls: with A not transposed C is m x n, which is already the size used.
TEST_F(MmGridY, coomm_atomic_nn_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::coo, coo_at, op_n, op_n, 43, row_ord, false, 29, 37});
}

TEST_F(MmGridY, coomm_segmented_atomic_16bit_k_greater_than_m)
{
    check_all_16bit(handle, {Format::coo, coo_sa, op_n, op_n, 43, row_ord, false, 29, 37});
}
