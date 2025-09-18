/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "flops.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipsparselt/hipsparselt.h>

inline void extract_metadata(unsigned char metadata, int& a, int& b, int& c, int& d)
{
    a = metadata & 0x03;
    b = (metadata >> 2) & 0x03;
    c = ((metadata >> 4) & 0x03);
    d = ((metadata >> 6) & 0x03);
}

void print_metadata_strided_batched(const char*    name,
                                    unsigned char* A,
                                    int64_t        n1,
                                    int64_t        n2,
                                    int64_t        n3,
                                    int64_t        s1,
                                    int64_t        s2,
                                    int64_t        s3)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    hipsparselt_cout << "---------- " << name << " ----------\n";
    int max_size = 128;

    for(int i3 = 0; i3 < n3 && i3 < max_size; i3++)
    {
        for(int i1 = 0; i1 < n1 && i1 < max_size; i1++)
        {
            for(int i2 = 0; i2 < n2 && i2 < max_size; i2++)
            {
                int a, b, c, d;
                extract_metadata((A[(i1 * s1) + (i2 * s2) + (i3 * s3)]), a, b, c, d);
                hipsparselt_cout << a << "," << b << "," << c << "," << d << "|";
            }
            hipsparselt_cout << "\n";
        }
        if(i3 < (n3 - 1) && i3 < (max_size - 1))
            hipsparselt_cout << "\n";
    }
    hipsparselt_cout << std::flush;
}

template <typename T>
void self_validate(T*             A,
                   T*             C,
                   unsigned char* M,
                   int64_t        n1,
                   int64_t        n2,
                   int64_t        n3,
                   int64_t        s1,
                   int64_t        s2,
                   int64_t        s3,
                   int64_t        c_n1,
                   int64_t        c_n2,
                   int64_t        c_n3,
                   int64_t        c_s1,
                   int64_t        c_s2,
                   int64_t        c_s3,
                   int64_t        m_n1,
                   int64_t        m_n2,
                   int64_t        m_n3,
                   int64_t        m_s1,
                   int64_t        m_s2,
                   int64_t        m_s3,
                   bool           sparse_b)
{
    // n1, n2, n3 are matrix dimensions, sometimes called m, n, batch_count
    // s1, s1, s3 are matrix strides, sometimes called 1, lda, stride_a
    using c_type = std::conditional_t<std::is_same<__half, T>::value
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
                                          || std::is_same<hipsparselt_fp8_e4m3, T>::value
                                          || std::is_same<hipsparselt_fp8_e5m2, T>::value
#endif
                                      ,float,
                                      T>;
    for(int i3 = 0; i3 < m_n3; i3++)
    {
        for(int i1 = 0; i1 < m_n1; i1++)
        {
            for(int i2 = 0; i2 < m_n2; i2++)
            {
                auto m_pos = (i1 * m_s1) + (i2 * m_s2) + (i3 * m_s3);
                int  idx[4];
                extract_metadata(M[m_pos], idx[0], idx[1], idx[2], idx[3]);
                idx[2] += 4;
                idx[3] += 4;
                int m_idx = 0;
                for(int i = 0; i < 8; i++)
                {
                    int64_t a_pos, c_pos;
                    if(!sparse_b)
                    {
                        a_pos = (i1 * s1) + ((i2 * 8 + i) * s2) + (i3 * s3);
                        c_pos = (i1 * c_s1) + ((i2 * 4 + m_idx) * c_s2) + (i3 * c_s3);
                    }
                    else
                    {
                        a_pos = ((i1 * 8 + i) * s1) + (i2 * s2) + (i3 * s3);
                        c_pos = ((i1 * 4 + m_idx) * c_s1) + (i2 * c_s2) + (i3 * c_s3);
                    }

                    T a = A[a_pos];
                    T b = static_cast<T>(0.0f);
                    if(i == idx[m_idx])
                    {

                        b = C[c_pos];
                        m_idx++;
                    }
                    CHECK_SUCCESS(static_cast<c_type>(a) == static_cast<c_type>(b));
                }
            }
        }
    }
}

inline unsigned char generate_metadata(int a, int b, int c, int d)
{
    unsigned char metadata = (a)&0x03;
    metadata |= (b << 2) & 0x0C;
    metadata |= ((c - 4) << 4) & 0x30;
    metadata |= (((d - 4) << 6)) & 0xC0;
    return metadata;
}

template <typename Ti, typename Tc>
void compress(const Ti*      in,
              Ti*            out,
              unsigned char* metadata,
              int64_t        m,
              int64_t        n,
              int64_t        stride1,
              int64_t        stride2,
              int64_t        stride_b,
              int64_t        c_stride1,
              int64_t        c_stride2,
              int64_t        c_stride_b,
              int64_t        m_stride1,
              int64_t        m_stride2,
              int64_t        m_stride_b,
              int            num_batches)
{
    constexpr int tiles_y = 8;
    using c_type          = std::conditional_t<std::is_same<__half, Ti>::value
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
                                          || std::is_same<hipsparselt_fp8_e4m3, Ti>::value
                                          || std::is_same<hipsparselt_fp8_e5m2, Ti>::value
#endif
                                      ,float,
                                      Ti>;

    for(int b = 0; b < num_batches; b++)
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j += tiles_y)
            {
                Ti  values[4];
                int idx[4];
                int m_idx = 0;

                auto valid_data = [&](int midx, int index, Ti value) {
                    idx[midx]    = index;
                    values[midx] = value;
                };

                for(int k = 0; k < tiles_y; k++)
                {
                    int64_t offset = b * stride_b + i * stride1 + (j + k) * stride2;
                    Ti      value  = in[offset];

                    if(m_idx > 4)
                    {
                        hipsparselt_cerr << "Err - The given matrix is not a 2:4 sparse matrix"
                                         << std::endl;
                        CHECK_SUCCESS(false);
                    }

                    if((k == 3 && m_idx == 0) || (k == 7 && m_idx == 2))
                    {
                        offset = b * stride_b + i * stride1 + (j + k - 1) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k - 1, value);
                    }
                    if((k == 3 && m_idx == 1) || (k == 7 && m_idx == 3)
                       || static_cast<c_type>(value) != static_cast<c_type>(0.0f))
                    {
                        offset = b * stride_b + i * stride1 + (j + k) * stride2;
                        value  = in[offset];
                        valid_data(m_idx++, k, value);
                    }
                }
                for(int k = 0; k < 4; k++)
                {
                    int64_t c_offset = b * c_stride_b + i * c_stride1 + (j / 2 + k) * c_stride2;
                    out[c_offset]    = values[k];
                }

                unsigned char md = generate_metadata(idx[0], idx[1], idx[2], idx[3]);

                int64_t metadata_offset   = b * m_stride_b + i * m_stride1 + (j / 8) * m_stride2;
                metadata[metadata_offset] = md;
            }
        }
}

template <typename Ti, typename To, typename Tc>
void testing_compress_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // allocate memory on device
    device_vector<Ti> dA(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    hipsparseOrder_t            order = HIPSPARSE_ORDER_COL;
    hipsparseLtHandle_t      handle_;
    hipsparseLtMatDescriptor_t matA_;
    hipsparseLtMatmulPlan_t plan_;

    hipsparselt_local_handle    handle{arg};
    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, order);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, order);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, order);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, order);
    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);
    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t                        workspace_size, compressed_size, compress_buffer_size;
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);

    hipsparseLtMatmulGetWorkspace(handle, plan, &workspace_size);

    hipStream_t stream = nullptr;

    if(arg.func_version == 1)
    {
        // test version 1
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(nullptr, plan, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(&handle_, plan, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, nullptr, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, &plan_, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, plan, nullptr, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size, nullptr),
                                HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_SUCCESS);

        device_vector<Ti> dA_1(compressed_size);
        device_vector<Ti> dA_ws(compress_buffer_size);
        CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
        CHECK_DEVICE_ALLOCATION(dA_ws.memcheck());

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(nullptr, plan, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(&handle_, plan, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(handle, nullptr, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(handle, &plan_, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(handle, plan, nullptr, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress(handle, plan, dA_1, nullptr, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);
    }
    else if (arg.func_version == 2)
    {
        // test version 2
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(nullptr, matA, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(&handle_, matA, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, nullptr, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, &matA_, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, matB, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_NOT_SUPPORTED);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, matA, nullptr, &compress_buffer_size),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, matA, &compressed_size, nullptr),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(handle, matA, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_SUCCESS);

        device_vector<Ti> dA_1(compressed_size);
        device_vector<Ti> dA_ws(compress_buffer_size);
        CHECK_DEVICE_ALLOCATION(dA_1.memcheck());
        CHECK_DEVICE_ALLOCATION(dA_ws.memcheck());

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(nullptr, matA, true, transA, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(&handle_, matA, true, transA, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(handle, nullptr, true, transA, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(handle, &matA_, true, transA, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(handle, matB, true, transA, dA, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_NOT_SUPPORTED);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(handle, matA, true, transA, nullptr, dA_1, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompress2(handle, matA, true, transA, dA_1, nullptr, dA_ws, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);
    }
}

template <typename Ti,
          typename To,
          typename Tc,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_compress(const Arguments& arg)
{
    hipsparseOperation_t transA = char_to_hipsparselt_operation(arg.transA);
    hipsparseOperation_t transB = char_to_hipsparselt_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = arg.K;

    int64_t lda = arg.lda;
    int64_t ldb = arg.ldb;
    int64_t ldc = arg.ldc;
    int64_t ldd = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used                = 0.0;
    double                   hipsparselt_error_c = 0.0;
    double                   hipsparselt_error_m = 0.0;
    bool                     HMM                 = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipsparseOrder_t orderA = char_to_hipsparselt_order(arg.orderA);
    hipsparseOrder_t orderB = char_to_hipsparselt_order(arg.orderB);
    hipsparseOrder_t orderC = char_to_hipsparselt_order(arg.orderC);
    hipsparseOrder_t orderD = char_to_hipsparselt_order(arg.orderD);

    int64_t A_row = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K;
    int64_t A_col = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M;
    int64_t B_row = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : N;
    int64_t B_col = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K;

    auto adjust_row_col = [](int64_t& row, int64_t& col, int64_t ld, hipsparseOperation_t& trans) {
        if(row > ld)
        {
            if(col > ld)
                return false;
            std::swap(row, col);
            trans = (trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                        ? HIPSPARSE_OPERATION_TRANSPOSE
                        : HIPSPARSE_OPERATION_NON_TRANSPOSE;
        }
        return true;
    };

    if(orderA == HIPSPARSE_ORDER_COL)
    {
        if(!adjust_row_col(A_row, A_col, lda, transA))
            return;
    }
    else
    {
        if(!adjust_row_col(A_col, A_row, lda, transA))
            return;
    }
    if(orderB == HIPSPARSE_ORDER_COL)
    {
        if(!adjust_row_col(B_row, B_col, ldb, transB))
            return;
    }
    else
    {
        if(!adjust_row_col(B_col, B_row, ldb, transB))
            return;
    }

    int64_t stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : lda;
    int64_t stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda : 1;

    int64_t stride_1_b = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : ldb;
    int64_t stride_2_b = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? ldb : 1;

    constexpr bool do_batched         = (btype == hipsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == hipsparselt_batch_type::strided_batched);
    int            num_batches        = (do_batched || do_strided_batched ? arg.batch_count : 1);
    int64_t        stride_a           = do_strided_batched              ? arg.stride_a
                                        : orderA == HIPSPARSE_ORDER_COL ? lda * A_col
                                                                        : lda * A_row;
    int64_t        stride_b           = do_strided_batched              ? arg.stride_b
                                        : orderB == HIPSPARSE_ORDER_COL ? ldb * B_col
                                                                        : ldb * B_row;
    int64_t        stride_c           = do_strided_batched              ? arg.stride_c
                                        : orderC == HIPSPARSE_ORDER_COL ? ldc * N
                                                                        : ldc * M;
    int64_t        stride_d           = do_strided_batched              ? arg.stride_d
                                        : orderD == HIPSPARSE_ORDER_COL ? ldd * N
                                                                        : ldc * M;

    hipsparselt_local_mat_descr matA(arg.sparse_b ? hipsparselt_matrix_type_dense
                                                  : hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     orderA);
    hipsparselt_local_mat_descr matB(arg.sparse_b ? hipsparselt_matrix_type_structured
                                                  : hipsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     orderB);

    hipsparselt_local_mat_descr matAv2(arg.sparse_b ? hipsparselt_matrix_type_dense
                                                    : hipsparselt_matrix_type_structured,
                                       handle,
                                       A_row,
                                       A_col,
                                       lda,
                                       arg.a_type,
                                       orderA);
    hipsparselt_local_mat_descr matBv2(arg.sparse_b ? hipsparselt_matrix_type_structured
                                                    : hipsparselt_matrix_type_dense,
                                       handle,
                                       B_row,
                                       B_col,
                                       ldb,
                                       arg.b_type,
                                       orderB);

    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, orderC);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldd, arg.d_type, orderD);

    hipsparseStatus_t eStatus = expected_hipsparse_status_of_matrix_size(
        arg.a_type, A_row, A_col, lda, orderA, !arg.sparse_b);
    EXPECT_HIPSPARSE_STATUS(matA.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(
        arg.b_type, B_row, B_col, ldb, orderB, arg.sparse_b);
    EXPECT_HIPSPARSE_STATUS(matB.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.c_type, M, N, ldc, orderC);
    EXPECT_HIPSPARSE_STATUS(matC.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    eStatus = expected_hipsparse_status_of_matrix_size(arg.d_type, M, N, ldd, orderD);
    EXPECT_HIPSPARSE_STATUS(matD.status(), eStatus);
    if(eStatus != HIPSPARSE_STATUS_SUCCESS)
        return;

    if(do_batched || do_strided_batched)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matAv2, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matBv2, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
    }
    if(do_strided_batched)
    {
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_a, A_row, A_col, lda, orderA);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_b, B_row, B_col, ldb, orderB);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_a, A_row, A_col, lda, orderA);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matAv2, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_b, B_row, B_col, ldb, orderB);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matBv2, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_c, M, N, ldc, orderC);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_c, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatus = expected_hipsparse_status_of_matrix_stride(stride_d, M, N, ldd, orderD);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_d, sizeof(int64_t)),
            eStatus);
        if(eStatus != HIPSPARSE_STATUS_SUCCESS)
            return;
    }

    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    hipsparselt_local_matmul_alg_selection alg_sel(handle, matmul, HIPSPARSELT_MATMUL_ALG_DEFAULT);

    size_t                        workspace_size, compressed_size, compress_buffer_size;
    hipsparselt_local_matmul_plan plan(handle, matmul, alg_sel);

    hipsparseLtMatmulGetWorkspace(handle, plan, &workspace_size);

    if(arg.func_version == 1)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize(handle, plan, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_SUCCESS);
    }
    else if(arg.func_version == 2)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMACompressedSize2(
                handle, arg.sparse_b ? matBv2 : matAv2, &compressed_size, &compress_buffer_size),
            HIPSPARSE_STATUS_SUCCESS);
    }
    const size_t size_A             = stride_a == 0
                                          ? (orderA == HIPSPARSE_ORDER_COL ? A_col * lda : A_row * lda)
                                          : stride_a * num_batches;
    const size_t size_A_pruned_copy = arg.unit_check || arg.norm_check ? size_A : 0;

    const size_t size_B             = stride_b == 0
                                          ? (orderB == HIPSPARSE_ORDER_COL ? B_col * ldb : B_row * ldb)
                                          : stride_b * num_batches;
    const size_t size_B_pruned_copy = arg.unit_check || arg.norm_check ? size_B : 0;

    const size_t size_compressed_copy = arg.unit_check || arg.norm_check ? compressed_size : 0;

    // allocate memory on device
    device_vector<Ti>            dT(arg.sparse_b ? size_B : size_A, 1, HMM);
    device_vector<unsigned char> dT_compressd(compressed_size, 1, HMM);
    device_vector<unsigned char> dT_compressBuffer(compress_buffer_size, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dT.memcheck());
    CHECK_DEVICE_ALLOCATION(dT_compressd.memcheck());
    CHECK_DEVICE_ALLOCATION(dT_compressBuffer.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti>            hT(arg.sparse_b ? size_B : size_A);
    host_vector<Ti>            hT_pruned(arg.sparse_b ? size_B_pruned_copy : size_A_pruned_copy);
    host_vector<unsigned char> hT_gold(size_compressed_copy);
    host_vector<unsigned char> hT_1(size_compressed_copy);

    hipsparselt_seedrand();

    size_t T_row, T_col, ldt, stride_t;
    if(!arg.sparse_b)
    {
        T_row    = orderA == HIPSPARSE_ORDER_COL ? A_row : A_col;
        T_col    = orderA == HIPSPARSE_ORDER_COL ? A_col : A_row;
        ldt      = lda;
        stride_t = stride_a;
    }
    else
    {
        T_row    = orderB == HIPSPARSE_ORDER_COL ? B_row : B_col;
        T_col    = orderB == HIPSPARSE_ORDER_COL ? B_col : B_row;
        ldt      = ldb;
        stride_t = stride_b;
    }

    // Initial Data on CPU
    if(arg.initialization == hipsparselt_initialization::rand_int)
    {
        hipsparselt_init<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::trig_float)
    {
        hipsparselt_init_sin<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::hpl)
    {
        hipsparselt_init_hpl<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::special)
    {
        hipsparselt_init_alt_impl_big<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dT.transfer_from(hT));

    if(arg.func_version == 1)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(
                handle, matmul, dT, dT, hipsparseLtPruneAlg_t(arg.prune_algo), stream),
            HIPSPARSE_STATUS_SUCCESS);
    }
    else if(arg.func_version == 2)
    {
        EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMAPrune2(handle,
                                                       arg.sparse_b ? matBv2 : matAv2,
                                                       !arg.sparse_b,
                                                       arg.sparse_b ? transB : transA,
                                                       dT,
                                                       dT,
                                                       hipsparseLtPruneAlg_t(arg.prune_algo),
                                                       stream),
                                HIPSPARSE_STATUS_SUCCESS);
    }

    if(arg.unit_check || arg.norm_check)
    {
        int64_t          row, col, stride_1, stride_2, stride;
        int64_t          c_row, c_col, c_ld, c_stride_1, c_stride_2, c_stride_r, c_stride;
        int64_t          m_row, m_col, m_ld, m_stride_1, m_stride_2, m_stride_r, m_stride;
        hipsparseOrder_t order;

        if(!arg.sparse_b)
        {
            order = orderA;

            row      = M;
            col      = K;
            stride_1 = stride_1_a;
            stride_2 = stride_2_a;
            stride   = stride_a;

            //compressd matrix
            c_row      = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K / 2;
            c_col      = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K / 2 : M;
            c_ld       = orderA == HIPSPARSE_ORDER_COL ? c_row : c_col;
            c_stride_1 = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : c_ld;
            c_stride_2 = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? c_ld : 1;
            c_stride_r = K / 2 * M;
            c_stride   = stride_a == 0 ? 0 : c_stride_r;

            //metadata
            m_row      = M;
            m_col      = K / 8;
            m_ld       = orderA == HIPSPARSE_ORDER_COL ? m_row : m_col; //M;
            m_stride_1 = m_col; //K / 8;
            m_stride_2 = 1;
            m_stride_r = m_row * m_col;
            m_stride   = stride_a == 0 ? 0 : m_stride_r;
        }
        else
        {
            order = orderB;

            row      = K;
            col      = N;
            stride_1 = stride_1_b;
            stride_2 = stride_2_b;
            stride   = stride_b;

            //compressd matrix
            c_row      = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K / 2 : N;
            c_col      = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K / 2;
            c_ld       = orderB == HIPSPARSE_ORDER_COL ? c_row : c_col;
            c_stride_1 = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : c_ld;
            c_stride_2 = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? c_ld : 1;
            c_stride_r = K / 2 * N;
            c_stride   = stride_b == 0 ? 0 : c_stride_r;

            //metadata
            m_row      = K / 8;
            m_col      = N;
            m_ld       = orderB == HIPSPARSE_ORDER_COL ? m_row : m_col; //K / 8;
            m_stride_1 = 1;
            m_stride_2 = m_row;
            m_stride_r = m_row * m_col;
            m_stride   = stride_b == 0 ? 0 : m_stride_r;
        }

        if(order == HIPSPARSE_ORDER_ROW)
        {
            std::swap(stride_1, stride_2);
            std::swap(c_row, c_col);
            std::swap(c_stride_1, c_stride_2);
            std::swap(m_row, m_col);
            //std::swap(m_stride_1, m_stride_2);
        }

        auto metadata_offset = c_stride_r * sizeof(Ti) * (stride == 0 ? 1 : num_batches);

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hT_pruned.transfer_from(dT));

        if(arg.func_version == 1)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dT, dT_compressd, dT_compressBuffer, stream),
                HIPSPARSE_STATUS_SUCCESS);
        else if(arg.func_version == 2)
            EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMACompress2(handle,
                                                              arg.sparse_b ? matBv2 : matAv2,
                                                              !arg.sparse_b,
                                                              arg.sparse_b ? transB : transA,
                                                              dT,
                                                              dT_compressd,
                                                              dT_compressBuffer,
                                                              stream),
                                    HIPSPARSE_STATUS_SUCCESS);

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hT_1.transfer_from(dT_compressd));

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        if(!arg.sparse_b)
            compress<Ti, Tc>(hT_pruned,
                             reinterpret_cast<Ti*>(hT_gold.data()),
                             hT_gold.data() + metadata_offset,
                             row,
                             col,
                             stride_1,
                             stride_2,
                             stride,
                             c_stride_1,
                             c_stride_2,
                             c_stride,
                             m_stride_1,
                             m_stride_2,
                             m_stride,
                             num_batches);
        else
            compress<Ti, Tc>(hT_pruned,
                             reinterpret_cast<Ti*>(hT_gold.data()),
                             hT_gold.data() + metadata_offset,
                             col,
                             row,
                             stride_2,
                             stride_1,
                             stride,
                             c_stride_2,
                             c_stride_1,
                             c_stride,
                             m_stride_2,
                             m_stride_1,
                             m_stride,
                             num_batches);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        // check host error and norm
        if(arg.unit_check)
        {
#ifdef __HIP_PLATFORM_AMD__
            self_validate<Ti>(hT_pruned,
                              reinterpret_cast<Ti*>(hT_gold.data()),
                              hT_1 + metadata_offset,
                              row,
                              col,
                              num_batches,
                              stride_1,
                              stride_2,
                              stride,
                              c_row,
                              c_col,
                              num_batches,
                              c_stride_1,
                              c_stride_2,
                              c_stride,
                              order == HIPSPARSE_ORDER_COL ? m_row : m_col,
                              order == HIPSPARSE_ORDER_COL ? m_col : m_row,
                              num_batches,
                              m_stride_1,
                              m_stride_2,
                              m_stride,
                              arg.sparse_b);
#endif
            unit_check_general<Ti>(c_row,
                                   c_col,
                                   c_ld,
                                   c_stride,
                                   reinterpret_cast<Ti*>(hT_gold.data()),
                                   reinterpret_cast<Ti*>(hT_1.data()),
                                   num_batches);
// cusparselt' metadata has different layout so skip metadata check.
#ifdef __HIP_PLATFORM_AMD__
            unit_check_general<int8_t>(m_row,
                                       m_col,
                                       m_ld,
                                       m_stride,
                                       reinterpret_cast<int8_t*>(hT_gold + metadata_offset),
                                       reinterpret_cast<int8_t*>(hT_1 + metadata_offset),
                                       num_batches);
#endif
        }
        if(arg.norm_check)
        {
            hipsparselt_error_c = unit_check_diff<Ti>(c_row,
                                                      c_col,
                                                      c_ld,
                                                      c_stride,
                                                      reinterpret_cast<Ti*>(hT_gold.data()),
                                                      reinterpret_cast<Ti*>(hT_1.data()),
                                                      num_batches);
// cusparselt' metadata has different layout so skip metadata check.
#ifdef __HIP_PLATFORM_AMD__
            hipsparselt_error_m
                = unit_check_diff<int8_t>(m_row,
                                          m_col,
                                          m_ld,
                                          m_stride,
                                          reinterpret_cast<int8_t*>(hT_gold + metadata_offset),
                                          reinterpret_cast<int8_t*>(hT_1 + metadata_offset),
                                          num_batches);
#endif
        }
        //print_strided_batched("Pruned", &hT_pruned[0], T_row, T_col, num_batches, 1, ldt, stride_t);
        //print_strided_batched("Compress Host", reinterpret_cast<Ti*>(hT_gold.data()), c_row, c_col, num_batches, 1, c_ld, c_stride);
        //print_strided_batched("Compress Device", reinterpret_cast<Ti*>(hT_1.data()), c_row, c_col, num_batches, 1, c_ld, c_stride);
        //hipsparselt_cout <<  metadata_offset<< "," << m_row<< "," << m_col<< "," << num_batches<< "," << 1<< "," << m_ld<< "," << m_stride << std::endl;
        //print_metadata_strided_batched("Metadata Host", reinterpret_cast<unsigned char*>(hT_gold.data()+ metadata_offset), m_row, m_col, num_batches, 1, m_ld, m_stride);
        //print_metadata_strided_batched("Metadata Device", reinterpret_cast<unsigned char*>(hT_1.data())+ metadata_offset, m_row, m_col, num_batches, 1, m_ld, m_stride);
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dT, dT_compressd, dT_compressBuffer, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMACompress(handle, plan, dT, dT_compressd, dT_compressBuffer, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_lda, e_stride_a, e_batch_count>{}
            .log_args<float>(hipsparselt_cout,
                             arg,
                             gpu_time_used,
                             ArgumentLogging::NA_value,
                             ArgumentLogging::NA_value,
                             cpu_time_used,
                             hipsparselt_error_c,
                             hipsparselt_error_m);
    }
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
