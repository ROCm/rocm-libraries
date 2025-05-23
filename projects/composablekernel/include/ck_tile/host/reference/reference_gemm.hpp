// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <thread>

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void reference_gemm(const HostTensor<ADataType>& a_m_k,
                                 const HostTensor<BDataType>& b_k_n,
                                 HostTensor<CDataType>& c_m_n,
                                 const AElementOp& a_element_op     = {},
                                 const BElementOp& b_element_op     = {},
                                 const ACCElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(1);
    const std::size_t K = a_m_k.get_length(1);

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;

        for(std::size_t k = 0; k < K; ++k)
        {
            AccDataType v_a;
            AccDataType v_b;
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = a_element_op(a_m_k(m, k));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(a_element_op(a_m_k(m, k)));
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                const pk_int4_t pk_val  = b_element_op(b_k_n(k, n));
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(pk_val);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(b_element_op(b_k_n(k, n)));
            }
            v_acc += v_a * v_b;
        }

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
__global__ void naive_gemm_kernel(ADataType* A,
                                  BDataType* B,
                                  CDataType* C,
                                  ck_tile::index_t M,
                                  ck_tile::index_t N,
                                  ck_tile::index_t K,
                                  ck_tile::index_t strideA,
                                  ck_tile::index_t strideB,
                                  ck_tile::index_t strideC)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N; // Compute row index
    int col = idx % N; // Compute column index

    if(row < M && col < N)
    {
        AccDataType acc = 0.0;
        for(int k = 0; k < K; ++k)
        {
            constexpr index_t packed_size_a = ck_tile::numeric_traits<ADataType>::PackedSize;
            constexpr index_t packed_size_b = ck_tile::numeric_traits<BDataType>::PackedSize;
            // Adjust indexing based on matrix layout
            int a_index = (std::is_same_v<LayoutA, tensor_layout::gemm::RowMajor>)
                              ? row * strideA + k
                              : k * strideA + row;
            int b_index = (std::is_same_v<LayoutB, tensor_layout::gemm::ColumnMajor>)
                              ? col * strideB + k
                              : k * strideB + col;

            AccDataType v_a;
            AccDataType v_b;
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(A[a_index / packed_size_a]);
                if(k % 2 == 1)
                    v_a = fp32_val.hi;
                else
                    v_a = fp32_val.lo;
            }
            else
            {
                v_a = ck_tile::type_convert<AccDataType>(A[a_index]);
            }
            if constexpr(std::is_same_v<BDataType, pk_int4_t>)
            {
                const fp32x2_t fp32_val = pk_int4_t_to_fp32x2_t(B[b_index / packed_size_b]);
                if(k % 2 == 1)
                    v_b = fp32_val.hi;
                else
                    v_b = fp32_val.lo;
            }
            else
            {
                v_b = ck_tile::type_convert<AccDataType>(B[b_index]);
            }
            acc += v_a * v_b;
        }

        int c_index = (std::is_same_v<LayoutC, tensor_layout::gemm::RowMajor>)
                          ? row * strideC + col
                          : col * strideC + row;
        C[c_index]  = ck_tile::type_convert<CDataType>(acc);
    }
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void reference_gemm_gpu(ADataType* a_ptr,
                        BDataType* b_ptr,
                        CDataType* c_ptr,
                        index_t M,
                        index_t N,
                        index_t K,
                        index_t stride_a,
                        index_t stride_b,
                        index_t stride_c)
{
    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    naive_gemm_kernel<ADataType, BDataType, AccDataType, CDataType, LayoutA, LayoutB, LayoutC>
        <<<numBlocks, numThreadsPerBlock>>>(
            a_ptr, b_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c);

    return;
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename LayoutA,
          typename LayoutB,
          typename LayoutC>
void reference_batched_gemm_gpu(ADataType* a_ptr,
                                BDataType* b_ptr,
                                CDataType* c_ptr,
                                index_t M,
                                index_t N,
                                index_t K,
                                index_t stride_a,
                                index_t stride_b,
                                index_t stride_c,
                                index_t batch_stride_A,
                                index_t batch_stride_B,
                                index_t batch_stride_C,
                                index_t batch_count)
{
    int totalElements      = M * N;
    int numThreadsPerBlock = 256; // Common choice for threads per block
    int numBlocks          = (totalElements + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for(index_t batch_id = 0; batch_id < batch_count; ++batch_id)
    {
        ADataType* d_ATemp = a_ptr + batch_id * batch_stride_A;
        BDataType* d_BTemp = b_ptr + batch_id * batch_stride_B;
        CDataType* d_CTemp = c_ptr + batch_id * batch_stride_C;
        naive_gemm_kernel<ADataType, BDataType, AccDataType, CDataType, LayoutA, LayoutB, LayoutC>
            <<<numBlocks, numThreadsPerBlock>>>(
                d_ATemp, d_BTemp, d_CTemp, M, N, K, stride_a, stride_b, stride_c);
    }

    return;
}
} // namespace ck_tile
