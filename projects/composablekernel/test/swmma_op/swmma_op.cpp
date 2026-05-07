// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "test/swmma_op/swmma_op_util.hpp"

static ck::index_t test_case_id = -1;

static ck::index_t case_id = 0;

template <typename Src1Type,
          typename Src2Type,
          typename DstType,
          ck::index_t AccVecSize,
          typename GPUAccType,
          typename CPUAccType,
          ck::index_t M,
          ck::index_t N,
          ck::index_t K>
bool run_test()
{
    case_id++;

    if(test_case_id != -1 && (test_case_id + 1) != case_id)
    {
        return true;
    }

    using Row         = ck::tensor_layout::gemm::RowMajor;
    using Col         = ck::tensor_layout::gemm::ColumnMajor;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    bool pass         = true;

    const auto matmul_default = ck::swmma_op_util::
        matmul<Src1Type, Src2Type, GPUAccType, DstType, M, N, K, (K >> 5), false>;

    const auto swmma_kernel_container = std::make_tuple(matmul_default);

    ck::static_for<0, 1, 1>{}([&](auto i) {
        pass &=
            ck::swmma_op_util::TestSwmma<decltype(std::get<ck::Number<i>{}>(
                                             swmma_kernel_container)),
                                         Src1Type,
                                         Src2Type,
                                         DstType,
                                         GPUAccType,
                                         CPUAccType,
                                         decltype(Row{}),
                                         decltype(Col{}),
                                         decltype(Row{}),
                                         PassThrough,
                                         PassThrough,
                                         PassThrough,
                                         AccVecSize,
                                         M,
                                         N,
                                         K>{}(std::get<ck::Number<i>{}>(swmma_kernel_container));
    });

    return pass;
}
int main(int argc, char* argv[])
{
    bool pass = true;
    if(argc > 1)
    {
        test_case_id = atoi(argv[1]);
    }
    // clang-format off
    //              |   Src1Type|       Src2Type|      DstType|  DstVecSize|  GPUAccType   | CPUAccType|   M|   N|   K|
    pass &= run_test< ck::half_t,      ck::half_t,      float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_F16
    pass &= run_test< ck::half_t,      ck::half_t,     ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  32>(); // V_SWMMA_F16_16X16_F16
    pass &= run_test< ck::bhalf_t,     ck::bhalf_t,     float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_BF16
    pass &= run_test< ck::bhalf_t,     ck::bhalf_t,    ck::bhalf_t,     8,    ck::bhalf_t,   float,       16,  16,  32>(); // V_SWMMA_BF16_16X16_BF16
    pass &= run_test< ck::f8_t,        ck::f8_t,        float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_FP8_FP8
    pass &= run_test< ck::f8_t,        ck::f8_t,        float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16X16_FP8_FP8
    pass &= run_test< ck::bf8_t,       ck::bf8_t,       float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_BF8_BF8
    pass &= run_test< ck::bf8_t,       ck::bf8_t,       float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16X16_BF8_BF8
    pass &= run_test< ck::bf8_t,       ck::bf8_t,      ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  32>(); // V_SWMMA_F16_16X16_BF8_BF8
    pass &= run_test< ck::bf8_t,       ck::bf8_t,      ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  64>(); // V_SWMMA_F16_16X16_BF8_BF8
    pass &= run_test< ck::f8_t,        ck::f8_t,       ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  32>(); // V_SWMMA_F16_16X16_FP8_FP8
    pass &= run_test< ck::f8_t,        ck::f8_t,       ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  64>(); // V_SWMMA_F16_16X16_FP8_FP8
    pass &= run_test< ck::f8_t,        ck::bf8_t,       float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_FP8_BF8
    pass &= run_test< ck::f8_t,        ck::bf8_t,       float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16X16_FP8_BF8
    pass &= run_test< ck::bf8_t,       ck::f8_t,        float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16X16_BF8_FP8
    pass &= run_test< ck::bf8_t,       ck::f8_t,        float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16X16_BF8_FP8
    pass &= run_test< ck::f8_t,        ck::bf8_t,      ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  32>(); // V_SWMMA_F16_16X16_FP8_BF8
    pass &= run_test< ck::f8_t,        ck::bf8_t,      ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  64>(); // V_SWMMA_F16_16X16_FP8_BF8
    pass &= run_test< ck::bf8_t,       ck::f8_t,       ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  32>(); // V_SWMMA_F16_16X16_BF8_FP8
    pass &= run_test< ck::bf8_t,       ck::f8_t,       ck::half_t,      8,    ck::half_t, ck::half_t,     16,  16,  64>(); // V_SWMMA_F16_16X16_BF8_FP8
    pass &= run_test< int8_t,          int8_t,         int32_t,         8,     int32_t,     int32_t,      16,  16,  32>(); // V_SWMMA_I32_16x16_IU8
    pass &= run_test< int8_t,          int8_t,         int32_t,         8,     int32_t,     int32_t,      16,  16,  64>(); // V_SWMMA_I32_16x16_IU8
    pass &= run_test< int8_t,          int8_t,          float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16x16_IU8
    pass &= run_test< int8_t,          int8_t,          float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16x16_IU8
    pass &= run_test< int8_t,         uint8_t,         int32_t,         8,     int32_t,     int32_t,      16,  16,  32>(); // V_SWMMA_I32_16x16_IU8
    pass &= run_test<uint8_t,          int8_t,         int32_t,         8,     int32_t,     int32_t,      16,  16,  64>(); // V_SWMMA_I32_16x16_IU8
    pass &= run_test<uint8_t,         uint8_t,          float,          8,       float,      float,       16,  16,  32>(); // V_SWMMA_F32_16x16_IU8
    pass &= run_test< int8_t,         uint8_t,          float,          8,       float,      float,       16,  16,  64>(); // V_SWMMA_F32_16x16_IU8
    pass &= run_test< int8_t,          int8_t,          float,          8,     int32_t,     int32_t,      16,  16,  32>(); // V_SWMMA_F32I32_16X16_IU8
    pass &= run_test< int8_t,          int8_t,          float,          8,     int32_t,     int32_t,      16,  16,  64>(); // V_SWMMA_F32I32_16X16_IU8
    #ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    pass &= run_test< ck::int4_t,     ck::int4_t,      int32_t,         8,     int32_t,     int32_t,      16,  16,  32>(); // V_SWMMA_I32_16x16_IU4
    pass &= run_test< ck::int4_t,     ck::int4_t,      int32_t,         8,     int32_t,     int32_t,      16,  16,  64>(); // V_SWMMA_I32_16x16_IU4
    pass &= run_test< ck::int4_t,     ck::int4_t,       float,          8,       float,       float,      16,  16,  32>(); // V_SWMMA_F32_16x16_IU4
    pass &= run_test< ck::int4_t,     ck::int4_t,       float,          8,       float,       float,      16,  16,  64>(); // V_SWMMA_F32_16x16_IU4
    pass &= run_test< ck::int4_t,     ck::int4_t,       float,          8,     int32_t,     int32_t,      16,  16,  32>(); // V_SWMMA_F32I32_16X16_IU4
    pass &= run_test< ck::int4_t,     ck::int4_t,       float,          8,     int32_t,     int32_t,      16,  16,  64>(); // V_SWMMA_F32I32_16X16_IU4
    #endif
    // clang-format on

    std::cout << "TestGemm ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass;
}
