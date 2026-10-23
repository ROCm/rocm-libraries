// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// getAllAlgos() must list algorithms in solution-index order. For sizes the
// logic files do not cover, hipblasLtMatmulAlgoGetHeuristic returns the first
// supported entry of that same list, so an order that followed heap addresses
// let identical processes -- e.g. the ranks of a data-parallel job -- run
// numerically different kernels for the same GEMM (ROCM-29721).
//
// The suite name carries the "pre_checkin" token so the ctest presets in
// clients/tests/test_categories.yaml select it.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace
{
    struct GemmTypes
    {
        hipDataType          ab;
        hipDataType          cd;
        hipblasComputeType_t compute;
    };

    TEST(AlgoOrder_pre_checkin, GetAllAlgosIsSortedBySolutionIndex)
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
            GTEST_SKIP() << "No GPU available";

        hipblasLtHandle_t handle;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        const hipblasOperation_t ops[]   = {HIPBLAS_OP_N, HIPBLAS_OP_T};
        const GemmTypes          types[] = {{HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F},
                                            {HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F},
                                            {HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F}};

        size_t checked = 0;
        for(auto opA : ops)
            for(auto opB : ops)
                for(const auto& t : types)
                {
                    std::vector<hipblasLtMatmulHeuristicResult_t> algos;
                    if(hipblaslt_ext::getAllAlgos(handle,
                                                  hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                  opA,
                                                  opB,
                                                  t.ab,
                                                  t.ab,
                                                  t.cd,
                                                  t.cd,
                                                  t.compute,
                                                  algos)
                       != HIPBLAS_STATUS_SUCCESS)
                        continue;

                    std::vector<int> indices;
                    indices.reserve(algos.size());
                    for(auto& result : algos)
                        indices.push_back(hipblaslt_ext::getIndexFromAlgo(result.algo));

                    const auto unordered = std::adjacent_find(
                        indices.begin(), indices.end(), std::greater_equal<int>());
                    EXPECT_TRUE(unordered == indices.end())
                        << "opA=" << opA << " opB=" << opB << " abType=" << t.ab
                        << ": solution index " << *unordered << " at position "
                        << (unordered - indices.begin()) << " is not below the next one";
                    checked += indices.size();
                }

        hipblasLtDestroy(handle);

        if(checked == 0)
            GTEST_SKIP() << "No algorithms available on this device";
    }
} // namespace
