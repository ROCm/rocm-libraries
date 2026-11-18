// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// getUidFromAlgo must report the SolutionUID that TensileCreateLibrary stored
// for each TensileLite solution. The build rejects a zero UID and a UID shared
// by two solution indices, so every listed TensileLite algo has a distinct
// non-zero UID.
//
// The suite name carries the "pre_checkin" token so the ctest presets in
// clients/tests/test_categories.yaml select it.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <map>
#include <vector>

namespace
{
    TEST(SolutionUid_pre_checkin, GetUidFromAlgoIsUniquePerSolution)
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
            GTEST_SKIP() << "No GPU available";

        hipblasLtHandle_t handle;
        ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

        const hipDataType types[] = {HIP_R_32F, HIP_R_16F, HIP_R_16BF};

        std::map<uint64_t, int> indexByUid;
        for(auto type : types)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> algos;
            if(hipblaslt_ext::getAllAlgos(handle,
                                          hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                          HIPBLAS_OP_N,
                                          HIPBLAS_OP_N,
                                          type,
                                          type,
                                          type,
                                          type,
                                          HIPBLAS_COMPUTE_32F,
                                          algos)
               != HIPBLAS_STATUS_SUCCESS)
                continue;

            for(auto& result : algos)
            {
                const int      index = hipblaslt_ext::getIndexFromAlgo(result.algo);
                const uint64_t uid   = hipblaslt_ext::getUidFromAlgo(handle, result.algo);
                if(index < 0)
                {
                    EXPECT_EQ(uid, 0u) << "non-TensileLite algo " << index;
                    continue;
                }
                EXPECT_NE(uid, 0u) << "solution index " << index;
                auto [it, inserted] = indexByUid.emplace(uid, index);
                EXPECT_TRUE(inserted || it->second == index)
                    << "UID " << uid << " reported for solution indices " << it->second << " and "
                    << index;
            }
        }

        hipblasLtDestroy(handle);

        if(indexByUid.empty())
            GTEST_SKIP() << "No TensileLite algorithms available on this device";
    }
} // namespace
