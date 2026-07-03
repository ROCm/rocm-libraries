/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

// Unit tests for the host-only hipTensor stub (hiptensor_stub.cpp).

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <gtest/gtest.h>

#include <hiptensor/hiptensor.h>

namespace
{
    // Every operational API returns NOT_SUPPORTED. Each lambda performs one
    // call; the harness checks the status uniformly so a new API added to this
    // list is covered by the same contract assertion.
    using ApiCall = hiptensorStatus_t (*)();

    hiptensorStatus_t callCreate()
    {
        hiptensorHandle_t* handle = nullptr;
        return hiptensorCreate(handle);
    }

    hiptensorStatus_t callDestroy()
    {
        return hiptensorDestroy(nullptr);
    }

    hiptensorStatus_t callResizePlanCache()
    {
        return hiptensorHandleResizePlanCache(nullptr, 0u);
    }

    hiptensorStatus_t callWritePlanCacheToFile()
    {
        return hiptensorHandleWritePlanCacheToFile(nullptr, "");
    }

    hiptensorStatus_t callReadPlanCacheFromFile()
    {
        return hiptensorHandleReadPlanCacheFromFile(nullptr, "", nullptr);
    }

    hiptensorStatus_t callWriteKernelCacheToFile()
    {
        return hiptensorWriteKernelCacheToFile(nullptr, "");
    }

    hiptensorStatus_t callReadKernelCacheFromFile()
    {
        return hiptensorReadKernelCacheFromFile(nullptr, "");
    }

    hiptensorStatus_t callCreateTensorDescriptor()
    {
        return hiptensorCreateTensorDescriptor(
            nullptr, nullptr, 0u, nullptr, nullptr, HIPTENSOR_R_32F, 0u);
    }

    hiptensorStatus_t callDestroyTensorDescriptor()
    {
        return hiptensorDestroyTensorDescriptor(nullptr);
    }

    hiptensorStatus_t callCreateContraction()
    {
        return hiptensorCreateContraction(nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_OP_IDENTITY,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_OP_IDENTITY,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_OP_IDENTITY,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callDestroyOperationDescriptor()
    {
        return hiptensorDestroyOperationDescriptor(nullptr);
    }

    hiptensorStatus_t callOperationDescriptorSetAttribute()
    {
        return hiptensorOperationDescriptorSetAttribute(
            nullptr, nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, nullptr, 0u);
    }

    hiptensorStatus_t callOperationDescriptorGetAttribute()
    {
        return hiptensorOperationDescriptorGetAttribute(
            nullptr, nullptr, HIPTENSOR_OPERATION_DESCRIPTOR_TAG, nullptr, 0u);
    }

    hiptensorStatus_t callCreatePlanPreference()
    {
        return hiptensorCreatePlanPreference(
            nullptr, nullptr, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);
    }

    hiptensorStatus_t callDestroyPlanPreference()
    {
        return hiptensorDestroyPlanPreference(nullptr);
    }

    hiptensorStatus_t callPlanPreferenceSetAttribute()
    {
        return hiptensorPlanPreferenceSetAttribute(
            nullptr, nullptr, HIPTENSOR_PLAN_PREFERENCE_ALGO, nullptr, 0u);
    }

    hiptensorStatus_t callPlanGetAttribute()
    {
        return hiptensorPlanGetAttribute(
            nullptr, nullptr, HIPTENSOR_PLAN_REQUIRED_WORKSPACE, nullptr, 0u);
    }

    hiptensorStatus_t callEstimateWorkspaceSize()
    {
        return hiptensorEstimateWorkspaceSize(
            nullptr, nullptr, nullptr, HIPTENSOR_WORKSPACE_DEFAULT, nullptr);
    }

    hiptensorStatus_t callCreatePermutation()
    {
        return hiptensorCreatePermutation(nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_OP_IDENTITY,
                                          nullptr,
                                          nullptr,
                                          HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callCreatePlan()
    {
        return hiptensorCreatePlan(nullptr, nullptr, nullptr, nullptr, 0u);
    }

    hiptensorStatus_t callDestroyPlan()
    {
        return hiptensorDestroyPlan(nullptr);
    }

    hiptensorStatus_t callContract()
    {
        return hiptensorContract(nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 nullptr,
                                 0u,
                                 nullptr);
    }

    hiptensorStatus_t callCreateContractionTrinary()
    {
        return hiptensorCreateContractionTrinary(nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callContractTrinary()
    {
        return hiptensorContractTrinary(nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        0u,
                                        nullptr);
    }

    hiptensorStatus_t callPermute()
    {
        return hiptensorPermute(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    hiptensorStatus_t callCreateElementwiseBinary()
    {
        return hiptensorCreateElementwiseBinary(nullptr,
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                HIPTENSOR_OP_IDENTITY,
                                                nullptr,
                                                nullptr,
                                                HIPTENSOR_OP_IDENTITY,
                                                nullptr,
                                                nullptr,
                                                HIPTENSOR_OP_ADD,
                                                HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callElementwiseBinaryExecute()
    {
        return hiptensorElementwiseBinaryExecute(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    hiptensorStatus_t callCreateElementwiseTrinary()
    {
        return hiptensorCreateElementwiseTrinary(nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_IDENTITY,
                                                 nullptr,
                                                 nullptr,
                                                 HIPTENSOR_OP_ADD,
                                                 HIPTENSOR_OP_ADD,
                                                 HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callElementwiseTrinaryExecute()
    {
        return hiptensorElementwiseTrinaryExecute(nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr);
    }

    hiptensorStatus_t callCreateReduction()
    {
        return hiptensorCreateReduction(nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        HIPTENSOR_OP_IDENTITY,
                                        nullptr,
                                        nullptr,
                                        HIPTENSOR_OP_IDENTITY,
                                        nullptr,
                                        nullptr,
                                        HIPTENSOR_OP_ADD,
                                        HIPTENSOR_COMPUTE_DESC_NONE);
    }

    hiptensorStatus_t callReduce()
    {
        return hiptensorReduce(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0u, nullptr);
    }

    hiptensorStatus_t callLoggerSetCallback()
    {
        return hiptensorLoggerSetCallback(nullptr);
    }

    hiptensorStatus_t callLoggerSetFile()
    {
        return hiptensorLoggerSetFile(nullptr);
    }

    hiptensorStatus_t callLoggerOpenFile()
    {
        return hiptensorLoggerOpenFile("");
    }

    hiptensorStatus_t callLoggerSetLevel()
    {
        return hiptensorLoggerSetLevel(HIPTENSOR_LOG_LEVEL_OFF);
    }

    hiptensorStatus_t callLoggerSetMask()
    {
        return hiptensorLoggerSetMask(0);
    }

    hiptensorStatus_t callLoggerForceDisable()
    {
        return hiptensorLoggerForceDisable();
    }

    struct NamedApiCall
    {
        const char* name;
        ApiCall     call;
    };

    const NamedApiCall kNotSupportedApis[] = {
        {"hiptensorCreate", callCreate},
        {"hiptensorDestroy", callDestroy},
        {"hiptensorHandleResizePlanCache", callResizePlanCache},
        {"hiptensorHandleWritePlanCacheToFile", callWritePlanCacheToFile},
        {"hiptensorHandleReadPlanCacheFromFile", callReadPlanCacheFromFile},
        {"hiptensorWriteKernelCacheToFile", callWriteKernelCacheToFile},
        {"hiptensorReadKernelCacheFromFile", callReadKernelCacheFromFile},
        {"hiptensorCreateTensorDescriptor", callCreateTensorDescriptor},
        {"hiptensorDestroyTensorDescriptor", callDestroyTensorDescriptor},
        {"hiptensorCreateContraction", callCreateContraction},
        {"hiptensorDestroyOperationDescriptor", callDestroyOperationDescriptor},
        {"hiptensorOperationDescriptorSetAttribute", callOperationDescriptorSetAttribute},
        {"hiptensorOperationDescriptorGetAttribute", callOperationDescriptorGetAttribute},
        {"hiptensorCreatePlanPreference", callCreatePlanPreference},
        {"hiptensorDestroyPlanPreference", callDestroyPlanPreference},
        {"hiptensorPlanPreferenceSetAttribute", callPlanPreferenceSetAttribute},
        {"hiptensorPlanGetAttribute", callPlanGetAttribute},
        {"hiptensorEstimateWorkspaceSize", callEstimateWorkspaceSize},
        {"hiptensorCreatePermutation", callCreatePermutation},
        {"hiptensorCreatePlan", callCreatePlan},
        {"hiptensorDestroyPlan", callDestroyPlan},
        {"hiptensorContract", callContract},
        {"hiptensorCreateContractionTrinary", callCreateContractionTrinary},
        {"hiptensorContractTrinary", callContractTrinary},
        {"hiptensorPermute", callPermute},
        {"hiptensorCreateElementwiseBinary", callCreateElementwiseBinary},
        {"hiptensorElementwiseBinaryExecute", callElementwiseBinaryExecute},
        {"hiptensorCreateElementwiseTrinary", callCreateElementwiseTrinary},
        {"hiptensorElementwiseTrinaryExecute", callElementwiseTrinaryExecute},
        {"hiptensorCreateReduction", callCreateReduction},
        {"hiptensorReduce", callReduce},
        {"hiptensorLoggerSetCallback", callLoggerSetCallback},
        {"hiptensorLoggerSetFile", callLoggerSetFile},
        {"hiptensorLoggerOpenFile", callLoggerOpenFile},
        {"hiptensorLoggerSetLevel", callLoggerSetLevel},
        {"hiptensorLoggerSetMask", callLoggerSetMask},
        {"hiptensorLoggerForceDisable", callLoggerForceDisable},
    };
}

class HiptensorStubNotSupportedTest : public ::testing::TestWithParam<NamedApiCall>
{
};

TEST_P(HiptensorStubNotSupportedTest, ReturnsNotSupported)
{
    const auto& api = GetParam();
    EXPECT_EQ(api.call(), HIPTENSOR_STATUS_NOT_SUPPORTED) << api.name;
}

INSTANTIATE_TEST_SUITE_P(StubApi,
                         HiptensorStubNotSupportedTest,
                         ::testing::ValuesIn(kNotSupportedApis),
                         [](const ::testing::TestParamInfo<NamedApiCall>& info) {
                             return info.param.name;
                         });

// hiptensorGetErrorString keeps its normal behavior in the stub: it must map
// every status to the matching name and fall back to UNKNOWN for unknown codes.
TEST(HiptensorStubErrorStringTest, MapsKnownStatuses)
{
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_SUCCESS), "HIPTENSOR_STATUS_SUCCESS");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_NOT_SUPPORTED),
                 "HIPTENSOR_STATUS_NOT_SUPPORTED");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_HIP_ERROR), "HIPTENSOR_STATUS_HIP_ERROR");
    EXPECT_STREQ(hiptensorGetErrorString(HIPTENSOR_STATUS_IO_ERROR), "HIPTENSOR_STATUS_IO_ERROR");
}

TEST(HiptensorStubErrorStringTest, UnknownStatusFallsBack)
{
    EXPECT_STREQ(hiptensorGetErrorString(static_cast<hiptensorStatus_t>(0x7fffffff)),
                 "HIPTENSOR_STATUS_UNKNOWN");
}

// The version helpers are pure and remain meaningful even in the stub.
TEST(HiptensorStubVersionTest, HiprtVersionReportsUnavailable)
{
    EXPECT_EQ(hiptensorGetHiprtVersion(), -1);
}

TEST(HiptensorStubVersionTest, VersionMatchesCompiledMacros)
{
    const size_t expected = HIPTENSOR_MAJOR_VERSION * 1000000 + HIPTENSOR_MINOR_VERSION * 1000
                            + HIPTENSOR_PATCH_VERSION;
    EXPECT_EQ(hiptensorGetVersion(), expected);
}
