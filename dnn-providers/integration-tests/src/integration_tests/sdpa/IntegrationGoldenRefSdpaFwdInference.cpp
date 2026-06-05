// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/GoldenReferenceCpu.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

template <class T>
class TestCpuSdpaFwdGoldenReference : public TestGoldenReferenceCpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(sdpa::getToleranceFwd<T>(), sdpa::getToleranceFwd<T>());
    }
};

class TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchBfp16
    : public TestCpuSdpaFwdGoldenReference<bfloat16>
{
};

TEST_P(TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchBfp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchBfp16,
                         getGoldenReferenceParams("quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch"));

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
