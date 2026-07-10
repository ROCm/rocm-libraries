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

class TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchStatsBfp16
    : public TestCpuSdpaFwdGoldenReference<bfloat16>
{
};

TEST_P(TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchStatsBfp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestCpuSdpaFwdGoldenReferenceBf16Hd128NomaskBatchStatsBfp16,
    getGoldenReferenceParams("quick/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"));

class TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchFp16
    : public TestCpuSdpaFwdGoldenReference<half>
{
};

TEST_P(TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchFp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchFp16,
                         getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch"));

class TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchStatsFp16
    : public TestCpuSdpaFwdGoldenReference<half>
{
};

TEST_P(TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchStatsFp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestCpuSdpaFwdGoldenReferenceFp16Hd128NomaskBatchStatsFp16,
    getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"));

// --- standard tier (Medium, Gqa — larger sequence lengths and GQA ratios) ---

class TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatch
    : public TestCpuSdpaFwdGoldenReference<bfloat16>
{
};

TEST_P(TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatch, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatch,
                         getGoldenReferenceParams("standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch"));

class TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatchStats
    : public TestCpuSdpaFwdGoldenReference<bfloat16>
{
};

TEST_P(TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatchStats, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestCpuSdpaFwdGoldenRefStdBf16Hd128NomaskBatchStats,
    getGoldenReferenceParams("standard/SdpaFwd/bhsd/bf16/hd128_nomask_batch_stats"));

class TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatch : public TestCpuSdpaFwdGoldenReference<half>
{
};

TEST_P(TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatch, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatch,
                         getGoldenReferenceParams("standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch"));

class TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatchStats
    : public TestCpuSdpaFwdGoldenReference<half>
{
};

TEST_P(TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatchStats, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestCpuSdpaFwdGoldenRefStdFp16Hd128NomaskBatchStats,
    getGoldenReferenceParams("standard/SdpaFwd/bhsd/fp16/hd128_nomask_batch_stats"));

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
