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
class TestCpuBatchnormFwdInferenceGoldenReference : public TestGoldenReferenceCpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(batchnorm::getToleranceInference<T>(),
                                        batchnorm::getToleranceInference<T>());
    }
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32
    : public TestCpuBatchnormFwdInferenceGoldenReference<float>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16
    : public TestCpuBatchnormFwdInferenceGoldenReference<half>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16
    : public TestCpuBatchnormFwdInferenceGoldenReference<bfloat16>
{
};

class TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32
    : public TestCpuBatchnormFwdInferenceGoldenReference<float>
{
};

// Nchw Fp32------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp32,
                         getGoldenReferenceParams("quick/BatchnormFwdInference/nchw/fp32"));

// Nchw Fp16------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwFp16,
                         getGoldenReferenceParams("quick/BatchnormFwdInference/nchw/fp16"));

// Nchw Bfp16------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNchwBfp16,
                         getGoldenReferenceParams("quick/BatchnormFwdInference/nchw/bfp16"));

// Ncdhw Fp32------------
TEST_P(TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuBatchnormFwdInferenceGoldenReferenceNcdhwFp32,
                         getGoldenReferenceParams("quick/BatchnormFwdInference/ncdhw/fp32"));

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
