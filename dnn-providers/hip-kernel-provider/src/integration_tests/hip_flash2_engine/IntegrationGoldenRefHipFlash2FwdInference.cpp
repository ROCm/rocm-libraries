// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Integration tests for HipFlash2Engine (FP16 SDPA forward pass).
// Follows the golden reference test pattern from PR #7962.
//
// Run with: ninja integration-check --gtest_filter="*HipFlash2*"

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/GoldenReferenceCpu.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

// ── FP16 MHA causal ──────────────────────────────────────────────────────────
// Tests the primary use case: FP16 causal attention matching the performance
// shapes validated on MI300X (71.27 TFLOPS) and MI325X (78.98 TFLOPS).

template <class T>
class TestCpuHipFlash2FwdGoldenReference : public TestGoldenReferenceCpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(sdpa::getToleranceFwd<T>(), sdpa::getToleranceFwd<T>());
    }
};

// MHA FP16 causal seq=2048 D=128 — validated on MI300X/MI325X/MI355X
class TestCpuHipFlash2FwdFP16Hd128CausalMha : public TestCpuHipFlash2FwdGoldenReference<half>
{
};

TEST_P(TestCpuHipFlash2FwdFP16Hd128CausalMha, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuHipFlash2FwdFP16Hd128CausalMha,
                         getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd128_causal_mha"));

// GQA FP16 causal seq=4096 D=128 — validated on MI325X (78.97 TFLOPS)
class TestCpuHipFlash2FwdFP16Hd128CausalGqa4 : public TestCpuHipFlash2FwdGoldenReference<half>
{
};

TEST_P(TestCpuHipFlash2FwdFP16Hd128CausalGqa4, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuHipFlash2FwdFP16Hd128CausalGqa4,
                         getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd128_causal_gqa4"));

// D=64 FP16 causal — validated on MI325X (87.85 TFLOPS)
class TestCpuHipFlash2FwdFP16Hd64Causal : public TestCpuHipFlash2FwdGoldenReference<half>
{
};

TEST_P(TestCpuHipFlash2FwdFP16Hd64Causal, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(,
                         TestCpuHipFlash2FwdFP16Hd64Causal,
                         getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd64_causal"));

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
