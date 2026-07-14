// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Integration tests for HipFlash2Engine (FP16 SDPA forward pass).
// Uses the GPU golden-reference harness so the actual HIP kernel is exercised.
// Descriptors are pinned to HIP_FLASH2_ENGINE_ID to prevent silent fallback to
// another engine (e.g. asm_sdpa) and ensure numeric correctness is validated
// end-to-end through hipModuleLoad + hipModuleLaunchKernel.
//
// Run with: ninja integration-check --gtest_filter="*HipFlash2*"
// Requires: gfx942 or gfx950, ENABLE_HIP_FLASH2_ENGINE=ON

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "harness/GoldenReferenceGpu.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

// ── Engine-pinned GPU golden reference base class ─────────────────────────────
// Inherits from TestGoldenReferenceGpu so inference runs on the real GPU kernel.
// HIP_FLASH2_ENGINE_ID pins execution to HipFlash2Engine — any shape rejected by
// isApplicable (wrong arch, wrong dtype, decode shape) will SKIP rather than
// silently fall back to another engine.
template <class T>
class TestGpuHipFlash2FwdGoldenReference : public TestGoldenReferenceGpu
{
public:
    void testSuite()
    {
        return goldenReferenceTestSuite(HIP_FLASH2_ENGINE_ID,
                                        sdpa::getToleranceFwd<T>(),
                                        sdpa::getToleranceFwd<T>());
    }
};

// ── MHA FP16 causal seq=2048 D=128 ───────────────────────────────────────────
// Validated on MI300X (71.27 TFLOPS), MI325X (78.98 TFLOPS), MI355X (153.83 TFLOPS)
class TestGpuHipFlash2FwdFP16Hd128CausalMha
    : public TestGpuHipFlash2FwdGoldenReference<half>
{
};

TEST_P(TestGpuHipFlash2FwdFP16Hd128CausalMha, Correctness)
{
    testSuite();
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestGpuHipFlash2FwdFP16Hd128CausalMha,
    getGoldenReferenceParams("quick/SdpaFwd/bhsd/fp16/hd128_causal_mha"));

// ── GQA-4 FP16 causal seq=4096 D=128 ───────────�