// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// What hipdnn_golden_data_tests registers is decided by planGoldenDataValidation():
// these bundles in this session produce these tests. The contract is the list of
// test kinds and names, so that is what these cases compare.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "MatmulGraphTestUtils.hpp"
#include "ReferenceGraphFixtures.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/reference-validation/GoldenDataPlan.hpp"

using hipdnn_integration_tests::ReferenceExecutorType;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_integration_tests::bundle::fixtures::buildBatchnormGraph;
using hipdnn_integration_tests::bundle::fixtures::buildSdpaGraph;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

using Planned = std::tuple<PlannedKind, std::string, std::string>;

std::shared_ptr<IntegrationTestBundle> graphBundle(flatbuffers::DetachedBuffer graph,
                                                   bool hasGoldenOutputs)
{
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->graphBuffer = std::move(graph);
    bundle->hasGoldenOutputs = hasGoldenOutputs;
    return bundle;
}

detail::LoadedBundle loaded(const std::string& suite,
                            const std::string& test,
                            flatbuffers::DetachedBuffer graph,
                            bool hasGoldenOutputs = true)
{
    detail::LoadedBundle bundle;
    bundle.jsonPath = std::filesystem::path("bundles") / (suite + ".json");
    bundle.suiteName = suite;
    bundle.testName = test;
    bundle.bundle = graphBundle(std::move(graph), hasGoldenOutputs);
    return bundle;
}

// Matmul is in neither reference's op set, so no lane can cover it.
flatbuffers::DetachedBuffer buildMatmulGraph()
{
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const std::vector<int64_t> strides = {32, 4, 2, 1};
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    return hipdnn_integration_tests::test_utils::createMatmulGraph(10,
                                                                   11,
                                                                   12,
                                                                   dims,
                                                                   strides,
                                                                   dims,
                                                                   strides,
                                                                   dims,
                                                                   strides,
                                                                   DataType::FLOAT,
                                                                   DataType::FLOAT,
                                                                   DataType::FLOAT,
                                                                   DataType::FLOAT)
        .Release();
}

GoldenDataSession session(bool cpu, bool gpu, bool gpuHasDevice)
{
    GoldenDataSession s;
    s.cpuSelected = cpu;
    s.gpuSelected = gpu;
    s.gpuHasDevice = gpuHasDevice;
    return s;
}

std::vector<Planned> kindsAndNames(const GoldenDataPlan& plan)
{
    std::vector<Planned> out;
    out.reserve(plan.tests.size());
    for(const auto& test : plan.tests)
    {
        out.emplace_back(test.kind, test.suiteName, test.testName);
    }
    return out;
}

class TestGoldenDataPlan : public ::testing::Test
{
protected:
    // One bundle for each way a lane can judge a bundle:
    //   0 batchnorm            CPU validates; outside the GPU set
    //   1 costly Sdpa          too costly for CPU; GPU validates
    //   2 varlen Sdpa          a known gap for both references
    //   3 fp8 batch Sdpa       CPU validates; a known gap for GPU
    //   4 no golden outputs    nothing to validate
    //   5 matmul               outside both sets
    std::vector<detail::LoadedBundle> _bundles;

    void SetUp() override
    {
        _bundles.push_back(loaded("quick_BatchnormFwdInference_x", "Small", buildBatchnormGraph()));
        _bundles.push_back(loaded("quick_SdpaFwd_x", "Large", buildSdpaGraph(8192)));
        _bundles.push_back(loaded(
            "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small", "Small", buildSdpaGraph(256)));
        _bundles.push_back(loaded(
            "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small", "Small", buildSdpaGraph(256)));
        _bundles.push_back(loaded("quick_SdpaFwd_y", "Small", buildSdpaGraph(256), false));
        _bundles.push_back(loaded("quick_Matmul_x", "Small", buildMatmulGraph()));
    }
};

} // namespace

// The common CI shape. The costly bundle is left to the GPU lane, which runs, so
// the CPU lane registers nothing for it. The varlen bundle is a known gap in both
// lanes: both assert the decline, and it counts as accounted for even though no
// lane compares its golden data. Only the matmul, which no lane covers, fails.
TEST_F(TestGoldenDataPlan, BothLanesWithADeviceCoverEachBundleOnceOrFailIt)
{
    const auto plan = planGoldenDataValidation(_bundles, session(true, true, true));

    const std::vector<Planned> expected{
        {PlannedKind::VALIDATE, "quick_BatchnormFwdInference_x_CpuRef", "Small"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_CpuRef",
         "Small"},
        {PlannedKind::VALIDATE, "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_CpuRef", "Small"},
        {PlannedKind::VALIDATE, "quick_SdpaFwd_x_GpuRef", "Large"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_GpuRef",
         "Small"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_GpuRef",
         "Small"},
        {PlannedKind::FAIL, "quick_Matmul_x_Unvalidated", "Small"},
    };
    EXPECT_EQ(kindsAndNames(plan), expected);
}

// A selected GPU lane with no device covers nothing, so the CPU lane can no longer
// leave the costly bundle to it: that bundle gets a skipping test naming the gap.
// The GPU lane still registers its tests; they skip in SetUp().
TEST_F(TestGoldenDataPlan, NoDeviceTurnsTheCostExclusionIntoASkipNamingTheGap)
{
    const auto plan = planGoldenDataValidation(_bundles, session(true, true, false));

    const std::vector<Planned> expected{
        {PlannedKind::VALIDATE, "quick_BatchnormFwdInference_x_CpuRef", "Small"},
        {PlannedKind::SKIP, "quick_SdpaFwd_x_CpuRef", "Large"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_CpuRef",
         "Small"},
        {PlannedKind::VALIDATE, "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_CpuRef", "Small"},
        {PlannedKind::VALIDATE, "quick_SdpaFwd_x_GpuRef", "Large"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_GpuRef",
         "Small"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_GpuRef",
         "Small"},
        {PlannedKind::FAIL, "quick_Matmul_x_Unvalidated", "Small"},
    };
    EXPECT_EQ(kindsAndNames(plan), expected);
}

// With one --reference the other lane is out of scope by the caller's choice, so
// nothing is cross-checked: the matmul gets no _Unvalidated failure.
TEST_F(TestGoldenDataPlan, ASingleLaneChecksNothingAcrossLanes)
{
    const std::vector<Planned> cpuOnly{
        {PlannedKind::VALIDATE, "quick_BatchnormFwdInference_x_CpuRef", "Small"},
        {PlannedKind::SKIP, "quick_SdpaFwd_x_CpuRef", "Large"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_CpuRef",
         "Small"},
        {PlannedKind::VALIDATE, "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_CpuRef", "Small"},
    };
    EXPECT_EQ(kindsAndNames(planGoldenDataValidation(_bundles, session(true, false, true))),
              cpuOnly);

    const std::vector<Planned> gpuOnly{
        {PlannedKind::VALIDATE, "quick_SdpaFwd_x_GpuRef", "Large"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small_GpuRef",
         "Small"},
        {PlannedKind::EXPECT_DECLINE,
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small_GpuRef",
         "Small"},
    };
    EXPECT_EQ(kindsAndNames(planGoldenDataValidation(_bundles, session(false, true, true))),
              gpuOnly);
}

// The harness asserts whatever entry the plan hands it and looks nothing up, so the
// entry has to be the one for that test's own lane and bundle.
TEST_F(TestGoldenDataPlan, EachExpectedDeclineCarriesItsOwnLanesGapEntry)
{
    const auto plan = planGoldenDataValidation(_bundles, session(true, true, true));

    for(const auto& test : plan.tests)
    {
        SCOPED_TRACE(test.suiteName + "." + test.testName);
        if(test.kind != PlannedKind::EXPECT_DECLINE)
        {
            EXPECT_FALSE(test.gap.has_value());
            continue;
        }
        const auto& bundle = _bundles.at(test.bundleIndex);
        ASSERT_TRUE(test.gap.has_value());
        ASSERT_TRUE(test.reference.has_value());
        EXPECT_EQ(test.gap->reference, *test.reference);
        EXPECT_EQ(std::string(test.gap->bundleId), bundle.suiteName + "." + bundle.testName);
    }
}

// The failure has to say what each lane did with the bundle, or it cannot be acted
// on: "outside both op sets" and "left to a GPU lane that cannot run it" are fixed
// in different places.
TEST_F(TestGoldenDataPlan, AnUnvalidatedFailureNamesEachLanesVerdictAndTheBundle)
{
    const auto plan = planGoldenDataValidation(_bundles, session(true, true, true));

    ASSERT_FALSE(plan.tests.empty());
    const auto& failure = plan.tests.back();
    ASSERT_EQ(failure.kind, PlannedKind::FAIL);
    EXPECT_NE(failure.message.find(std::string("CpuRef: ") + toString(LaneVerdict::OUTSIDE_OP_SET)),
              std::string::npos)
        << failure.message;
    EXPECT_NE(failure.message.find(std::string("GpuRef: ") + toString(LaneVerdict::OUTSIDE_OP_SET)),
              std::string::npos)
        << failure.message;
    EXPECT_NE(failure.message.find(_bundles.at(failure.bundleIndex).jsonPath.string()),
              std::string::npos)
        << failure.message;
}

// What one lane makes of one bundle, before any session fact is applied. The order
// matters: a gap entry only changes the outcome for a bundle the lane would
// otherwise validate, so a costly shape stays TOO_COSTLY even when it is listed.
TEST(TestGoldenDataPlanJudgeLane, EachBundleGetsTheVerdictItsLaneEarns)
{
    struct Row
    {
        const char* label;
        std::shared_ptr<IntegrationTestBundle> bundle;
        const char* bundleId;
        ReferenceExecutorType reference;
        LaneVerdict expected;
    };

    const std::vector<Row> rows{
        {"no golden outputs",
         graphBundle(buildSdpaGraph(256), false),
         "quick_SdpaFwd_x.Small",
         ReferenceExecutorType::CPU,
         LaneVerdict::NO_GOLDEN_OUTPUTS},
        {"op outside the reference's set",
         graphBundle(buildBatchnormGraph(), true),
         "quick_BatchnormFwdInference_x.Small",
         ReferenceExecutorType::GPU,
         LaneVerdict::OUTSIDE_OP_SET},
        {"too costly for the CPU reference",
         graphBundle(buildSdpaGraph(8192), true),
         "quick_SdpaFwd_x.Small",
         ReferenceExecutorType::CPU,
         LaneVerdict::TOO_COSTLY},
        {"cost is checked before a gap entry",
         graphBundle(buildSdpaGraph(8192), true),
         "quick_SdpaFwd_bhsd_bf16_hd128_causal_group_Small.Small",
         ReferenceExecutorType::CPU,
         LaneVerdict::TOO_COSTLY},
        {"a gap entry for this reference",
         graphBundle(buildSdpaGraph(256), true),
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small.Small",
         ReferenceExecutorType::GPU,
         LaneVerdict::KNOWN_GAP},
        {"a gap entry for the other reference only",
         graphBundle(buildSdpaGraph(256), true),
         "quick_SdpaFwd_bhsd_fp8_hd128_causal_batch_Small.Small",
         ReferenceExecutorType::CPU,
         LaneVerdict::VALIDATE},
        {"the cost gate is CPU-only",
         graphBundle(buildSdpaGraph(8192), true),
         "standard_SdpaFwd_x.Medium",
         ReferenceExecutorType::GPU,
         LaneVerdict::VALIDATE},
    };

    for(const auto& row : rows)
    {
        SCOPED_TRACE(row.label);
        EXPECT_EQ(judgeLane(*row.bundle, row.bundleId, row.reference), row.expected);
    }
}

// NOLINTEND(readability-identifier-naming)
