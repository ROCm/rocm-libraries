// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdPlan.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace ::testing;
using namespace hipdnn_sdk_test_utils;

TEST(TestSdpaFwdPlan, ExecutePlan)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] — standard MHA (numHeads == numKvHeads)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         /*leftBound=*/-1,
                         /*rightBound=*/-1,
                         /*topLeftAlignment=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor);

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlan, ExecutePlanWithCausalMask)
{
    // [B=1, H=2, Sq=4, Skv=4, D=8] with causal mask
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle, DataType::FLOAT, /*causalMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    const auto& tensorMap = graphWrapper.getTensorMap();

    SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                         *tensorMap.at(nodeAttributes->k_tensor_uid()),
                         *tensorMap.at(nodeAttributes->v_tensor_uid()),
                         *tensorMap.at(nodeAttributes->o_tensor_uid()),
                         std::nullopt,
                         /*leftBound=*/-1,
                         /*rightBound=*/0,
                         /*topLeftAlignment=*/true);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();

    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            /*causalMask=*/true);

    SdpaFwdPlan<float, float, float, float> patient(std::move(params));
    patient.execute(variantPack);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor));
}

TEST(TestSdpaFwdPlanBuilder, ExecutePlanWithAsymmetricWindow)
{
    // Regression test for a bug where SdpaFwdPlanBuilder forwarded the same value
    // (left_bound) for both leftBound and rightBound to CpuFpReferenceSdpa::forward,
    // ignoring right_bound. With leftBound == rightBound (e.g. for symmetric windows
    // or causal masks where both are -1/0), that bug is invisible. This test uses an
    // asymmetric window (leftBound=2, rightBound=1) so that any confusion of the two
    // bounds produces a different attention pattern and a divergent output.
    //
    // Compares the output of the dispatched plan (graph → SdpaFwdPlanBuilder →
    // SdpaFwdPlan::execute) against a direct call to CpuFpReferenceSdpa::forward
    // with the same asymmetric window parameters.
    //
    // [B=1, H=2, Sq=4, Skv=4, D=8] with leftBound=2, rightBound=1, TopLeft alignment.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    constexpr int64_t LEFT_BOUND = 2;
    constexpr int64_t RIGHT_BOUND = 1;

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> planTensorBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> directTensorBundle(qDims, kDims, vDims, seed);

    auto graphTuple = buildSdpaFwdGraph(planTensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/LEFT_BOUND,
                                        /*rightBound=*/RIGHT_BOUND,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Build the plan through SdpaFwdPlanBuilder so the dispatcher's left/right bound
    // extraction is exercised (this is where the original bug lived).
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    auto plan = planBuilder.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[nodeAttributes->q_tensor_uid()] = planTensorBundle.qTensor.memory().hostData();
    variantPack[nodeAttributes->k_tensor_uid()] = planTensorBundle.kTensor.memory().hostData();
    variantPack[nodeAttributes->v_tensor_uid()] = planTensorBundle.vTensor.memory().hostData();
    variantPack[nodeAttributes->o_tensor_uid()] = planTensorBundle.oTensor.memory().hostData();
    plan->execute(variantPack);

    // Direct CPU reference with the same asymmetric window parameters.
    const hipdnn_data_sdk::utilities::TensorBase<float>* noMask = nullptr;
    CpuFpReferenceSdpa::forward<float, float, float, float>(directTensorBundle.qTensor,
                                                            directTensorBundle.kTensor,
                                                            directTensorBundle.vTensor,
                                                            directTensorBundle.oTensor,
                                                            std::nullopt,
                                                            noMask,
                                                            LEFT_BOUND,
                                                            RIGHT_BOUND,
                                                            /*topLeftAlignment=*/true);

    const float tolerance = 1e-5f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(
        cpuRefOutputValidation.allClose(directTensorBundle.oTensor, planTensorBundle.oTensor))
        << "Plan output (via SdpaFwdPlanBuilder) does not match direct CpuFpReferenceSdpa "
           "with leftBound="
        << LEFT_BOUND << ", rightBound=" << RIGHT_BOUND
        << ". This indicates the dispatcher is not forwarding the two bounds distinctly.";
}

TEST(TestSdpaFwdPlanBuilder, PlanConstruction)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        patient;
    auto builtPlan = patient.buildNodePlan(graphWrapper, graphWrapper.getNode(0));

    const bool result
        = dynamic_cast<SdpaFwdPlan<float, float, float, float>*>(builtPlan.get()) != nullptr;
    EXPECT_TRUE(result);
}

TEST(TestSdpaFwdPlanBuilder, IsApplicable)
{
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    // Correct data types: applicable
    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        floatPlanBuilder;
    EXPECT_TRUE(
        floatPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Mismatched data types: not applicable
    const SdpaFwdPlanBuilder<DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF>
        halfPlanBuilder;
    EXPECT_FALSE(
        halfPlanBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()));

    // Missing tensor in map: not applicable
    auto tensorMapCopy = graphWrapper.getTensorMap();
    const auto* nodeAttributes = graphWrapper.getNode(0).attributes_as_SdpaAttributes();
    tensorMapCopy.erase(nodeAttributes->k_tensor_uid());
    EXPECT_FALSE(floatPlanBuilder.isApplicable(graphWrapper.getNode(0), tensorMapCopy));
}

TEST(TestSdpaFwdPlanBuilder, IsApplicableRejectsAlibiMask)
{
    // SdpaFwdPlanBuilder does not implement ALiBi positional encoding, so it must
    // refuse to claim nodes that have alibi_mask=true. Otherwise a graph with ALiBi
    // would silently fall through to a CPU plan that ignores the ALiBi attribute
    // and produces wrong results.
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims, /*seed=*/1);

    auto graphTuple = buildSdpaFwdGraph(tensorBundle,
                                        DataType::FLOAT,
                                        /*causalMask=*/false,
                                        /*causalMaskBottomRight=*/false,
                                        /*leftBound=*/std::nullopt,
                                        /*rightBound=*/std::nullopt,
                                        hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                                        /*alibiMask=*/true);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const GraphWrapper graphWrapper(serializedGraph.data(), serializedGraph.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;
    EXPECT_FALSE(planBuilder.isApplicable(graphWrapper.getNode(0), graphWrapper.getTensorMap()))
        << "SdpaFwdPlanBuilder must reject nodes with alibi_mask=true";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskMatchesExplicitTopLeftBounds)
{
    // The dispatcher in SdpaFwdPlanBuilder maps the deprecated causal_mask=true flag
    // to (leftBound=-1, rightBound=0, TOP_LEFT). This test verifies that mapping by
    // running two graphs that should produce bit-for-bit identical output:
    //   (a) causal_mask=true                         (deprecated path)
    //   (b) leftBound=-1, rightBound=0, TOP_LEFT     (modern path)
    const std::vector<int64_t> qDims = {1, 2, 4, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/true);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, TOP_LEFT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Both code paths feed the same arguments into CpuFpReferenceSdpa::forward, so
    // results must match to within bit-for-bit tolerance.
    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask=true should produce identical output to "
           "leftBound=-1, rightBound=0, TOP_LEFT alignment.";
}

TEST(TestSdpaFwdPlanBuilder, DeprecatedCausalMaskBottomRightMatchesExplicitBottomRightBounds)
{
    // The dispatcher maps the deprecated causal_mask_bottom_right=true flag to
    // (leftBound=-1, rightBound=0, BOTTOM_RIGHT). Verify by comparing against an
    // explicit bottom-right window configuration. Use Sq != Skv so that TOP_LEFT
    // and BOTTOM_RIGHT alignments produce different output, catching any bug
    // where the dispatcher forgets to set the alignment to BOTTOM_RIGHT.
    const std::vector<int64_t> qDims = {1, 2, 2, 8};
    const std::vector<int64_t> kDims = {1, 2, 4, 8};
    const std::vector<int64_t> vDims = {1, 2, 4, 8};

    const unsigned int seed = getGlobalTestSeed();
    SdpaFwdTensorBundle<float> deprecatedBundle(qDims, kDims, vDims, seed);
    SdpaFwdTensorBundle<float> explicitBundle(qDims, kDims, vDims, seed);

    // (a) Deprecated causal_mask_bottom_right=true
    auto deprecatedGraphTuple = buildSdpaFwdGraph(deprecatedBundle,
                                                  DataType::FLOAT,
                                                  /*causalMask=*/false,
                                                  /*causalMaskBottomRight=*/true,
                                                  /*leftBound=*/std::nullopt,
                                                  /*rightBound=*/std::nullopt,
                                                  hipdnn_frontend::DiagonalAlignment::TOP_LEFT);
    auto& deprecatedGraph = std::get<0>(deprecatedGraphTuple);
    auto [depBin, depErr] = deprecatedGraph->to_binary();
    ASSERT_TRUE(depErr.is_good()) << depErr.get_message();
    const GraphWrapper depWrapper(depBin.data(), depBin.size());

    // (b) Explicit (leftBound=-1, rightBound=0, BOTTOM_RIGHT)
    auto explicitGraphTuple = buildSdpaFwdGraph(explicitBundle,
                                                DataType::FLOAT,
                                                /*causalMask=*/false,
                                                /*causalMaskBottomRight=*/false,
                                                /*leftBound=*/-1,
                                                /*rightBound=*/0,
                                                hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    auto& explicitGraph = std::get<0>(explicitGraphTuple);
    auto [expBin, expErr] = explicitGraph->to_binary();
    ASSERT_TRUE(expErr.is_good()) << expErr.get_message();
    const GraphWrapper expWrapper(expBin.data(), expBin.size());

    const SdpaFwdPlanBuilder<DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT>
        planBuilder;

    // Execute deprecated-path plan
    {
        auto plan = planBuilder.buildNodePlan(depWrapper, depWrapper.getNode(0));
        const auto* attrs = depWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = deprecatedBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = deprecatedBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = deprecatedBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = deprecatedBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    // Execute explicit-bounds plan
    {
        auto plan = planBuilder.buildNodePlan(expWrapper, expWrapper.getNode(0));
        const auto* attrs = expWrapper.getNode(0).attributes_as_SdpaAttributes();
        std::unordered_map<int64_t, void*> vp;
        vp[attrs->q_tensor_uid()] = explicitBundle.qTensor.memory().hostData();
        vp[attrs->k_tensor_uid()] = explicitBundle.kTensor.memory().hostData();
        vp[attrs->v_tensor_uid()] = explicitBundle.vTensor.memory().hostData();
        vp[attrs->o_tensor_uid()] = explicitBundle.oTensor.memory().hostData();
        plan->execute(vp);
    }

    const float tolerance = 0.0f;
    const CpuFpReferenceValidation<float> cpuRefOutputValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefOutputValidation.allClose(deprecatedBundle.oTensor, explicitBundle.oTensor))
        << "Deprecated causal_mask_bottom_right=true should produce identical output to "
           "leftBound=-1, rightBound=0, BOTTOM_RIGHT alignment.";
}
