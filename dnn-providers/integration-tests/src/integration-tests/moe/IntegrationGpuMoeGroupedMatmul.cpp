// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/hip_runtime.h>

#include <optional>
#include <ostream>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

// `routedRows` is the length of the routing tensors, and for GATHER also the
// output row count. NONE and SCATTER require it to equal `tokenRows`.
struct MoeTestCase
{
    int64_t experts;
    int64_t batch;
    int64_t hiddenK;
    int64_t weightN;
    int64_t tokenRows;
    int64_t routedRows;
    MoeGroupedMatmulMode mode;
    int32_t topK;
    unsigned int seed;

    /// Rows the group offsets partition: routed rows under GATHER, token rows otherwise.
    int64_t rowsTotal() const
    {
        return mode == MoeGroupedMatmulMode::GATHER ? routedRows : tokenRows;
    }

    friend std::ostream& operator<<(std::ostream& os, const MoeTestCase& tc)
    {
        return os << "experts=" << tc.experts << ",batch=" << tc.batch << ",K=" << tc.hiddenK
                  << ",N=" << tc.weightN << ",rows=" << tc.tokenRows << ",routed=" << tc.routedRows
                  << ",mode=" << static_cast<int>(tc.mode) << ",topK=" << tc.topK
                  << ",seed=" << tc.seed;
    }
};

std::vector<MoeTestCase> getMoeTestCases()
{
    return {
        MoeTestCase{4, 1, 64, 32, 32, 32, MoeGroupedMatmulMode::NONE, 0, 1},
        MoeTestCase{4, 1, 64, 32, 32, 32, MoeGroupedMatmulMode::SCATTER, 2, 1},
        MoeTestCase{4, 1, 64, 32, 32, 32, MoeGroupedMatmulMode::GATHER, 0, 1},
        MoeTestCase{2, 2, 64, 32, 32, 32, MoeGroupedMatmulMode::NONE, 0, 1}, // batched
        MoeTestCase{4, 1, 64, 32, 30, 30, MoeGroupedMatmulMode::NONE, 0, 1}, // rows % experts != 0
        // GATHER over a subset of the tokens: routed rows != token rows, so the
        // output is shorter than the token tensor and the offsets partition the
        // routed rows rather than the token rows.
        MoeTestCase{4, 1, 64, 32, 32, 16, MoeGroupedMatmulMode::GATHER, 0, 1},
    };
}

template <typename DataType>
class Moe : public IntegrationGraphVerificationHarness<DataType, MoeTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> output;
        std::shared_ptr<graph::TensorAttributes> firstTokenOffset;
        std::shared_ptr<graph::TensorAttributes> tokenIndex;
        std::shared_ptr<graph::TensorAttributes> tokenKs;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const MoeTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("MoeGroupedMatmulTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        const std::vector<int64_t> tokenDims = {1, tc.tokenRows, tc.hiddenK};
        auto tokenAttr = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("token", dataType, tokenDims, generateStrides(tokenDims)));

        const std::vector<int64_t> weightDims = {tc.experts, tc.hiddenK, tc.weightN};
        auto weightAttr = std::make_shared<graph::TensorAttributes>(graph::makeTensorAttributes(
            "weight", dataType, weightDims, generateStrides(weightDims)));

        const int64_t groupCount = tc.batch * tc.experts;
        const std::vector<int64_t> offsetDims = {groupCount, 1, 1};
        auto firstTokenOffsetAttr = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("first_token_offset",
                                        hipdnn_frontend::DataType::INT32,
                                        offsetDims,
                                        generateStrides(offsetDims)));

        const std::vector<int64_t> routingDims = {1, tc.routedRows, 1};
        std::shared_ptr<graph::TensorAttributes> tokenIndexAttr;
        if(tc.mode != MoeGroupedMatmulMode::NONE)
        {
            tokenIndexAttr = std::make_shared<graph::TensorAttributes>(
                graph::makeTensorAttributes("token_index",
                                            hipdnn_frontend::DataType::INT32,
                                            routingDims,
                                            generateStrides(routingDims)));
        }
        std::shared_ptr<graph::TensorAttributes> tokenKsAttr;
        if(tc.mode == MoeGroupedMatmulMode::SCATTER)
        {
            tokenKsAttr = std::make_shared<graph::TensorAttributes>(
                graph::makeTensorAttributes("token_ks",
                                            hipdnn_frontend::DataType::INT32,
                                            routingDims,
                                            generateStrides(routingDims)));
        }

        graph::MoeGroupedMatmulAttributes moeAttrs;
        moeAttrs.set_name("MoeGroupedMatmul_integration");
        moeAttrs.set_compute_data_type(graphObj.get_compute_data_type());
        moeAttrs.set_mode(tc.mode);
        moeAttrs.set_top_k(tc.topK);

        auto outputAttr = graphObj.moe_grouped_matmul(
            tokenAttr, weightAttr, firstTokenOffsetAttr, tokenIndexAttr, tokenKsAttr, moeAttrs);
        outputAttr->set_output(true);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            throw std::runtime_error("Failed to validate graph: " + validateResult.get_message());
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(
            std::move(graphObj),
            GraphOutputs{outputAttr, firstTokenOffsetAttr, tokenIndexAttr, tokenKsAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        _firstTokenOffsetUid = outputs.firstTokenOffset->get_uid();
        _tokenIndexUid = outputs.tokenIndex ? std::optional<int64_t>(outputs.tokenIndex->get_uid())
                                            : std::nullopt;
        _tokenKsUid
            = outputs.tokenKs ? std::optional<int64_t>(outputs.tokenKs->get_uid()) : std::nullopt;

        this->registerValidator(outputs.output, this->getTolerance(graphObj, outputs.output));

        this->inputFillRecipes().setGlobalSeed(testCase.seed);
        this->verifyGraph(graphObj);
    }

    // The generic synthesizer fills every INT32 leaf with arbitrary values,
    // which are invalid routing indices/offsets. Overwrite the routing tensors
    // with the same deterministic valid routing the test-SDK bundle uses
    // (MoeGroupedMatmulTensorBundle::setDefaultRouting), after the base
    // implementation has synthesized (and sentinel-filled the output of)
    // everything else.
    hipdnn_integration_tests::FillResult
        initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle) override
    {
        auto result = IntegrationGraphVerificationHarness<DataType, MoeTestCase>::initializeBundle(
            graph, bundle);
        if(!result.filled)
        {
            return result;
        }

        const auto& testCase = this->GetParam();

        auto& offsetTensor = static_cast<hipdnn_data_sdk::utilities::TensorBase<int32_t>&>(
            bundle.getTensor(_firstTokenOffsetUid));
        const int64_t groupCount = offsetTensor.dims()[0];
        const int64_t rowsTotal = testCase.rowsTotal();
        for(int64_t g = 0; g < groupCount; ++g)
        {
            offsetTensor.setHostValue(static_cast<int32_t>((g * rowsTotal) / groupCount),
                                      {g, 0, 0});
        }
        offsetTensor.markHostModified();

        // Both permutations below are deliberately non-identity. With identity
        // routing a provider that ignored the routing tensors entirely would still
        // match the reference, which is the one thing this suite exists to catch.
        if(_tokenIndexUid.has_value())
        {
            auto& tokenIndexTensor = static_cast<hipdnn_data_sdk::utilities::TensorBase<int32_t>&>(
                bundle.getTensor(*_tokenIndexUid));
            for(int64_t r = 0; r < testCase.routedRows; ++r)
            {
                // GATHER reads token row `reversed`; SCATTER writes output row
                // `reversed` via dst = tokenIndex * topK + tokenKs.
                const int64_t reversed = testCase.tokenRows - 1 - r;
                const int32_t value = (testCase.mode == MoeGroupedMatmulMode::GATHER)
                                          ? static_cast<int32_t>(reversed)
                                          : static_cast<int32_t>(reversed / testCase.topK);
                tokenIndexTensor.setHostValue(value, {0, r, 0});
            }
            tokenIndexTensor.markHostModified();
        }

        if(_tokenKsUid.has_value())
        {
            auto& tokenKsTensor = static_cast<hipdnn_data_sdk::utilities::TensorBase<int32_t>&>(
                bundle.getTensor(*_tokenKsUid));
            for(int64_t r = 0; r < testCase.routedRows; ++r)
            {
                const int64_t reversed = testCase.tokenRows - 1 - r;
                tokenKsTensor.setHostValue(static_cast<int32_t>(reversed % testCase.topK),
                                           {0, r, 0});
            }
            tokenKsTensor.markHostModified();
        }

        return result;
    }

private:
    int64_t _firstTokenOffsetUid = -1;
    std::optional<int64_t> _tokenIndexUid;
    std::optional<int64_t> _tokenKsUid;
};

using IntegrationGpuMoeGroupedMatmulFp32 = Moe<float>;
using IntegrationGpuMoeGroupedMatmulFp16 = Moe<half>;
using IntegrationGpuMoeGroupedMatmulBf16 = Moe<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulFp32);
TEST_P(IntegrationGpuMoeGroupedMatmulFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulFp16);
TEST_P(IntegrationGpuMoeGroupedMatmulFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulBf16);
TEST_P(IntegrationGpuMoeGroupedMatmulBf16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulFp32,
                         testing::ValuesIn(getMoeTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulFp16,
                         testing::ValuesIn(getMoeTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulBf16,
                         testing::ValuesIn(getMoeTestCases()));
