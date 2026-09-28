// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Every case here reports SKIPPED today: no provider registers an engine for
// MoeGroupedMatmulBwd, so the harness skips with "No engine supports this graph".
// That is expected, not an unwired suite -- the first provider engine to land is
// exercised the moment it registers, and --fail-on-unsupported turns the skip into
// a hard failure. Numerical coverage lives with the CPU reference in
// hipdnn_test_sdk_tests.

#include <hip/hip_runtime.h>

#include <ostream>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

struct MoeBwdTestCase
{
    int64_t experts;
    int64_t hiddenK;
    int64_t outputN;
    int64_t tokenRows;
    unsigned int seed;

    friend std::ostream& operator<<(std::ostream& os, const MoeBwdTestCase& tc)
    {
        return os << "experts=" << tc.experts << ",K=" << tc.hiddenK << ",N=" << tc.outputN
                  << ",rows=" << tc.tokenRows << ",seed=" << tc.seed;
    }
};

std::vector<MoeBwdTestCase> getMoeBwdTestCases()
{
    return {
        MoeBwdTestCase{4, 64, 32, 32, 1}, MoeBwdTestCase{4, 64, 32, 30, 1}, // rows % experts != 0
    };
}

template <typename DataType>
class MoeBwd : public IntegrationGraphVerificationHarness<DataType, MoeBwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> dweight;
        std::shared_ptr<graph::TensorAttributes> firstTokenOffset;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const MoeBwdTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("MoeGroupedMatmulBwdTest");

        auto dataType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(dataType)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(dataType);

        const std::vector<int64_t> doutputDims = {1, tc.tokenRows, tc.outputN};
        auto doutputAttr = std::make_shared<graph::TensorAttributes>(graph::makeTensorAttributes(
            "doutput", dataType, doutputDims, generateStrides(doutputDims)));

        const std::vector<int64_t> tokenDims = {1, tc.tokenRows, tc.hiddenK};
        auto tokenAttr = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("token", dataType, tokenDims, generateStrides(tokenDims)));

        const std::vector<int64_t> offsetDims = {tc.experts, 1, 1};
        auto firstTokenOffsetAttr = std::make_shared<graph::TensorAttributes>(
            graph::makeTensorAttributes("first_token_offset",
                                        hipdnn_frontend::DataType::INT32,
                                        offsetDims,
                                        generateStrides(offsetDims)));

        graph::MoeGroupedMatmulBwdAttributes moeAttrs;
        moeAttrs.set_name("MoeGroupedMatmulBwd_integration");
        moeAttrs.set_compute_data_type(graphObj.get_compute_data_type());

        auto dweightAttr = graphObj.moe_grouped_matmul_bwd(
            doutputAttr, tokenAttr, firstTokenOffsetAttr, moeAttrs);
        dweightAttr->set_output(true);

        // dweight's dims and strides are left unset so MoeGroupedMatmulBwdNode infers
        // [experts, K, N] with column-major strides; the harness reads both back off the
        // attribute after validate() to size and lay out the output buffer.

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

        return std::make_pair(std::move(graphObj), GraphOutputs{dweightAttr, firstTokenOffsetAttr});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        _firstTokenOffsetUid = outputs.firstTokenOffset->get_uid();

        this->registerValidator(outputs.dweight, this->getTolerance(graphObj, outputs.dweight));

        this->inputFillRecipes().setGlobalSeed(testCase.seed);
        this->verifyGraph(graphObj);
    }

    // The generic synthesizer fills every INT32 leaf with arbitrary values, which
    // are invalid (non-monotonic, possibly out-of-range) offsets. Overwrite
    // first_token_offset with the same deterministic valid partition the
    // test-SDK bundle uses (defaultMoeGroupedMatmulBwdRoutingOffset), after the
    // base implementation has synthesized (and sentinel-filled the output of)
    // everything else.
    hipdnn_integration_tests::FillResult
        initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                         hipdnn_test_sdk::utilities::GraphTensorBundle& bundle) override
    {
        auto result
            = IntegrationGraphVerificationHarness<DataType, MoeBwdTestCase>::initializeBundle(
                graph, bundle);
        if(!result.filled)
        {
            return result;
        }

        const auto& testCase = this->GetParam();

        auto& offsetTensor = static_cast<hipdnn_data_sdk::utilities::TensorBase<int32_t>&>(
            bundle.getTensor(_firstTokenOffsetUid));
        const int64_t experts = offsetTensor.dims()[0];
        for(int64_t e = 0; e < experts; ++e)
        {
            offsetTensor.setHostValue(
                hipdnn_test_sdk::utilities::defaultMoeGroupedMatmulBwdRoutingOffset(
                    e, testCase.tokenRows, experts),
                {e, 0, 0});
        }
        offsetTensor.markHostModified();

        return result;
    }

private:
    int64_t _firstTokenOffsetUid = -1;
};

using IntegrationGpuMoeGroupedMatmulBwdFp32 = MoeBwd<float>;
using IntegrationGpuMoeGroupedMatmulBwdFp16 = MoeBwd<half>;
using IntegrationGpuMoeGroupedMatmulBwdBf16 = MoeBwd<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulBwdFp32);
TEST_P(IntegrationGpuMoeGroupedMatmulBwdFp32, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulBwdFp16);
TEST_P(IntegrationGpuMoeGroupedMatmulBwdFp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuMoeGroupedMatmulBwdBf16);
TEST_P(IntegrationGpuMoeGroupedMatmulBwdBf16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulBwdFp32,
                         testing::ValuesIn(getMoeBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulBwdFp16,
                         testing::ValuesIn(getMoeBwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuMoeGroupedMatmulBwdBf16,
                         testing::ValuesIn(getMoeBwdTestCases()));
