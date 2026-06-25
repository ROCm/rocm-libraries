// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The frontend Graph::sdpa() builder is gated behind HIPDNN_ENABLE_SDPA; this
// translation unit compiles to nothing when SDPA is disabled.
#ifdef HIPDNN_ENABLE_SDPA

#include <cmath>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

namespace
{

// Mask variants exercised by the suite. Kept to the forward feature surface the
// production SDPA engines (AITER ASM on gfx942/gfx950, CK elsewhere) accept, so
// the dispatch path is genuinely exercised rather than skipped.
enum class SdpaMask
{
    NONE,
    CAUSAL_TOP_LEFT,
};

struct SdpaFwdTestCase
{
    int64_t batch;
    int64_t headsQ;
    int64_t headsKv; // < headsQ exercises GQA/MQA; must divide headsQ
    int64_t seqLenQ;
    int64_t seqLenKv;
    int64_t headDim;
    SdpaMask mask;
    unsigned int seed;
    std::string note;
};

// Minimal green surface: plain MHA, causal, and GQA at head dim 128 (the head
// dim the production forward kernels ship). Head counts are multiples of 8 so
// the AITER ASM kernel does not downgrade. bf16 dispatches to AITER ASM;
// fp16 dispatches to the CK forward engine.
std::vector<SdpaFwdTestCase> getSdpaFwdTestCases()
{
    return {
        {2, 8, 8, 256, 256, 128, SdpaMask::NONE, 0xC0FFEE, "mha"},
        {2, 8, 8, 256, 256, 128, SdpaMask::CAUSAL_TOP_LEFT, 0xBEEF, "causal_top_left"},
        {2, 8, 2, 256, 256, 128, SdpaMask::NONE, 0xF00D, "gqa"},
    };
}

template <typename DataType>
class SdpaForward : public IntegrationGraphVerificationHarness<DataType, SdpaFwdTestCase>
{
public:
    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> o;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const SdpaFwdTestCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("SdpaForwardTest");

        const auto ioType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(ioType);

        const std::vector<int64_t> qDims{tc.batch, tc.headsQ, tc.seqLenQ, tc.headDim};
        const std::vector<int64_t> kDims{tc.batch, tc.headsKv, tc.seqLenKv, tc.headDim};
        const std::vector<int64_t> vDims{tc.batch, tc.headsKv, tc.seqLenKv, tc.headDim};

        auto makeIo = [&](const std::string& name, const std::vector<int64_t>& dims) {
            return std::make_shared<graph::TensorAttributes>(
                graph::makeTensorAttributes(name, ioType, dims, generateStrides(dims)));
        };

        auto q = makeIo("Q", qDims);
        auto k = makeIo("K", kDims);
        auto v = makeIo("V", vDims);

        graph::SdpaAttributes sdpaAttrs;
        sdpaAttrs.set_attn_scale_value(1.0f / std::sqrt(static_cast<float>(tc.headDim)));
        if(tc.mask == SdpaMask::CAUSAL_TOP_LEFT)
        {
            sdpaAttrs.set_causal_mask(true);
        }

        auto [o, stats] = graphObj.sdpa(q, k, v, sdpaAttrs);
        o->set_output(true);

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

        return std::make_pair(std::move(graphObj), GraphOutputs{o});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->registerValidator(outputs.o, this->getTolerance(graphObj, outputs.o));

        this->setTestCaseNote(testCase.note);
        this->verifyGraph(graphObj, testCase.seed);
    }
};

using IntegrationGpuSdpaFwdBfp16 = SdpaForward<bfloat16>;
using IntegrationGpuSdpaFwdFp16 = SdpaForward<half>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaFwdBfp16);
TEST_P(IntegrationGpuSdpaFwdBfp16, Correctness)
{
    runGraphTest();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaFwdFp16);
TEST_P(IntegrationGpuSdpaFwdFp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaFwdBfp16,
                         testing::ValuesIn(getSdpaFwdTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaFwdFp16,
                         testing::ValuesIn(getSdpaFwdTestCases()));

#endif // HIPDNN_ENABLE_SDPA
