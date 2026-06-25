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

// Mask variants exercised by the suite. Kept to the AITER ASM forward feature
// surface that actually ships for gfx942/gfx950: no-mask and bottom-right
// causal. The gfx942 forward kernels provide a bottom-right causal variant
// only; top-left causal has no kernel and the engine declines it.
enum class SdpaMask
{
    NONE,
    CAUSAL_BOTTOM_RIGHT,
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

// Engine-agnostic forward coverage at head dim 128 (the head dim the production
// forward kernels ship), bf16 — the only data type the AITER ASM forward engine
// provides kernels for. nhead_q is a multiple of 8 so the AITER ASM kernel does
// not downgrade; nhead_kv only has to divide nhead_q (GQA/MQA).
std::vector<SdpaFwdTestCase> getSdpaFwdTestCases()
{
    return {
        {2, 8, 8, 256, 256, 128, SdpaMask::NONE, 0xC0FFEE, "mha"},
        {2, 8, 8, 256, 256, 128, SdpaMask::CAUSAL_BOTTOM_RIGHT, 0xBEEF, "causal_bottom_right"},
        {2, 8, 2, 256, 256, 128, SdpaMask::NONE, 0xF00D, "gqa"},
        // GQA collapsed to a single KV head: distinct broadcast codepath.
        {2, 8, 1, 256, 256, 128, SdpaMask::NONE, 0xCAFE, "mqa"},
        // Non-square seqlens: the only shape that makes bottom-right causal
        // numerically distinct from top-left, so it actually exercises the
        // diagonal-alignment semantics the attributes request.
        {2, 8, 8, 128, 256, 128, SdpaMask::CAUSAL_BOTTOM_RIGHT, 0xD00D, "causal_br_nonsquare"},
        // Seqlens off any natural tile boundary: exercises the remainder/mask
        // edge that the power-of-two cases skip.
        {2, 8, 8, 200, 200, 128, SdpaMask::NONE, 0xABCD, "mha_remainder_seqlen"},
        // Sub-tile seqlen: the whole problem fits in a single partial tile,
        // exercising tail padding the >=128 shapes never hit.
        {2, 8, 8, 64, 64, 128, SdpaMask::NONE, 0x5EED, "mha_subtile_seqlen"},
        // Causal on an off-tile seqlen: combines the diagonal-mask path with the
        // remainder path, an interaction neither tested alone covers.
        {2, 8, 8, 200, 200, 128, SdpaMask::CAUSAL_BOTTOM_RIGHT, 0x1234, "causal_br_remainder"},
        // Bottom-right causal with seqQ > seqKv: opposite anchoring boundary from
        // the seqQ < seqKv case (early query rows attend to all keys).
        {2, 8, 8, 256, 128, 128, SdpaMask::CAUSAL_BOTTOM_RIGHT, 0x9A9A, "causal_br_q_gt_kv"},
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
        if(tc.mask == SdpaMask::CAUSAL_BOTTOM_RIGHT)
        {
            // Modern causal form: unbounded left, right bound at the diagonal,
            // anchored bottom-right — the variant the gfx942 forward kernel ships.
            sdpaAttrs.set_diagonal_band_right_bound(0).set_diagonal_alignment(
                DiagonalAlignment::BOTTOM_RIGHT);
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

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaFwdBfp16);
TEST_P(IntegrationGpuSdpaFwdBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaFwdBfp16,
                         testing::ValuesIn(getSdpaFwdTestCases()));

#endif // HIPDNN_ENABLE_SDPA
