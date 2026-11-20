// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hip_kernel_provider_common/SdpaConfigConstants.hpp>
#include <hip_kernel_provider_common/SdpaConfigEnumerations.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../../engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "../IntegrationGraphVerificationHarness.hpp"
#include "AsmSdpaConfigHelpers.hpp"
#include "asm_fmha_v3_fwd_configs.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::test_utilities;
using namespace asm_sdpa_engine;

namespace
{

// Names the config helper assigns to the fp8 q/k/v descale tensors; the fill below
// selects them by these names (see initializeBundle).
constexpr const char* K_DESCALE_Q_NAME = "descale_q";
constexpr const char* K_DESCALE_K_NAME = "descale_k";
constexpr const char* K_DESCALE_V_NAME = "descale_v";

// Fixed, positive, DISTINCT dequantization factors for the q/k/v descales. Distinct so a
// q/k/v pointer swap changes the result and fails the test; fixed (not randomized) so the
// same host value feeds both the GPU kernel and the CPU reference and the fp8 3e-2 bound
// stays deterministic.
constexpr float K_DESCALE_Q_FILL = 0.25f;
constexpr float K_DESCALE_K_FILL = 0.5f;
constexpr float K_DESCALE_V_FILL = 0.75f;

// The per-batch token counts a group-mode case writes into its offset tables, and the
// ragged offset tables of a cu_seq_len case: each batch's physical first token, then the
// end of the last batch's slot. A shared-offset-table case uses the Q lengths on both
// sides; a cu_seq_len case writes the lengths into its cumulative tables, of which a shared
// cu_seq_len case has only the Q one. In the physical starts batch 0 starts past token 0
// and every slot ends past its logical length (the Q length too on the K/V side, for a
// shared cu_seq_len case), so a kernel that takes a token's position from the cumulative
// lengths instead of these starts reads and writes the wrong rows. The slots fit the
// B * S token capacity of each side (configToTestCase gives group cases S_q = S_kv = 512).
struct GroupLengthSet
{
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
    std::vector<int32_t> physicalStartsQ;
    std::vector<int32_t> physicalStartsKv;
};

const GroupLengthSet& groupLengthSet(GroupLengths lengths)
{
    // Batch 2. The lengths differ between batches, are not multiples of any ts_qo or ts_kv,
    // and all exceed the largest group ts_qo (256), so every batch spans at least two
    // Q-tiles of every group row and the causal tile merging (tgDiv) changes the launch
    // grid. Every K/V length is at least its Q length, so bottom-right causal masks no row
    // out fully.
    static const GroupLengthSet s_standard
        = {{401, 260}, {483, 299}, {16, 480, 760}, {24, 560, 880}};
    // Batch 3. Batch 0 has 401 queries over 300 keys, so under bottom-right causal its first
    // 101 query rows see no key (O = 0, LSE = -inf). Batch 1 has no queries; batch 2 has no
    // keys, so all of its rows see no key under every mask.
    static const GroupLengthSet s_edge
        = {{401, 0, 260}, {300, 299, 0}, {16, 480, 560, 900}, {24, 440, 760, 1100}};
    return lengths == GroupLengths::EDGE ? s_edge : s_standard;
}

// Writes @p values into the INT32 token table @p tensor.
void fillTable(hipdnn_data_sdk::utilities::ITensor& tensor, const std::vector<int32_t>& values)
{
    if(tensor.elementCount() != values.size())
    {
        throw std::runtime_error("Token table has " + std::to_string(tensor.elementCount())
                                 + " entries; the group-mode values need "
                                 + std::to_string(values.size()));
    }
    tensor.fillWithData(values.data(), values.size() * sizeof(int32_t));
}

// Writes the exclusive prefix sum of @p lengths into @p tensor: the [B+1] token seqstarts
// AITER's group mode reads, or the cumulative lengths of cu_seq_len_q/kv.
void fillSeqStarts(hipdnn_data_sdk::utilities::ITensor& tensor, const std::vector<int32_t>& lengths)
{
    std::vector<int32_t> starts(lengths.size() + 1, 0);
    for(size_t b = 0; b < lengths.size(); ++b)
    {
        starts[b + 1] = starts[b] + lengths[b];
    }
    fillTable(tensor, starts);
}

// Checks that every slot of @p starts holds its batch's logical length with a gap after it,
// and that the slots fit @p capacity tokens.
void assertGappedSlots(const std::vector<int32_t>& starts,
                       const std::vector<int32_t>& lengths,
                       int64_t capacity)
{
    ASSERT_EQ(starts.size(), lengths.size() + 1);
    ASSERT_GT(starts[0], 0);
    for(size_t b = 0; b < lengths.size(); ++b)
    {
        ASSERT_LT(starts[b] + lengths[b], starts[b + 1]) << "batch " << b;
    }
    ASSERT_LE(starts.back(), capacity);
}

/**
 * @brief Test fixture that runs a GraphTestCase on ASM_SDPA_ENGINE and checks it against
 * the CPU reference.
 */
template <typename DataType>
class IntegrationSdpaFwd : public IntegrationGraphVerificationHarness<DataType, GraphTestCase>
{
protected:
    // Fill inputs, identifying the fp8 q/k/v descales and the group-mode token tables
    // BY NAME rather than by element count. Element-count selection is fragile: it
    // silently breaks once per-(batch, KV-head) descales land (they stop being
    // element-count 1) and it cannot separate the descales from the equally-scalar
    // attention scale. Descales are dequantization factors, so they get the fixed,
    // positive, DISTINCT fill constants defined above (see K_DESCALE_*_FILL for the
    // distinct-and-fixed rationale). Offset tables get the token seqstarts of the case's
    // groupLengthSet lengths, or its physical starts in a cu_seq_len case, whose cumulative
    // tables (and the reference graph's seq_len tensors) get the lengths.
    // Every other tensor gets the plain random fill.
    void initializeBundle(const hipdnn_frontend::graph::Graph& graph,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        std::unordered_map<int64_t, std::string> uidToName;
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeInputTensorAttributes())
            {
                uidToName.emplace(tensorAttr->get_uid(), tensorAttr->get_name());
                if(tensorAttr->has_ragged_offset())
                {
                    const auto offset = tensorAttr->get_ragged_offset();
                    uidToName.emplace(offset->get_uid(), offset->get_name());
                }
            }
        });

        const GroupLengthSet& lengths = groupLengthSet(this->GetParam().groupLengths);
        for(auto& tensorPair : bundle.tensors)
        {
            const auto nameIt = uidToName.find(tensorPair.first);
            const bool hasName = (nameIt != uidToName.end());
            if(hasName && nameIt->second == K_DESCALE_Q_NAME)
            {
                bundle.randomizeTensor(tensorPair.first, K_DESCALE_Q_FILL, K_DESCALE_Q_FILL, seed);
            }
            else if(hasName && nameIt->second == K_DESCALE_K_NAME)
            {
                bundle.randomizeTensor(tensorPair.first, K_DESCALE_K_FILL, K_DESCALE_K_FILL, seed);
            }
            else if(hasName && nameIt->second == K_DESCALE_V_NAME)
            {
                bundle.randomizeTensor(tensorPair.first, K_DESCALE_V_FILL, K_DESCALE_V_FILL, seed);
            }
            else if(hasName && nameIt->second == K_QO_RAGGED_OFFSET_NAME)
            {
                if(this->GetParam().cuSeqLens)
                {
                    fillTable(*tensorPair.second, lengths.physicalStartsQ);
                }
                else
                {
                    fillSeqStarts(*tensorPair.second, lengths.seqLensQ);
                }
            }
            else if(hasName && nameIt->second == K_KV_RAGGED_OFFSET_NAME)
            {
                if(this->GetParam().cuSeqLens)
                {
                    fillTable(*tensorPair.second, lengths.physicalStartsKv);
                }
                else
                {
                    fillSeqStarts(*tensorPair.second, lengths.seqLensKv);
                }
            }
            else if(hasName && nameIt->second == K_CU_SEQ_LEN_Q_NAME)
            {
                fillSeqStarts(*tensorPair.second, lengths.seqLensQ);
            }
            else if(hasName && nameIt->second == K_CU_SEQ_LEN_KV_NAME)
            {
                fillSeqStarts(*tensorPair.second, lengths.seqLensKv);
            }
            else if(hasName && nameIt->second == K_SEQ_LEN_Q_NAME)
            {
                fillTable(*tensorPair.second, lengths.seqLensQ);
            }
            else if(hasName && nameIt->second == K_SEQ_LEN_KV_NAME)
            {
                fillTable(*tensorPair.second, lengths.seqLensKv);
            }
            else
            {
                bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
            }
        }
    }

    void runGraphTest(float tolerance)
    {
        auto deviceString = hip_kernel_provider_common::getDeviceString(this->stream());
        const GraphTestCase& testCase = this->GetParam();

        // Skip if device is not supported
        if(testCase.arch != deviceString)
        {
            GTEST_SKIP() << "Skipped: Test case requires " << testCase.arch
                         << " but current device architecture is " << deviceString;
        }

        const SdpaFwdGraph built = buildSdpaFwdGraph(testCase);
        const auto& graph = built.graph;
        const auto& stats = built.stats;

        auto validationResult = graph->validate();

        ASSERT_TRUE(validationResult.is_good())
            << "Graph validation failed for config: " << testCase.name << " - "
            << validationResult.get_message();

        // Group-mode cases fill their offset tables from groupLengthSet and are checked
        // against the CPU ragged reference. IntegrationGpuSdpaForwardRagged.cpp adds
        // frontend-built packed cases.
        if(testCase.config.mode == hip_kernel_provider_common::BatchMode::GROUP)
        {
            const GroupLengthSet& lengths = groupLengthSet(testCase.groupLengths);
            // The offset tables must fit the dense allocation the graph describes.
            ASSERT_EQ(testCase.batch, static_cast<int64_t>(lengths.seqLensQ.size()));
            ASSERT_EQ(testCase.batch, static_cast<int64_t>(lengths.seqLensKv.size()));
            ASSERT_LE(*std::max_element(lengths.seqLensQ.begin(), lengths.seqLensQ.end()),
                      testCase.seqQ);
            ASSERT_LE(*std::max_element(lengths.seqLensKv.begin(), lengths.seqLensKv.end()),
                      testCase.seqKv);
            if(testCase.groupLengths == GroupLengths::STANDARD)
            {
                // Each batch must span at least two Q-tiles of this row.
                ASSERT_GT(*std::min_element(lengths.seqLensQ.begin(), lengths.seqLensQ.end()),
                          testCase.config.ts_qo);
            }
            if(testCase.cuSeqLens)
            {
                // A shared cu_seq_len table gives K/V the Q lengths.
                const auto& kvLengths
                    = testCase.sharedCuSeqLen ? lengths.seqLensQ : lengths.seqLensKv;
                ASSERT_NO_FATAL_FAILURE(assertGappedSlots(
                    lengths.physicalStartsQ, lengths.seqLensQ, testCase.batch * testCase.seqQ));
                ASSERT_NO_FATAL_FAILURE(assertGappedSlots(
                    lengths.physicalStartsKv, kvLengths, testCase.batch * testCase.seqKv));
            }
        }

        // The CPU reference declines cu_seq_len_q/kv, so a cu_seq_len case runs it on the
        // same graph with the per-batch lengths, over the same tensors. Neither side writes
        // the rows in the gaps between slots; both bundles seed them alike, so comparing whole
        // outputs also checks that the engine leaves them untouched.
        std::shared_ptr<hipdnn_frontend::graph::Graph> referenceGraph;
        if(testCase.cuSeqLens)
        {
            referenceGraph = buildSdpaFwdReferenceGraph(testCase).graph;
            const auto referenceValidation = referenceGraph->validate();
            ASSERT_TRUE(referenceValidation.is_good())
                << "Reference graph validation failed for config: " << testCase.name << " - "
                << referenceValidation.get_message();
        }
        const auto contextOwner
            = referenceGraph ? std::make_unique<GraphVerificationContext>(*graph, *referenceGraph)
                             : std::make_unique<GraphVerificationContext>(*graph);
        GraphVerificationContext& context = *contextOwner;

        // Register output tensor validator
        graph->visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(tensorAttr->get_is_virtual())
                {
                    continue;
                }
                if(tensorAttr == stats)
                {
                    // A row that sees no key (fully masked by a causal mask, or in a batch
                    // with no keys) has a log-sum-exp of -inf in both the CPU reference and
                    // the device result.
                    this->registerValidator(context,
                                            tensorAttr,
                                            createAllCloseMatchingInfinitiesValidator(
                                                frontendToSdkDataType(tensorAttr->get_data_type()),
                                                tolerance,
                                                tolerance));
                }
                else
                {
                    this->registerValidator(context, tensorAttr, tolerance);
                }
            }
        });

        // Pinned to the ASM engine: other hip-kernel-provider engines serve some of these
        // graphs, so a heuristic build could pass while this engine declines its own row.
        ASSERT_NO_FATAL_FAILURE(this->buildPlansForEngine(*graph, ASM_SDPA_ENGINE_ID, {}));
        this->verifyBuiltGraph(context, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

using IntegrationGpuSdpaFwdBf16 = IntegrationSdpaFwd<bfloat16>;

/**
 * @brief Test fixture for shape-sweep tests parameterized on SdpaFwdTestCase.
 *
 * Builds a forward SDPA graph from explicit tensor dimensions, enabling
 * shape sweeps, GQA, and asymmetric sequence-length testing.
 */
template <typename DataType>
class IntegrationSdpaFwdShapeSweep
    : public IntegrationGraphVerificationHarness<DataType, SdpaFwdTestCase>
{
protected:
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            bundle.randomizeTensor(tensorPair.first, _minVal, _maxVal, seed);
        }
    }

    void runGraphTest(float tolerance)
    {
        // MI300/MI308 coverage: Both MI300 and MI308 report "gfx942" via getDeviceString()
        // but load different .co files (MI300/ vs MI308/ subdirectory in getKernelCoPath()).
        // Integration tests exercise whichever device the CI agent has. Full coverage
        // requires CI agents with both MI300 and MI308 hardware.
        auto deviceString = hip_kernel_provider_common::getDeviceString(this->stream());
        const SdpaFwdTestCase& testCase = this->GetParam();

        if(testCase.arch != deviceString)
        {
            GTEST_SKIP() << "Skipped: Test case requires " << testCase.arch
                         << " but current device architecture is " << deviceString;
        }

        const auto dataType = getDataTypeEnumFromType<DataType>();

        Graph graph;
        graph.set_io_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);

        auto q = std::make_shared<TensorAttributes>();
        q->set_dim(testCase.qDims)
            .set_stride(generateStrides(testCase.qDims))
            .set_data_type(dataType);

        auto k = std::make_shared<TensorAttributes>();
        k->set_dim(testCase.kDims)
            .set_stride(generateStrides(testCase.kDims))
            .set_data_type(dataType);

        auto v = std::make_shared<TensorAttributes>();
        v->set_dim(testCase.vDims)
            .set_stride(generateStrides(testCase.vDims))
            .set_data_type(dataType);

        SdpaAttributes attributes;
        attributes.set_name("SdpaFwdShapeSweepTest");

        // Configure mask: rightBound >= 0 indicates a causal/sliding-window mask.
        // leftBound is always forwarded (the -1 sentinel means "unbounded left").
        if(testCase.rightBound >= 0)
        {
            attributes.set_diagonal_band_left_bound(testCase.leftBound);
            attributes.set_diagonal_band_right_bound(testCase.rightBound);
            attributes.set_diagonal_alignment(testCase.topLeftAlignment
                                                  ? DiagonalAlignment::TOP_LEFT
                                                  : DiagonalAlignment::BOTTOM_RIGHT);
        }

        auto [o, stats] = graph.sdpa(q, k, v, attributes);

        o->set_output(true).set_data_type(dataType);

        auto validationResult = graph.validate();
        ASSERT_TRUE(validationResult.is_good()) << validationResult.get_message();

        GraphVerificationContext context(graph);
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& tensorAttr : node.getNodeOutputTensorAttributes())
            {
                if(!tensorAttr->get_is_virtual())
                {
                    this->registerValidator(context, tensorAttr, tolerance);
                }
            }
        });

        this->verifyGraph(context, 0);
    }

    float _minVal = -1.0;
    float _maxVal = 1.0;
};

std::vector<SdpaFwdTestCase> getSdpaFwdShapeSweepTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 NO_MASK cases ---
    // Minimal batch/heads
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // Multi-batch
    cases.emplace_back(
        std::vector<int64_t>{4, 2, 128, 128}, std::vector<int64_t>{4, 2, 128, 128}, "gfx942");
    // Long Q, short KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // Short Q, long KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 512, 128}, "gfx942");
    // HD192 kernel
    cases.emplace_back(
        std::vector<int64_t>{1, 2, 256, 192}, std::vector<int64_t>{1, 2, 128, 128}, "gfx942");

    // --- gfx942 BOTTOM_RIGHT_CAUSAL cases ---
    // Causal basic
    cases.emplace_back(std::vector<int64_t>{1, 1, 256, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Causal multi-batch
    cases.emplace_back(std::vector<int64_t>{2, 4, 256, 128},
                       std::vector<int64_t>{2, 4, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Causal HD192
    cases.emplace_back(std::vector<int64_t>{1, 1, 128, 192},
                       std::vector<int64_t>{1, 1, 128, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 NO_MASK cases ---
    // Minimal batch/heads
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // Multi-batch
    cases.emplace_back(
        std::vector<int64_t>{4, 2, 128, 128}, std::vector<int64_t>{4, 2, 128, 128}, "gfx950");
    // Long Q, short KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // Short Q, long KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 512, 128}, "gfx950");
    // HD192 kernel
    cases.emplace_back(
        std::vector<int64_t>{1, 2, 256, 192}, std::vector<int64_t>{1, 2, 128, 128}, "gfx950");

    return cases;
}

std::vector<SdpaFwdTestCase> getSdpaFwdGqaTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 GQA NO_MASK ---
    // GQA ratio=4
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // GQA ratio=2
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 2, 256, 128}, "gfx942");
    // GQA ratio=8
    cases.emplace_back(
        std::vector<int64_t>{1, 8, 128, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // GQA + asymmetric seq
    cases.emplace_back(
        std::vector<int64_t>{2, 4, 256, 128}, std::vector<int64_t>{2, 1, 128, 128}, "gfx942");
    // GQA + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 192}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");

    // --- gfx942 GQA BOTTOM_RIGHT_CAUSAL ---
    cases.emplace_back(std::vector<int64_t>{1, 4, 256, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 GQA NO_MASK ---
    // GQA ratio=4
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // GQA ratio=2
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 128}, std::vector<int64_t>{1, 2, 256, 128}, "gfx950");
    // GQA ratio=8
    cases.emplace_back(
        std::vector<int64_t>{1, 8, 128, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // GQA + asymmetric seq
    cases.emplace_back(
        std::vector<int64_t>{2, 4, 256, 128}, std::vector<int64_t>{2, 1, 128, 128}, "gfx950");
    // GQA + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 4, 256, 192}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");

    return cases;
}

std::vector<SdpaFwdTestCase> getSdpaFwdAsymSeqTestCases()
{
    std::vector<SdpaFwdTestCase> cases;

    // --- gfx942 NO_MASK ---
    // Q > KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx942");
    // KV > Q
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx942");
    // Large asymmetry
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 64, 128}, "gfx942");
    // Asym + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 192}, std::vector<int64_t>{1, 1, 512, 128}, "gfx942");

    // --- gfx942 BOTTOM_RIGHT_CAUSAL ---
    // Asym + causal
    cases.emplace_back(std::vector<int64_t>{1, 1, 256, 128},
                       std::vector<int64_t>{1, 1, 128, 128},
                       "gfx942",
                       -1,
                       0,
                       false);
    // Reversed asym + causal
    cases.emplace_back(std::vector<int64_t>{1, 1, 128, 128},
                       std::vector<int64_t>{1, 1, 256, 128},
                       "gfx942",
                       -1,
                       0,
                       false);

    // --- gfx950 NO_MASK ---
    // Q > KV
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 128}, std::vector<int64_t>{1, 1, 128, 128}, "gfx950");
    // KV > Q
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 128, 128}, std::vector<int64_t>{1, 1, 256, 128}, "gfx950");
    // Large asymmetry
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 512, 128}, std::vector<int64_t>{1, 1, 64, 128}, "gfx950");
    // Asym + HD192
    cases.emplace_back(
        std::vector<int64_t>{1, 1, 256, 192}, std::vector<int64_t>{1, 1, 512, 128}, "gfx950");

    return cases;
}

using IntegrationGpuSdpaFwdShapeSweepBf16 = IntegrationSdpaFwdShapeSweep<bfloat16>;

// Partition the kernel configs by precision so the bf16 and fp8 suites each run
// against their own kernels.
std::vector<GraphTestCase> getGraphTestCasesByPrecision(bool fp8)
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        const bool isFp8 = (config.dtype == hip_kernel_provider_common::config::FP8BF16);
        if(isFp8 == fp8)
        {
            cases.push_back(configToTestCase(config));
        }
    }
    return cases;
}

// Every batch-mode kernel paired with stats (getGroupStatsTestCases pairs the group-mode
// ones). The causal rows run at S_q > S_kv with bottom-right alignment, so their first
// S_q - S_kv query rows see no key and their LSE is -inf. Each causal row also runs a
// top-left causal mask at S_q == S_kv, which the engine serves with the same bottom-right
// kernel.
std::vector<GraphTestCase> getBatchModeStatsTestCases()
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(config.mode != hip_kernel_provider_common::BatchMode::BATCH)
        {
            continue;
        }
        cases.push_back(configToTestCase(config, /*withStats=*/true));
        if(static_cast<plan_utils::MaskType>(config.mask)
           == plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL)
        {
            auto topLeftConfig = config;
            topLeftConfig.mask = static_cast<int>(plan_utils::MaskType::TOP_LEFT_CAUSAL);
            auto topLeft = configToTestCase(topLeftConfig, /*withStats=*/true);
            topLeft.seqKv = topLeft.seqQ;
            cases.push_back(topLeft);
        }
    }
    return cases;
}

bool isGroupRow(const fmha_v3_fwdConfig& config)
{
    return config.mode == hip_kernel_provider_common::BatchMode::GROUP;
}

bool isBottomRightCausalRow(const fmha_v3_fwdConfig& config)
{
    return static_cast<plan_utils::MaskType>(config.mask)
           == plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL;
}

// configToTestCase for a group row over @p lengths, with the batch count of that set.
GraphTestCase groupTestCase(const fmha_v3_fwdConfig& config, bool withStats, GroupLengths lengths)
{
    auto testCase = configToTestCase(config, withStats);
    testCase.groupLengths = lengths;
    testCase.batch = static_cast<int64_t>(groupLengthSet(lengths).seqLensQ.size());
    return testCase;
}

// Each causal group row under a top-left causal mask, with Q, K, V and O on one offset
// table: every batch's Q and K/V lengths are then equal, where the two causal alignments
// agree, so the engine serves the top-left mask with the row's bottom-right kernel.
std::vector<GraphTestCase> getGroupTopLeftCausalTestCases()
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(!isGroupRow(config) || !isBottomRightCausalRow(config))
        {
            continue;
        }
        auto topLeftConfig = config;
        topLeftConfig.mask = static_cast<int>(plan_utils::MaskType::TOP_LEFT_CAUSAL);
        auto topLeft = configToTestCase(topLeftConfig);
        topLeft.sharedOffsetTable = true;
        topLeft.name += "SharedOffsets";
        cases.push_back(topLeft);
    }
    return cases;
}

// Every group row over @p lengths, without stats.
std::vector<GraphTestCase> getGroupTestCases(GroupLengths lengths)
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(isGroupRow(config))
        {
            cases.push_back(groupTestCase(config, /*withStats=*/false, lengths));
        }
    }
    return cases;
}

// Every group row over @p lengths with its stats as AITER's varlen LSE [H, total_q] on the
// Q/O offset table (buildSdpaFwdGraph), against the CPU ragged reference.
std::vector<GraphTestCase> getGroupStatsTestCases(GroupLengths lengths)
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(isGroupRow(config))
        {
            cases.push_back(groupTestCase(config, /*withStats=*/true, lengths));
        }
    }
    return cases;
}

// Every group row over @p lengths with cu_seq_len_q/kv over the set's gapped physical
// slots, without and with stats: each row is its own kernel object, and each must start a
// batch's LSE at the physical Q start, not the logical one. Each causal group row also runs
// under a top-left causal mask with one cu_seq_len table for both sides and separate K/V
// offsets, so the per-batch lengths are equal while the K/V slots differ from the Q slots.
std::vector<GraphTestCase> getGroupCuSeqLenTestCases(GroupLengths lengths)
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(!isGroupRow(config))
        {
            continue;
        }
        for(const bool withStats : {false, true})
        {
            auto padded = groupTestCase(config, withStats, lengths);
            padded.cuSeqLens = true;
            padded.name += "CuSeqLen";
            cases.push_back(padded);
        }
        if(isBottomRightCausalRow(config))
        {
            auto topLeftConfig = config;
            topLeftConfig.mask = static_cast<int>(plan_utils::MaskType::TOP_LEFT_CAUSAL);
            auto topLeft = groupTestCase(topLeftConfig, /*withStats=*/false, lengths);
            topLeft.cuSeqLens = true;
            topLeft.sharedCuSeqLen = true;
            topLeft.name += "SharedCuSeqLen";
            cases.push_back(topLeft);
        }
    }
    return cases;
}

// Every group row over GroupLengths::EDGE: a batch whose first rows see no key under
// bottom-right causal, a batch with no queries and a batch with no keys, on the offset
// tables without and with stats and through cu_seq_len over gapped slots.
std::vector<GraphTestCase> getGroupEdgeLengthTestCases()
{
    std::vector<GraphTestCase> cases = getGroupTestCases(GroupLengths::EDGE);
    const auto stats = getGroupStatsTestCases(GroupLengths::EDGE);
    cases.insert(cases.end(), stats.begin(), stats.end());
    const auto cuSeqLen = getGroupCuSeqLenTestCases(GroupLengths::EDGE);
    cases.insert(cases.end(), cuSeqLen.begin(), cuSeqLen.end());
    return cases;
}

// Every fp8 row with mma_core_mode set to each FP8 path value cuDNN writes: FP8_E4M3 (the
// sdpa_fp8() default) and FP8_E5M2. The mode names the MMA path, not the FP8_E4M3_FNUZ input
// encoding.
std::vector<GraphTestCase> getFp8MmaCoreModeTestCases()
{
    std::vector<GraphTestCase> cases;
    for(const auto& [key, config] : cfg_fmha_fwd)
    {
        if(config.dtype != hip_kernel_provider_common::config::FP8BF16)
        {
            continue;
        }
        for(const auto& [mode, suffix] :
            {std::pair{hipdnn_frontend::DataType::FP8_E4M3, "MmaCoreModeFp8E4m3"},
             std::pair{hipdnn_frontend::DataType::FP8_E5M2, "MmaCoreModeFp8E5m2"}})
        {
            auto fp8Mode = configToTestCase(config);
            fp8Mode.mmaCoreMode = mode;
            fp8Mode.name += suffix;
            cases.push_back(fp8Mode);
        }
    }
    return cases;
}

} // namespace

TEST_P(IntegrationGpuSdpaFwdBf16, Correctness)
{
    // One fixture covers both precisions (fp8 output is BF16, so the harness DataType
    // stays bfloat16 either way). FP8 (E4M3 FNUZ) inputs decode to identical values on
    // GPU and CPU, so the only divergence is fp8 quantization propagating through the
    // score/softmax/PV chain. That exceeds the bf16 1e-2 bound near a causal-mask
    // diagonal (fewer averaging terms => coarser fp8 rounding shows through); 3e-2
    // covers it. The dynamic SDPA tolerance model does not yet account for fp8 (see
    // DynamicTolerancesSdpa.hpp), so this fixed fp8 bound is the interim; tighten once
    // that lands.
    const bool isFp8 = (GetParam().config.dtype == hip_kernel_provider_common::config::FP8BF16);
    const float tolerance = isFp8 ? 3e-2f : 1e-2f;
    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGraphTestCasesByPrecision(/*fp8=*/false)),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(SmokeFp8,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGraphTestCasesByPrecision(/*fp8=*/true)),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(SmokeStats,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getBatchModeStatsTestCases()),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(SmokeFp8MmaCoreMode,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getFp8MmaCoreModeTestCases()),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(GroupTopLeftCausal,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGroupTopLeftCausalTestCases()),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(GroupStats,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGroupStatsTestCases(GroupLengths::STANDARD)),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(GroupCuSeqLen,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGroupCuSeqLenTestCases(GroupLengths::STANDARD)),
                         GraphTestCase::getName);

INSTANTIATE_TEST_SUITE_P(GroupEdgeLengths,
                         IntegrationGpuSdpaFwdBf16,
                         testing::ValuesIn(getGroupEdgeLengthTestCases()),
                         GraphTestCase::getName);

TEST_P(IntegrationGpuSdpaFwdShapeSweepBf16, Correctness)
{
    auto tolerance = 1e-2f;
    runGraphTest(tolerance);
}

INSTANTIATE_TEST_SUITE_P(ShapeSweep,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdShapeSweepTestCases()),
                         SdpaFwdTestCase::getName);

INSTANTIATE_TEST_SUITE_P(Gqa,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdGqaTestCases()),
                         SdpaFwdTestCase::getName);

INSTANTIATE_TEST_SUITE_P(AsymmetricSeq,
                         IntegrationGpuSdpaFwdShapeSweepBf16,
                         testing::ValuesIn(getSdpaFwdAsymSeqTestCases()),
                         SdpaFwdTestCase::getName);
