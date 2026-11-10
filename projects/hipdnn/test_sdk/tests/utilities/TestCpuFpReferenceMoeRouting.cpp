// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeRouting.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using hipdnn_data_sdk::utilities::generateStrides;
using hipdnn_data_sdk::utilities::Tensor;
using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;

namespace
{

constexpr int32_t K_UNTOUCHED = -7;
constexpr float K_UNTOUCHED_FLOAT = -1000.0F;

template <typename Type>
Tensor<Type> createTensor(const std::vector<int64_t>& dims)
{
    return Tensor<Type>(dims, generateStrides(dims));
}

template <typename Type>
void setValues(Tensor<Type>& tensor, const std::vector<int64_t>& values)
{
    ASSERT_EQ(static_cast<size_t>(tensor.elementCount()), values.size());
    auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = static_cast<Type>(values[i]);
    }
}

template <typename Type>
std::vector<int64_t> values(const Tensor<Type>& tensor)
{
    std::vector<int64_t> result;
    const auto* data = tensor.memory().hostData();
    for(size_t i = 0; i < tensor.elementCount(); ++i)
    {
        result.push_back(static_cast<int64_t>(data[i]));
    }
    return result;
}

template <typename Callable>
void expectRejectedWith(Callable&& call, const std::string& needle)
{
    try
    {
        call();
        ADD_FAILURE() << "expected std::runtime_error containing \"" << needle << "\"";
    }
    catch(const std::runtime_error& error)
    {
        EXPECT_NE(std::string(error.what()).find(needle), std::string::npos) << error.what();
    }
}

/// Routing outputs for `ids` ([tokens, topK]) over `experts`, with token_index / token_ks
/// pre-filled with K_UNTOUCHED so unwritten rows are observable.
template <typename OffsetType>
struct Routed
{
    Tensor<OffsetType> offsets;
    Tensor<int32_t> tokenIndex;
    Tensor<int32_t> tokenKs;
};

template <typename OffsetType>
Routed<OffsetType>
    route(int64_t tokens, int64_t topK, int64_t experts, const std::vector<int64_t>& ids)
{
    auto topkIds = createTensor<int32_t>({tokens, topK});
    setValues(topkIds, ids);
    Routed<OffsetType> routed{createTensor<OffsetType>({experts + 1, 1, 1}),
                              createTensor<int32_t>({1, tokens * topK, 1}),
                              createTensor<int32_t>({1, tokens * topK, 1})};
    routed.tokenIndex.fillWithValue(K_UNTOUCHED);
    routed.tokenKs.fillWithValue(K_UNTOUCHED);

    CpuFpReferenceMoeRouting::route<OffsetType>(
        topkIds, experts, routed.offsets, routed.tokenIndex, routed.tokenKs);
    return routed;
}

template <typename OffsetType>
void runSortsByExpertThenTokenThenSlot()
{
    // ids [[2, 0], [0, 1], [2, 2]] over 3 experts:
    //   expert 0: (0, 1), (1, 0); expert 1: (1, 1); expert 2: (0, 0), (2, 0), (2, 1).
    const auto routed = route<OffsetType>(3, 2, 3, {2, 0, 0, 1, 2, 2});

    EXPECT_EQ(values(routed.offsets), (std::vector<int64_t>{0, 2, 3, 6}));
    EXPECT_EQ(values(routed.tokenIndex), (std::vector<int64_t>{0, 1, 1, 0, 2, 2}));
    EXPECT_EQ(values(routed.tokenKs), (std::vector<int64_t>{1, 0, 1, 0, 0, 1}));
}

} // namespace

TEST(TestCpuFpReferenceMoeRouting, SortsByExpertThenTokenThenSlotInt32Offsets)
{
    runSortsByExpertThenTokenThenSlot<int32_t>();
}

TEST(TestCpuFpReferenceMoeRouting, SortsByExpertThenTokenThenSlotInt64Offsets)
{
    runSortsByExpertThenTokenThenSlot<int64_t>();
}

TEST(TestCpuFpReferenceMoeRouting, AllToOneExpertKeepsSourceOrderAndTies)
{
    // Every slot picks expert 1, including both slots of each token (ties): the rows keep
    // the (s, k) order and experts 0 and 2 get empty groups.
    const auto routed = route<int32_t>(2, 2, 3, {1, 1, 1, 1});

    EXPECT_EQ(values(routed.offsets), (std::vector<int64_t>{0, 0, 4, 4}));
    EXPECT_EQ(values(routed.tokenIndex), (std::vector<int64_t>{0, 0, 1, 1}));
    EXPECT_EQ(values(routed.tokenKs), (std::vector<int64_t>{0, 1, 0, 1}));
}

TEST(TestCpuFpReferenceMoeRouting, OutOfRangeIdsAreNotRoutedAndTrailingRowsUntouched)
{
    // Expert parallelism: ids 5 and -1 belong to no local expert.
    //   expert 0: (0, 0), (2, 1); expert 1: (1, 1), (2, 0).
    const auto routed = route<int32_t>(3, 2, 2, {0, 5, -1, 1, 1, 0});

    EXPECT_EQ(values(routed.offsets), (std::vector<int64_t>{0, 2, 4}));
    EXPECT_EQ(values(routed.tokenIndex),
              (std::vector<int64_t>{0, 2, 1, 2, K_UNTOUCHED, K_UNTOUCHED}));
    EXPECT_EQ(values(routed.tokenKs), (std::vector<int64_t>{0, 1, 1, 0, K_UNTOUCHED, K_UNTOUCHED}));
}

TEST(TestCpuFpReferenceMoeRouting, NoLocalIdsRoutesNothing)
{
    const auto routed = route<int64_t>(2, 1, 2, {2, -3});

    EXPECT_EQ(values(routed.offsets), (std::vector<int64_t>{0, 0, 0}));
    EXPECT_EQ(values(routed.tokenIndex), (std::vector<int64_t>{K_UNTOUCHED, K_UNTOUCHED}));
    EXPECT_EQ(values(routed.tokenKs), (std::vector<int64_t>{K_UNTOUCHED, K_UNTOUCHED}));
}

TEST(TestCpuFpReferenceMoeRouting, SingleExpert)
{
    // E = 1: ids other than 0 are out of range.
    const auto routed = route<int32_t>(2, 2, 1, {0, 3, 0, 0});

    EXPECT_EQ(values(routed.offsets), (std::vector<int64_t>{0, 3}));
    EXPECT_EQ(values(routed.tokenIndex), (std::vector<int64_t>{0, 1, 1, K_UNTOUCHED}));
    EXPECT_EQ(values(routed.tokenKs), (std::vector<int64_t>{0, 0, 1, K_UNTOUCHED}));
}

TEST(TestCpuFpReferenceMoeRouting, InvalidShapesThrow)
{
    auto topkIds = createTensor<int32_t>({2, 2});
    setValues(topkIds, {0, 1, 1, 0});
    auto offsets = createTensor<int32_t>({3, 1, 1});
    auto rows = createTensor<int32_t>({1, 4, 1});
    auto rowsKs = createTensor<int32_t>({1, 4, 1});

    expectRejectedWith(
        [&] { CpuFpReferenceMoeRouting::route<int32_t>(topkIds, 0, offsets, rows, rowsKs); },
        "num_experts must be at least 1");

    auto badOffsets = createTensor<int32_t>({2, 1, 1});
    expectRejectedWith(
        [&] { CpuFpReferenceMoeRouting::route<int32_t>(topkIds, 2, badOffsets, rows, rowsKs); },
        "first_token_offset must have shape");

    auto badRows = createTensor<int32_t>({1, 3, 1});
    expectRejectedWith(
        [&] { CpuFpReferenceMoeRouting::route<int32_t>(topkIds, 2, offsets, badRows, rowsKs); },
        "token_index must have shape");
    expectRejectedWith(
        [&] { CpuFpReferenceMoeRouting::route<int32_t>(topkIds, 2, offsets, rows, badRows); },
        "token_ks must have shape");

    auto rank3Ids = createTensor<int32_t>({1, 2, 2});
    expectRejectedWith(
        [&] { CpuFpReferenceMoeRouting::route<int32_t>(rank3Ids, 2, offsets, rows, rowsKs); },
        "topk_ids must have shape");
}

/* ========== Contract: routing outputs feed MoE grouped matmul GATHER then SCATTER ========== */

namespace
{

/// x [1, S, K] -> per (s, k) with local expert e: y[s * topK + k] = (x[s] @ w1[e]) @ w2[e],
/// computed via MoeRouting -> MoeGroupedMatmul GATHER (w1) -> SCATTER (w2), and checked
/// against the direct per-token expert matmul. Rows of non-local (s, k) stay untouched.
void runRoutingFeedsGatherThenScatter(const std::vector<int64_t>& ids)
{
    constexpr int64_t TOKENS = 3;
    constexpr int64_t TOP_K = 2;
    constexpr int64_t EXPERTS = 2;
    constexpr int64_t HIDDEN = 2; // K
    constexpr int64_t INTER = 3; // H
    constexpr int64_t OUT = 2; // N
    constexpr int64_t ROUTED = TOKENS * TOP_K;

    const std::vector<float> xValues = {1, 2, -1, 3, 2, 0};
    // w1[e] is [K, H], w2[e] is [H, N], row-major.
    const std::vector<float> w1Values = {1, 0, 2, 0, 1, -1, /* e1 */ 2, 1, 0, -1, 1, 1};
    const std::vector<float> w2Values = {1, 2, 0, 1, -1, 0, /* e1 */ 0, 1, 1, 0, 2, -2};

    auto x = createTensor<float>({1, TOKENS, HIDDEN});
    auto w1 = createTensor<float>({EXPERTS, HIDDEN, INTER});
    auto w2 = createTensor<float>({EXPERTS, INTER, OUT});
    std::copy(xValues.begin(), xValues.end(), x.memory().hostData());
    std::copy(w1Values.begin(), w1Values.end(), w1.memory().hostData());
    std::copy(w2Values.begin(), w2Values.end(), w2.memory().hostData());

    // Routing outputs, passed verbatim: [E + 1, 1, 1] explicit boundaries and
    // [1, S * top_k, 1] row indices.
    const auto routed = route<int32_t>(TOKENS, TOP_K, EXPERTS, ids);

    auto gathered = createTensor<float>({1, ROUTED, INTER});
    gathered.fillWithValue(K_UNTOUCHED_FLOAT);
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(
        x, w1, routed.offsets, gathered, Mode::GATHER, 0, &routed.tokenIndex);

    auto y = createTensor<float>({1, ROUTED, OUT});
    y.fillWithValue(K_UNTOUCHED_FLOAT);
    CpuFpReferenceMoeGroupedMatmul::forward<float, float, float, float>(gathered,
                                                                        w2,
                                                                        routed.offsets,
                                                                        y,
                                                                        Mode::SCATTER,
                                                                        static_cast<int32_t>(TOP_K),
                                                                        &routed.tokenIndex,
                                                                        &routed.tokenKs);

    for(int64_t s = 0; s < TOKENS; ++s)
    {
        for(int64_t k = 0; k < TOP_K; ++k)
        {
            const int64_t expert = ids[static_cast<size_t>(s * TOP_K + k)];
            for(int64_t n = 0; n < OUT; ++n)
            {
                float expected = K_UNTOUCHED_FLOAT;
                if(expert >= 0 && expert < EXPERTS)
                {
                    expected = 0.0F;
                    for(int64_t h = 0; h < INTER; ++h)
                    {
                        float hidden = 0.0F;
                        for(int64_t kk = 0; kk < HIDDEN; ++kk)
                        {
                            hidden += xValues[static_cast<size_t>(s * HIDDEN + kk)]
                                      * w1Values[static_cast<size_t>((expert * HIDDEN + kk) * INTER
                                                                     + h)];
                        }
                        expected += hidden
                                    * w2Values[static_cast<size_t>((expert * INTER + h) * OUT + n)];
                    }
                }
                EXPECT_EQ(y.getHostValue({0, s * TOP_K + k, n}), expected)
                    << "s=" << s << " k=" << k << " n=" << n;
            }
        }
    }
}

} // namespace

TEST(TestCpuFpReferenceMoeRouting, RoutingFeedsGatherThenScatter)
{
    runRoutingFeedsGatherThenScatter({1, 0, 0, 1, 1, 1});
}

TEST(TestCpuFpReferenceMoeRouting, RoutingFeedsGatherThenScatterWithExpertParallelism)
{
    // Ids 2 and -1 are not local: their output rows stay untouched, and the routed rows past
    // the final boundary are never read by GATHER or SCATTER.
    runRoutingFeedsGatherThenScatter({1, 2, -1, 0, 1, 1});
}
