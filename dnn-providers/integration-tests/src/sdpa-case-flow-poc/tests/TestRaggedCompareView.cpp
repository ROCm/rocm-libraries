// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include "RaggedCompareView.hpp"

namespace
{

using hipdnn_data_sdk::utilities::ITensor;
using hipdnn_data_sdk::utilities::Tensor;
using hipdnn_flatbuffers_sdk::data_objects::DataType;
using Validation = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float>;

// Two sequences in full 4-token slots of a token-major [B 2, H 1, S 4, D 2] output; sequence 0
// holds 3 live tokens and sequence 1 holds 1.
constexpr int64_t SLOT_TOKENS = 4;
constexpr int64_t HEAD_DIM = 2;
constexpr int64_t SEQUENCE_STRIDE = HEAD_DIM;
const std::vector<int64_t> DIMS = {2, 1, SLOT_TOKENS, HEAD_DIM};
const std::vector<int64_t> STRIDES = {SLOT_TOKENS * HEAD_DIM, HEAD_DIM, SEQUENCE_STRIDE, 1};
const std::vector<int32_t> LIVE_TOKENS = {3, 1};

std::shared_ptr<ITensor> int32Tensor(const std::vector<int32_t>& values)
{
    auto tensor = std::make_shared<Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(values.size()), 1, 1, 1},
        std::vector<int64_t>{1, 1, 1, 1});
    auto* data = static_cast<int32_t*>(tensor->rawHostData());
    for(size_t i = 0; i < values.size(); ++i)
    {
        data[i] = values[i];
    }
    return tensor;
}

// Live tokens hold 0.25 x their element index; every slack token holds NaN.
std::shared_ptr<Tensor<float>> outputWithNanSlack()
{
    auto tensor = std::make_shared<Tensor<float>>(DIMS, STRIDES);
    auto* data = static_cast<float*>(tensor->rawHostData());
    for(int64_t b = 0; b < DIMS[0]; ++b)
    {
        for(int64_t s = 0; s < SLOT_TOKENS; ++s)
        {
            for(int64_t d = 0; d < HEAD_DIM; ++d)
            {
                const int64_t i = b * STRIDES[0] + s * STRIDES[2] + d;
                data[i] = s < LIVE_TOKENS[static_cast<size_t>(b)]
                              ? 0.25f * static_cast<float>(i)
                              : std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
    return tensor;
}

struct RaggedPair
{
    std::shared_ptr<Tensor<float>> expected = outputWithNanSlack();
    std::shared_ptr<Tensor<float>> actual = outputWithNanSlack();
    std::shared_ptr<ITensor> offsets
        = int32Tensor({0,
                       static_cast<int32_t>(SLOT_TOKENS* HEAD_DIM),
                       static_cast<int32_t>(2 * SLOT_TOKENS * HEAD_DIM)});
    std::shared_ptr<ITensor> lengths = int32Tensor(LIVE_TOKENS);

    bool viewsAgree()
    {
        auto expectedView
            = sdpa_case_flow::raggedView(*expected, DataType::FLOAT, offsets, 1, lengths);
        auto actualView = sdpa_case_flow::raggedView(*actual, DataType::FLOAT, offsets, 1, lengths);
        return Validation(1e-3f, 1e-3f).allClose(*expectedView, *actualView);
    }

    float& actualAt(int64_t b, int64_t s, int64_t d)
    {
        return static_cast<float*>(actual->rawHostData())[b * STRIDES[0] + s * STRIDES[2] + d];
    }
};

} // namespace

TEST(TestRaggedCompareView, DenseCompareFailsOnNanSlackInBothOutputs)
{
    RaggedPair pair;
    EXPECT_FALSE(Validation(1e-3f, 1e-3f).allClose(*pair.expected, *pair.actual));
}

TEST(TestRaggedCompareView, RaggedViewsSkipNanSlackInBothOutputs)
{
    RaggedPair pair;
    EXPECT_TRUE(pair.viewsAgree());
}

TEST(TestRaggedCompareView, RaggedViewsSkipSlackThatDiffers)
{
    RaggedPair pair;
    for(int64_t s = LIVE_TOKENS[1]; s < SLOT_TOKENS; ++s)
    {
        for(int64_t d = 0; d < HEAD_DIM; ++d)
        {
            static_cast<float*>(pair.expected->rawHostData())[STRIDES[0] + s * STRIDES[2] + d]
                = 0.0f;
            pair.actualAt(1, s, d) = 7.0f;
        }
    }
    EXPECT_TRUE(pair.viewsAgree());
}

TEST(TestRaggedCompareView, RaggedViewsFailOnADifferentLiveValue)
{
    RaggedPair pair;
    pair.actualAt(0, 2, 1) += 1.0f;
    EXPECT_FALSE(pair.viewsAgree());
}

TEST(TestRaggedCompareView, RaggedViewsFailOnNanInOneLiveValue)
{
    RaggedPair pair;
    pair.actualAt(1, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(pair.viewsAgree());
}

TEST(TestRaggedCompareView, RaggedViewsFailWhenALiveTokenIsLeftUnwritten)
{
    RaggedPair pair;
    pair.actualAt(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    pair.actualAt(0, 0, 1) = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(pair.viewsAgree());
}
