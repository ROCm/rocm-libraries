// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_sdk_test_utils;
using namespace ::testing;

namespace
{

const std::vector<DataType> ALL_MASK_DTYPES
    = {DataType::UNSET, DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};

SdpaBwdSignatureKey makeKey(DataType allTensors, DataType mask)
{
    return SdpaBwdSignatureKey{allTensors,
                               allTensors,
                               allTensors,
                               allTensors,
                               allTensors,
                               allTensors,
                               allTensors,
                               allTensors,
                               mask};
}

} // namespace

TEST(TestSdpaBwdSignatureKey, EqualityOperator)
{
    EXPECT_TRUE(makeKey(DataType::FLOAT, DataType::UNSET)
                == makeKey(DataType::FLOAT, DataType::UNSET));
    EXPECT_FALSE(makeKey(DataType::FLOAT, DataType::UNSET)
                 == makeKey(DataType::HALF, DataType::UNSET));

    const SdpaBwdSignatureKey dvDiffers{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF};
    EXPECT_FALSE(makeKey(DataType::FLOAT, DataType::UNSET) == dvDiffers);
}

TEST(TestSdpaBwdSignatureKey, MaskDataTypeParticipatesInEquality)
{
    const SdpaBwdSignatureKey noMask{DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT,
                                     DataType::FLOAT};
    EXPECT_TRUE(noMask == makeKey(DataType::FLOAT, DataType::UNSET));

    for(const auto lhsMask : ALL_MASK_DTYPES)
    {
        for(const auto rhsMask : ALL_MASK_DTYPES)
        {
            EXPECT_EQ(makeKey(DataType::FLOAT, lhsMask) == makeKey(DataType::FLOAT, rhsMask),
                      lhsMask == rhsMask)
                << "lhs mask " << lhsMask << ", rhs mask " << rhsMask;
        }
    }
}

TEST(TestSdpaBwdSignatureKey, MaskDataTypeParticipatesInHash)
{
    std::unordered_set<std::size_t> hashes;
    for(const auto mask : ALL_MASK_DTYPES)
    {
        hashes.insert(makeKey(DataType::HALF, mask).hashSelf());
    }
    EXPECT_EQ(hashes.size(), ALL_MASK_DTYPES.size());
}

// Every registered tensor-dtype combination resolves with the no-mask key and with each supported
// mask dtype; a mask dtype outside that set resolves to nothing.
TEST(TestSdpaBwdSignatureKey, PlanBuildersCoverSupportedMaskDtypes)
{
    const auto planBuilders = SdpaBwdSignatureKey::getPlanBuilders();

    for(const auto tensorDtype : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto mask : ALL_MASK_DTYPES)
        {
            EXPECT_NE(planBuilders.find(makeKey(tensorDtype, mask)), planBuilders.end())
                << "tensors=" << tensorDtype << ", mask=" << mask;
        }
        for(const auto unsupportedMask : {DataType::INT32, DataType::FP8_E4M3, DataType::DOUBLE})
        {
            EXPECT_EQ(planBuilders.find(makeKey(tensorDtype, unsupportedMask)), planBuilders.end())
                << "tensors=" << tensorDtype << ", mask=" << unsupportedMask;
        }
    }
}

template <class MaskT>
class SdpaBwdSignatureKeyMask : public Test
{
};

using MaskTypes
    = Types<float, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16, int32_t>;
TYPED_TEST_SUITE(SdpaBwdSignatureKeyMask, MaskTypes, );

TYPED_TEST(SdpaBwdSignatureKeyMask, NodeConstructorReadsDeclaredMaskDtype)
{
    SdpaBwdTensorBundle<float, TypeParam> tensorBundle({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    tensorBundle.maskTensor.emplace(std::vector<int64_t>{1, 1, 1, 1});
    auto graphTuple = buildSdpaBwdGraph(tensorBundle, DataType::FLOAT);
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
        serializedGraph.data(), serializedGraph.size());

    const SdpaBwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == makeKey(DataType::FLOAT, nativeTypeToDataType<TypeParam>()));
}

TEST(TestSdpaBwdSignatureKey, NodeConstructorWithoutMaskYieldsUnset)
{
    SdpaBwdTensorBundle<float> tensorBundle({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    auto graphTuple = buildSdpaBwdGraph(tensorBundle, DataType::FLOAT);
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
        serializedGraph.data(), serializedGraph.size());

    const SdpaBwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == makeKey(DataType::FLOAT, DataType::UNSET));
}
