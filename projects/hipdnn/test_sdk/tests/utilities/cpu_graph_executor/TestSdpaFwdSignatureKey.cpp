// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;
using namespace ::testing;

TEST(TestSdpaFwdSignatureKey, EqualityOperator)
{
    const SdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const SdpaFwdSignatureKey key3{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    const SdpaFwdSignatureKey key4{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    const SdpaFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key6{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    const SdpaFwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key8{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const SdpaFwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestSdpaFwdSignatureKey, HashFunction)
{
    const SdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const SdpaFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const SdpaFwdSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestSdpaFwdSignatureKey, Copy)
{
    const SdpaFwdSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::FLOAT};
    const SdpaFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.qDataType, DataType::FLOAT);
    EXPECT_EQ(copied.kDataType, DataType::HALF);
    EXPECT_EQ(copied.vDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.oDataType, DataType::FLOAT);
}

TEST(TestSdpaFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const SdpaFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    const std::vector<int64_t> qDims = {1, 1, 1, 1};
    const std::vector<int64_t> kDims = {1, 1, 1, 1};
    const std::vector<int64_t> vDims = {1, 1, 1, 1};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims);
    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const SdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}

namespace
{
const std::vector<DataType> ALL_MASK_DTYPES
    = {DataType::UNSET, DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};
}

TEST(TestSdpaFwdSignatureKey, MaskDataTypeParticipatesInEquality)
{
    const SdpaFwdSignatureKey noMask{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey explicitNoMask{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_TRUE(noMask == explicitNoMask);

    for(const auto lhsMask : ALL_MASK_DTYPES)
    {
        for(const auto rhsMask : ALL_MASK_DTYPES)
        {
            const SdpaFwdSignatureKey lhs{
                DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, lhsMask};
            const SdpaFwdSignatureKey rhs{
                DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, rhsMask};
            EXPECT_EQ(lhs == rhs, lhsMask == rhsMask)
                << "lhs mask " << lhsMask << ", rhs mask " << rhsMask;
        }
    }
}

TEST(TestSdpaFwdSignatureKey, MaskDataTypeParticipatesInHash)
{
    std::unordered_set<std::size_t> hashes;
    for(const auto mask : ALL_MASK_DTYPES)
    {
        const SdpaFwdSignatureKey key{
            DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, mask};
        hashes.insert(key.hashSelf());
    }
    EXPECT_EQ(hashes.size(), ALL_MASK_DTYPES.size());
}

// Every Q/K/V/O combination resolves with the no-mask key and with each supported mask dtype;
// a mask dtype outside that set resolves to nothing.
TEST(TestSdpaFwdSignatureKey, PlanBuildersCoverSupportedMaskDtypes)
{
    const auto planBuilders = SdpaFwdSignatureKey::getPlanBuilders();

    const std::vector<std::array<DataType, 4>> qkvoCombinations = {
        {DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT},
        {DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF},
        {DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16},
        {DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT},
        {DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT},
        {DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16},
        {DataType::FP8_E4M3_FNUZ,
         DataType::FP8_E4M3_FNUZ,
         DataType::FP8_E4M3_FNUZ,
         DataType::BFLOAT16},
    };

    for(const auto& [q, k, v, o] : qkvoCombinations)
    {
        for(const auto mask : ALL_MASK_DTYPES)
        {
            EXPECT_NE(planBuilders.find(SdpaFwdSignatureKey{q, k, v, o, mask}), planBuilders.end())
                << "q=" << q << ", o=" << o << ", mask=" << mask;
        }
        for(const auto unsupportedMask : {DataType::INT32, DataType::FP8_E4M3, DataType::DOUBLE})
        {
            EXPECT_EQ(planBuilders.find(SdpaFwdSignatureKey{q, k, v, o, unsupportedMask}),
                      planBuilders.end())
                << "q=" << q << ", o=" << o << ", mask=" << unsupportedMask;
        }
    }
}

template <class MaskT>
class SdpaFwdSignatureKeyMask : public Test
{
};

using MaskTypes
    = Types<float, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16, int32_t>;
TYPED_TEST_SUITE(SdpaFwdSignatureKeyMask, MaskTypes, );

TYPED_TEST(SdpaFwdSignatureKeyMask, NodeConstructorReadsDeclaredMaskDtype)
{
    SdpaFwdTensorBundle<float, TypeParam> tensorBundle({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    tensorBundle.maskTensor.emplace(std::vector<int64_t>{1, 1, 1, 1});
    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
        serializedGraph.data(), serializedGraph.size());

    const SdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    const SdpaFwdSignatureKey expectedKey{DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          nativeTypeToDataType<TypeParam>()};
    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestSdpaFwdSignatureKey, NodeConstructorWithoutMaskYieldsUnset)
{
    SdpaFwdTensorBundle<float> tensorBundle({1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1});
    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrap(
        serializedGraph.data(), serializedGraph.size());

    const SdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_EQ(keyFromNode.maskDataType, DataType::UNSET);
}
