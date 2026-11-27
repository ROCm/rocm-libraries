// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <string>
#include <utility>
#include <vector>

#include "MiopenApi.hpp"
#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "MiopenPoolingDescriptor.hpp"

using namespace miopen_plugin;

struct PoolingDescriptorCase
{
    std::string name;
    miopenPoolingMode_t mode;
    std::vector<int> window;
    std::vector<int> padding;
    std::vector<int> stride;
    miopenIndexType_t indexType;
};

namespace
{

std::vector<PoolingDescriptorCase> getRoundTripCases()
{
    return {
        {"Max2dUint8", miopenPoolingMax, {2, 3}, {1, 0}, {2, 1}, miopenIndexUint8},
        {"Max2dUint32", miopenPoolingMax, {3, 2}, {0, 1}, {1, 2}, miopenIndexUint32},
        {"Average2d", miopenPoolingAverage, {3, 3}, {1, 1}, {2, 2}, miopenIndexUint8},
        {"AverageInclusive2d",
         miopenPoolingAverageInclusive,
         {2, 2},
         {0, 0},
         {2, 2},
         miopenIndexUint16},
        {"Max3dUint32", miopenPoolingMax, {2, 3, 4}, {1, 0, 2}, {2, 1, 3}, miopenIndexUint32},
        {"Max3dUint64", miopenPoolingMax, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, miopenIndexUint64},
        {"Average3d", miopenPoolingAverage, {3, 3, 3}, {1, 1, 1}, {1, 2, 3}, miopenIndexUint8},
        {"AverageInclusive3d",
         miopenPoolingAverageInclusive,
         {2, 4, 2},
         {1, 2, 1},
         {2, 2, 2},
         miopenIndexUint32},
    };
}

void expectBadParam(const std::vector<int>& window,
                    const std::vector<int>& padding,
                    const std::vector<int>& stride)
{
    try
    {
        const MiopenPoolingDescriptor descriptor(
            miopenPoolingMax, window, padding, stride, miopenIndexUint8);
        ADD_FAILURE() << "expected HipdnnPluginException";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_EQ(e.getStatus(), HIPDNN_PLUGIN_STATUS_BAD_PARAM);
    }
}

} // namespace

class TestMiopenPoolingDescriptorRoundTrip : public ::testing::TestWithParam<PoolingDescriptorCase>
{
};

TEST_P(TestMiopenPoolingDescriptorRoundTrip, DescriptorReportsConfiguredValues)
{
    const auto& param = GetParam();
    const MiopenPoolingDescriptor descriptor(
        param.mode, param.window, param.padding, param.stride, param.indexType);
    ASSERT_NE(descriptor.poolingDescriptor(), nullptr);

    const auto spatialRank = static_cast<int>(param.window.size());
    miopenPoolingMode_t mode{};
    int nbDims = 0;
    std::vector<int> window(param.window.size(), -1);
    std::vector<int> padding(param.window.size(), -1);
    std::vector<int> stride(param.window.size(), -1);
    ASSERT_EQ(miopenGetNdPoolingDescriptor(descriptor.poolingDescriptor(),
                                           spatialRank,
                                           &mode,
                                           &nbDims,
                                           window.data(),
                                           padding.data(),
                                           stride.data()),
              miopenStatusSuccess);

    EXPECT_EQ(mode, param.mode);
    EXPECT_EQ(nbDims, spatialRank);
    EXPECT_EQ(window, param.window);
    EXPECT_EQ(padding, param.padding);
    EXPECT_EQ(stride, param.stride);

    // miopenSetNdPoolingDescriptor resets both of these, so they only survive if the descriptor
    // applies them afterwards.
    miopenIndexType_t indexType{};
    ASSERT_EQ(miopenGetPoolingIndexType(descriptor.poolingDescriptor(), &indexType),
              miopenStatusSuccess);
    EXPECT_EQ(indexType, param.indexType);

    // Image mode for 2D too, where MIOpen's own default is mask mode.
    miopenPoolingWorkspaceIndexMode_t indexMode{};
    ASSERT_EQ(miopenGetPoolingWorkSpaceIndexMode(descriptor.poolingDescriptor(), &indexMode),
              miopenStatusSuccess);
    EXPECT_EQ(indexMode, miopenPoolingWorkspaceIndexImage);
}

INSTANTIATE_TEST_SUITE_P(AllCases,
                         TestMiopenPoolingDescriptorRoundTrip,
                         ::testing::ValuesIn(getRoundTripCases()),
                         [](const ::testing::TestParamInfo<PoolingDescriptorCase>& info) {
                             return info.param.name;
                         });

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForPaddingLengthMismatch)
{
    expectBadParam({2, 2}, {0}, {2, 2});
}

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForStrideLengthMismatch)
{
    expectBadParam({2, 2}, {0, 0}, {2, 2, 2});
}

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForWindowLengthMismatch)
{
    expectBadParam({2, 2, 2}, {0, 0}, {2, 2});
}

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForNoSpatialDims)
{
    expectBadParam({}, {}, {});
}

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForOneSpatialDim)
{
    expectBadParam({2}, {0}, {2});
}

TEST(TestMiopenPoolingDescriptor, ThrowsBadParamForFourSpatialDims)
{
    expectBadParam({2, 2, 2, 2}, {0, 0, 0, 0}, {1, 1, 1, 1});
}

TEST(TestMiopenPoolingDescriptor, MoveConstructorTransfersOwnershipAndNullsSource)
{
    MiopenPoolingDescriptor source(miopenPoolingMax, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint32);
    auto* const handle = source.poolingDescriptor();
    ASSERT_NE(handle, nullptr);

    const MiopenPoolingDescriptor destination(std::move(source));

    EXPECT_EQ(destination.poolingDescriptor(), handle);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.poolingDescriptor(), nullptr);

    miopenIndexType_t indexType{};
    ASSERT_EQ(miopenGetPoolingIndexType(destination.poolingDescriptor(), &indexType),
              miopenStatusSuccess);
    EXPECT_EQ(indexType, miopenIndexUint32);
}

TEST(TestMiopenPoolingDescriptor, MoveAssignmentTransfersOwnershipAndNullsSource)
{
    MiopenPoolingDescriptor source(
        miopenPoolingAverage, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, miopenIndexUint8);
    MiopenPoolingDescriptor destination(
        miopenPoolingMax, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint32);
    auto* const handle = source.poolingDescriptor();
    ASSERT_NE(handle, nullptr);
    ASSERT_NE(destination.poolingDescriptor(), handle);

    destination = std::move(source);

    EXPECT_EQ(destination.poolingDescriptor(), handle);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.poolingDescriptor(), nullptr);

    miopenPoolingMode_t mode{};
    int nbDims = 0;
    ASSERT_EQ(miopenGetNdPoolingDescriptor(
                  destination.poolingDescriptor(), 0, &mode, &nbDims, nullptr, nullptr, nullptr),
              miopenStatusSuccess);
    EXPECT_EQ(mode, miopenPoolingAverage);
    EXPECT_EQ(nbDims, 3);
}

TEST(TestMiopenPoolingDescriptor, MoveAssignmentFromMovedFromSourceLeavesBothNull)
{
    MiopenPoolingDescriptor first(miopenPoolingMax, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8);
    const MiopenPoolingDescriptor keeper(std::move(first));
    MiopenPoolingDescriptor second(miopenPoolingMax, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    second = std::move(first);

    EXPECT_EQ(second.poolingDescriptor(), nullptr);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(first.poolingDescriptor(), nullptr);
    EXPECT_NE(keeper.poolingDescriptor(), nullptr);
}

TEST(TestMiopenPoolingDescriptor, SelfMoveAssignmentKeepsDescriptor)
{
    MiopenPoolingDescriptor descriptor(miopenPoolingMax, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8);
    auto* const handle = descriptor.poolingDescriptor();

    // Through a reference so the compiler does not reject the self-move outright.
    auto& alias = descriptor;
    descriptor = std::move(alias);

    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(descriptor.poolingDescriptor(), handle);
}
