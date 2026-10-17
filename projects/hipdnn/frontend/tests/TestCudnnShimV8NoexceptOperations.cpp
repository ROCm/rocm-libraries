// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Operation builder status-return path: with exceptions disabled a rejected
// operation keeps its status and message instead of throwing.
#ifndef NV_CUDNN_DISABLE_EXCEPTION
#error "TestCudnnShimV8NoexceptOperations.cpp must be built with NV_CUDNN_DISABLE_EXCEPTION"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::DataType_t;
using fe::DescriptorType_t;
using ::testing::HasSubstr;

using Dims = std::vector<std::int64_t>;

const Dims X_DIMS = {2, 4, 8, 8};

class TestCudnnShimV8NoexceptOperations : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        std::array<char, 1024> message{};
        cudnnGetLastErrorString(message.data(), message.size());
    }

    static fe::Tensor
        tensor(std::int64_t uid, const Dims& dims, DataType_t dataType = DataType_t::FLOAT)
    {
        Dims strides(dims.size(), 1);
        for(std::size_t i = dims.size() - 1; i > 0; --i)
        {
            strides[i - 1] = strides[i] * dims[i];
        }
        return fe::TensorBuilder()
            .setDataType(dataType)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .build();
    }
};

TEST_F(TestCudnnShimV8NoexceptOperations, UnloweredOpSetsStatus)
{
    const Dims channel = {1, 4, 1, 1};
    const fe::Operation genStats
        = fe::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR)
              .setxDesc(tensor(1, X_DIMS))
              .setSumDesc(tensor(2, channel))
              .setSqSumDesc(tensor(3, channel))
              .build();

    EXPECT_EQ(genStats.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(std::string(genStats.get_error()), HasSubstr("cudnnFinalize Failed"));
    EXPECT_THAT(std::string(genStats.get_error()),
                HasSubstr("CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR is not supported"));
    EXPECT_EQ(genStats.getTag(), "GenStats");

    // Upstream's paged-cache-load builder never sets a tag.
    const fe::Operation pagedCacheLoad
        = fe::OperationBuilder(DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)
              .setcontainerDesc(tensor(1, {8, 2, 16, 64}))
              .setpageTableDesc(tensor(2, {2, 1, 4, 1}, DataType_t::INT32))
              .setsequenceDesc(tensor(3, {2, 1, 1, 1}, DataType_t::INT32))
              .setyDesc(tensor(4, {2, 2, 64, 64}))
              .build();

    EXPECT_EQ(pagedCacheLoad.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(std::string(pagedCacheLoad.get_error()),
                HasSubstr("CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR is not supported"));
    EXPECT_EQ(pagedCacheLoad.getTag(), "");
}

TEST_F(TestCudnnShimV8NoexceptOperations, WrongKindSetterSetsStatus)
{
    const fe::PointWiseDesc relu = fe::PointWiseDescBuilder()
                                       .setMode(fe::PointwiseMode_t::RELU_FWD)
                                       .setComputeType(DataType_t::FLOAT)
                                       .build();
    fe::OperationBuilder pointwiseBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    pointwiseBuilder.setpwDesc(relu)
        .setxDesc(tensor(1, X_DIMS))
        .setwDesc(tensor(2, {8, 4, 3, 3}))
        .setyDesc(tensor(3, X_DIMS));
    const fe::Operation onPointwise = pointwiseBuilder.build();

    // build() reports the status the setter recorded and creates no descriptor.
    EXPECT_EQ(onPointwise.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_THAT(std::string(onPointwise.get_error()),
                HasSubstr("Operation not initialized properly"));
    EXPECT_EQ(onPointwise.get_desc(), nullptr);

    fe::OperationBuilder convBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    convBuilder.setbDesc(tensor(4, {1, 8, 1, 1}));
    const fe::Operation onConv = convBuilder.build();

    EXPECT_EQ(onConv.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(onConv.get_desc(), nullptr);
}

} // namespace
