// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builder status-return path: with exceptions disabled a failed build() leaves
// its status and message on the returned descriptor instead of throwing.
#ifndef NV_CUDNN_DISABLE_EXCEPTION
#error "TestCudnnShimV8NoExcept.cpp must be built with NV_CUDNN_DISABLE_EXCEPTION"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using ::testing::HasSubstr;

constexpr std::array<int64_t, 4> DIMS{1, 2, 3, 4};
constexpr std::array<int64_t, 4> STRIDES{24, 12, 4, 1};

class TestCudnnShimV8NoExcept : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        clearLastError();
    }

    static void clearLastError()
    {
        std::array<char, 1024> message{};
        cudnnGetLastErrorString(message.data(), message.size());
    }

    static fe::TensorBuilder& describeTensor(fe::TensorBuilder& builder)
    {
        return builder.setDataType(fe::DataType_t::FLOAT)
            .setDim(static_cast<int64_t>(DIMS.size()), DIMS.data())
            .setStride(static_cast<int64_t>(STRIDES.size()), STRIDES.data())
            .setId(1);
    }
};

TEST_F(TestCudnnShimV8NoExcept, FailedPrecheckReturnsStatus)
{
    fe::TensorBuilder builder;
    describeTensor(builder);

    const fe::Tensor tensor = builder.build();

    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_THAT(std::string(tensor.get_error()), HasSubstr("CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT"));
}

TEST_F(TestCudnnShimV8NoExcept, FailedFinalizeCarriesShimMessage)
{
    fe::TensorBuilder builder;
    describeTensor(builder).setAlignment(16).setVectorCountAndDimension(4, 1);

    const fe::Tensor tensor = builder.build();

    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(std::string(tensor.get_error()),
                HasSubstr("CUDNN_ATTR_TENSOR_VECTOR_COUNT other than 1 is not supported"));
}

TEST_F(TestCudnnShimV8NoExcept, SuccessfulBuildReportsSuccess)
{
    fe::TensorBuilder tensorBuilder;
    describeTensor(tensorBuilder).setAlignment(16);
    const fe::Tensor tensor = tensorBuilder.build();

    constexpr std::array<int64_t, 2> PADDING{0, 0};
    constexpr std::array<int64_t, 2> UNIT{1, 1};
    fe::ConvDescBuilder convBuilder;
    convBuilder.setComputeType(fe::DataType_t::FLOAT)
        .setMathMode(fe::ConvolutionMode_t::CROSS_CORRELATION)
        .setSpatialDimCount(static_cast<int64_t>(PADDING.size()))
        .setSpatialStride(static_cast<int64_t>(UNIT.size()), UNIT.data())
        .setPrePadding(static_cast<int64_t>(PADDING.size()), PADDING.data())
        .setPostPadding(static_cast<int64_t>(PADDING.size()), PADDING.data())
        .setDilation(static_cast<int64_t>(UNIT.size()), UNIT.data());
    const fe::ConvDesc conv = convBuilder.build();

    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_SUCCESS) << tensor.get_error();
    EXPECT_NE(tensor.get_raw_desc(), nullptr);
    EXPECT_EQ(conv.get_status(), CUDNN_STATUS_SUCCESS) << conv.get_error();
    EXPECT_NE(conv.get_raw_desc(), nullptr);
}

} // namespace
