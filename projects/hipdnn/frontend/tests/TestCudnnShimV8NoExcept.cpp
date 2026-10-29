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

#include "fake_backend/ShimV8BuilderFixture.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using ::testing::HasSubstr;

constexpr std::array<int64_t, 4> DIMS{1, 2, 3, 4};
constexpr std::array<int64_t, 4> STRIDES{24, 12, 4, 1};

class TestCudnnShimV8NoExcept : public hipdnn_shim_test::ShimV8BuilderFixture
{
protected:
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
    EXPECT_THAT(std::string(tensor.get_error()), HasSubstr("TensorBuilder: byte alignment"));
}

TEST_F(TestCudnnShimV8NoExcept, RetryAfterFailedPrecheckSucceeds)
{
    fe::TensorBuilder builder;
    describeTensor(builder);
    {
        const fe::Tensor first = builder.build();
        ASSERT_EQ(first.get_status(), CUDNN_STATUS_BAD_PARAM);
        // An empty placeholder, so an operation slot reads it as set but not built.
        EXPECT_NE(first.get_desc(), nullptr);
        EXPECT_EQ(fe::detail::v8::recordOf<fe::detail::v8::TensorRecord>(first.get_desc()),
                  nullptr);
    }

    const fe::Tensor retry = builder.setAlignment(16).build();
    EXPECT_EQ(retry.get_status(), CUDNN_STATUS_SUCCESS) << retry.get_error();
    EXPECT_NE(fe::detail::v8::recordOf<fe::detail::v8::TensorRecord>(retry.get_desc()), nullptr);
}

TEST_F(TestCudnnShimV8NoExcept, RetryKeepsRaggedOffsetAndConstant)
{
    constexpr std::array<int64_t, 1> ONE{1};
    fe::TensorBuilder raggedBuilder;
    raggedBuilder.setDataType(fe::DataType_t::INT32)
        .setDim(1, ONE.data())
        .setStride(1, ONE.data())
        .setId(9)
        .setAlignment(16);
    auto ragged = std::make_shared<fe::Tensor>(raggedBuilder.build());
    ASSERT_EQ(ragged->get_status(), CUDNN_STATUS_SUCCESS) << ragged->get_error();

    struct Case
    {
        const char* label;
        std::function<void(fe::TensorBuilder&)> breakIt;
        std::function<void(fe::TensorBuilder&)> fix;
    };
    const std::vector<Case> cases = {
        {"failed precheck",
         [](fe::TensorBuilder&) {},
         [](fe::TensorBuilder& b) { b.setAlignment(16); }},
        {"failed core check",
         [](fe::TensorBuilder& b) { b.setAlignment(16).setVectorCountAndDimension(4, 1); },
         [](fe::TensorBuilder& b) { b.setVectorCountAndDimension(1, -1); }},
    };
    for(const Case& c : cases)
    {
        SCOPED_TRACE(c.label);
        fe::TensorBuilder builder;
        describeTensor(builder).setRaggedOffset(ragged).setConstValue(2.5F);
        c.breakIt(builder);
        {
            // Binding by value moves the failed tensor's members out of the builder.
            const fe::Tensor first = builder.build();
            ASSERT_NE(first.get_status(), CUDNN_STATUS_SUCCESS);
        }
        c.fix(builder);

        const fe::Tensor retry = builder.build();
        ASSERT_EQ(retry.get_status(), CUDNN_STATUS_SUCCESS) << retry.get_error();
        const auto record
            = fe::detail::v8::recordOf<fe::detail::v8::TensorRecord>(retry.get_desc());
        ASSERT_NE(record, nullptr);
        EXPECT_EQ(record->raggedOffset(),
                  fe::detail::v8::recordOf<fe::detail::v8::TensorRecord>(ragged->get_desc()));
        ASSERT_TRUE(record->constantValue().has_value());
        ASSERT_EQ(record->constantValue()->size(), sizeof(float));
        float constant = 0.0F;
        std::memcpy(&constant, record->constantValue()->data(), sizeof(constant));
        EXPECT_EQ(constant, 2.5F);
    }
}

TEST_F(TestCudnnShimV8NoExcept, FailedFinalizeCarriesShimMessage)
{
    fe::TensorBuilder builder;
    describeTensor(builder).setAlignment(16).setVectorCountAndDimension(4, 1);

    const fe::Tensor tensor = builder.build();

    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(std::string(tensor.get_error()),
                HasSubstr("TensorBuilder: vector count other than 1 is not supported"));
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

    // Descriptors carry a shim record and no hipDNN descriptor.
    EXPECT_EQ(tensor.get_status(), CUDNN_STATUS_SUCCESS) << tensor.get_error();
    EXPECT_EQ(tensor.get_raw_desc(), nullptr);
    EXPECT_NE(fe::detail::v8::recordOf<fe::detail::v8::TensorRecord>(tensor.get_desc()), nullptr);
    EXPECT_EQ(conv.get_status(), CUDNN_STATUS_SUCCESS) << conv.get_error();
    EXPECT_EQ(conv.get_raw_desc(), nullptr);
    EXPECT_NE(fe::detail::v8::recordOf<fe::detail::v8::ConvolutionRecord>(conv.get_desc()),
              nullptr);
}

} // namespace
