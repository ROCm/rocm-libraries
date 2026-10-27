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

#include "fake_backend/ShimV8BuilderFixture.hpp"

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

using hipdnn_shim_test::Dims;

const Dims X_DIMS = {2, 4, 8, 8};

class TestCudnnShimV8NoexceptOperations : public hipdnn_shim_test::ShimV8BuilderFixture
{
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
    EXPECT_THAT(std::string(genStats.get_error()),
                HasSubstr("gen stats operations are not supported"));
    EXPECT_EQ(genStats.getTag(), "GenStats");

    // Upstream's paged-cache-load builder never sets a tag.
    const fe::Operation pagedCacheLoad
        = fe::OperationBuilder(DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)
              .setcontainerDesc(tensor(1, {8, 2, 16, 64}))
              .setpageTableDesc(tensor(2, {2, 1, 4, 1}, false, DataType_t::INT32))
              .setsequenceDesc(tensor(3, {2, 1, 1, 1}, false, DataType_t::INT32))
              .setyDesc(tensor(4, {2, 2, 64, 64}))
              .build();

    EXPECT_EQ(pagedCacheLoad.get_status(), CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_THAT(std::string(pagedCacheLoad.get_error()),
                HasSubstr("paged cache load operations are not supported"));
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

    // build() reports the status the setter recorded and creates no descriptor, so the raw
    // descriptor reads as null rather than dereferencing the missing pointer.
    EXPECT_EQ(onPointwise.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_THAT(std::string(onPointwise.get_error()),
                HasSubstr("Operation not initialized properly"));
    EXPECT_EQ(onPointwise.get_desc(), nullptr);
    EXPECT_EQ(onPointwise.get_raw_desc(), nullptr);

    fe::OperationBuilder convBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    convBuilder.setbDesc(tensor(4, {1, 8, 1, 1}));
    const fe::Operation onConv = convBuilder.build();

    EXPECT_EQ(onConv.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(onConv.get_desc(), nullptr);
    EXPECT_EQ(onConv.get_raw_desc(), nullptr);
}

// A failed tensor in an optional slot is a caller error, not an absent slot: a
// built index tensor would be lowered, a built X or M override NOT_SUPPORTED.
TEST_F(TestCudnnShimV8NoexceptOperations, FailedTensorInOptionalSlotIsBadParam)
{
    const Dims dims = {2, 4, 8, 8};
    const Dims strides = {256, 64, 8, 1};
    // No alignment: TensorBuilder rejects it.
    const fe::Tensor failed = fe::TensorBuilder()
                                  .setDataType(DataType_t::FLOAT)
                                  .setDim(4, dims.data())
                                  .setStride(4, strides.data())
                                  .setId(9)
                                  .build();
    ASSERT_EQ(failed.get_status(), CUDNN_STATUS_BAD_PARAM);

    const std::array<std::int64_t, 2> window = {2, 2};
    const std::array<std::int64_t, 2> zeros = {0, 0};
    const fe::ResampleDesc_v8 pool = fe::ResampleDescBuilder_v8()
                                         .setResampleMode(fe::ResampleMode_t::MAXPOOL)
                                         .setComputeType(DataType_t::FLOAT)
                                         .setPaddingMode(fe::PaddingMode_t::NEG_INF_PAD)
                                         .setSpatialDim(2, window.data())
                                         .setPrePadding(2, zeros.data())
                                         .setPostPadding(2, zeros.data())
                                         .setSpatialStride(2, window.data())
                                         .build();
    ASSERT_EQ(pool.get_status(), CUDNN_STATUS_SUCCESS) << pool.get_error();
    const Dims pooled = {2, 4, 4, 4};

    const fe::Operation forward
        = fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR)
              .setResampleDesc(pool)
              .setxDesc(tensor(1, X_DIMS))
              .setyDesc(tensor(2, pooled))
              .setidxDesc(failed)
              .build();
    EXPECT_EQ(forward.get_status(), CUDNN_STATUS_BAD_PARAM);
    EXPECT_THAT(std::string(forward.get_error()), HasSubstr("index tensor is not built"));
    EXPECT_EQ(forward.get_raw_desc(), nullptr);

    const fe::Operation backward
        = fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR)
              .setResampleDesc(pool)
              .setdxDesc(tensor(1, X_DIMS))
              .setdyDesc(tensor(2, pooled))
              .setidxDesc(tensor(3, pooled, false, DataType_t::INT32))
              .setxDesc(failed)
              .build();
    EXPECT_EQ(backward.get_status(), CUDNN_STATUS_BAD_PARAM);

    const fe::MatMulDesc matmul = fe::MatMulDescBuilder().setComputeType(DataType_t::FLOAT).build();
    const fe::Operation product
        = fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
              .setmatmulDesc(matmul)
              .setaMatDesc(tensor(1, {2, 4, 8}))
              .setbMatDesc(tensor(2, {2, 8, 16}))
              .setcMatDesc(tensor(3, {2, 4, 16}))
              .setmOverrideDesc(failed)
              .build();
    EXPECT_EQ(product.get_status(), CUDNN_STATUS_BAD_PARAM);
}

} // namespace
