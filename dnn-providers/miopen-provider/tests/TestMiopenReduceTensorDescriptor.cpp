// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <utility>

#include <gtest/gtest.h>

#include "MiopenApi.hpp"
#include "MiopenReduceTensorDescriptor.hpp"

using namespace miopen_plugin;

namespace
{

// Everything miopenGetReduceTensorDescriptor reports about a descriptor.
struct ReduceDescriptorFields
{
    miopenReduceTensorOp_t op{};
    miopenDataType_t compType{};
    miopenNanPropagation_t nanOpt{};
    miopenReduceTensorIndices_t indices{};
    miopenIndicesType_t indicesType{};
};

ReduceDescriptorFields queryFields(miopenReduceTensorDescriptor_t descriptor)
{
    ReduceDescriptorFields fields;
    EXPECT_EQ(miopenGetReduceTensorDescriptor(descriptor,
                                              &fields.op,
                                              &fields.compType,
                                              &fields.nanOpt,
                                              &fields.indices,
                                              &fields.indicesType),
              miopenStatusSuccess);
    return fields;
}

} // namespace

// NORM2 is not op 0 (ADD), so a descriptor that ignored the op would show. The op mapping per
// reduction mode is pinned by TestMiopenReductionChecks.
TEST(TestMiopenReduceTensorDescriptor, ConfiguresLegacyReduceWithoutIndices)
{
    const MiopenReduceTensorDescriptor descriptor(MIOPEN_REDUCE_TENSOR_NORM2);
    ASSERT_NE(descriptor.reduceTensorDescriptor(), nullptr);

    const auto fields = queryFields(descriptor.reduceTensorDescriptor());
    EXPECT_EQ(fields.op, MIOPEN_REDUCE_TENSOR_NORM2);
    EXPECT_EQ(fields.compType, miopenFloat);
    EXPECT_EQ(fields.nanOpt, MIOPEN_PROPAGATE_NAN);
    EXPECT_EQ(fields.indices, MIOPEN_REDUCE_TENSOR_NO_INDICES);
    EXPECT_EQ(fields.indicesType, MIOPEN_32BIT_INDICES);
}

TEST(TestMiopenReduceTensorDescriptor, MoveConstructorTransfersOwnership)
{
    MiopenReduceTensorDescriptor source(MIOPEN_REDUCE_TENSOR_AMAX);
    auto* const raw = source.reduceTensorDescriptor();

    const MiopenReduceTensorDescriptor destination(std::move(source));

    EXPECT_EQ(destination.reduceTensorDescriptor(), raw);
    EXPECT_EQ(source.reduceTensorDescriptor(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(queryFields(destination.reduceTensorDescriptor()).op, MIOPEN_REDUCE_TENSOR_AMAX);
}

TEST(TestMiopenReduceTensorDescriptor, MoveAssignmentTransfersOwnership)
{
    MiopenReduceTensorDescriptor source(MIOPEN_REDUCE_TENSOR_NORM2);
    MiopenReduceTensorDescriptor destination(MIOPEN_REDUCE_TENSOR_ADD);
    auto* const raw = source.reduceTensorDescriptor();
    ASSERT_NE(destination.reduceTensorDescriptor(), raw);

    destination = std::move(source);

    EXPECT_EQ(destination.reduceTensorDescriptor(), raw);
    EXPECT_EQ(source.reduceTensorDescriptor(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(queryFields(destination.reduceTensorDescriptor()).op, MIOPEN_REDUCE_TENSOR_NORM2);
}

TEST(TestMiopenReduceTensorDescriptor, MoveAssignmentIntoMovedFromDescriptorTransfersOwnership)
{
    MiopenReduceTensorDescriptor source(MIOPEN_REDUCE_TENSOR_MIN);
    MiopenReduceTensorDescriptor destination(MIOPEN_REDUCE_TENSOR_MAX);
    const MiopenReduceTensorDescriptor holder(std::move(destination));
    auto* const raw = source.reduceTensorDescriptor();

    // destination holds no descriptor now, so the assignment has nothing to release.
    destination = std::move(source); // NOLINT(bugprone-use-after-move)

    EXPECT_EQ(destination.reduceTensorDescriptor(), raw);
    EXPECT_EQ(queryFields(destination.reduceTensorDescriptor()).op, MIOPEN_REDUCE_TENSOR_MIN);
}

TEST(TestMiopenReduceTensorDescriptor, SelfMoveAssignmentKeepsDescriptor)
{
    MiopenReduceTensorDescriptor descriptor(MIOPEN_REDUCE_TENSOR_AVG);
    auto* const raw = descriptor.reduceTensorDescriptor();

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
    descriptor = std::move(descriptor);
#pragma clang diagnostic pop

    EXPECT_EQ(descriptor.reduceTensorDescriptor(), raw); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(queryFields(descriptor.reduceTensorDescriptor()).op, MIOPEN_REDUCE_TENSOR_AVG);
}
