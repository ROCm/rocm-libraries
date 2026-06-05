// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_gpu_ref/ShallowGpuTensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

using hipdnn_data_sdk::utilities::MemoryLocation;
using hipdnn_data_sdk::utilities::Workspace;
using hipdnn_gpu_ref::ShallowGpuTensor;

namespace
{
size_t computeElementSpace(const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
{
    return static_cast<size_t>(
        std::inner_product(dims.begin(),
                           dims.end(),
                           strides.begin(),
                           int64_t{1},
                           std::plus<>(),
                           [](int64_t len, int64_t stride) { return (len - 1) * stride; }));
}
} // namespace

TEST(TestGpuShallowTensor, PackedTensorReportsCorrectShape)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 3, 2, 2};
    const std::vector<int64_t> strides = {12, 4, 2, 1};

    const auto elementSpace = computeElementSpace(dims, strides);
    const Workspace workspace(elementSpace * sizeof(float));

    const ShallowGpuTensor<float> tensor(workspace.get(), dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_TRUE(tensor.isPacked());
    EXPECT_EQ(tensor.elementCount(), 12u);
    EXPECT_EQ(tensor.elementSpace(), elementSpace);
}

TEST(TestGpuShallowTensor, NonPackedTensorReportsCorrectElementSpace)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    const auto elementSpace = computeElementSpace(dims, strides);
    const Workspace workspace(elementSpace * sizeof(float));

    const ShallowGpuTensor<float> tensor(workspace.get(), dims, strides);

    EXPECT_EQ(tensor.elementCount(), 16u);
    EXPECT_EQ(tensor.elementSpace(), elementSpace);
    EXPECT_FALSE(tensor.isPacked());
}

TEST(TestGpuShallowTensor, MemoryExposesDevicePointer)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_EQ(tensor.memory().deviceData(), workspace.get());
    EXPECT_EQ(tensor.memory().location(), MemoryLocation::DEVICE);
}

TEST(TestGpuShallowTensor, HostFillOperationsThrow)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_THROW(tensor.fillWithValue(1.0f), std::runtime_error);
    EXPECT_THROW(tensor.fillWithRandomValues(-1.0f, 1.0f, 42), std::runtime_error);

    std::array<float, 4> hostData = {1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_THROW(tensor.fillWithData(hostData.data(), sizeof(hostData)), std::runtime_error);
}

TEST(TestGpuShallowTensor, HostMemoryAccessThrows)
{
    SKIP_IF_NO_DEVICES();

    const Workspace workspace(4 * sizeof(float));

    ShallowGpuTensor<float> tensor(workspace.get(), {1, 1, 2, 2}, {4, 4, 2, 1});

    auto& mem = tensor.memory();
    EXPECT_THROW(mem.hostData(), std::runtime_error);
    EXPECT_THROW(mem.hostDataAsync(), std::runtime_error);
    EXPECT_THROW(mem.markHostModified(), std::runtime_error);
}

TEST(TestGpuShallowTensor, MoveConstruction)
{
    SKIP_IF_NO_DEVICES();

    const std::vector<int64_t> dims = {1, 1, 3, 3};
    const std::vector<int64_t> strides = {9, 9, 3, 1};

    const auto elementSpace = computeElementSpace(dims, strides);
    const Workspace workspace(elementSpace * sizeof(float));

    ShallowGpuTensor<float> source(workspace.get(), dims, strides);
    ShallowGpuTensor<float> dest(std::move(source));

    EXPECT_EQ(dest.dims(), dims);
    EXPECT_EQ(dest.strides(), strides);
    EXPECT_EQ(dest.memory().deviceData(), workspace.get());
    EXPECT_EQ(dest.elementCount(), 9u);

    // Source memory is emptied — intentionally inspecting moved-from state
    EXPECT_EQ(source.memory().deviceData(), nullptr); // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(source.memory().count(), 0u);
    EXPECT_TRUE(source.memory().empty());
}
