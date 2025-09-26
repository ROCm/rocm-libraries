// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <stdexcept>
#include <vector>

using namespace hipdnn_sdk::utilities;

TEST(TestShallowTensor, ConstructionAndShape)
{
    std::array<float, 12> backing{};
    std::vector<int64_t> dims = {1, 3, 2, 2}; // N C H W
    std::vector<int64_t> strides = {12, 4, 2, 1}; // custom (not contiguous standard)
    ShallowTensor<float> tensor(backing.data(), dims, strides);

    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
}

TEST(TestShallowTensor, MemoryAccessHostOnly)
{
    std::array<int, 6> backing = {1, 2, 3, 4, 5, 6};
    std::vector<int64_t> dims = {1, 1, 2, 3};
    std::vector<int64_t> strides = {6, 6, 3, 1};
    ShallowTensor<int> tensor(backing.data(), dims, strides);

    auto& mem = tensor.memory();
    auto* hostPtr = mem.hostData();
    EXPECT_EQ(hostPtr, backing.data());
    EXPECT_EQ(mem.location(), MemoryLocation::HOST);
}

TEST(TestShallowTensor, FillWithValueThrows)
{
    std::array<int, 4> backing = {7, 8, 9, 10};
    ShallowTensor<int> tensor(backing.data(), {1, 1, 2, 2}, {4, 4, 2, 1});

    EXPECT_THROW(tensor.fillWithValue(123), std::runtime_error);
}

TEST(TestShallowTensor, FillWithRandomValuesThrows)
{
    std::array<float, 5> backing = {0.f, 1.f, 2.f, 3.f, 4.f};
    ShallowTensor<float> tensor(backing.data(), {1, 1, 1, 5}, {5, 5, 5, 1});

    EXPECT_THROW(tensor.fillWithRandomValues(-1.f, 1.f, 1337), std::runtime_error);
}

TEST(TestShallowTensor, DeviceAccessThrows)
{
    std::array<float, 2> backing = {0.f, 1.f};
    ShallowTensor<float> tensor(backing.data(), {1, 1, 1, 2}, {2, 2, 2, 1});
    auto& mem = tensor.memory();
    EXPECT_THROW(mem.deviceData(), std::runtime_error);
    EXPECT_THROW(mem.deviceDataAsync(), std::runtime_error);
}
