// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <gtest/gtest.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <stdexcept>
#include <vector>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

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

#ifndef NDEBUG
TEST(TestShallowTensor, OutOfBoundsAccessThrows)
{
    std::array<int, 6> backing = {1, 2, 3, 4, 5, 6};
    ShallowTensor<int> tensor(backing.data(), {1, 1, 2, 3}, {6, 6, 3, 1});
    EXPECT_THROW(tensor.getHostValue(20), std::out_of_range);
    EXPECT_THROW(tensor.setHostValue(25, 28), std::out_of_range);
}
#endif

// Sparse (strided) tensor test
TEST(TestShallowTensor, SparseTensorCreationAndUsage)
{
    // Create a sparse tensor with dims {2,2,2,2} and strides {2,4,8,16}
    // This represents a non-packed layout with gaps in memory
    std::vector<int64_t> dims = {2, 2, 2, 2};
    std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor(dims, strides);

    // Test setting and getting values at different indices
    tensor.fillWithValue(0.0f);

    // Set values at specific logical indices
    tensor.setHostValue(10.0f, 0, 0, 0, 0); // Offset: 0
    tensor.setHostValue(20.0f, 1, 0, 0, 0); // Offset: 1*2 = 2
    tensor.setHostValue(30.0f, 0, 1, 0, 0); // Offset: 1*4 = 4
    tensor.setHostValue(40.0f, 0, 0, 1, 0); // Offset: 1*8 = 8
    tensor.setHostValue(50.0f, 0, 0, 0, 1); // Offset: 1*16 = 16
    tensor.setHostValue(99.0f, 1, 1, 1, 1); // Offset: 2+4+8+16 = 30

    ShallowTensor<float> shallow(tensor.rawHostData(), tensor.dims(), tensor.strides());

    // Verify properties
    EXPECT_EQ(shallow.dims(), dims);
    EXPECT_EQ(shallow.strides(), strides);
    EXPECT_EQ(shallow.elementCount(), 16); // Logical elements: 2*2*2*2

    // But calculateElementSpace returns sum of (dim-1)*stride
    // = (2-1)*2 + (2-1)*4 + (2-1)*8 + (2-1)*16 = 2+4+8+16 = 30
    // Add the init value of 1 and you get 31
    EXPECT_EQ(shallow.elementSpace(), 31);

    // Verify it's not packed
    EXPECT_FALSE(shallow.isPacked());

    //checking the shallow tensor of tensor matches should validate the setup of shallow
    //tensor for sparse data.
    CpuFpReferenceValidation<float> refValidation;
    EXPECT_TRUE(refValidation.allClose(tensor, shallow));
}
