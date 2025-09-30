// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "descriptors/BackendDescriptor.hpp"
#include "descriptors/EngineDescriptor.hpp"
#include "descriptors/ScopedDescriptor.hpp"
#include "mocks/MockDescriptor.hpp"

#include <gtest/gtest.h>
#include <memory>

using namespace ::testing;
using namespace hipdnn_backend;

TEST(TestBackendDescriptor, PackAndUnpackDescriptorWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();

    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));
    ASSERT_NE(packed.get(), nullptr);

    auto unpacked = HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
        packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mockPtr.get());
    ASSERT_EQ(unpacked->getType(), HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
}

TEST(TestBackendDescriptor, AsDescriptorCastsCorrectly)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));

    auto result = packed.get()->asDescriptor<MockDescriptor<EngineDescriptor>>();
    ASSERT_EQ(result.get(), mockPtr.get());
}

TEST(TestBackendDescriptor, UnpackDescriptorFromArrayWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    ScopedDescriptor packed(HipdnnBackendDescriptor::packDescriptor(mockPtr));

    void* arrayOfElements = &packed.descriptor;
    auto unpacked = HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
        arrayOfElements, HIPDNN_STATUS_INTERNAL_ERROR, "fail");
    ASSERT_EQ(unpacked.get(), mockPtr.get());
}

TEST(TestBackendDescriptor, PackDescriptorToArrayWorks)
{
    auto mockPtr = std::make_shared<MockDescriptor<EngineDescriptor>>();
    hipdnnBackendDescriptor_t desc = nullptr;
    void* arrayOfElements = &desc;
    HipdnnBackendDescriptor::packDescriptor(mockPtr, arrayOfElements);
    ScopedDescriptor scoped(desc);

    ASSERT_NE(desc, nullptr);
}

TEST(TestBackendDescriptor, UnpackDescriptorThrowsOnNullDescriptor)
{
    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                static_cast<HipdnnBackendDescriptor*>(nullptr),
                HIPDNN_STATUS_INTERNAL_ERROR,
                "fail");
        },
        HipdnnException);
}

TEST(TestBackendDescriptor, UnpackDescriptorThrowsOnNullPrivateDescriptor)
{
    ScopedDescriptor packed(new HipdnnBackendDescriptor());

    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                packed.get(), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        HipdnnException);
}

TEST(TestBackendDescriptor, UnpackDescriptorFromArrayThrowsOnNullArray)
{
    EXPECT_THROW(
        {
            HipdnnBackendDescriptor::unpackDescriptor<MockDescriptor<EngineDescriptor>>(
                static_cast<void*>(nullptr), HIPDNN_STATUS_INTERNAL_ERROR, "fail");
        },
        HipdnnException);
}
