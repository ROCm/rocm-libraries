// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/backend/ScopedHipdnnBackendDescriptor.hpp>
#include <hipdnn_sdk/utilities/StringUtil.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

using namespace hipdnn_frontend;
using namespace ::testing;

class TestScopedHipdnnBackendDescriptor : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, getLastErrorString(_, _))
            .WillByDefault([](char* errorString, size_t size) {
                std::string fakeError = "Fake backend error";
                hipdnn_sdk::utilities::copyMaxSizeWithNullTerminator(
                    errorString, fakeError.c_str(), size - 1);
            });
    }
    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

TEST_F(TestScopedHipdnnBackendDescriptor, DefaultConstructorIsInvalid)
{
    auto desc = ScopedHipdnnBackendDescriptor();
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, ConstructWithRawDescriptorIsValid)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1234);
    auto desc = ScopedHipdnnBackendDescriptor(fakeDesc);

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fakeDesc);
}

TEST_F(TestScopedHipdnnBackendDescriptor, ConstructWithTypeSuccess)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5678);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fakeDesc);
}

TEST_F(TestScopedHipdnnBackendDescriptor, ConstructWithTypeFailure)
{
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, _))
        .WillOnce([](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t*) {
            return HIPDNN_STATUS_BAD_PARAM;
        });
    EXPECT_CALL(*_mockBackend, getLastErrorString).Times(AnyNumber()); // Uninteresting call

    auto desc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, ConstructWithSerializedGraphSuccess)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x4321);
    std::vector<uint8_t> fakeGraph{1, 2, 3, 4};
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptor_t* out, const uint8_t*, size_t) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc = ScopedHipdnnBackendDescriptor(fakeGraph.data(), fakeGraph.size());
    EXPECT_TRUE(desc.valid());
    EXPECT_EQ(desc.get(), fakeDesc);
}

TEST_F(TestScopedHipdnnBackendDescriptor, ConstructWithSerializedGraphFailure)
{
    std::vector<uint8_t> fakeGraph{1, 2, 3, 4};
    EXPECT_CALL(*_mockBackend, backendCreateAndDeserializeGraphExt(_, _, _))
        .WillOnce([](hipdnnBackendDescriptor_t*, const uint8_t*, size_t) {
            return HIPDNN_STATUS_BAD_PARAM;
        });
    EXPECT_CALL(*_mockBackend, getLastErrorString).Times(AnyNumber()); // Uninteresting call

    auto desc = ScopedHipdnnBackendDescriptor(fakeGraph.data(), fakeGraph.size());
    EXPECT_FALSE(desc.valid());
    EXPECT_EQ(desc.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, DestructorDestroysDescriptorIfValid)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x9999);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    {
        auto desc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        EXPECT_TRUE(desc.valid());
    }
}

TEST_F(TestScopedHipdnnBackendDescriptor, DestructorHandlesDestroyFailure)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x8888);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));
    EXPECT_CALL(*_mockBackend, getLastErrorString).Times(AnyNumber()); // Uninteresting call

    {
        auto desc = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        EXPECT_TRUE(desc.valid());
    }
}

TEST_F(TestScopedHipdnnBackendDescriptor, MoveConstructorTransfersOwnership)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x1111);
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    EXPECT_TRUE(desc1.valid());
    auto raw = desc1.get();

    auto desc2 = ScopedHipdnnBackendDescriptor(std::move(desc1));
    EXPECT_TRUE(desc2.valid());
    EXPECT_EQ(desc2.get(), raw);
    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, MoveAssignmentTransfersOwnership)
{
    auto fakeDesc1 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x2222);
    auto fakeDesc2 = reinterpret_cast<hipdnnBackendDescriptor_t>(0x3333);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc1](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc1;
            return HIPDNN_STATUS_SUCCESS;
        })
        .WillOnce([&fakeDesc2](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc2;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc1))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc2))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto desc2 = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);

    desc2 = std::move(desc1);

    EXPECT_TRUE(desc2.valid());
    EXPECT_EQ(desc2.get(), fakeDesc1);
    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, MoveFromInvalidDescriptor)
{
    auto desc1 = ScopedHipdnnBackendDescriptor();
    auto desc2 = ScopedHipdnnBackendDescriptor(std::move(desc1));
    EXPECT_FALSE(desc2.valid());
    EXPECT_EQ(desc2.get(), nullptr);
}

TEST_F(TestScopedHipdnnBackendDescriptor, MoveAssignFromInvalidDescriptor)
{
    auto fakeDesc = reinterpret_cast<hipdnnBackendDescriptor_t>(0x5555);

    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR, _))
        .WillOnce([&fakeDesc](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* out) {
            *out = fakeDesc;
            return HIPDNN_STATUS_SUCCESS;
        });

    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(fakeDesc))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    auto desc1 = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto desc2 = ScopedHipdnnBackendDescriptor();

    desc1 = std::move(desc2);

    EXPECT_FALSE(desc1.valid());
    EXPECT_EQ(desc1.get(), nullptr);
}
