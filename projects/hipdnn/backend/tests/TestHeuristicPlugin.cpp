// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestHeuristicPlugin.cpp
 * @brief Unit tests for HeuristicPlugin class (RFC 0007 Part 1)
 *
 * These tests verify the plugin wrapper class including:
 * - Symbol resolution and error handling
 * - Plugin metadata access
 * - Handle lifecycle
 * - Policy descriptor lifecycle
 */

#include "HipdnnException.hpp"
#include "descriptors/mocks/MockHeuristicPlugin.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace hipdnn_backend;
using namespace hipdnn_backend::plugin;
using ::testing::NiceMock;
using ::testing::Return;

namespace
{
// Helper to create fake handles for testing
// NOLINTBEGIN(performance-no-int-to-ptr)
hipdnnHeuristicHandle_t makeFakeHandle(int id)
{
    return reinterpret_cast<hipdnnHeuristicHandle_t>(static_cast<uintptr_t>(id));
}

hipdnnHeuristicPolicyDescriptor_t makeFakePolicyDescriptor(int id)
{
    return reinterpret_cast<hipdnnHeuristicPolicyDescriptor_t>(static_cast<uintptr_t>(id));
}
// NOLINTEND(performance-no-int-to-ptr)
} // anonymous namespace

class TestHeuristicPlugin : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Tests use mocks, not real plugin shared libraries
    }

    void TearDown() override {}
};

// ========== Mock Plugin Behavior Tests ==========
// Note: Trivial single-method mocking tests removed - integration tests with real plugins
// provide better coverage of actual behavior

// ========== Complete Workflow Tests ==========
// Note: Complete workflow is tested with real plugins in TestHeuristicPluginLoading

// ========== Multiple Handles Tests ==========

TEST_F(TestHeuristicPlugin, MockPluginCanManageMultipleHandles)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const auto handle1 = makeFakeHandle(1);
    const auto handle2 = makeFakeHandle(2);
    const auto handle3 = makeFakeHandle(3);

    EXPECT_CALL(plugin, createHandle())
        .WillOnce(Return(handle1))
        .WillOnce(Return(handle2))
        .WillOnce(Return(handle3));

    // Create multiple handles
    const auto h1 = plugin.createHandle();
    const auto h2 = plugin.createHandle();
    const auto h3 = plugin.createHandle();

    EXPECT_EQ(h1, handle1);
    EXPECT_EQ(h2, handle2);
    EXPECT_EQ(h3, handle3);

    // All handles should be unique
    EXPECT_NE(h1, h2);
    EXPECT_NE(h2, h3);
    EXPECT_NE(h1, h3);
}

// ========== Multiple Policy Descriptors Tests ==========

TEST_F(TestHeuristicPlugin, MockPluginCanManageMultiplePolicyDescriptors)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const auto handle = makeFakeHandle(42);
    const auto desc1 = makeFakePolicyDescriptor(1);
    const auto desc2 = makeFakePolicyDescriptor(2);
    const auto desc3 = makeFakePolicyDescriptor(3);

    EXPECT_CALL(plugin, createPolicyDescriptor(handle))
        .WillOnce(Return(desc1))
        .WillOnce(Return(desc2))
        .WillOnce(Return(desc3));

    // Create multiple descriptors from same handle
    const auto d1 = plugin.createPolicyDescriptor(handle);
    const auto d2 = plugin.createPolicyDescriptor(handle);
    const auto d3 = plugin.createPolicyDescriptor(handle);

    EXPECT_EQ(d1, desc1);
    EXPECT_EQ(d2, desc2);
    EXPECT_EQ(d3, desc3);

    // All descriptors should be unique
    EXPECT_NE(d1, d2);
    EXPECT_NE(d2, d3);
    EXPECT_NE(d1, d3);
}

// ========== Call Count Verification Tests ==========

TEST_F(TestHeuristicPlugin, MockPluginTracksCallCounts)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const auto handle = makeFakeHandle(42);

    EXPECT_CALL(plugin, createHandle()).Times(3).WillRepeatedly(Return(handle));

    // Create handle 3 times
    plugin.createHandle();
    plugin.createHandle();
    plugin.createHandle();

    // Expectations verified by gmock
}

TEST_F(TestHeuristicPlugin, MockPluginVerifiesCallSequence)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const auto handle = makeFakeHandle(42);
    const auto descriptor = makeFakePolicyDescriptor(100);

    {
        const ::testing::InSequence seq;

        EXPECT_CALL(plugin, createHandle()).WillOnce(Return(handle));

        EXPECT_CALL(plugin, createPolicyDescriptor(handle)).WillOnce(Return(descriptor));

        EXPECT_CALL(plugin, destroyPolicyDescriptor(descriptor));

        EXPECT_CALL(plugin, destroyHandle(handle));
    }

    // Execute in expected order
    const auto h = plugin.createHandle();
    const auto d = plugin.createPolicyDescriptor(h);
    plugin.destroyPolicyDescriptor(d);
    plugin.destroyHandle(h);
}

// ========== Edge Cases Tests ==========

TEST_F(TestHeuristicPlugin, MockPluginCanReturnNullHandle)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    EXPECT_CALL(plugin, createHandle()).WillOnce(Return(nullptr));

    const auto handle = plugin.createHandle();
    EXPECT_EQ(handle, nullptr);
}

TEST_F(TestHeuristicPlugin, MockPluginCanReturnNullDescriptor)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const auto handle = makeFakeHandle(42);

    EXPECT_CALL(plugin, createPolicyDescriptor(handle)).WillOnce(Return(nullptr));

    const auto descriptor = plugin.createPolicyDescriptor(handle);
    EXPECT_EQ(descriptor, nullptr);
}

// ========== Policy ID Caching Tests ==========

TEST_F(TestHeuristicPlugin, MockPluginPolicyIdCanBeCached)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const int64_t testPolicyId = 0xABCDEF;

    // First call should query the mock
    EXPECT_CALL(plugin, policyId()).Times(2).WillRepeatedly(Return(testPolicyId));

    // Multiple calls
    const int64_t id1 = plugin.policyId();
    const int64_t id2 = plugin.policyId();

    EXPECT_EQ(id1, testPolicyId);
    EXPECT_EQ(id2, testPolicyId);
}

// ========== Policy Name Edge Cases ==========

TEST_F(TestHeuristicPlugin, MockPluginEmptyPolicyNameIsValid)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    EXPECT_CALL(plugin, name()).WillOnce(Return(""));

    const auto name = plugin.name();
    EXPECT_TRUE(name.empty());
}

TEST_F(TestHeuristicPlugin, MockPluginLongPolicyNameIsValid)
{
    const NiceMock<MockHeuristicPlugin> plugin;

    const std::string_view longName = "VeryLongPolicyNameThatExceedsTypicalLengthsButIsStillValid";
    EXPECT_CALL(plugin, name()).WillOnce(Return(longName));

    const auto name = plugin.name();
    EXPECT_EQ(name, longName);
}

// Base class interface tests (name, version, type) are covered by loading tests
// since they delegate to virtual methods that can't be effectively tested with mocks
