// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <gmock/gmock.h>

#include "harness/IReferenceExecutors.hpp"
#include "harness/IReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests
{

class MockReferenceGraphExecutor : public IReferenceGraphExecutor
{
public:
    MOCK_METHOD(bool, isApplicable, (void* graphBuffer, size_t size), (override));
    MOCK_METHOD(void,
                execute,
                (void* graphBuffer, size_t size, const VariantPack& variantPack),
                (override));

    MOCK_METHOD(bool, requiresDeviceMemory, (), (const, override));
};

class MockReferenceExecutors : public IReferenceExecutors
{
public:
    MOCK_METHOD(IReferenceGraphExecutor&, get, (ReferenceExecutorType type), (override));
};

} // namespace hipdnn_integration_tests
