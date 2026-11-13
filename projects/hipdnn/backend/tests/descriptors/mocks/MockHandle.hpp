// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "handle/Handle.hpp"
#include <gmock/gmock.h>

struct MockHandle : hipdnnHandle
{
    MockHandle()
    {
        // Behave like a real handle unless a test programs a device: resolve from getStream().
        ON_CALL(*this, getDevice).WillByDefault([this](int* deviceId) {
            return hipdnnHandle::getDevice(deviceId);
        });
    }

    MOCK_METHOD(void, setStream, (hipStream_t stream), (override));
    MOCK_METHOD(hipStream_t, getStream, (), (const, override));
    MOCK_METHOD(hipError_t, getDevice, (int* deviceId), (const, override));
    MOCK_METHOD(std::shared_ptr<hipdnn_backend::plugin::EnginePluginResourceManager>,
                getPluginResourceManager,
                (),
                (const, override));
    MOCK_METHOD(std::shared_ptr<hipdnn_backend::plugin::HeuristicPluginResourceManager>,
                getHeuristicPluginResourceManager,
                (),
                (const, override));
};
