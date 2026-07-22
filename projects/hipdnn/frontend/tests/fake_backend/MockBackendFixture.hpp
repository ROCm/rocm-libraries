// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Reusable mock-backend gtest fixture for the cuDNN-shim tests. Mirrors the
// TestGraph.cpp fixture pattern: installs a NiceMock IHipdnnBackend that answers
// descriptor create/set/finalize/destroy with success and hands out stable fake
// descriptor pointers, so a graph can be lowered to the backend without a device.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "MockHipdnnBackend.hpp"

#include <array>
#include <cstddef>
#include <memory>

namespace hipdnn_shim_test
{

class ShimMockBackendFixture : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    cudnnHandle_t _handle = nullptr;
    std::array<char, 256> _fakeDescs{};
    std::size_t _nextFakeDescIdx = 0;

    void SetUp() override
    {
        using ::testing::_;
        using ::testing::Return;

        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
        _handle = reinterpret_cast<cudnnHandle_t>(0x12345678);

        _nextFakeDescIdx = 0;
        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t, hipdnnBackendDescriptor_t* desc) {
                *desc = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    &_fakeDescs[_nextFakeDescIdx++ % _fakeDescs.size()]);
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

} // namespace hipdnn_shim_test
