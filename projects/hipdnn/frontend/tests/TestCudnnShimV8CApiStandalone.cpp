// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// cudnn_graph.h must come first: this TU proves it compiles standalone.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace
{

using ::testing::_;
using ::testing::Return;

class TestCudnnShimV8CApiStandalone : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::StrictMock<Mock_hipdnn_backend>> _mockBackend;

    const cudnnHandle_t _handle = reinterpret_cast<cudnnHandle_t>(0x1000);
    const cudnnBackendDescriptor_t _plan = reinterpret_cast<cudnnBackendDescriptor_t>(0x2000);
    const cudnnBackendDescriptor_t _pack = reinterpret_cast<cudnnBackendDescriptor_t>(0x3000);

    void SetUp() override
    {
        _mockBackend = std::make_shared<::testing::StrictMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};

static_assert(std::is_same_v<cudnnBackendDescriptor_t, hipdnnBackendDescriptor_t>);

TEST_F(TestCudnnShimV8CApiStandalone, ExecuteForwardsOnceWithSameArguments)
{
    EXPECT_CALL(*_mockBackend, backendExecute(_handle, _plan, _pack))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));

    EXPECT_EQ(cudnnBackendExecute(_handle, _plan, _pack), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8CApiStandalone, ExecuteTranslatesStatus)
{
    const std::array<std::pair<hipdnnStatus_t, cudnnStatus_t>, 5> cases = {{
        {HIPDNN_STATUS_BAD_PARAM, CUDNN_STATUS_BAD_PARAM},
        {HIPDNN_STATUS_NOT_SUPPORTED, CUDNN_STATUS_NOT_SUPPORTED},
        {HIPDNN_STATUS_EXECUTION_FAILED, CUDNN_STATUS_EXECUTION_FAILED},
        {HIPDNN_STATUS_INTERNAL_ERROR, CUDNN_STATUS_INTERNAL_ERROR},
        {HIPDNN_STATUS_ALLOC_FAILED, CUDNN_STATUS_ALLOC_FAILED},
    }};
    for(const auto& [hipdnnStatus, cudnnStatus] : cases)
    {
        EXPECT_CALL(*_mockBackend, backendExecute(_handle, _plan, _pack))
            .WillOnce(Return(hipdnnStatus));
        EXPECT_EQ(cudnnBackendExecute(_handle, _plan, _pack), cudnnStatus) << hipdnnStatus;
    }
}

// Null descriptors are forwarded too; hipDNN decides.
TEST_F(TestCudnnShimV8CApiStandalone, ExecuteForwardsNullDescriptors)
{
    EXPECT_CALL(*_mockBackend, backendExecute(nullptr, nullptr, nullptr))
        .WillOnce(Return(HIPDNN_STATUS_BAD_PARAM));

    EXPECT_EQ(cudnnBackendExecute(nullptr, nullptr, nullptr), CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8CApiStandalone, ExecuteNeverThrows)
{
    EXPECT_CALL(*_mockBackend, backendExecute(_, _, _))
        .WillOnce([](hipdnnHandle_t,
                     hipdnnBackendDescriptor_t,
                     hipdnnBackendDescriptor_t) -> hipdnnStatus_t { throw std::bad_alloc(); })
        .WillOnce([](hipdnnHandle_t, hipdnnBackendDescriptor_t, hipdnnBackendDescriptor_t)
                      -> hipdnnStatus_t { throw std::runtime_error("backend failure"); });

    EXPECT_EQ(cudnnBackendExecute(_handle, _plan, _pack), CUDNN_STATUS_ALLOC_FAILED);
    EXPECT_EQ(cudnnBackendExecute(_handle, _plan, _pack), CUDNN_STATUS_INTERNAL_ERROR);
}

TEST_F(TestCudnnShimV8CApiStandalone, LastErrorStringIsHipdnns)
{
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _))
        .WillOnce([](char* message, std::size_t size) {
            const std::string text = "hipDNN said no";
            ASSERT_GT(size, text.size());
            std::memcpy(message, text.c_str(), text.size() + 1);
        });

    std::array<char, 64> message{};
    message[0] = 'x';
    cudnnGetLastErrorString(message.data(), message.size());
    EXPECT_STREQ(message.data(), "hipDNN said no");

    // No buffer: nothing to fill, and hipDNN is not asked.
    cudnnGetLastErrorString(nullptr, 16);
    cudnnGetLastErrorString(message.data(), 0);
}

} // namespace
