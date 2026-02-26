// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"
#include <hipdnn_frontend/detail/BackendWrapper.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace ::testing;

TEST(TestBackendInterface, TryToUseBackendInterfaceGetVersionFails)
{
    auto mockBackend = std::make_shared<Mock_hipdnn_backend>();
    EXPECT_CALL(*mockBackend, versionExt(testing::_))
        .WillOnce(::testing::Return(hipdnnStatus_t::HIPDNN_STATUS_INTERNAL_ERROR));

    EXPECT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(
        tryToUseBackendInterface(mockBackend)));
}

TEST(TestBackendInterface, TryToUseBackendInterfaceSuccess)
{
    auto mockBackend = std::make_shared<Mock_hipdnn_backend>();
    EXPECT_CALL(*mockBackend, versionExt(testing::_))
        .WillOnce(::testing::Return(hipdnnStatus_t::HIPDNN_STATUS_SUCCESS));

    EXPECT_TRUE(
        std::dynamic_pointer_cast<Mock_hipdnn_backend>(tryToUseBackendInterface(mockBackend)));
}
