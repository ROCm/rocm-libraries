// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/version.h>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;
using namespace ::testing;

namespace
{
// NOLINTNEXTLINE(bugprone-throwing-static-initialization) test constant
const std::string SUCCESS_VERSION = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + ".-1.0";
}

TEST(TestBackendInterface, TryToUseBackendInterfaceSuccess)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<HipdnnBackendWrapper>(
        tryToUseBackendInterface(SUCCESS_VERSION.c_str())));
}

TEST(TestBackendInterface, TryToUseBackendInterfaceMajorVersionMismatch)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(
        tryToUseBackendInterface("-1.0.0.TWEAK")));
}

TEST(TestBackendInterface, TryToUseBackendInterfaceBadlyFormedVersion)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(
        tryToUseBackendInterface("CantParseThis")));
}

TEST(TestBackendInterface, TryToUseBackendInterfaceNullptr)
{
    EXPECT_TRUE(
        std::dynamic_pointer_cast<IncompatibleBackendWrapper>(tryToUseBackendInterface(nullptr)));
}

TEST(TestBackendInterface, VersionEqualsVersionString)
{
    EXPECT_EQ(hipdnnBackend()->version(),
              Version{std::string_view(hipdnnBackend()->versionString())});
}
