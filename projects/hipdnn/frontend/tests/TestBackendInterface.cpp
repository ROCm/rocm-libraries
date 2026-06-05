// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "fake_backend/MockHipdnnBackend.hpp"
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/version.h>

#include <array>

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

TEST(TestBackendInterface, BackendGetSerializedExecutionPlanExtForwardsToBackend)
{
    HipdnnBackendWrapper backendWrapper(Version{std::string_view(hipdnnVersionString_ext())});
    size_t planByteSize = 0;

    EXPECT_EQ(
        backendWrapper.backendGetSerializedExecutionPlanExt(nullptr, 0, &planByteSize, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST(TestBackendInterface, BackendCreateAndDeserializeExecutionPlanExtForwardsToBackend)
{
    HipdnnBackendWrapper backendWrapper(Version{std::string_view(hipdnnVersionString_ext())});
    hipdnnBackendDescriptor_t descriptor = nullptr;
    const std::array<uint8_t, 1> serializedPlan{0};

    EXPECT_EQ(backendWrapper.backendCreateAndDeserializeExecutionPlanExt(
                  nullptr, &descriptor, serializedPlan.data(), serializedPlan.size()),
              HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}
