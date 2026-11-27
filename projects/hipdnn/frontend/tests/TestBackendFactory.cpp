// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/HipdnnDirectBackendWrapper.hpp>
#include <hipdnn_frontend/version_number.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;

namespace
{
// NOLINTNEXTLINE(bugprone-throwing-static-initialization) test constant
const std::string SUCCESS_VERSION = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + ".0.0";
// NOLINTNEXTLINE(bugprone-throwing-static-initialization) test constant
const std::string MAJOR_MISMATCH_VERSION
    = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR + 1) + ".0.0.TWEAK";
}

TEST(TestBackendFactory, TryToUseBackendInterfaceSuccessCreatesDirectWrapper)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<HipdnnDirectBackendWrapper>(
        tryToUseBackendInterface(SUCCESS_VERSION.c_str())));
}

TEST(TestBackendFactory, TryToUseBackendInterfaceMajorVersionMismatchCreatesIncompatibleWrapper)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(
        tryToUseBackendInterface(MAJOR_MISMATCH_VERSION.c_str())));
}

TEST(TestBackendFactory, TryToUseBackendInterfaceBadlyFormedVersionCreatesIncompatibleWrapper)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(
        tryToUseBackendInterface("CantParseThis")));
}

TEST(TestBackendFactory, TryToUseBackendInterfaceNullptrCreatesIncompatibleWrapper)
{
    EXPECT_TRUE(
        std::dynamic_pointer_cast<IncompatibleBackendWrapper>(tryToUseBackendInterface(nullptr)));
}

TEST(TestBackendFactory, TryToUseBackendInterfaceVersionMismatchReportsBothVersions)
{
    const std::string backendVersion
        = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR + 1) + ".0.0.TWEAK";
    const auto backend = tryToUseBackendInterface(backendVersion.c_str());
    ASSERT_TRUE(std::dynamic_pointer_cast<IncompatibleBackendWrapper>(backend));

    std::array<char, 256> message{};
    backend->getLastErrorString(message.data(), message.size());
    const std::string reason(message.data());
    const std::string frontendVersion = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + "."
                                        + std::to_string(HIPDNN_FRONTEND_VERSION_MINOR) + "."
                                        + std::to_string(HIPDNN_FRONTEND_VERSION_PATCH);
    EXPECT_NE(reason.find("'" + backendVersion + "'"), std::string::npos) << reason;
    EXPECT_NE(reason.find(frontendVersion), std::string::npos) << reason;
}

TEST(TestBackendFactory, GetOrCreateInstanceReturnsPresetInstanceWithoutCallingFactory)
{
    IHipdnnBackend::resetInstance();
    const auto presetBackend = std::make_shared<IncompatibleBackendWrapper>();
    IHipdnnBackend::setInstance(presetBackend);

    int factoryCalls = 0;
    const auto backend = IHipdnnBackend::getOrCreateInstance([&] {
        ++factoryCalls;
        return tryToUseBackendInterface(SUCCESS_VERSION.c_str());
    });

    EXPECT_EQ(backend, presetBackend);
    EXPECT_EQ(factoryCalls, 0);

    IHipdnnBackend::resetInstance();
}

TEST(TestBackendFactory, GetOrCreateInstanceCachesFirstInitialization)
{
    IHipdnnBackend::resetInstance();
    auto createdBackend = std::make_shared<IncompatibleBackendWrapper>();

    int factoryCalls = 0;
    const auto firstBackend = IHipdnnBackend::getOrCreateInstance([&] {
        ++factoryCalls;
        return createdBackend;
    });
    const auto secondBackend = IHipdnnBackend::getOrCreateInstance([&] {
        ++factoryCalls;
        return tryToUseBackendInterface(SUCCESS_VERSION.c_str());
    });

    EXPECT_EQ(firstBackend, createdBackend);
    EXPECT_EQ(secondBackend, createdBackend);
    EXPECT_EQ(factoryCalls, 1);

    IHipdnnBackend::resetInstance();
}

TEST(TestBackendFactory, HipdnnBackendReturnsPresetInstance)
{
    IHipdnnBackend::resetInstance();
    const auto presetBackend = std::make_shared<IncompatibleBackendWrapper>();
    IHipdnnBackend::setInstance(presetBackend);

    EXPECT_EQ(hipdnnBackend(), presetBackend);

    IHipdnnBackend::resetInstance();
}

TEST(TestBackendFactory, HipdnnBackendCreatesDirectWrapperWithBackendVersion)
{
    IHipdnnBackend::resetInstance();

    const auto backend = hipdnnBackend();
    EXPECT_TRUE(std::dynamic_pointer_cast<HipdnnDirectBackendWrapper>(backend));
    EXPECT_EQ(backend->version(), Version{std::string_view(backend->versionString())});

    IHipdnnBackend::resetInstance();
}
