// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Run without a system-wide backend (ldconfig cache or /opt/rocm/lib) so the loader's
// fallback cannot mask a resolution failure.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

#include <filesystem>
#include <memory>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;

namespace
{

class TestRuntimeLoadBackendFactory : public testing::Test
{
protected:
    void SetUp() override
    {
#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
        const bool expectBackend = true;
#else
        const bool expectBackend = false;
#endif
        if(expectBackend)
        {
            ASSERT_NE(backendLibraryHandle(), nullptr);
        }
        else if(backendLibraryHandle() == nullptr)
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }

        IHipdnnBackend::resetInstance();
        _backend = hipdnnBackend();
        if(_backend->versionString()[0] == '\0')
        {
            if(expectBackend)
            {
                FAIL() << "hipDNN backend library was found, but runtime symbol loading failed";
            }
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
    }

    std::shared_ptr<IHipdnnBackend> _backend;
};

std::filesystem::path normalized(const std::filesystem::path& path)
{
    std::error_code failed;
    const auto resolved = std::filesystem::weakly_canonical(path, failed);
    return failed ? path : resolved;
}

} // namespace

TEST_F(TestRuntimeLoadBackendFactory, TryToUseDynamicBackendInterfaceCreatesDynamicWrapper)
{
    EXPECT_TRUE(
        std::dynamic_pointer_cast<HipdnnDynamicBackendWrapper>(tryToUseDynamicBackendInterface()));
}

TEST_F(TestRuntimeLoadBackendFactory, HipdnnBackendCreatesDynamicWrapper)
{
    EXPECT_TRUE(std::dynamic_pointer_cast<HipdnnDynamicBackendWrapper>(_backend));
}

TEST_F(TestRuntimeLoadBackendFactory, HipdnnBackendUsesBackendVersion)
{
    EXPECT_EQ(_backend->version(), Version{std::string_view(_backend->versionString())});
}

// A non-null handle alone could hide a stale backend found somewhere else entirely.
// This holds in any layout, ordinary or custom -- CMAKE_INSTALL_LIBDIR is not assumed:
// the path reported is the file the loader actually mapped, and that backend answers
// real calls.
TEST_F(TestRuntimeLoadBackendFactory, ResolvedPathNamesTheBackendActuallyLoaded)
{
    const auto resolved = resolveBackendLibraryPath();
    ASSERT_FALSE(resolved.empty()) << "the backend library was loaded, but no path was resolved";

    const auto handle = backendLibraryHandle();
    ASSERT_NE(handle, nullptr);
    const auto origin = getLoadedLibraryOrigin(handle);
    EXPECT_TRUE(std::filesystem::is_regular_file(origin / getLibraryName("hipdnn_backend")))
        << "the loaded backend's origin holds no backend library: " << origin;
    // The loader-search tier resolves to a bare name, leaving no directory of the
    // resolver's own to compare against, so the comparison below is skipped rather
    // than failed. The check above still holds there. Asserting it unconditionally is
    // what made this test reject a correctly loading build under a custom libdir.
    if(resolved.has_parent_path())
    {
        EXPECT_TRUE(pathCompEq(normalized(resolved).parent_path(), normalized(origin)))
            << "resolved " << normalized(resolved) << ", loaded from " << normalized(origin);
    }

    EXPECT_EQ(_backend->version(), Version{std::string_view(_backend->versionString())});
}
