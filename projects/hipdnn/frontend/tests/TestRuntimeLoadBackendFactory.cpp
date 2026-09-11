// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Precondition for ResolvesBackendWithinItsOwnTree below, and for the
// hipdnn_frontend_dynamic_load_tests_asan_preload ctest entry that gives this suite its
// teeth: no hipDNN backend may be visible to the loader's own search. `ldconfig -p` must
// mention no hipdnn library and no /opt/rocm/lib/libhipdnn_backend.so may exist. With one
// present, the bare-soname fallback tier can satisfy the suite for the wrong reason -- the
// resolved-path assertion below is what makes such a violation visible rather than silent.

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
        ASSERT_NE(backendLibraryHandle(), nullptr);
#else
        if(backendLibraryHandle() == nullptr)
        {
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
        }
#endif

        IHipdnnBackend::resetInstance();
        _backend = hipdnnBackend();
        if(_backend->versionString()[0] == '\0')
        {
#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
            FAIL() << "hipDNN backend library was found, but runtime symbol loading failed";
#else
            GTEST_SKIP() << "hipDNN backend library is not available for runtime symbol loading";
#endif
        }
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
    }

    std::shared_ptr<IHipdnnBackend> _backend;
};

/// An address guaranteed to live inside this test executable, so the loader can be asked
/// where the executable itself came from.
void testExecutableAnchor() {}

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

// The point of the suite in the ASan-preload configuration: the backend must be found by a
// path the frontend computed, not by a loader search the ASan dlopen interceptor performs
// against its own RUNPATH. A non-null handle alone cannot tell those apart, so compare the
// resolved path against the two self-relative locations -- beside the executable, and in
// the sibling library directory -- that the build and install trees actually use. A pass
// through the HIP-anchor or bare-soname tier lands somewhere else and fails here.
TEST_F(TestRuntimeLoadBackendFactory, ResolvesBackendWithinItsOwnTree)
{
    const auto resolved = resolveBackendLibraryPath();
    ASSERT_FALSE(resolved.empty()) << "the backend library was loaded, but no path was resolved";

    const auto selfDirectory
        = getLoadedLibraryDirectoryForAddress(reinterpret_cast<const void*>(&testExecutableAnchor));
    const auto libraryName = getLibraryName("hipdnn_backend");
    const auto besideExecutable = normalized(selfDirectory / libraryName);
    const auto inSiblingLibraryDirectory
        = normalized(selfDirectory.parent_path() / HIPDNN_TEST_INSTALL_LIBDIR / libraryName);
    const auto actual = normalized(resolved);

    EXPECT_TRUE(actual == besideExecutable || actual == inSiblingLibraryDirectory)
        << "resolved " << actual << ", expected " << besideExecutable << " or "
        << inSiblingLibraryDirectory
        << ". A path outside this tree means the backend was found by the HIP anchor or by "
           "the loader's own search, so the self-relative resolution is untested here -- "
           "check that no hipDNN backend is installed system-wide.";
}
