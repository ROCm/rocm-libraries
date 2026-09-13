// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Run without a system-wide backend (ldconfig cache or /opt/rocm/lib) so the loader's
// fallback cannot mask a self-relative resolution failure.

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/DynamicBackendLibrary.hpp>
#include <hipdnn_frontend/detail/HipdnnDynamicBackendWrapper.hpp>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string_view>

using namespace hipdnn_frontend::detail;
using namespace hipdnn_data_sdk::utilities;

namespace
{

#if defined(__linux__)
testing::AssertionResult asanPreloadInterceptsDlopen()
{
    void* const effectiveDlopen = dlsym(RTLD_DEFAULT, "dlopen");
    void* const asanInit = dlsym(RTLD_DEFAULT, "__asan_init");
    if(effectiveDlopen == nullptr || asanInit == nullptr)
    {
        return testing::AssertionFailure() << "missing effective dlopen or __asan_init";
    }

    Dl_info dlopenInfo{};
    Dl_info asanInfo{};
    void* dlopenOwner = nullptr;
    void* asanOwner = nullptr;
    if(dladdr1(effectiveDlopen, &dlopenInfo, &dlopenOwner, RTLD_DL_LINKMAP) == 0
       || dladdr1(asanInit, &asanInfo, &asanOwner, RTLD_DL_LINKMAP) == 0 || dlopenOwner == nullptr
       || asanOwner == nullptr)
    {
        return testing::AssertionFailure() << "cannot identify effective symbol owners";
    }

    const std::unique_ptr<void, decltype(&closeLibrary)> mainHandle(
        dlopen(nullptr, RTLD_NOW | RTLD_LOCAL), &closeLibrary);
    link_map* mainMap = nullptr;
    if(mainHandle == nullptr
       || dlinfo(mainHandle.get(), RTLD_DI_LINKMAP, static_cast<void*>(&mainMap)) != 0
       || mainMap == nullptr)
    {
        return testing::AssertionFailure() << "cannot identify the main executable's loader map";
    }

    const auto* map = static_cast<const link_map*>(dlopenOwner);
    if(dlopenOwner != asanOwner || dlopenOwner == mainMap || map->l_name == nullptr
       || map->l_name[0] == '\0')
    {
        return testing::AssertionFailure()
               << "effective dlopen and __asan_init must share a non-main loader map; dlopen="
               << dlopenOwner << ", __asan_init=" << asanOwner << ", main=" << mainMap;
    }
    std::fprintf(stderr, "ASan dlopen interception: map=%p, object=%s\n", dlopenOwner, map->l_name);
    return testing::AssertionSuccess();
}
#endif

class TestRuntimeLoadBackendFactory : public testing::Test
{
protected:
    void SetUp() override
    {
        bool expectBackend = getEnv("HIPDNN_TEST_EXPECT_ASAN_PRELOAD") == "1";
#if defined(__linux__)
        if(expectBackend)
        {
            ASSERT_TRUE(asanPreloadInterceptsDlopen());
        }
#endif
#ifdef HIPDNN_TEST_EXPECT_BACKEND_LIBRARY
        expectBackend = true;
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

#ifdef _WIN32
class TestRuntimeLoadBackendNativeOverride : public testing::Test
{
protected:
    void SetUp() override
    {
        _previousOverride = getEnvW(L"HIPDNN_BACKEND_LIBRARY_PATH", L"\x01unset");
        const UINT codePage = GetACP();
        RecordProperty("ansi_code_page", static_cast<int>(codePage));
        std::fprintf(stderr, "Windows native override ACP=%u\n", codePage);
        ASSERT_NE(codePage, CP_UTF8) << "this regression requires a non-UTF-8 ANSI code page";

        constexpr const wchar_t* NATIVE_NAME = L"\u6D4B\u8BD5_\u0416_\u03A9_\U0001F9EA";
        BOOL usedDefault = FALSE;
        ASSERT_GT(
            WideCharToMultiByte(
                CP_ACP, WC_NO_BEST_FIT_CHARS, NATIVE_NAME, -1, nullptr, 0, nullptr, &usedDefault),
            0);
        ASSERT_TRUE(usedDefault)
            << "the fixture name must be unrepresentable in the ANSI code page";

        const auto executableDirectory = getCurrentExecutableDirectory();
        const auto libraryName = getLibraryName("hipdnn_backend");
        auto source = executableDirectory / libraryName;
        if(!std::filesystem::exists(source))
        {
            source = executableDirectory.parent_path() / HIPDNN_TEST_INSTALL_LIBDIR / libraryName;
        }
        ASSERT_TRUE(std::filesystem::is_regular_file(source)) << source;
        const auto directory = executableDirectory / "hipdnn_native_override_fixture" / NATIVE_NAME;
        std::filesystem::create_directories(directory);
        _expectedPath = normalized(directory / libraryName);
        std::filesystem::copy_file(
            source, _expectedPath, std::filesystem::copy_options::overwrite_existing);
        ASSERT_TRUE(SetEnvironmentVariableW(L"HIPDNN_BACKEND_LIBRARY_PATH", directory.c_str()));
    }

    void TearDown() override
    {
        IHipdnnBackend::resetInstance();
        SetEnvironmentVariableW(L"HIPDNN_BACKEND_LIBRARY_PATH",
                                _previousOverride == L"\x01unset" ? nullptr
                                                                  : _previousOverride.c_str());
        // CTest removes the test-owned directory after this process releases the DLL.
    }

    std::wstring _previousOverride;
    std::filesystem::path _expectedPath;
};
#endif

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

// Under ASan preload, a non-null handle alone could hide a loader fallback.
// Require a self-relative path in the build or relocated install tree.
TEST_F(TestRuntimeLoadBackendFactory, ResolvesBackendWithinItsOwnTree)
{
    const auto resolved = resolveBackendLibraryPath();
    ASSERT_FALSE(resolved.empty()) << "the backend library was loaded, but no path was resolved";

    const auto selfDirectory = getCurrentExecutableDirectory();
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

#ifdef _WIN32
TEST_F(TestRuntimeLoadBackendNativeOverride, DISABLED_LoadsBackendFromNativeEnvironmentPath)
{
    const auto resolved = resolveBackendLibraryPath();
    RecordProperty("expected_native_path", _expectedPath.u8string());
    RecordProperty("selected_native_path", resolved.u8string());
    std::fprintf(stderr,
                 "Windows native override expected=%s, selected=%s\n",
                 _expectedPath.u8string().c_str(),
                 resolved.u8string().c_str());
    ASSERT_EQ(normalized(resolved).native(), _expectedPath.native());
    ASSERT_NE(backendLibraryHandle(), nullptr);
    EXPECT_EQ(getLoadedLibraryOrigin(backendLibraryHandle()).native(),
              _expectedPath.parent_path().native());

    const auto backend = hipdnnBackend();
    ASSERT_TRUE(std::dynamic_pointer_cast<HipdnnDynamicBackendWrapper>(backend));
    ASSERT_NE(backend->versionString()[0], '\0');
    EXPECT_EQ(backend->version(), Version{std::string_view(backend->versionString())});
}
#endif
