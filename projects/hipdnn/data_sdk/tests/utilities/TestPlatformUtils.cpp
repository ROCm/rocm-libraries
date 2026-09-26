// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <type_traits>

#if defined(__linux__)
#include <climits>
#include <link.h>
#include <unistd.h>
#endif

TEST(TestPlatformUtils, PathCompEqIdenticalPaths)
{
    const std::filesystem::path path1 = "/home/user/project";
    const std::filesystem::path path2 = "/home/user/project";

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqDifferentPaths)
{
    const std::filesystem::path path1 = "/home/user/project1";
    const std::filesystem::path path2 = "/home/user/project2";

    EXPECT_FALSE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

TEST(TestPlatformUtils, PathCompEqEmptyPaths)
{
    const std::filesystem::path path1;
    const std::filesystem::path path2;

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(path1, path2));
}

#ifdef _WIN32
TEST(TestPlatformUtils, PathCompEqNativeUnicodePaths)
{
    const std::filesystem::path path = L"C:\\\u6D4B\u8BD5_\U0001F9EA\\\u03A9\u0416.dll";
    const std::filesystem::path same = L"c:\\\u6D4B\u8BD5_\U0001F9EA\\\u03C9\u0436.DLL";
    const std::filesystem::path different = L"c:\\\u6D4B\u8BD5_\U0001F9EA\\\u03C9\u0437.DLL";

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(path, same));
    EXPECT_FALSE(hipdnn_data_sdk::utilities::pathCompEq(path, different));
}
#endif

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryReturnsValidPath)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_FALSE(execDir.empty());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryExists)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::exists(execDir));
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsAbsolute)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(execDir.is_absolute());
}

TEST(TestPlatformUtils, GetCurrentExecutableDirectoryIsDirectory)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    EXPECT_TRUE(std::filesystem::is_directory(execDir));
}

#if defined(__linux__)
TEST(TestPlatformUtils, GetCurrentExecutableDirectoryContainsExecutable)
{
    auto execDir = hipdnn_data_sdk::utilities::getCurrentExecutableDirectory();

    std::array<char, PATH_MAX> execPath{};
    const ssize_t len = readlink("/proc/self/exe", execPath.data(), PATH_MAX);
    ASSERT_NE(len, -1);

    const std::filesystem::path actualExecPath(
        std::string(execPath.data(), static_cast<size_t>(len)));

    EXPECT_TRUE(std::filesystem::exists(execDir / actualExecPath.filename()));
}

#endif // defined(__linux__)

// getLibraryName tests

TEST(TestPlatformUtils, GetLibraryNameBasicName)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("foo");

#if defined(__linux__)
    EXPECT_EQ(result, "libfoo.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, "foo.dll");
#endif
}

TEST(TestPlatformUtils, GetLibraryNameEmptyName)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("");

#if defined(__linux__)
    EXPECT_EQ(result, "lib.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, ".dll");
#endif
}

TEST(TestPlatformUtils, GetLibraryNameWithDots)
{
    auto result = hipdnn_data_sdk::utilities::getLibraryName("foo.bar");

#if defined(__linux__)
    EXPECT_EQ(result, "libfoo.bar.so");
#elif defined(_WIN32)
    EXPECT_EQ(result, "foo.bar.dll");
#endif
}

// getExecutableName tests

TEST(TestPlatformUtils, GetExecutableNameBasicName)
{
    auto result = hipdnn_data_sdk::utilities::getExecutableName("app");

#if defined(__linux__)
    EXPECT_EQ(result, "app");
#elif defined(_WIN32)
    EXPECT_EQ(result, "app.exe");
#endif
}

TEST(TestPlatformUtils, GetExecutableNameEmptyName)
{
    auto result = hipdnn_data_sdk::utilities::getExecutableName("");

#if defined(__linux__)
    EXPECT_EQ(result, "");
#elif defined(_WIN32)
    EXPECT_EQ(result, ".exe");
#endif
}

// getEnv tests

TEST(TestPlatformUtils, GetEnvReturnsValue)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_GETENV", "test_value");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_GETENV");

    EXPECT_EQ(result, "test_value");
}

TEST(TestPlatformUtils, GetEnvReturnsDefaultWhenUnset)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET");

    auto result
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET", "default_value");

    EXPECT_EQ(result, "default_value");
}

TEST(TestPlatformUtils, GetEnvReturnsEmptyWhenUnsetNoDefault)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET2");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_UNSET2");

    EXPECT_EQ(result, "");
}

TEST(TestPlatformUtils, GetEnvReturnsEmptyStringValue)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_EMPTY", "");

    auto result
        = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_EMPTY", "default_value");

    EXPECT_EQ(result, "");
}

// setEnv tests

TEST(TestPlatformUtils, SetEnvSetsValue)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_SET");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_SET", "new_value");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_SET");

    EXPECT_EQ(result, "new_value");

    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_SET");
}

TEST(TestPlatformUtils, SetEnvNullValueDoesNotSetVariable)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_NULL_SET");
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_NULL_SET", nullptr);

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_NULL_SET");

    EXPECT_EQ(result, "");
}

TEST(TestPlatformUtils, SetEnvOverwritesExisting)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_OVERWRITE", "original");

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_OVERWRITE", "updated");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_OVERWRITE");

    EXPECT_EQ(result, "updated");
}

// unsetEnv tests

TEST(TestPlatformUtils, UnsetEnvRemovesVariable)
{
    hipdnn_data_sdk::utilities::setEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE", "to_remove");
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_REMOVE");

    EXPECT_EQ(result, "");
}

TEST(TestPlatformUtils, UnsetEnvNoOpOnMissing)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_NONEXISTENT");

    auto result = hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_NONEXISTENT");

    EXPECT_EQ(result, "");
}

// expandUser tests

#if defined(__linux__)

TEST(TestPlatformUtils, ExpandUserLeadingTildeExpandsToHome)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("~/foo");

    EXPECT_EQ(result, "/home/testuser/foo");
}

TEST(TestPlatformUtils, ExpandUserBareTildeExpandsToHome)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("~");

    EXPECT_EQ(result, "/home/testuser");
}

TEST(TestPlatformUtils, ExpandUserEmbeddedTildeIsReturnedVerbatim)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("/tmp/a~b");

    EXPECT_EQ(result, "/tmp/a~b");
}

TEST(TestPlatformUtils, ExpandUserNamedUserIsNotExpanded)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("~otheruser/foo");

    EXPECT_EQ(result, "~otheruser/foo");
}

TEST(TestPlatformUtils, ExpandUserUnsetHomeReturnsInputUnchanged)
{
    hipdnn_data_sdk::utilities::unsetEnv("HOME");

    std::string result;
    EXPECT_NO_THROW(result = hipdnn_data_sdk::utilities::expandUser("~/foo"));

    EXPECT_EQ(result, "~/foo");
}

TEST(TestPlatformUtils, ExpandUserEmptyHomeReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME", "");

    std::string result;
    EXPECT_NO_THROW(result = hipdnn_data_sdk::utilities::expandUser("~/foo"));

    EXPECT_EQ(result, "~/foo");
}

TEST(TestPlatformUtils, ExpandUserNoLeadingTokenReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("relative/path");

    EXPECT_EQ(result, "relative/path");
}

TEST(TestPlatformUtils, ExpandUserEmptyInputReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("HOME",
                                                                             "/home/testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("");

    EXPECT_EQ(result, "");
}

#elif defined(_WIN32)

TEST(TestPlatformUtils, ExpandUserLeadingTildeExpandsToUserProfile)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("~\\foo");

    EXPECT_EQ(result, "C:\\Users\\testuser\\foo");
}

TEST(TestPlatformUtils, ExpandUserLeadingUserProfileTokenExpands)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("%USERPROFILE%\\foo");

    EXPECT_EQ(result, "C:\\Users\\testuser\\foo");
}

TEST(TestPlatformUtils, ExpandUserEmbeddedTildeIsReturnedVerbatim)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("C:\\tmp\\a~b");

    EXPECT_EQ(result, "C:\\tmp\\a~b");
}

TEST(TestPlatformUtils, ExpandUserNamedUserIsNotExpanded)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("~otheruser\\foo");

    EXPECT_EQ(result, "~otheruser\\foo");
}

TEST(TestPlatformUtils, ExpandUserUnsetUserProfileReturnsInputUnchanged)
{
    hipdnn_data_sdk::utilities::unsetEnv("USERPROFILE");

    std::string result;
    EXPECT_NO_THROW(result = hipdnn_data_sdk::utilities::expandUser("~\\foo"));

    EXPECT_EQ(result, "~\\foo");
}

TEST(TestPlatformUtils, ExpandUserEmptyUserProfileReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE", "");

    std::string result;
    EXPECT_NO_THROW(result = hipdnn_data_sdk::utilities::expandUser("~\\foo"));

    EXPECT_EQ(result, "~\\foo");
}

TEST(TestPlatformUtils, ExpandUserNoLeadingTokenReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("relative\\path");

    EXPECT_EQ(result, "relative\\path");
}

TEST(TestPlatformUtils, ExpandUserEmptyInputReturnsInputUnchanged)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter("USERPROFILE",
                                                                             "C:\\Users\\testuser");

    auto result = hipdnn_data_sdk::utilities::expandUser("");

    EXPECT_EQ(result, "");
}

#endif // defined(__linux__) / defined(_WIN32)

TEST(TestPlatformUtils, IsSecureExecutionFalseForAnOrdinaryProcess)
{
    // Assumes a test process launched without set-ID or capability elevation.
    EXPECT_FALSE(hipdnn_data_sdk::utilities::isSecureExecution());
}

// Checks ordinary-process compatibility, not secure-execution hardening.
TEST(TestPlatformUtils, GetSecureEnvMatchesGetEnvOutsideSecureExecution)
{
    const hipdnn_test_sdk::utilities::ScopedEnvironmentVariableSetter setter(
        "HIPDNN_TEST_PLATFORMUTILS_SECURE", "secure_value");

    EXPECT_EQ(hipdnn_data_sdk::utilities::getSecureEnv("HIPDNN_TEST_PLATFORMUTILS_SECURE"),
              hipdnn_data_sdk::utilities::getEnv("HIPDNN_TEST_PLATFORMUTILS_SECURE"));
}

TEST(TestPlatformUtils, GetSecureEnvReturnsDefaultWhenUnset)
{
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_TEST_PLATFORMUTILS_SECURE_UNSET");

    EXPECT_EQ(hipdnn_data_sdk::utilities::getSecureEnv("HIPDNN_TEST_PLATFORMUTILS_SECURE_UNSET",
                                                       "default_value"),
              "default_value");
}

TEST(TestPlatformUtils, GetLoadedLibraryOriginRejectsNullHandle)
{
    EXPECT_THROW(hipdnn_data_sdk::utilities::getLoadedLibraryOrigin(nullptr), std::runtime_error);
}

#if defined(__linux__)

namespace
{

/// Absolute path of some shared object already mapped into this process.
std::filesystem::path anyLoadedLibraryPath()
{
    std::filesystem::path found;
    dl_iterate_phdr(
        [](struct dl_phdr_info* info, [[maybe_unused]] size_t size, void* data) -> int {
            if(info->dlpi_name == nullptr || info->dlpi_name[0] != '/')
            {
                return 0;
            }
            std::error_code failed;
            if(!std::filesystem::is_regular_file(info->dlpi_name, failed) || failed)
            {
                return 0;
            }
            *static_cast<std::filesystem::path*>(data) = info->dlpi_name;
            return 1;
        },
        &found);
    return found;
}

class TestPlatformUtilsLibraryOrigin : public testing::Test
{
protected:
    void SetUp() override
    {
        _root = std::filesystem::temp_directory_path()
                / ("hipdnn_library_origin_" + std::to_string(getpid()));
        ASSERT_TRUE(std::filesystem::create_directory(_root));
    }

    void TearDown() override
    {
        std::error_code failed;
        std::filesystem::remove_all(_root, failed);
    }

    std::filesystem::path _root;
};

} // namespace

TEST(TestPlatformUtils, GetLoadedLibraryOriginReportsTheDirectoryTheLibraryCameFrom)
{
    const std::filesystem::path library = anyLoadedLibraryPath();
    ASSERT_FALSE(library.empty()) << "no loaded shared object to interrogate";

    const std::unique_ptr<void, decltype(&hipdnn_data_sdk::utilities::closeLibrary)> handle(
        hipdnn_data_sdk::utilities::openLibrary(library),
        &hipdnn_data_sdk::utilities::closeLibrary);

    const auto origin = hipdnn_data_sdk::utilities::getLoadedLibraryOrigin(handle.get());

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(
        origin, std::filesystem::weakly_canonical(library).parent_path()));
}

TEST_F(TestPlatformUtilsLibraryOrigin, RejectsRelativeLoaderName)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> sourceHandle(
        utilities::openLibrary("libm.so.6"), &utilities::closeLibrary);
    link_map* sourceMap = nullptr;
    ASSERT_EQ(dlinfo(sourceHandle.get(), RTLD_DI_LINKMAP, static_cast<void*>(&sourceMap)), 0);
    ASSERT_NE(sourceMap, nullptr);
    ASSERT_NE(sourceMap->l_name, nullptr);
    const std::filesystem::path source(sourceMap->l_name);
    ASSERT_TRUE(source.is_absolute());
    const auto library = _root / "relative-origin.so";
    ASSERT_TRUE(std::filesystem::copy_file(source, library));
    const auto relative = std::filesystem::relative(library);
    ASSERT_TRUE(relative.is_relative());

    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> handle(
        utilities::openLibrary(relative), &utilities::closeLibrary);
    EXPECT_THROW(utilities::getLoadedLibraryOrigin(handle.get()), std::runtime_error);
}

// The self tier is computed from this helper, so a relative owner name must be
// refused here too: canonicalizing it would name whatever tree the process has
// since chdir'd into.
TEST_F(TestPlatformUtilsLibraryOrigin, RejectsRelativeAddressOwner)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> sourceHandle(
        utilities::openLibrary("libm.so.6"), &utilities::closeLibrary);
    link_map* sourceMap = nullptr;
    ASSERT_EQ(dlinfo(sourceHandle.get(), RTLD_DI_LINKMAP, static_cast<void*>(&sourceMap)), 0);
    ASSERT_NE(sourceMap, nullptr);
    ASSERT_NE(sourceMap->l_name, nullptr);
    const std::filesystem::path source(sourceMap->l_name);
    ASSERT_TRUE(source.is_absolute());
    const auto library = _root / "relative-address-owner.so";
    ASSERT_TRUE(std::filesystem::copy_file(source, library));
    const auto relative = std::filesystem::relative(library);
    ASSERT_TRUE(relative.is_relative());

    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> handle(
        utilities::openLibrary(relative), &utilities::closeLibrary);
    const void* symbol = utilities::getSymbol(handle.get(), "cos");
    ASSERT_NE(symbol, nullptr);

    EXPECT_THROW(utilities::getLoadedLibraryDirectoryForAddress(symbol), std::runtime_error);
}

TEST(TestPlatformUtils, GetLoadedLibraryOriginRejectsMainExecutable)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> handle(
        dlopen(nullptr, RTLD_NOW | RTLD_LOCAL), &utilities::closeLibrary);
    ASSERT_NE(handle, nullptr);
    EXPECT_THROW(utilities::getLoadedLibraryOrigin(handle.get()), std::runtime_error);
}

TEST(TestPlatformUtils, GetLoadedLibraryDirectoryForAddressRejectsNullAddress)
{
    EXPECT_THROW(hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForAddress(nullptr),
                 std::runtime_error);
}

TEST(TestPlatformUtils, GetLoadedLibraryDirectoryForAddressResolvesLoadedLibrary)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    const std::unique_ptr<void, decltype(&utilities::closeLibrary)> handle(
        utilities::openLibrary("libm.so.6"), &utilities::closeLibrary);
    link_map* map = nullptr;
    ASSERT_EQ(dlinfo(handle.get(), RTLD_DI_LINKMAP, static_cast<void*>(&map)), 0);
    ASSERT_NE(map, nullptr);
    ASSERT_NE(map->l_name, nullptr);
    const auto expected = std::filesystem::weakly_canonical(map->l_name).parent_path();
    const void* symbol = utilities::getSymbol(handle.get(), "cos");
    ASSERT_NE(symbol, nullptr);

    EXPECT_EQ(utilities::getLoadedLibraryDirectoryForAddress(symbol), expected);
}

TEST(TestPlatformUtils, GetLoadedLibraryDirectoryForSymbolRejectsMissingSymbol)
{
    EXPECT_THROW(hipdnn_data_sdk::utilities::getLoadedLibraryDirectoryForSymbol(
                     "hipdnn_test_platform_utils_missing_symbol_12009"),
                 std::runtime_error);
}

TEST(TestPlatformUtils, GetLoadedLibraryDirectoryForSymbolClearsPriorLookupError)
{
    namespace utilities = hipdnn_data_sdk::utilities;
    const void* symbol = dlsym(RTLD_DEFAULT, "malloc");
    ASSERT_NE(symbol, nullptr);
    Dl_info info{};
    ASSERT_NE(dladdr(symbol, &info), 0);
    ASSERT_NE(info.dli_fname, nullptr);
    const auto expected = std::filesystem::weakly_canonical(info.dli_fname).parent_path();

    // Leave the failed lookup's dlerror pending for the utility to clear.
    ASSERT_EQ(
        utilities::getSymbol(RTLD_DEFAULT, "hipdnn_test_platform_utils_stale_lookup_error_12009"),
        nullptr);

    EXPECT_EQ(utilities::getLoadedLibraryDirectoryForSymbol("malloc"), expected);
}

#endif // defined(__linux__)

#ifdef _WIN32

namespace
{

struct RemoveTreeOnExit
{
    std::filesystem::path path;

    ~RemoveTreeOnExit()
    {
        std::error_code failed;
        std::filesystem::remove_all(path, failed);
    }
};

using ModuleHandle = std::unique_ptr<std::remove_pointer_t<HMODULE>,
                                     decltype(&hipdnn_data_sdk::utilities::closeLibrary)>;

/// A module loaded from a directory whose path exceeds MAX_PATH, and the directory the
/// lookups under test are expected to report. Declaration order is load-bearing: the
/// handle is released before the tree is removed, so FreeLibrary runs while the image
/// file is still there.
struct LongPathModule
{
    RemoveTreeOnExit cleanup{};
    ModuleHandle handle{nullptr, &hipdnn_data_sdk::utilities::closeLibrary};
    std::filesystem::path expectedDirectory{};
};

/// Copies a system DLL into a freshly created extended-length directory and loads it.
/// @p tag distinguishes both the directory and the copy's base name: the loader matches
/// an already-loaded module by base name before mapping a new image, so two probes
/// sharing a name would resolve to whichever was loaded first.
void loadModuleFromLongPath(const std::wstring& tag, LongPathModule& probe)
{
    namespace utilities = hipdnn_data_sdk::utilities;

    std::array<wchar_t, MAX_PATH> systemDirectory{};
    const UINT systemLength = GetSystemDirectoryW(systemDirectory.data(), MAX_PATH);
    ASSERT_GT(systemLength, 0u);
    ASSERT_LT(systemLength, static_cast<UINT>(MAX_PATH));
    // version.dll is not a KnownDLL, so the loader honors a copy's own path instead
    // of redirecting back to the system directory.
    const std::filesystem::path source
        = std::filesystem::path(std::wstring(systemDirectory.data(), systemLength)) / "version.dll";
    ASSERT_TRUE(std::filesystem::is_regular_file(source)) << source;

    // The \\?\ prefix opts these paths out of MAX_PATH regardless of the host's
    // long-path policy.
    std::wstring root = L"\\\\?\\" + std::filesystem::temp_directory_path().wstring();
    if(root.back() != L'\\')
    {
        root += L'\\';
    }
    root += L"hipdnn_long_" + tag + L'_' + std::to_wstring(GetCurrentProcessId());
    probe.cleanup.path = std::filesystem::path(root);

    std::wstring longDirectory = root;
    const std::wstring segment(120, L'p');
    while(longDirectory.size() < static_cast<size_t>(MAX_PATH) + 64)
    {
        longDirectory += L'\\';
        longDirectory += segment;
    }
    const std::filesystem::path directory(longDirectory);
    std::error_code failed;
    std::filesystem::create_directories(directory, failed);
    ASSERT_FALSE(failed) << failed.message();

    const auto library = directory / (L"hipdnn_long_" + tag + L"_probe.dll");
    std::filesystem::copy_file(
        source, library, std::filesystem::copy_options::overwrite_existing, failed);
    ASSERT_FALSE(failed) << failed.message();
    ASSERT_GT(library.native().size(), static_cast<size_t>(MAX_PATH));

    probe.handle.reset(utilities::openLibrary(library));
    ASSERT_NE(probe.handle, nullptr);

    std::error_code resolveFailed;
    const auto resolved = std::filesystem::weakly_canonical(library, resolveFailed);
    probe.expectedDirectory = (resolveFailed ? library : resolved).parent_path();
}

} // namespace

// The HIP anchor tier is dropped whenever this helper reports no origin, so a module
// loaded from an extended-length path must still yield its directory. The test
// executable itself stays at a short path, keeping this about the helper's buffer
// rather than any executable-path limit.
TEST(TestPlatformUtils, GetLoadedLibraryOriginResolvesAPathBeyondMaxPath)
{
    namespace utilities = hipdnn_data_sdk::utilities;

    LongPathModule probe;
    ASSERT_NO_FATAL_FAILURE(loadModuleFromLongPath(L"origin", probe));

    const auto origin = utilities::getLoadedLibraryOrigin(probe.handle.get());

    EXPECT_TRUE(utilities::pathCompEq(origin, probe.expectedDirectory))
        << origin << " != " << probe.expectedDirectory;
    EXPECT_GT(origin.native().size(), static_cast<size_t>(MAX_PATH));
}

// The self tier and the two directories derived from it are dropped whenever this helper
// throws, so a module loaded from an extended-length path must still yield its directory
// when it is reached by an address rather than by handle.
TEST(TestPlatformUtils, GetLoadedLibraryDirectoryForAddressResolvesAPathBeyondMaxPath)
{
    namespace utilities = hipdnn_data_sdk::utilities;

    LongPathModule probe;
    ASSERT_NO_FATAL_FAILURE(loadModuleFromLongPath(L"address", probe));

    // The module's base address is inside the loaded copy's image by construction;
    // an exported entry point may be a forwarder into another module.
    const void* address = reinterpret_cast<const void*>(probe.handle.get());

    const auto owning = utilities::getLoadedLibraryDirectoryForAddress(address);

    EXPECT_TRUE(utilities::pathCompEq(owning, probe.expectedDirectory))
        << owning << " != " << probe.expectedDirectory;
    EXPECT_GT(owning.native().size(), static_cast<size_t>(MAX_PATH));
}

#endif // _WIN32
