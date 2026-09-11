// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <filesystem>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <iostream>

#if defined(__linux__)
#include <array>
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

// isSecureExecution / getSecureEnv tests

TEST(TestPlatformUtils, IsSecureExecutionFalseForAnOrdinaryProcess)
{
    // A test binary is launched by the same user that owns it, with no set-uid/set-gid
    // bit and no inherited capabilities, which is the whole of what AT_SECURE reports.
    EXPECT_FALSE(hipdnn_data_sdk::utilities::isSecureExecution());
}

// Regression guard for ordinary processes only: it shows that routing a variable through
// getSecureEnv() does not change what an unprivileged caller sees. It says nothing about
// the hardening itself, which only takes effect when isSecureExecution() is true.
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

// getLoadedLibraryOrigin tests

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
            if(info->dlpi_name == nullptr || info->dlpi_name[0] == '\0')
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

} // namespace

TEST(TestPlatformUtils, GetLoadedLibraryOriginReportsTheDirectoryTheLibraryCameFrom)
{
    const std::filesystem::path library = anyLoadedLibraryPath();
    ASSERT_FALSE(library.empty()) << "no loaded shared object to interrogate";

    const auto handle = hipdnn_data_sdk::utilities::openLibrary(library);
    ASSERT_NE(handle, nullptr);

    const auto origin = hipdnn_data_sdk::utilities::getLoadedLibraryOrigin(handle);
    hipdnn_data_sdk::utilities::closeLibrary(handle);

    EXPECT_TRUE(hipdnn_data_sdk::utilities::pathCompEq(
        origin, std::filesystem::weakly_canonical(library).parent_path()));
}

#endif // defined(__linux__)
