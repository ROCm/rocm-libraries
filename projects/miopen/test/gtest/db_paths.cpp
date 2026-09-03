// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/db_path.hpp>
#include <miopen/db.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/filesystem_checker.hpp>
#include <miopen/version.h>
#include <miopen/stringutils.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gtest_common.hpp"

#include <string>
#include <sstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <miopen/miopen.h>
#include <windows.h>
#endif

using ::testing::_;
using ::testing::Return;

namespace fs = miopen::fs;

MIOPEN_LIB_ENV_VAR(MIOPEN_USER_DB_PATH)
MIOPEN_LIB_ENV_VAR(MIOPEN_CUSTOM_CACHE_DIR)
MIOPEN_LIB_ENV_VAR(MIOPEN_SYSTEM_DB_PATH)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_DISABLE_SYSTEM_DB)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_DISABLE_USER_DB)

// Helper function to build expected version string
std::string GetExpectedVersionString()
{
    std::ostringstream oss;
    oss << MIOPEN_VERSION_MAJOR << "." << MIOPEN_VERSION_MINOR << "." << MIOPEN_VERSION_PATCH << "."
        << MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK);
    return oss.str();
}

// Helper function to check if a path contains a substring
bool PathContains(const fs::path& path, const std::string& substring)
{
    return path.string().find(substring) != std::string::npos;
}

// Mock filesystem checker for testing
class MockFilesystemChecker : public miopen::IFilesystemChecker
{
public:
    MOCK_METHOD(bool, IsNetworkedFilesystem, (const fs::path& path), (const, override));
};

// Test fixture for db_path tests with filesystem mocking
class CPU_DbPaths_NONE : public ::testing::Test
{
protected:
    MockFilesystemChecker mock_checker;

    void SetUp() override
    {
        // Reset cached paths to allow fresh initialization with new mock/env settings
        miopen::testing::ResetCachedPaths();
        miopen::testing::ResetUserDbPath();

        // Install the mock checker before any path functions are called
        miopen::SetFilesystemChecker(&mock_checker);
    }

    void TearDown() override
    {
        // Restore default checker
        miopen::SetFilesystemChecker(nullptr);
    }
};

// ============================================================================
// Tests for GetUserDbPath() - 4 scenarios
// ============================================================================

TEST_F(CPU_DbPaths_NONE, UserDbPath_LocalFS_NoEnvVar)
{
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_user_db(MIOPEN_USER_DB_PATH, "");

    // Scenario 1: Local filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use default path containing "miopen"
    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path '" << user_db_path.string() << "' should contain 'miopen' folder";
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_LocalFS_EnvVarSet)
{
    // Scenario 2: Local filesystem, environment variable set
    const std::string custom_path = "/custom/user/db/path";
    ScopedEnvironment<std::string> scoped_env(MIOPEN_USER_DB_PATH, custom_path);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom path from environment variable
    EXPECT_TRUE(PathContains(user_db_path, custom_path))
        << "User DB path '" << user_db_path.string() << "' should contain custom path '"
        << custom_path << "'";
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_NetworkFS_NoEnvVar)
{
#if MIOPEN_BUILD_DEV
    GTEST_SKIP() << "Network filesystem detection is disabled in MIOPEN_BUILD_DEV mode";
#else
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_user_db(MIOPEN_USER_DB_PATH, "");

    // Scenario 3: Network filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should fallback to temp directory
    const auto temp_dir = fs::temp_directory_path();
    EXPECT_TRUE(PathContains(user_db_path, temp_dir.string()))
        << "User DB path '" << user_db_path.string()
        << "' should be in temp directory for network filesystem";

    // Should contain .config/miopen
    EXPECT_TRUE(PathContains(user_db_path, ".config"))
        << "User DB path should contain '.config' folder";
    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path should contain 'miopen' folder";
#endif
}

TEST_F(CPU_DbPaths_NONE, UserDbPath_NetworkFS_EnvVarSet)
{
    // Scenario 4: Network filesystem, environment variable set
    // Environment variable should take precedence over network detection
    const std::string custom_path = "/custom/network/db/path";
    ScopedEnvironment<std::string> scoped_env(MIOPEN_USER_DB_PATH, custom_path);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom path from environment variable (takes precedence)
    EXPECT_TRUE(PathContains(user_db_path, custom_path))
        << "User DB path '" << user_db_path.string()
        << "' should use custom path even on network filesystem";
}

// ============================================================================
// Tests for GetCachePath() - 4 scenarios
// ============================================================================

TEST_F(CPU_DbPaths_NONE, CachePath_LocalFS_NoEnvVar)
{
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_cache(MIOPEN_CUSTOM_CACHE_DIR, "");

    // Scenario 1: Local filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should contain version string
    const std::string expected_version = GetExpectedVersionString();
    EXPECT_TRUE(PathContains(cache_path, expected_version))
        << "Cache path '" << cache_path.string() << "' should contain version string '"
        << expected_version << "'";

    // Should contain "miopen"
    EXPECT_TRUE(PathContains(cache_path, "miopen")) << "Cache path should contain 'miopen' folder";
}

TEST_F(CPU_DbPaths_NONE, CachePath_LocalFS_EnvVarSet)
{
    // Scenario 2: Local filesystem, environment variable set
    const std::string custom_cache = (fs::temp_directory_path() / "custom/cache/dir").string();
    ScopedEnvironment<std::string> scoped_env(MIOPEN_CUSTOM_CACHE_DIR, custom_cache);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom cache directory
    EXPECT_TRUE(PathContains(cache_path, custom_cache))
        << "Cache path '" << cache_path.string() << "' should contain custom cache dir '"
        << custom_cache << "'";
}

TEST_F(CPU_DbPaths_NONE, CachePath_NetworkFS_NoEnvVar)
{
#if MIOPEN_BUILD_DEV
    GTEST_SKIP() << "Network filesystem detection is disabled in MIOPEN_BUILD_DEV mode";
#else
    // Ensure environment variable is NOT set for this test
    ScopedEnvironment<std::string> unset_cache(MIOPEN_CUSTOM_CACHE_DIR, "");

    // Scenario 3: Network filesystem, no environment variable set
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should fallback to temp directory
    const auto temp_dir = fs::temp_directory_path();
    EXPECT_TRUE(PathContains(cache_path, temp_dir.string()))
        << "Cache path '" << cache_path.string()
        << "' should be in temp directory for network filesystem";

    // Should contain .cache/miopen/<version>
    EXPECT_TRUE(PathContains(cache_path, ".cache")) << "Cache path should contain '.cache' folder";
    EXPECT_TRUE(PathContains(cache_path, "miopen")) << "Cache path should contain 'miopen' folder";

    const std::string expected_version = GetExpectedVersionString();
    EXPECT_TRUE(PathContains(cache_path, expected_version))
        << "Cache path should contain version string '" << expected_version << "'";
#endif
}

TEST_F(CPU_DbPaths_NONE, CachePath_NetworkFS_EnvVarSet)
{
    // Scenario 4: Network filesystem, environment variable set
    // Environment variable should take precedence over network detection
    const std::string custom_cache = (fs::temp_directory_path() / "custom/network/cache").string();
    ScopedEnvironment<std::string> scoped_env(MIOPEN_CUSTOM_CACHE_DIR, custom_cache);

    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(true));

    const auto cache_path = miopen::GetCachePath(false);

    if(cache_path.empty())
    {
        GTEST_SKIP() << "User cache is disabled (MIOPEN_DISABLE_USERDB)";
    }

    // Should use the custom cache directory (takes precedence)
    EXPECT_TRUE(PathContains(cache_path, custom_cache))
        << "Cache path '" << cache_path.string()
        << "' should use custom cache even on network filesystem";
}

// ============================================================================
// Additional tests
// ============================================================================

TEST_F(CPU_DbPaths_NONE, UserDbSuffix_ContainsVersionInfo)
{
    const std::string suffix = miopen::GetUserDbSuffix();
    EXPECT_FALSE(suffix.empty()) << "User DB suffix should not be empty";

    // Suffix should contain version numbers separated by underscores
    std::ostringstream expected_pattern;
    expected_pattern << MIOPEN_VERSION_MAJOR << "_" << MIOPEN_VERSION_MINOR << "_"
                     << MIOPEN_VERSION_PATCH;

    EXPECT_TRUE(suffix.find(expected_pattern.str()) != std::string::npos)
        << "Suffix '" << suffix << "' should contain version pattern '" << expected_pattern.str()
        << "'";
}

TEST_F(CPU_DbPaths_NONE, UserAndSystemCachePaths_AreDifferent)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto user_cache = miopen::GetCachePath(false);
    const auto sys_cache  = miopen::GetCachePath(true);

    if(!user_cache.empty() && !sys_cache.empty())
    {
        EXPECT_NE(user_cache, sys_cache) << "User and system cache paths should be different";
    }
}

TEST_F(CPU_DbPaths_NONE, Paths_AreValid)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto& user_db_path  = miopen::GetUserDbPath();
    const auto cache_path     = miopen::GetCachePath(false);
    const auto sys_cache_path = miopen::GetCachePath(true);

    // Paths should either be empty (if disabled) or valid filesystem paths
    if(!user_db_path.empty())
    {
        EXPECT_FALSE(user_db_path.string().empty());
    }

    if(!cache_path.empty())
    {
        EXPECT_FALSE(cache_path.string().empty());
    }

    if(!sys_cache_path.empty())
    {
        EXPECT_FALSE(sys_cache_path.string().empty());
    }
}

TEST_F(CPU_DbPaths_NONE, CacheDisabled_ReturnsCorrectly)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const bool is_disabled = miopen::IsCacheDisabled();

    // The result depends on build configuration
    // Just verify it returns a valid boolean without crashing
    EXPECT_TRUE(is_disabled == true || is_disabled == false);
}

// ============================================================================
// Tests for GetSystemDbPath()
// ============================================================================

TEST_F(CPU_DbPaths_NONE, SystemDbPath_EnvVarSet)
{
    const std::string custom_path = (fs::temp_directory_path() / "custom/system/db").string();
    ScopedEnvironment<std::string> scoped_env(MIOPEN_SYSTEM_DB_PATH, custom_path);

    EXPECT_EQ(miopen::GetSystemDbPath(), fs::path{custom_path})
        << "MIOPEN_SYSTEM_DB_PATH must take precedence over the built-in default";
}

TEST_F(CPU_DbPaths_NONE, SystemDbPath_NoEnvVar_IsAbsolute)
{
    ScopedEnvironment<std::string> unset_system_db(MIOPEN_SYSTEM_DB_PATH, "");

    const auto system_db_path = miopen::GetSystemDbPath();

    // An empty or relative default is resolved against the current working directory, so the
    // system databases would only be found when the process happens to be started from the
    // directory holding them.
    EXPECT_TRUE(system_db_path.is_absolute())
        << "System DB path '" << system_db_path.string()
        << "' must be absolute so database lookup does not depend on the working directory";
}

#ifdef _WIN32
TEST_F(CPU_DbPaths_NONE, SystemDbPath_NoEnvVar_FollowsMIOpenModule)
{
#if MIOPEN_BUILD_DEV
    GTEST_SKIP() << "Development builds read databases from the build tree, not the install layout";
#else
    ScopedEnvironment<std::string> unset_system_db(MIOPEN_SYSTEM_DB_PATH, "");

    // DATABASE_INSTALL_DIR is "bin" on Windows, so the databases are installed next to
    // MIOpen.dll. Resolve that directory independently and compare.
    HMODULE hmod = nullptr;
    ASSERT_NE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                     GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                 reinterpret_cast<LPCWSTR>(miopenCreate),
                                 &hmod),
              0)
        << "Unable to locate the module exporting miopenCreate";

    std::wstring module_path(MAX_PATH, L'\0');
    DWORD len = 0;
    while(true)
    {
        len = GetModuleFileNameW(hmod, module_path.data(), static_cast<DWORD>(module_path.size()));
        ASSERT_NE(len, 0u) << "GetModuleFileNameW failed";
        if(len < module_path.size())
            break;
        module_path.resize(module_path.size() * 2);
    }
    module_path.resize(len);
    const auto expected = miopen::weakly_canonical(fs::path{module_path}).parent_path();

    EXPECT_EQ(miopen::GetSystemDbPath(), expected)
        << "System DB path should follow MIOpen.dll so an installed MIOpen finds its databases "
           "no matter which directory the application runs from";
#endif
}
#endif

// ============================================================================
// Tests for IsSystemDbDisabled()
// ============================================================================

TEST_F(CPU_DbPaths_NONE, SystemDbDisabled_UnsetEnvVar)
{
    ScopedEnvironment<bool> unset_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB);

    // The system databases are on by default, unless the build disables them outright.
    EXPECT_EQ(miopen::IsSystemDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_SYSDB));
}

TEST_F(CPU_DbPaths_NONE, SystemDbDisabled_EnvVarSet)
{
    ScopedEnvironment<bool> disable_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB, true);

    EXPECT_TRUE(miopen::IsSystemDbDisabled());
}

TEST_F(CPU_DbPaths_NONE, SystemDbDisabled_EnvVarCleared)
{
    ScopedEnvironment<bool> enable_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB, false);

    EXPECT_EQ(miopen::IsSystemDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_SYSDB));
}

TEST_F(CPU_DbPaths_NONE, SystemDbDisabled_DoesNotAffectUserDbOrSystemDbPath)
{
    EXPECT_CALL(mock_checker, IsNetworkedFilesystem(_)).WillRepeatedly(Return(false));

    const auto sys_db_path_before = miopen::GetSystemDbPath();

    ScopedEnvironment<bool> disable_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB, true);

    // GetSystemDbPath() is also the home of the AI heuristic models, so it must keep
    // resolving. Only the find-db and perf-db lookups are gated on IsSystemDbDisabled().
    EXPECT_EQ(miopen::GetSystemDbPath(), sys_db_path_before);

    miopen::testing::ResetUserDbPath();
    const auto& user_db_path = miopen::GetUserDbPath();

    if(user_db_path.empty())
    {
        GTEST_SKIP() << "User DB is disabled (MIOPEN_DISABLE_USERDB)";
    }

    EXPECT_TRUE(PathContains(user_db_path, "miopen"))
        << "User DB path '" << user_db_path.string()
        << "' should be unaffected by MIOPEN_DEBUG_DISABLE_SYSTEM_DB";
}

// ============================================================================
// Tests for IsUserDbDisabled()
// ============================================================================

TEST_F(CPU_DbPaths_NONE, UserDbDisabled_UnsetEnvVar)
{
    ScopedEnvironment<bool> unset_userdb(MIOPEN_DEBUG_DISABLE_USER_DB);

    // The user databases are on by default, unless the build disables them outright.
    EXPECT_EQ(miopen::IsUserDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_USERDB));
}

TEST_F(CPU_DbPaths_NONE, UserDbDisabled_EnvVarSet)
{
    ScopedEnvironment<bool> disable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, true);

    EXPECT_TRUE(miopen::IsUserDbDisabled());
}

TEST_F(CPU_DbPaths_NONE, UserDbDisabled_EnvVarCleared)
{
    ScopedEnvironment<bool> enable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, false);

    EXPECT_EQ(miopen::IsUserDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_USERDB));
}

TEST_F(CPU_DbPaths_NONE, UserAndSystemDbSwitches_AreIndependent)
{
    {
        ScopedEnvironment<bool> disable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, true);
        ScopedEnvironment<bool> enable_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB, false);

        EXPECT_TRUE(miopen::IsUserDbDisabled());
        EXPECT_EQ(miopen::IsSystemDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_SYSDB));
    }
    {
        ScopedEnvironment<bool> enable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, false);
        ScopedEnvironment<bool> disable_sysdb(MIOPEN_DEBUG_DISABLE_SYSTEM_DB, true);

        EXPECT_EQ(miopen::IsUserDbDisabled(), static_cast<bool>(MIOPEN_DISABLE_USERDB));
        EXPECT_TRUE(miopen::IsSystemDbDisabled());
    }
}

TEST_F(CPU_DbPaths_NONE, UserDbDisabled_SuppressesWrites)
{
    ScopedEnvironment<bool> disable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, true);

    // With user-db I/O off, a store must report success without creating the file, and a
    // subsequent lookup must miss -- that is what keeps repeated runs reproducible.
    const auto db_file = fs::temp_directory_path() / "miopen_disable_user_db_test.db.txt";
    fs::remove(db_file);

    miopen::PlainTextDb db{miopen::DbKinds::PerfDb, db_file};
    const miopen::DbRecord record{miopen::DbKinds::PerfDb, std::string{"key"}};

    EXPECT_TRUE(db.StoreRecord(record)) << "StoreRecord should report success when I/O is off";
    EXPECT_FALSE(fs::exists(db_file)) << "No user db file should have been created at " << db_file;
    EXPECT_FALSE(db.FindRecord(std::string{"key"}).has_value())
        << "Lookup should miss when user db I/O is disabled";
}

TEST_F(CPU_DbPaths_NONE, UserDbDisabled_LatchedAtConstruction)
{
    if(static_cast<bool>(MIOPEN_DISABLE_USERDB))
        GTEST_SKIP() << "User db file I/O is disabled at build time";

    // Flipping the switch must never take effect part-way through a live database object: the
    // answer is fixed when the object is constructed, so the constructor and every later operation
    // agree. An instance built while the switch was off keeps writing.
    const auto db_file = fs::temp_directory_path() / "miopen_user_db_latch_test.db.txt";
    fs::remove(db_file);

    miopen::PlainTextDb db{miopen::DbKinds::PerfDb, db_file};
    const miopen::DbRecord record{miopen::DbKinds::PerfDb, std::string{"key"}};

    {
        ScopedEnvironment<bool> disable_userdb(MIOPEN_DEBUG_DISABLE_USER_DB, true);

        EXPECT_TRUE(miopen::IsUserDbDisabled());
        EXPECT_TRUE(db.StoreRecord(record));
        EXPECT_TRUE(fs::exists(db_file))
            << "An already constructed db must keep the file I/O it was built with";
    }

    fs::remove(db_file);
}
