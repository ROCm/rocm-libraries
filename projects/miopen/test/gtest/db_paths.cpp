// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/db_path.hpp>
#include <miopen/db.hpp>
#include <miopen/binary_cache.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/filesystem_checker.hpp>
#include <miopen/logger.hpp>
#include <miopen/tmp_dir.hpp>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "gtest_common.hpp"

#include <string>

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
    testing::NiceMock<MockFilesystemChecker> mock_checker;

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
        miopen::testing::ResetCachedPaths();
        miopen::testing::ResetUserDbPath();
        miopen::SetFilesystemChecker(nullptr);
    }
};

namespace {

class ScopedPathWarningLogging
{
    const bool previous_quiet = miopen::debug::LoggingQuiet;
    ScopedEnvironment<std::string> log_level{
        MIOPEN_LOG_LEVEL, std::to_string(static_cast<int>(miopen::LoggingLevel::Warning))};

public:
    ScopedPathWarningLogging() { miopen::debug::LoggingQuiet = false; }
    ~ScopedPathWarningLogging() { miopen::debug::LoggingQuiet = previous_quiet; }
};

class ScopedStderrCapture
{
    bool active = true;

public:
    ScopedStderrCapture() { testing::internal::CaptureStderr(); }
    ~ScopedStderrCapture()
    {
        if(active)
            testing::internal::GetCapturedStderr();
    }
    std::string Finish()
    {
        auto result = testing::internal::GetCapturedStderr();
        active      = false;
        return result;
    }
};

struct PathWarningCase
{
    bool cache;
    bool network;
    bool override_path;
    const char* name;
};

} // namespace

class CPU_DbPathWarnings_NONE : public CPU_DbPaths_NONE,
                                public testing::WithParamInterface<PathWarningCase>
{
protected:
    fs::path GetPath() const
    {
        return GetParam().cache ? miopen::GetCachePath(false) : miopen::GetUserDbPath();
    }

    void ResetPath()
    {
        miopen::testing::ResetCachedPaths();
        miopen::testing::ResetUserDbPath();
    }
};

TEST_P(CPU_DbPathWarnings_NONE, NetworkDiagnosticsAndPathSelection)
{
    const auto& param = GetParam();
    miopen::TmpDir temp{"db_path_warnings"};
    const auto custom_path = temp.path / "override";
    ScopedEnvironment<std::string> cache_env(
        MIOPEN_CUSTOM_CACHE_DIR, param.cache && param.override_path ? custom_path.string() : "");
    ScopedEnvironment<std::string> db_env(
        MIOPEN_USER_DB_PATH, !param.cache && param.override_path ? custom_path.string() : "");
    ScopedPathWarningLogging logging;

    // Resolve the normal path independently of the network condition. This also avoids
    // encoding build-specific default directories or cache version suffixes in the test.
    ON_CALL(mock_checker, IsNetworkedFilesystem(_)).WillByDefault(Return(false));
    const auto normal_path = GetPath();
    if(normal_path.empty())
        GTEST_SKIP() << "Requested user cache/database path is disabled in this build";
    if(param.override_path)
    {
        if(param.cache)
            EXPECT_EQ(normal_path.parent_path(), custom_path);
        else
            EXPECT_EQ(normal_path, custom_path);
    }
    ResetPath();
    ON_CALL(mock_checker, IsNetworkedFilesystem(_)).WillByDefault(Return(param.network));

    ScopedStderrCapture initial_capture;
    const auto selected_path = GetPath();
    const auto diagnostic    = initial_capture.Finish();

    if(param.network)
    {
        EXPECT_THAT(diagnostic, testing::HasSubstr("Warning"));
        EXPECT_THAT(diagnostic, testing::HasSubstr(normal_path.string()));
        EXPECT_THAT(
            diagnostic,
            testing::HasSubstr(param.cache ? "MIOPEN_CUSTOM_CACHE_DIR" : "MIOPEN_USER_DB_PATH"));
    }
    else
    {
        EXPECT_EQ(diagnostic, "");
    }

#if !MIOPEN_BUILD_DEV
    if(param.network && !param.override_path)
    {
        const auto fallback_root =
            fs::temp_directory_path() / (param.cache ? ".cache" : ".config") / "miopen";
        const auto expected = param.cache ? fallback_root / normal_path.filename() : fallback_root;
        EXPECT_EQ(selected_path, expected);
        EXPECT_THAT(diagnostic, testing::HasSubstr(selected_path.string()));
    }
    else
#endif
    {
        EXPECT_EQ(selected_path, normal_path);
    }

    ScopedStderrCapture repeat_capture;
    EXPECT_EQ(GetPath(), selected_path);
    EXPECT_EQ(repeat_capture.Finish(), "") << "Cached path access must not repeat diagnostics";
}

INSTANTIATE_TEST_SUITE_P(
    Unit,
    CPU_DbPathWarnings_NONE,
    testing::Values(PathWarningCase{false, true, true, "UserDbNetworkOverride"},
                    PathWarningCase{true, true, true, "CacheNetworkOverride"},
                    PathWarningCase{false, false, true, "UserDbLocalOverride"},
                    PathWarningCase{true, false, true, "CacheLocalOverride"},
                    PathWarningCase{false, true, false, "UserDbNetworkDefault"},
                    PathWarningCase{true, true, false, "CacheNetworkDefault"}),
    [](const testing::TestParamInfo<PathWarningCase>& info) { return info.param.name; });

// ============================================================================
// Additional tests
// ============================================================================

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

    // DATABASE_INSTALL_DIR is share/miopen/db on Windows too, under the prefix whose bin/
    // directory holds MIOpen.dll. Resolve that directory independently and compare.
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
    const auto expected =
        miopen::weakly_canonical(fs::path{module_path}).parent_path().parent_path() / "share" /
        "miopen" / "db";

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
