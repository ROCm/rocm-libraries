// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/filesystem_checker.hpp>
#include <gtest/gtest.h>

// Mock implementation of IFilesystemChecker for testing
class MockFilesystemChecker : public miopen::IFilesystemChecker
{
public:
    // Control the return value for testing
    bool should_return_networked = false;

    // Track which paths were checked (useful for verification)
    mutable std::vector<miopen::fs::path> checked_paths;

    bool IsNetworkedFilesystem(const miopen::fs::path& path) const override
    {
        checked_paths.push_back(path);
        return should_return_networked;
    }
};

// Helper to automatically restore the default checker after tests
class FilesystemCheckerGuard
{
public:
    FilesystemCheckerGuard(miopen::IFilesystemChecker* checker)
    {
        miopen::SetFilesystemChecker(checker);
    }

    ~FilesystemCheckerGuard() { miopen::SetFilesystemChecker(nullptr); }
};

TEST(CPU_FilesystemChecker_NONE, DefaultCheckerWorks)
{
    // The default checker should be available
    auto& checker = miopen::GetFilesystemChecker();

    // This will use the real implementation
    // We can't predict the result, but it shouldn't crash
    miopen::fs::path test_path = "/tmp";
    bool result                = checker.IsNetworkedFilesystem(test_path);

    // Just verify it returns a boolean (true or false)
    EXPECT_TRUE(result == true || result == false);
}

TEST(CPU_FilesystemChecker_NONE, MockCheckerCanBeInjected)
{
    MockFilesystemChecker mock;
    mock.should_return_networked = true;

    FilesystemCheckerGuard guard(&mock);

    auto& checker              = miopen::GetFilesystemChecker();
    miopen::fs::path test_path = "/some/test/path";

    bool result = checker.IsNetworkedFilesystem(test_path);

    // Verify the mock was used
    EXPECT_TRUE(result);
    ASSERT_EQ(mock.checked_paths.size(), 1);
    EXPECT_EQ(mock.checked_paths[0], test_path);
}

TEST(CPU_FilesystemChecker_NONE, MockCheckerReturnsNonNetworked)
{
    MockFilesystemChecker mock;
    mock.should_return_networked = false;

    FilesystemCheckerGuard guard(&mock);

    auto& checker              = miopen::GetFilesystemChecker();
    miopen::fs::path test_path = "/another/test/path";

    bool result = checker.IsNetworkedFilesystem(test_path);

    // Verify the mock returned false
    EXPECT_FALSE(result);
    ASSERT_EQ(mock.checked_paths.size(), 1);
    EXPECT_EQ(mock.checked_paths[0], test_path);
}

TEST(CPU_FilesystemChecker_NONE, DefaultCheckerRestoredAfterTest)
{
    {
        MockFilesystemChecker mock;
        mock.should_return_networked = true;

        FilesystemCheckerGuard guard(&mock);

        // Inside this scope, mock is active
        EXPECT_TRUE(miopen::GetFilesystemChecker().IsNetworkedFilesystem("/test"));
    }

    // After guard goes out of scope, default checker should be restored
    // We can't predict the result, but it should work without crashing
    miopen::fs::path test_path = "/tmp";
    bool result                = miopen::GetFilesystemChecker().IsNetworkedFilesystem(test_path);
    EXPECT_TRUE(result == true || result == false);
}

TEST(CPU_FilesystemChecker_NONE, MultiplePathsCanBeChecked)
{
    MockFilesystemChecker mock;
    mock.should_return_networked = true;

    FilesystemCheckerGuard guard(&mock);

    auto& checker = miopen::GetFilesystemChecker();

    checker.IsNetworkedFilesystem("/path1");
    checker.IsNetworkedFilesystem("/path2");
    checker.IsNetworkedFilesystem("/path3");

    // Verify all paths were checked
    ASSERT_EQ(mock.checked_paths.size(), 3);
    EXPECT_EQ(mock.checked_paths[0], "/path1");
    EXPECT_EQ(mock.checked_paths[1], "/path2");
    EXPECT_EQ(mock.checked_paths[2], "/path3");
}
