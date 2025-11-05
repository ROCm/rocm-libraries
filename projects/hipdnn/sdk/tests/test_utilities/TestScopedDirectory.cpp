// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/FileUtilities.hpp>

using namespace hipdnn_sdk::test_utilities;

TEST(TestScopedDirectory, CreatesAndDestroys)
{
    std::filesystem::path folderPath = "OIJIR44E";
    ASSERT_FALSE(std::filesystem::exists(folderPath));

    {
        ScopedDirectory temp(folderPath);
        ASSERT_TRUE(std::filesystem::exists(folderPath));
    }

    ASSERT_FALSE(std::filesystem::exists(folderPath));
}

TEST(TestScopedDirectory, ExistingPath)
{
    std::filesystem::path folderPath = "REPOIEHJv28";
    ASSERT_FALSE(std::filesystem::exists(folderPath));
    std::filesystem::create_directory(folderPath);
    ASSERT_TRUE(std::filesystem::exists(folderPath));

    EXPECT_THROW(ScopedDirectory{folderPath}, std::runtime_error);

    ASSERT_TRUE(std::filesystem::exists(folderPath));

    std::filesystem::remove_all(folderPath);
}
