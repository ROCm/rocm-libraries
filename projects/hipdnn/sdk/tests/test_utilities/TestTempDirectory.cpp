// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TempDirectory.hpp>

using namespace hipdnn_sdk::test_utilities;

TEST(TestTempDirectory, CreatesAndDestroys)
{
    std::filesystem::path folderPath = "OIJIR44E";
    ASSERT_FALSE(std::filesystem::exists(folderPath));

    {
        TempDirectory temp(folderPath);
        ASSERT_TRUE(std::filesystem::exists(folderPath));
    }

    ASSERT_FALSE(std::filesystem::exists(folderPath));
}

TEST(TestTempDirectory, ExistingPath)
{
    std::filesystem::path folderPath = "REPOIEHJv28";
    ASSERT_FALSE(std::filesystem::exists(folderPath));
    std::filesystem::create_directory(folderPath);
    ASSERT_TRUE(std::filesystem::exists(folderPath));

    EXPECT_THROW(TempDirectory{folderPath}, std::runtime_error);

    ASSERT_TRUE(std::filesystem::exists(folderPath));

    std::filesystem::remove_all(folderPath);
}
