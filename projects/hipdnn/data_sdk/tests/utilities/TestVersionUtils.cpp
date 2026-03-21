// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <random>

namespace hipdnn_data_sdk::utilities
{

TEST(TestVersionUtils, VersionStringConstructorValidInput)
{
    Version version;
    const std::vector<std::string> validVersions{"1.12.53", "1.12.53.aw939d94"};

    for(const auto& versionStr : validVersions)
    {
        ASSERT_NO_THROW(version = Version{versionStr});
        EXPECT_EQ(version, Version(1, 12, 53)) << "Version string = " << versionStr;
    }
}

TEST(TestVersionUtils, VersionTupleInvalidInput)
{
    Version version;
    const std::vector<std::string> invalidVersions{"1.12", "1.12.a53", "1", "", "Str1.12.53"};

    for(const auto& versionStr : invalidVersions)
    {
        EXPECT_THROW(version = Version(versionStr), std::invalid_argument)
            << "Version string = " << versionStr;
    }
}

namespace
{
std::vector<Version> randomVersions(size_t numberGenerated, unsigned int seed = 0)
{
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 100);
    std::vector<Version> versions;
    versions.reserve(numberGenerated);

    for(size_t i = 0; i < numberGenerated; i++)
    {
        versions.emplace_back(
            distribution(generator), distribution(generator), distribution(generator));
    }

    return versions;
}

std::vector<std::pair<Version, Version>> randomVersionCartesianProduct(size_t numberGenerated,
                                                                       unsigned int seed = 0)
{
    auto versions = randomVersions(15, seed);
    std::vector<std::pair<Version, Version>> versionCartesianProducts;
    versionCartesianProducts.reserve(numberGenerated * numberGenerated);

    for(const auto& v1 : versions)
    {
        for(const auto& v2 : versions)
        {
            versionCartesianProducts.emplace_back(v1, v2);
        }
    }

    return versionCartesianProducts;
}
} // namespace

TEST(TestVersionUtils, VersionTupleEquals)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 == version1, tuple0 == tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

TEST(TestVersionUtils, VersionTupleNotEqual)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 != version1, tuple0 != tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

TEST(TestVersionUtils, VersionTupleLessThan)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 < version1, tuple0 < tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

TEST(TestVersionUtils, VersionTupleLessThanOrEquals)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 <= version1, tuple0 <= tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

TEST(TestVersionUtils, VersionTupleGreaterThan)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 > version1, tuple0 > tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

TEST(TestVersionUtils, VersionTupleGreaterThanOrEquals)
{
    auto versionProduct = randomVersionCartesianProduct(8);

    for(auto [version0, version1] : versionProduct)
    {
        auto tuple0 = std::make_tuple(version0.major, version0.minor, version0.patch);
        auto tuple1 = std::make_tuple(version1.major, version1.minor, version1.patch);
        EXPECT_EQ(version0 >= version1, tuple0 >= tuple1)
            << "Version 0 = " << version0.str() << " Version 1 = " << version1.str();
    }
}

}
