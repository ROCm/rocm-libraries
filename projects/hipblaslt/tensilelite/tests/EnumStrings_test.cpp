// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/EnumStrings.hpp>

using namespace TensileLite;

// EnumStrings is a static class: no object can be created from it.
static_assert(!std::is_default_constructible_v<EnumStrings<CustomGridSize>>);

namespace
{
    // Every value in [0, count) must round-trip, have a distinct name, and agree with the
    // stream operators. A value added to an enum but not to its table fails here.
    template <typename Enum>
    void verifyRoundTrip(int count)
    {
        std::unordered_set<std::string> seen;
        for(int i = 0; i < count; i++)
        {
            auto const value = static_cast<Enum>(i);
            auto const name  = toString(value);

            EXPECT_FALSE(name.empty()) << "value " << i;
            EXPECT_TRUE(seen.insert(name).second) << "duplicate name " << name;
            EXPECT_EQ(enumFromString<Enum>(name), value) << name;

            std::ostringstream out;
            out << value;
            EXPECT_EQ(out.str(), name);

            std::istringstream in(name);
            Enum               parsed{};
            in >> parsed;
            EXPECT_EQ(parsed, value) << name;
        }
    }
}

TEST(EnumStringsTest, CustomArgSemanticRoundTrips)
{
    verifyRoundTrip<CustomArgSemantic>(static_cast<int>(CustomArgSemantic::COUNT));
}

TEST(EnumStringsTest, CustomGridSizeRoundTrips)
{
    verifyRoundTrip<CustomGridSize>(static_cast<int>(CustomGridSize::CustomGridSize_Count));
}

TEST(EnumStringsTest, CustomWorkspaceTypeRoundTrips)
{
    verifyRoundTrip<CustomWorkspaceType>(
        static_cast<int>(CustomWorkspaceType::CustomWorkspaceType_Count));
}

TEST(EnumStringsTest, NamesAreTheEnumeratorSpelling)
{
    EXPECT_EQ(toString(CustomArgSemantic::MagicShiftItersPerTile), "MagicShiftItersPerTile");
    EXPECT_EQ(toString(CustomGridSize::StreamKWithBatch), "StreamKWithBatch");
    EXPECT_EQ(toString(CustomWorkspaceType::StreamKWithReduction), "StreamKWithReduction");
}

TEST(EnumStringsTest, UnknownNamesThrowWithTheTypeName)
{
    try
    {
        enumFromString<CustomGridSize>("NotAGridSize");
        FAIL() << "expected an exception";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_STREQ(e.what(), "Invalid CustomGridSize value: NotAGridSize");
    }

    EXPECT_THROW(enumFromString<CustomArgSemantic>("NotASemantic"), std::runtime_error);
    EXPECT_THROW(enumFromString<CustomWorkspaceType>(""), std::runtime_error);

    std::istringstream in("NotAGridSize");
    CustomGridSize     value{};
    EXPECT_THROW(in >> value, std::runtime_error);
}

TEST(EnumStringsTest, OutOfRangeValuesThrow)
{
    try
    {
        toString(CustomGridSize::CustomGridSize_Count);
        FAIL() << "expected an exception";
    }
    catch(std::runtime_error const& e)
    {
        EXPECT_STREQ(e.what(), "Invalid CustomGridSize value: 14");
    }
    EXPECT_THROW(toString(static_cast<CustomGridSize>(-1)), std::runtime_error);
    EXPECT_THROW(toString(CustomWorkspaceType::CustomWorkspaceType_Count), std::runtime_error);
    EXPECT_THROW(toString(CustomArgSemantic::COUNT), std::runtime_error);
}

TEST(EnumStringsTest, CustomArgDefinitionParsesItsSemantic)
{
    std::istringstream   in("uint64 Synchronizer");
    CustomArgDefinition  arg{};
    in >> arg;
    EXPECT_EQ(arg.semantic, CustomArgSemantic::Synchronizer);

    std::istringstream bad("uint64 NotASemantic");
    EXPECT_THROW(bad >> arg, std::runtime_error);
}
