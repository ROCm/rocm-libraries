// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>

#include <msgpack.hpp>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/msgpack/MessagePack.hpp>

using namespace TensileLite;
using Serialization::MessagePackInput;

namespace
{
    template <typename T>
    T decodeEnum(msgpack::object const& obj)
    {
        T                value{};
        MessagePackInput reader(obj);
        reader.input(value);
        return value;
    }

    template <typename T>
    bool decodeEnumHasError(msgpack::object const& obj)
    {
        T                value{};
        MessagePackInput reader(obj);
        reader.input(value);
        return !reader.error.empty();
    }

    template <typename T>
    void verifyAllStringValues(int count,
                               std::function<std::string(int)> nameForIndex)
    {
        for(int i = 0; i < count; i++)
        {
            auto expected = static_cast<T>(i);
            auto name     = nameForIndex(i);

            msgpack::zone   zone;
            msgpack::object obj(name, zone);

            auto decoded = decodeEnum<T>(obj);
            EXPECT_EQ(decoded, expected)
                << "Mismatch for string \"" << name << "\" (index " << i << ")";
        }
    }
}

TEST(HashmapEnumLookupTest, CustomArgSemanticAllStrings)
{
    verifyAllStringValues<CustomArgSemantic>(
        static_cast<int>(CustomArgSemantic::COUNT),
        [](int i) { return toString(static_cast<CustomArgSemantic>(i)); });
}

TEST(HashmapEnumLookupTest, CustomArgTypeAllStrings)
{
    verifyAllStringValues<CustomArgType>(
        static_cast<int>(CustomArgType::CustomArgType_Count),
        [](int i) { return toString(static_cast<CustomArgType>(i)); });
}

TEST(HashmapEnumLookupTest, DataTypeAllStrings)
{
    verifyAllStringValues<rocisa::DataType>(
        static_cast<int>(rocisa::DataType::Count),
        [](int i) { return DataTypeInfo::Get(i).name; });
}

TEST(HashmapEnumLookupTest, CustomGridSizeAllStrings)
{
    verifyAllStringValues<CustomGridSize>(
        static_cast<int>(CustomGridSize::CustomGridSize_Count),
        [](int i) { return toString(static_cast<CustomGridSize>(i)); });
}

TEST(HashmapEnumLookupTest, CustomWorkspaceTypeAllStrings)
{
    verifyAllStringValues<CustomWorkspaceType>(
        static_cast<int>(CustomWorkspaceType::CustomWorkspaceType_Count),
        [](int i) { return toString(static_cast<CustomWorkspaceType>(i)); });
}

TEST(HashmapEnumLookupTest, IntegerStillDecodes)
{
    msgpack::zone   zone;
    msgpack::object obj(static_cast<int64_t>(CustomGridSize::TilesX), zone);

    auto decoded = decodeEnum<CustomGridSize>(obj);
    EXPECT_EQ(decoded, CustomGridSize::TilesX);
}

TEST(HashmapEnumLookupTest, UnknownStringGivesError)
{
    msgpack::zone   zone;
    msgpack::object obj(std::string("NoSuchEnumValue"), zone);

    EXPECT_TRUE(decodeEnumHasError<CustomArgSemantic>(obj));
    EXPECT_TRUE(decodeEnumHasError<CustomArgType>(obj));
    EXPECT_TRUE(decodeEnumHasError<CustomGridSize>(obj));
    EXPECT_TRUE(decodeEnumHasError<CustomWorkspaceType>(obj));
}
