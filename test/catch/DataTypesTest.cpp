#include <rocRoller/DataTypes/DataTypes.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

using namespace rocRoller;

namespace DataTypesTest
{
    TEST_CASE("DataTypes to and from strings", "[toString]")
    {
        STATIC_REQUIRE(CCountedEnum<DataType>);

        SECTION("1:1 mapping")
        {
            auto dataType
                = static_cast<DataType>(GENERATE(range(0, static_cast<int>(DataType::Count))));
            std::string str = toString(dataType);
            CHECK(dataType == fromString<DataType>(str));
        }

        SECTION("Special cases")
        {
            CHECK(DataType::Float == fromString<DataType>("Float"));
            CHECK(DataType::Float == fromString<DataType>("float"));
            CHECK(DataType::Half == fromString<DataType>("Half"));
            CHECK(DataType::Half == fromString<DataType>("half"));
            CHECK(DataType::Half == fromString<DataType>("FP16"));
            CHECK(DataType::Half == fromString<DataType>("fp16"));
            CHECK(DataType::BFloat16 == fromString<DataType>("BF16"));
            CHECK(DataType::BFloat16 == fromString<DataType>("bf16"));
            CHECK(DataType::FP8 == fromString<DataType>("fp8"));
            CHECK(DataType::FP8 == fromString<DataType>("FP8"));
            CHECK(DataType::BF8 == fromString<DataType>("BF8"));
            CHECK(DataType::BF8 == fromString<DataType>("bf8"));
        }
    }
}
