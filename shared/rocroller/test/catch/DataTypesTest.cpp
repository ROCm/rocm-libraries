/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <rocRoller/DataTypes/DataTypes.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

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

        SECTION("friendlyTypeName")
        {
            CHECK(friendlyTypeName<float>() == "Float");
            CHECK(friendlyTypeName<Half>() == "Half");
            CHECK_THAT(friendlyTypeName<std::vector<int>>(),
                       Catch::Matchers::ContainsSubstring("vector"));
        }
    }
}
