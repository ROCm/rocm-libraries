

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

TEST_CASE("concatenate_join joins different types", "[infrastructure][utils]")
{
    CHECK(rocRoller::concatenate_join(", ", "x", 5, rocRoller::DataType::Double) == "x, 5, Double");

    CHECK(rocRoller::concatenate_join(", ") == "");

    CHECK(rocRoller::concatenate_join(", ", 6) == "6");
}
