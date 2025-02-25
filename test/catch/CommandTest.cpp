
#include <rocRoller/Operations/Command.hpp>

#include <common/CommonGraphs.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace rocRoller;

namespace CommandTest
{
    TEST_CASE("Command equality", "[command]")
    {
        SECTION("GEMM/TileAdd")
        {
            auto example1 = *rocRollerTest::Graphs::GEMM<float>().getCommand();
            auto example2 = *rocRollerTest::Graphs::GEMM<float>().getCommand();
            auto example3 = *rocRollerTest::Graphs::GEMM<Half>().getCommand();
            auto example4 = *rocRollerTest::Graphs::TileDoubleAdd<Half>().getCommand();

            CHECK(example1 == example2);
            CHECK(example1 != example3);
            CHECK(example1 != example4);

            CHECK(example2 != example3);
            CHECK(example2 != example4);

            CHECK(example3 != example4);
        }
    }

    TEST_CASE("Command serialization", "[command]")
    {
        SECTION("GEMM")
        {
            auto example = rocRollerTest::Graphs::GEMM<float>();

            auto command0 = example.getCommand();
            auto yaml     = Command::toYAML(*command0);
            auto command1 = Command::fromYAML(yaml);

            CHECK((*command0) == command1);
        }

        SECTION("TileDoubleAdd")
        {
            auto example = rocRollerTest::Graphs::TileDoubleAdd<Half>();

            auto command0 = example.getCommand();
            auto yaml     = Command::toYAML(*command0);
            auto command1 = Command::fromYAML(yaml);

            CHECK((*command0) == command1);
        }
    }
}
