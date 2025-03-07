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
