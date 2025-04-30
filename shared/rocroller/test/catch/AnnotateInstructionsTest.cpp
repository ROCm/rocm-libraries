/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/Annotate.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>

TEST_CASE("AddComment works", "")
{
    using namespace rocRoller;
    using namespace Catch::Matchers;

    auto generatorOne = []() -> Generator<Instruction> {
        co_yield_(Instruction("v_add_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_sub_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_mul_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_add_f32", {}, {}, {}, ""));
    };

    for(auto inst : generatorOne().map(AddComment("foo")))
        CHECK_THAT(inst.comments(), Contains("foo"));

    auto generatorTwo = [&]() -> Generator<Instruction> {
        //
        co_yield generatorOne().map(AddComment("bar"));
    };

    for(auto inst : generatorTwo())
        CHECK_THAT(inst.comments(), Contains("bar"));

    for(auto inst : generatorTwo().map(AddComment("baz")))
        CHECK_THAT(inst.comments(), Contains("bar") && Contains("baz"));
}

TEST_CASE("AddControlOp works", "")
{
    using namespace rocRoller;
    using namespace Catch::Matchers;

    auto generatorOne = []() -> Generator<Instruction> {
        co_yield_(Instruction("v_add_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_sub_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_mul_u32", {}, {}, {}, ""));
        co_yield_(Instruction("v_add_f32", {}, {}, {}, ""));
    };

    for(auto inst : generatorOne().map(AddControlOp(3)))
        CHECK(inst.controlOps() == std::vector{3});

    auto generatorTwo = [&]() -> Generator<Instruction> {
        //
        co_yield generatorOne().map(AddControlOp(9));
    };

    for(auto inst : generatorTwo())
        CHECK(inst.controlOps() == std::vector{9});

    for(auto inst : generatorTwo().map(AddControlOp(4)))
        CHECK(inst.controlOps() == std::vector({9, 4}));
}
