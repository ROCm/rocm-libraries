/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#include <migraphx/param_utils.hpp>
#include <migraphx/ranges.hpp>
#include <random>
#include <test.hpp>

TEST_CASE(test_param_name)
{
    CHECK(migraphx::param_name(0) == "x0");
    CHECK(migraphx::param_name(1) == "x1");
    CHECK(migraphx::param_name(10) == "x:00010");
    CHECK(migraphx::param_name(11) == "x:00011");
    CHECK(migraphx::param_name(100) == "x:00100");
    CHECK(migraphx::param_name(101) == "x:00101");
    CHECK(migraphx::param_name(10011) == "x:10011");
    CHECK(migraphx::param_name(99999) == "x:99999");
    CHECK(test::throws([] { migraphx::param_name(100000); }));
    CHECK(test::throws([] { migraphx::param_name(100001); }));
}

TEST_CASE(test_param_name_sorted)
{
    auto pname = [](std::size_t i) { return migraphx::param_name(i); };
    std::vector<std::string> names;
    migraphx::transform(migraphx::range(8, 25), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(90, 130), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(990, 1030), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(9990, 10030), std::back_inserter(names), pname);
    migraphx::transform(migraphx::range(99990, 100000), std::back_inserter(names), pname);
    CHECK(std::is_sorted(names.begin(), names.end()));

    auto xnames = names;
    // Shuffled
    std::shuffle(xnames.begin(), xnames.end(), std::minstd_rand{});
    std::sort(xnames.begin(), xnames.end());
    EXPECT(xnames == names);
    // Reversed
    std::reverse(xnames.begin(), xnames.end());
    std::sort(xnames.begin(), xnames.end());
    EXPECT(xnames == names);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
