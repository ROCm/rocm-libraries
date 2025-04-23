/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
 */
#include <migraphx/stringutils.hpp>
#include <test.hpp>

TEST_CASE(interpolate_string_simple1)
{
    std::string input = "Hello ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_simple2)
{
    std::string input = "${hello}";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_unbalanced)
{
    std::string input = "${hello";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"hello", "bye"}}); }));
}

TEST_CASE(interpolate_string_extra_space)
{
    std::string input = "${  hello  }";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "bye");
}

TEST_CASE(interpolate_string_multiple)
{
    std::string input = "${h} ${w}!";
    auto s            = migraphx::interpolate_string(input, {{"w", "world"}, {"h", "Hello"}});
    EXPECT(s == "Hello world!");
}

TEST_CASE(interpolate_string_next)
{
    std::string input = "${hh}${ww}!";
    auto s            = migraphx::interpolate_string(input, {{"ww", "world"}, {"hh", "Hello"}});
    EXPECT(s == "Helloworld!");
}

TEST_CASE(interpolate_string_dollar_sign)
{
    std::string input = "$hello";
    auto s            = migraphx::interpolate_string(input, {{"hello", "bye"}});
    EXPECT(s == "$hello");
}

TEST_CASE(interpolate_string_missing)
{
    std::string input = "${hello}";
    EXPECT(test::throws([&] { migraphx::interpolate_string(input, {{"h", "bye"}}); }));
}

TEST_CASE(interpolate_string_custom1)
{
    std::string input = "****{{a}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{", "}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom2)
{
    std::string input = "****{{{a}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{", "}}}");
    EXPECT(s == "****b****");
}

TEST_CASE(interpolate_string_custom3)
{
    std::string input = "****{{{{a}}}}****";
    auto s            = migraphx::interpolate_string(input, {{"a", "b"}}, "{{{{", "}}}}");
    EXPECT(s == "****b****");
}

TEST_CASE(slit_string_simple1)
{
    std::string input = "one,two,three";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 3);
    EXPECT(resuts.front() == "one");
    EXPECT(resuts.back() == "three");
}

TEST_CASE(slit_string_simple2)
{
    std::string input = "one";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 1);
    EXPECT(resuts.front() == "one");
}

TEST_CASE(slit_string_simple3)
{
    std::string input = "one two three";
    auto resuts       = migraphx::split_string(input, ',');
    EXPECT(resuts.size() == 1);
    EXPECT(resuts.front() == "one two three");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
