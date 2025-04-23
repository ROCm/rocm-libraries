/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <test.hpp>

static bool glob_match(const std::string& input, const std::string& pattern)
{
    return test::glob_match(input.begin(), input.end(), pattern.begin(), pattern.end());
}

TEST_CASE(globbing)
{
    EXPECT(not glob_match("ab", "a"));
    EXPECT(not glob_match("ba", "a"));
    EXPECT(not glob_match("bac", "a"));
    EXPECT(glob_match("ab", "ab"));

    // Star loop
    EXPECT(glob_match("/foo/bar/baz/blig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("/foo/bar/baz/xlig/fig/blig", "/foo/*/blig"));
    EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", "a*a*a*a*a*a*a*a*b"));
    EXPECT(glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab",
                      "a*a*a*a*a*a*a*a**a*a*a*a*b"));
    EXPECT(glob_match("aabaabaab", "a*"));
    EXPECT(glob_match("aabaabaab", "a*b*ab"));
    EXPECT(glob_match("aabaabaab", "a*baab"));
    EXPECT(glob_match("aabaabaab", "aa*"));
    EXPECT(glob_match("aabaabaab", "aaba*"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("aabaabqqbaab", "a*baab"));
    EXPECT(glob_match("abcdd", "*d"));
    EXPECT(glob_match("abcdd", "*d*"));
    EXPECT(glob_match("daaadabadmanda", "da*da*da*"));
    EXPECT(glob_match("mississippi", "m*issip*"));
    EXPECT(glob_match("abc", "ab*c"));

    // Repeated star
    EXPECT(glob_match("aabaabqqbaab", "a****baab"));
    EXPECT(glob_match("abcdd", "***d"));
    EXPECT(glob_match("abcdd", "***d****"));
    EXPECT(not glob_match("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "a**z"));

    // Single wildcard
    EXPECT(glob_match("abc", "a?c"));
    EXPECT(not glob_match("abc", "ab?c"));

    // Special characters
    EXPECT(glob_match("test.foo[gpu]", "test.foo[gpu]"));
    EXPECT(glob_match("test.foo[gpu]", "test.foo[*]"));
    EXPECT(glob_match("test.foo[gpu]", "*[*"));

    EXPECT(glob_match("test.foo(gpu)", "test.foo(gpu)"));
    EXPECT(glob_match("test.foo(gpu)", "test.foo(*)"));
    EXPECT(glob_match("test.foo(gpu)", "*(*"));

    EXPECT(not glob_match("test.foog", "test.foo[gpu]"));
    EXPECT(not glob_match("test.foogpu", "test.foo[gpu]"));
    EXPECT(not glob_match("test_foo", "test.foo"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
