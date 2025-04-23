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
 */

#include <migraphx/fileutils.hpp>
#include <test.hpp>
#include <string_view>

namespace fs = migraphx::fs;

constexpr std::string_view baze_name{"test"};
constexpr std::string_view txt{".txt"};
constexpr std::string_view bz2{".bz2"};
constexpr std::string_view separator{": "};

#ifdef _WIN32
constexpr std::string_view executable_postfix{".exe"};
constexpr std::string_view library_prefix{""};
constexpr std::string_view shared_object_postfix{".dll"};
constexpr std::string_view static_library_postfix{".lib"};
constexpr std::string_view object_file_postfix{".obj"};
#else
constexpr std::string_view executable_postfix{""};
constexpr std::string_view library_prefix{"lib"};
constexpr std::string_view shared_object_postfix{".so"};
constexpr std::string_view static_library_postfix{".a"};
constexpr std::string_view object_file_postfix{".o"};
#endif

TEST_CASE(executable_filename)
{
    auto name = migraphx::make_executable_filename(baze_name);
    EXPECT(name == std::string{baze_name}.append(executable_postfix));
}

TEST_CASE(shared_object_filename)
{
    auto name = migraphx::make_shared_object_filename(baze_name);
    EXPECT(name == std::string{library_prefix}.append(baze_name).append(shared_object_postfix));
}

TEST_CASE(object_filename)
{
    auto name = migraphx::make_object_file_filename(baze_name);
    EXPECT(name == std::string{baze_name}.append(object_file_postfix));
}

TEST_CASE(static_library_filename)
{
    auto name = migraphx::make_static_library_filename(baze_name);
    EXPECT(name == std::string{library_prefix}.append(baze_name).append(static_library_postfix));
}

TEST_CASE(append_to_string)
{
    // 'using namespace' required for '+' operator
    using namespace migraphx::MIGRAPHX_INLINE_NS; // NOLINT
    auto cwd = fs::current_path();
    std::string prefix{baze_name};
    auto s1 = prefix + separator + cwd;
    EXPECT(s1 == prefix + separator + cwd.string());
    auto s2 = cwd + std::string{separator} + prefix;
    EXPECT(s2 == cwd.string() + separator + prefix);
}

TEST_CASE(append_file_extension)
{
    auto name    = fs::path{baze_name}.replace_extension(txt);
    auto updated = migraphx::MIGRAPHX_INLINE_NS::append_extension(name, bz2);
    EXPECT(updated == std::string{baze_name}.append(txt).append(bz2));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
