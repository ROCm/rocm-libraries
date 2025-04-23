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

#include <algorithm>
#include <array>
#include <climits>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>

#include <migraphx/tmp_dir.hpp>
#include <migraphx/env.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/process.hpp>
#include <migraphx/filesystem.hpp>

#ifndef _WIN32
#include <cstring>
#else
#include <io.h>
#include <fcntl.h>
#endif

#include "test.hpp"

static migraphx::fs::path executable; // NOLINT

constexpr std::string_view string_data =
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
    "sed do eiusmod tempor incididunt ut labore et dolore magna "
    "aliqua. Ut enim ad minim veniam, quis nostrud exercitation "
    "ullamco laboris nisi ut aliquip ex ea commodo consequat. "
    "Duis aute irure dolor in reprehenderit in voluptate velit "
    "esse cillum dolore eu fugiat nulla pariatur. Excepteur sint "
    "occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum.";

static std::vector<char> read_stdin()
{
    std::vector<char> result;
    std::array<char, 1024> buffer{};
    std::size_t len = 0;
#ifdef _WIN32
    // Set stream translation mode to BINARY to suppress translations.
    // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/setmode?view=msvc-170
    auto old_mode = _setmode(_fileno(stdin), _O_BINARY);
    if(old_mode == -1)
        throw std::runtime_error{"failure setting IO mode to binary"};
#endif
    while((len = std::fread(buffer.data(), 1, buffer.size(), stdin)) > 0)
    {
        if(std::ferror(stdin) != 0 and std::feof(stdin) == 0)
            throw std::runtime_error{std::strerror(errno)};

        result.insert(result.end(), buffer.begin(), buffer.begin() + len);
    }
#ifdef _WIN32
    // Reset to the previously set translation mode.
    _setmode(_fileno(stdin), old_mode);
#endif
    return result;
}

TEST_CASE(string_stdin)
{
    auto tmp = migraphx::tmp_dir{};
    auto out = (tmp.path / "output.txt").string();

    migraphx::process{executable, {"--stdin", out}}.write(
        [&](const auto& writer) { writer(string_data.data(), string_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out)};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(binary_stdin)
{
    std::random_device rd;
    std::independent_bits_engine<std::mt19937, CHAR_BIT, unsigned short> rbe(rd());

    std::vector<char> binary_data(4096);
    std::generate(binary_data.begin(), binary_data.end(), std::ref(rbe));

    auto tmp = migraphx::tmp_dir{};
    auto out = (tmp.path / "output.bin").string();

    migraphx::process{executable, {"--stdin", out}}.write(
        [&](const auto& writer) { writer(binary_data.data(), binary_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::vector<char> result{migraphx::read_buffer(out)};
    EXPECT(result == binary_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(read_stdout)
{
    std::string buffer;
    migraphx::process{executable, {"--stdout"}}.read([&buffer](const char* buf, std::size_t size) {
        buffer = std::string{buf, size};
    });
    EXPECT(buffer == string_data);
}

TEST_CASE(current_working_dir)
{
    constexpr auto filename = "output.txt";
    auto tmp                = migraphx::tmp_dir{};

    auto out = tmp.path / filename;

    migraphx::process{executable, {"--stdin", filename}}.cwd(tmp.path).write(
        [&](const auto& writer) { writer(string_data.data(), string_data.size()); });

    EXPECT(migraphx::fs::is_regular_file(out));

    std::string result{migraphx::read_string(out)};
    EXPECT(result == string_data);

    EXPECT(migraphx::fs::remove(out));
}

TEST_CASE(environment_variable)
{
    std::string buffer;
    migraphx::process{executable, {"--stdout"}}
        .env({"MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE=1"})
        .read([&buffer](const char* buf, std::size_t size) {
            buffer = std::string{buf, size};
        });
    std::string reversed(string_data);
    std::reverse(reversed.begin(), reversed.end());
    EXPECT(buffer == reversed);
}

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE)

int main(int argc, const char* argv[])
{
    if(argc > 1)
    {
        std::string arg = argv[1];
        if(arg == "--stdin")
        {
            migraphx::write_buffer(argv[2], read_stdin());
            return 0;
        }
        if(arg == "--stdout")
        {
            std::vector<char> result{string_data.begin(), string_data.end()};
            if(migraphx::enabled(MIGRAPHX_PROCESS_TEST_ENVIRONMENT_VARIABLE{}))
                std::reverse(result.begin(), result.end());
            std::fwrite(result.data(), 1, result.size(), stdout);
            return 0;
        }
    }
    else
    {
        executable = argv[0];
        test::run(argc, argv);
    }
}
