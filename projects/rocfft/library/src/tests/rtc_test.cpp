// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "../../../shared/environment.h"
#include "rtc_cache.h"
#include <array>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

#ifndef _WIN32
#include <errno.h> // program_invocation_name
#endif

namespace fs = std::filesystem;

static const char* simple_kernel_jit = R"(
extern "C" __global__ void simple_kernel(int* input)
{
    *input = 1337;
}
)";

// make sure RTC gracefully handles a helper process that crashes
TEST(rocfft_internal, rtc_helper_crash)
{
#ifdef _WIN32
    char filename[MAX_PATH];
    GetModuleFileNameA(NULL, filename, MAX_PATH);
    fs::path test_exe    = filename;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash.exe");
#else
    fs::path test_exe    = program_invocation_name;
    fs::path crasher_exe = test_exe.replace_filename("rtc_helper_crash");
#endif

    // don't touch the cache, to force compilation
    EnvironmentSetTemp env_read("ROCFFT_RTC_CACHE_READ_DISABLE", "1");
    EnvironmentSetTemp env_write("ROCFFT_RTC_CACHE_WRITE_DISABLE", "1");
    // force out-of-process compile
    EnvironmentSetTemp env_process("ROCFFT_RTC_PROCESS", "1");

    // build a trivial kernel for some specific arch
    auto generator_func = [](const std::string&) { return std::string(simple_kernel_jit); };
    kernel_src_gen_t generator(generator_func);
    const auto       code = RTCCache::cached_compile(
        "simple_kernel", "gfx1201", generator, std::array<char, 32>{}, crasher_exe.string());

    // we should get compiled code back
    ASSERT_FALSE(code.empty());
}

TEST(rocfft_internal, rtc_cache_store_then_get)
{
    RTCCache cache;

    const std::string    name = "internal_test_kernel";
    const std::string    arch = "gfx942";
    std::array<char, 32> sum{};
    sum[0]                       = 'a';
    const std::vector<char> code = {'c', 'o', 'd', 'e'};

    // nothing stored yet
    EXPECT_TRUE(cache.get_code_object(name, arch, sum).empty());

    cache.store_code_object(name, arch, sum, code);
    EXPECT_EQ(cache.get_code_object(name, arch, sum), code);

    // a different name, architecture or generator checksum is a different kernel
    EXPECT_TRUE(cache.get_code_object("other_kernel", arch, sum).empty());
    EXPECT_TRUE(cache.get_code_object(name, "gfx90a", sum).empty());

    std::array<char, 32> other_sum{};
    other_sum[0] = 'b';
    EXPECT_TRUE(cache.get_code_object(name, arch, other_sum).empty());
}

TEST(rocfft_internal, rtc_cache_store_replaces)
{
    RTCCache cache;

    const std::string    name = "internal_test_replace";
    const std::string    arch = "gfx942";
    std::array<char, 32> sum{};

    cache.store_code_object(name, arch, sum, {'o', 'l', 'd'});
    cache.store_code_object(name, arch, sum, {'n', 'e', 'w'});

    EXPECT_EQ(cache.get_code_object(name, arch, sum), (std::vector<char>{'n', 'e', 'w'}));
}

TEST(rocfft_internal, rtc_cache_serialize_round_trip)
{
    const std::string       name = "internal_test_serialize";
    const std::string       arch = "gfx942";
    std::array<char, 32>    sum{};
    const std::vector<char> code = {'s', 'e', 'r'};

    void*  buffer     = nullptr;
    size_t buffer_len = 0;
    {
        RTCCache source;
        source.store_code_object(name, arch, sum, code);
        ASSERT_EQ(source.serialize(&buffer, &buffer_len), rocfft_status_success);
    }
    ASSERT_NE(buffer, nullptr);
    ASSERT_GT(buffer_len, 0u);

    RTCCache destination;
    EXPECT_TRUE(destination.get_code_object(name, arch, sum).empty());
    ASSERT_EQ(destination.deserialize(buffer, buffer_len), rocfft_status_success);
    EXPECT_EQ(destination.get_code_object(name, arch, sum), code);

    RTCCache::serialize_free(buffer);
}

TEST(rocfft_internal, rtc_cache_corrupt_file_is_survivable)
{
    const auto path = fs::temp_directory_path() / "rocfft_internal_test_corrupt_cache.db";
    {
        std::ofstream out(path, std::ios::binary);
        out << "this is not a sqlite database";
    }

    EnvironmentSetTemp env_path("ROCFFT_RTC_CACHE_PATH", path.string().c_str());

    std::array<char, 32> sum{};
    EXPECT_NO_THROW({
        RTCCache cache;
        EXPECT_TRUE(cache.get_code_object("internal_test_corrupt", "gfx942", sum).empty());
    });

    fs::remove(path);
}
