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

#include "solution_map.h"

#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
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

namespace fs = std::filesystem;

// Files that ship with the library, listed by the build.
#ifndef ROCFFT_TEST_SOLUTION_MAP_DIR
#define ROCFFT_TEST_SOLUTION_MAP_DIR ""
#endif

static std::vector<fs::path> shipped_solution_map_files()
{
    std::vector<fs::path> files;
    const fs::path        dir(ROCFFT_TEST_SOLUTION_MAP_DIR);
    if(dir.empty() || !fs::is_directory(dir))
        return files;
    for(const auto& entry : fs::directory_iterator(dir))
        if(entry.path().extension() == ".dat")
            files.push_back(entry.path());
    return files;
}

TEST(rocfft_internal, shipped_solution_maps_are_readable)
{
    auto files = shipped_solution_map_files();
    ASSERT_FALSE(files.empty()) << "no solution map files found in "
                                << ROCFFT_TEST_SOLUTION_MAP_DIR;

    auto& map = solution_map::get_solution_map();
    for(const auto& file : files)
    {
        // read into the scratch map so the built-in solutions stay untouched
        EXPECT_TRUE(map.read_solution_map_data(file, false)) << "could not read " << file;
    }
}

TEST(rocfft_internal, solution_map_write_then_read)
{
    auto& map = solution_map::get_solution_map();

    const auto path = fs::temp_directory_path() / "rocfft_internal_test_solution_map.dat";
    fs::remove(path);

    ASSERT_TRUE(map.write_solution_map_data(path, true, true));
    ASSERT_TRUE(fs::exists(path));
    ASSERT_GT(fs::file_size(path), 0u);

    EXPECT_TRUE(map.read_solution_map_data(path, false));

    fs::remove(path);
}

TEST(rocfft_internal, solution_map_missing_file)
{
    auto& map = solution_map::get_solution_map();

    const auto path = fs::temp_directory_path() / "rocfft_internal_test_no_such_file.dat";
    fs::remove(path);

    EXPECT_FALSE(map.read_solution_map_data(path, false));
}

TEST(rocfft_internal, solution_map_empty_file)
{
    auto& map = solution_map::get_solution_map();

    const auto path = fs::temp_directory_path() / "rocfft_internal_test_empty.dat";
    {
        std::ofstream out(path);
    }

    EXPECT_FALSE(map.read_solution_map_data(path, false));

    fs::remove(path);
}

TEST(rocfft_internal, solution_map_truncated_file)
{
    auto files = shipped_solution_map_files();
    ASSERT_FALSE(files.empty());

    std::stringstream contents;
    contents << std::ifstream(files.front()).rdbuf();
    const auto text = contents.str();
    ASSERT_GT(text.size(), 2u);

    const auto path = fs::temp_directory_path() / "rocfft_internal_test_truncated.dat";
    {
        std::ofstream out(path);
        out << text.substr(0, text.size() / 2);
    }

    auto& map = solution_map::get_solution_map();
    EXPECT_FALSE(map.read_solution_map_data(path, false));

    fs::remove(path);
}
