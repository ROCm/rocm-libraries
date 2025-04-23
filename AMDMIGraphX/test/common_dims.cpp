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
#include <migraphx/common_dims.hpp>
#include <test.hpp>

using axes_map = std::vector<std::vector<std::size_t>>;

TEST_CASE(common_d1_less)
{
    auto cd = migraphx::common_dims::compute({2, 32, 40, 8}, {2, 1280, 8});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1}, {2}, {3}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1, 2}, {3}});
}

static void verify_common(const migraphx::common_dims& cd)
{
    EXPECT(cd.get_dimensions_for({2, 32, 40, 8, 8}) == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.get_dimensions_for({64, 2560}) == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.get_dimensions_for({2, 32, 1}) == std::vector<std::size_t>{2, 32, 1, 1, 1});
    EXPECT(cd.get_dimensions_for({2, 1, 2560}) == std::vector<std::size_t>{2, 1, 40, 8, 8});
    EXPECT(cd.get_dimensions_for({2, 8, 2560}) == std::vector<std::size_t>{2, 8, 40, 8, 8});
    EXPECT(cd.get_dimensions_for({2, 1, 8, 8}) == std::vector<std::size_t>{2, 1, 1, 8, 8});
    EXPECT(cd.get_dimensions_for({2, 32, 8}).empty());
    EXPECT(cd.get_dimensions_for({2, 8, 8, 8}).empty());
    EXPECT(cd.get_dimensions_for({2, 1, 40, 8, 8}) == std::vector<std::size_t>{2, 1, 40, 8, 8});
    EXPECT(cd.get_dimensions_for({2, 32, 256, 8, 8}) == std::vector<std::size_t>{2, 32, 256, 8, 8});
}

TEST_CASE(common1)
{
    auto cd = migraphx::common_dims::compute({2, 32, 2560}, {2, 1280, 8, 8});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1}, {2, 3, 4}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1, 2}, {3}, {4}});
    verify_common(cd);
}

TEST_CASE(common2)
{
    auto cd = migraphx::common_dims::compute({2, 1280, 8, 8}, {2, 32, 2560});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 40, 8, 8});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1, 2}, {3}, {4}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1}, {2, 3, 4}});

    verify_common(cd);
}

TEST_CASE(common3)
{
    auto cd = migraphx::common_dims::compute({2, 32, 4096}, {4, 16, 64, 64});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 2, 16, 64, 64});
    EXPECT(cd.axes_map1 == axes_map{{0}, {1, 2}, {3, 4}});
    EXPECT(cd.axes_map2 == axes_map{{0, 1}, {2}, {3}, {4}});
}

TEST_CASE(common4)
{
    auto cd = migraphx::common_dims::compute({4, 16, 64, 64}, {2, 32, 4096});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 2, 16, 64, 64});
    EXPECT(cd.axes_map1 == axes_map{{0, 1}, {2}, {3}, {4}});
    EXPECT(cd.axes_map2 == axes_map{{0}, {1, 2}, {3, 4}});
}

TEST_CASE(common_same_dims)
{
    auto cd = migraphx::common_dims::compute({{2, 32, 4}}, {64, 2, 2});
    EXPECT(cd.dims == std::vector<std::size_t>{2, 32, 2, 2});
    EXPECT(cd.get_dimensions_for({64, 2, 2}) == std::vector<std::size_t>{2, 32, 2, 2});
    EXPECT(cd.get_dimensions_for({2, 32, 4}) == std::vector<std::size_t>{2, 32, 2, 2});
    // TODO: CHeck for similiarity
    EXPECT(cd.get_dimensions_for({2, 32, 1}).empty());
    EXPECT(cd.get_dimensions_for({64, 2, 1}).empty());
}

TEST_CASE(common_error1)
{
    auto cd = migraphx::common_dims::compute({6, 35}, {3, 7, 2, 5});
    EXPECT(cd.dims.empty());
}

TEST_CASE(common_error2)
{
    auto cd = migraphx::common_dims::compute({3, 7, 2, 5}, {6, 35});
    EXPECT(cd.dims.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
