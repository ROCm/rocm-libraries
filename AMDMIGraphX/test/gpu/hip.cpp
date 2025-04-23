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

#include <test.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/target.hpp>

TEST_CASE(tuple_from_gpu)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 4}};
    std::vector<float> p1_data = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    std::vector<int> p2_data   = {1, 2, 3, 4, 5, 6, 7, 8};
    auto p1                    = migraphx::argument{s1, p1_data.data()};
    auto p2                    = migraphx::argument{s2, p2_data.data()};
    auto p1_gpu                = migraphx::gpu::to_gpu(p1);
    auto p2_gpu                = migraphx::gpu::to_gpu(p2);
    auto p_tuple               = migraphx::gpu::from_gpu(migraphx::argument({p1_gpu, p2_gpu}));
    std::vector<migraphx::argument> results = p_tuple.get_sub_objects();
    std::vector<float> result1;
    results[0].visit([&](auto output) { result1.assign(output.begin(), output.end()); });
    std::vector<int> result2;
    results[1].visit([&](auto output) { result2.assign(output.begin(), output.end()); });
    EXPECT(result1 == p1_data);
    EXPECT(result2 == p2_data);
}

TEST_CASE(tuple_to_gpu)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 3}};
    migraphx::shape s2{migraphx::shape::int32_type, {2, 4}};
    std::vector<float> p1_data              = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    std::vector<int> p2_data                = {1, 2, 3, 4, 5, 6, 7, 8};
    auto p1                                 = migraphx::argument{s1, p1_data.data()};
    auto p2                                 = migraphx::argument{s2, p2_data.data()};
    auto p_gpu                              = migraphx::gpu::to_gpu(migraphx::argument({p1, p2}));
    auto p_host                             = migraphx::gpu::from_gpu(p_gpu);
    std::vector<migraphx::argument> results = p_host.get_sub_objects();
    std::vector<float> result1;
    results[0].visit([&](auto output) { result1.assign(output.begin(), output.end()); });
    std::vector<int> result2;
    results[1].visit([&](auto output) { result2.assign(output.begin(), output.end()); });
    EXPECT(result1 == p1_data);
    EXPECT(result2 == p2_data);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
