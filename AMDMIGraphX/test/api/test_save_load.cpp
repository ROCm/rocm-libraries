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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_save_default)
{
    std::string filename = "migraphx_api_load_save.mxr";
    auto p1              = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto s1              = p1.get_output_shapes();
    migraphx::save(p1, filename.c_str());
    auto p2 = migraphx::load(filename.c_str());
    auto s2 = p2.get_output_shapes();
    EXPECT(s1.size() == s2.size());
    EXPECT(bool{s1.front() == s2.front()});
    EXPECT(bool{p1.sort() == p2.sort()});
    std::remove(filename.c_str());
}

TEST_CASE(load_save_json)
{
    std::string filename = "migraphx_api_load_save.json";
    auto p1              = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto s1              = p1.get_output_shapes();
    migraphx::file_options options;
    options.set_file_format("json");

    migraphx::save(p1, filename.c_str(), options);
    auto p2 = migraphx::load(filename.c_str(), options);
    auto s2 = p2.get_output_shapes();
    EXPECT(s1.size() == s2.size());
    EXPECT(bool{s1.front() == s2.front()});
    EXPECT(bool{p1.sort() == p2.sort()});
    std::remove(filename.c_str());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
