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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(load_tf)
{
    auto p      = migraphx::parse_tf("models/add_test.pb");
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
}

TEST_CASE(load_tf_default_dim)
{
    migraphx::tf_options tf_options;
    size_t batch = 2;
    tf_options.set_default_dim_value(batch);
    tf_options.set_nhwc();
    auto p      = migraphx::parse_tf("models/conv_batch_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
    CHECK(shapes.front().lengths().front() == batch);
}

TEST_CASE(load_tf_param_shape)
{
    migraphx::tf_options tf_options;
    std::vector<size_t> new_shape{1, 3};
    tf_options.set_input_parameter_shape("0", new_shape);
    tf_options.set_input_parameter_shape("1", new_shape);
    auto p      = migraphx::parse_tf("models/add_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 1);
    CHECK(shapes.front().lengths() == new_shape);
}

TEST_CASE(load_tf_multi_outputs)
{
    migraphx::tf_options tf_options;
    tf_options.set_output_names({"relu", "tanh"});
    auto p      = migraphx::parse_tf("models/multi_output_test.pb", tf_options);
    auto shapes = p.get_output_shapes();
    CHECK(shapes.size() == 2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
