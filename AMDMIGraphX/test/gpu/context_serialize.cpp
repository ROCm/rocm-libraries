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
#include <iostream>
#include <vector>
#include <migraphx/verify.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include "test.hpp"

TEST_CASE(gpu_context_serialize)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};

    auto v = ctx.to_value();
    EXPECT(v.size() == 2);

    EXPECT(v.contains("events"));
    EXPECT(v.at("events").without_key().to<std::size_t>() == 0);

    EXPECT(v.contains("streams"));
    EXPECT(v.at("streams").without_key().to<std::size_t>() == 3);

    migraphx::gpu::context g_ctx;
    g_ctx.from_value(v);

    auto v1 = g_ctx.to_value();
    EXPECT(v == v1);
}

TEST_CASE(context_queue)
{
    migraphx::context ctx = migraphx::gpu::context{0, 3};
    EXPECT(ctx.get_queue().get<hipStream_t>() != nullptr);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
