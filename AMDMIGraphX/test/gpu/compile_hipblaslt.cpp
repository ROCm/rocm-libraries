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

#include <migraphx/gpu/compile_hipblaslt.hpp>
#include <migraphx/gpu/hip_gemm.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/op/allocate.hpp>
#include <migraphx/register_op.hpp>
#include <test.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)

#if MIGRAPHX_USE_HIPBLASLT
static void run_lowering(migraphx::module& m, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m, {migraphx::gpu::lowering{&ctx, offload_copy}});
}

TEST_CASE(hipblaslt_op)
{
    if(not(migraphx::string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "rocblas") and
       migraphx::gpu::hipblaslt_supported() and not migraphx::gpu::gfx_default_rocblas())
    {
        migraphx::module m1;
        {
            migraphx::shape sa{migraphx::shape::float_type, {4, 2}};
            migraphx::shape sb{migraphx::shape::float_type, {2, 3}};
            migraphx::shape s_output{migraphx::shape::float_type, {4, 3}};
            auto a                     = m1.add_parameter("a", sa);
            auto b                     = m1.add_parameter("b", sb);
            migraphx::operation dot_op = migraphx::make_op("dot");
            m1.add_instruction(dot_op, a, b);
        }

        run_lowering(m1);
        migraphx::module m2;
        {
            auto a = m2.add_parameter("a", {migraphx::shape::float_type, {4, 2}});
            auto b = m2.add_parameter("b", {migraphx::shape::float_type, {2, 3}});

            migraphx::shape output_shape{migraphx::shape::float_type, {4, 3}, {3, 1}};

            // Add an allocate instruction for the output
            auto output = m2.add_instruction(migraphx::op::allocate{output_shape, std::nullopt});

            migraphx::op::dot dot_instance;
            migraphx::gpu::hipblaslt_op hipblaslt_operator;
            hipblaslt_operator.op = migraphx::gpu::hip_gemm<migraphx::op::dot>{dot_instance, 1, 0};
            m2.add_instruction(hipblaslt_operator, a, b, output);
        }
        EXPECT(m1 == m2);
    }
}
#endif

int main(int argc, const char* argv[]) { test::run(argc, argv); }
