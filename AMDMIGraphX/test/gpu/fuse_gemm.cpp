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

#include <basic_ops.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/compile_hipblaslt.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/gpu/gemm.hpp>
#include <migraphx/gpu/hip_gemm.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/allocate.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <pointwise.hpp>
#include <test.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)

#if MIGRAPHX_USE_HIPBLASLT
static void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        *p.get_main_module(),
        {migraphx::auto_contiguous{}, migraphx::gpu::lowering{&ctx, offload_copy}});
}

static void run_fuse_ops(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(gemm_pointwise_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add = add_pointwise(p1, "main:pointwise0", {dot, x}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_lowering(p1);
    run_fuse_ops(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);

        auto output = mm->add_instruction(migraphx::op::allocate{s, std::nullopt});

        if(not(migraphx::string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "rocblas") and
           migraphx::gpu::hipblaslt_supported() and not migraphx::gpu::gfx_default_rocblas())
        {
            migraphx::op::dot dot_instance;
            migraphx::gpu::hipblaslt_op hipblaslt_operator;
            hipblaslt_operator.op = migraphx::gpu::hip_gemm<migraphx::op::dot>{dot_instance, 1, 1};
            auto add              = mm->add_instruction(hipblaslt_operator, a, b, x, output);
            mm->add_return({add});
        }
        else
        {
            auto gemm_oper =
                migraphx::make_op("gpu::gemm",
                                  {{"alpha", 1},
                                   {"beta", 1},
                                   {"compute_fp32", migraphx::gpu::get_compute_fp32_flag()}});
            auto add = mm->add_instruction(gemm_oper, a, b, x, output);
            mm->add_return({add});
        }
    }
    EXPECT(p1.sort() == p2.sort());
}
#endif

int main(int argc, const char* argv[]) { test::run(argc, argv); }
