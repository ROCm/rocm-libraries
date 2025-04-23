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
#include <migraphx/gpu/allocation_model.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/adjust_allocation.hpp>
#include <migraphx/auto_contiguous.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/replace_allocate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/tanh.hpp>
#include <migraphx/op/transpose.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/make_op.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include "make_precompile_op.hpp"

// Treat some operators as compilable to enable lowering
MIGRAPHX_GPU_TEST_PRECOMPILE("add", "mul", "convert")

static void run_lowering(migraphx::program& p, bool offload_copy = false)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(
        *p.get_main_module(),
        {migraphx::auto_contiguous{},
         migraphx::gpu::lowering{&ctx, offload_copy},
         migraphx::dead_code_elimination{},
         migraphx::eliminate_contiguous{"gpu::contiguous"},
         migraphx::dead_code_elimination{},
         migraphx::replace_allocate{migraphx::gpu::gpu_allocation_model{}, offload_copy},
         migraphx::dead_code_elimination{}});
}

TEST_CASE(tanh_shape)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x   = mm->add_parameter("x", s);
        auto tx  = mm->add_instruction(migraphx::op::transpose{{1, 0}}, x);
        auto txh = mm->add_instruction(migraphx::op::tanh{}, tx);
        auto sum = mm->add_instruction(migraphx::op::add{}, txh, txh);
        mm->add_instruction(migraphx::op::contiguous{}, sum);

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_program();
    EXPECT(p1 == p2);

    run_lowering(p1);
    run_lowering(p2);

    EXPECT(p1 == p2);

    for(auto ins : iterator_for(*p1.get_main_module()))
    {
        if(ins->name() == "hip::allocate")
        {
            migraphx::shape new_s{migraphx::shape::float_type, {3, 2}, {1, 3}};
            ins->replace(migraphx::gpu::hip_allocate{new_s});
        }
    }
    EXPECT(p1 != p2);

    migraphx::run_passes(*p2.get_main_module(),
                         {migraphx::adjust_allocation{migraphx::gpu::gpu_allocation_model{}},
                          migraphx::dead_code_elimination{}});
    EXPECT(p1 == p2);
}

TEST_CASE(no_copy_dead_param)
{
    auto create_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_parameter("y", s);
        auto sum = mm->add_instruction(migraphx::make_op("add"), x, x);
        mm->add_return({sum});

        return p;
    };

    auto create_gpu_program = [] {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", s);
        mm->add_parameter("y", s);
        auto xb = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto gx = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), x, xb);
        auto ab = mm->add_instruction(migraphx::make_op("hip::allocate", {{"shape", to_value(s)}}));
        auto sum = mm->add_instruction(make_precompile_op("add"), gx, gx, ab);
        auto r   = mm->add_instruction(migraphx::make_op("hip::copy_from_gpu"), sum);
        mm->add_return({r});

        return p;
    };

    auto p1 = create_program();
    auto p2 = create_gpu_program();

    run_lowering(p1, true);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
