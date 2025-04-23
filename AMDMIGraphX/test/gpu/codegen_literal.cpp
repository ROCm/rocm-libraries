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
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>
#include <migraphx/make_op.hpp>

static void run_prog(migraphx::program p,
                     const migraphx::target& t,
                     migraphx::parameter_map& m_in,
                     std::vector<float>& res)
{
    p.compile(t);
    migraphx::parameter_map m;
    for(auto&& x : p.get_parameter_shapes())
    {
        if(m_in.count(x.first) > 0)
        {
            m[x.first] = t.copy_to(m_in[x.first]);
        }
        else
        {
            m[x.first] = t.allocate(x.second);
        }
    }

    auto result = t.copy_from(p.eval(m).back());
    result.visit([&](auto v) { res.assign(v.begin(), v.end()); });
}

// This test ensures that the codegen path doesn't round up literals,
// otherwise there are accuracy differences compared to ref.
// The values being passed in are 0.5 * (1/0.00787402),
// and after rounding must equal 63, not 64.
TEST_CASE(mul_literal_round_test)
{

    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s0{migraphx::shape::float_type, {1}};
    auto l0 = mm->add_parameter("a", s0);
    auto l1 = mm->add_literal(1 / 0.00787402f);

    auto mul   = mm->add_instruction(migraphx::make_op("mul"), l0, l1);
    auto round = mm->add_instruction(migraphx::make_op("nearbyint"), mul);

    mm->add_return({round});

    migraphx::parameter_map m;
    std::vector<float> a = {0.5f};

    m["a"] = migraphx::argument{s0, a.data()};
    std::vector<float> ref_result;
    migraphx::target ref_t = migraphx::make_target("ref");
    run_prog(p, ref_t, m, ref_result);

    std::vector<float> gpu_result;
    migraphx::target gpu_t = migraphx::make_target("gpu");
    run_prog(p, gpu_t, m, gpu_result);

    EXPECT(migraphx::verify::verify_rms_range(gpu_result, ref_result));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
