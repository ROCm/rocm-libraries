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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <quantize_helpers.hpp>

struct test_fp8_ocp_to_fnuz_gemm : verify_program<test_fp8_ocp_to_fnuz_gemm>
{
    using fp8e4m3fn   = migraphx::fp8::fp8e4m3fn;
    using fp8e4m3fnuz = migraphx::fp8::fp8e4m3fnuz;
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                           = p.get_main_module();
        std::vector<std::size_t> data_lens = {2, 2};
        migraphx::shape data_shape{migraphx::shape::float_type, data_lens};
        auto a     = mm->add_parameter("a", data_shape);
        auto b     = mm->add_parameter("b", data_shape);
        auto scale = mm->add_literal(0.5f);
        std::vector<fp8e4m3fn> data;
        data.push_back(fp8e4m3fn{0.f});
        auto zero =
            mm->add_literal(migraphx::shape{migraphx::shape::fp8e4m3fn_type, {1}, {0}}, data);

        auto qa = add_quantize_op(*mm, "quantizelinear", a, scale, zero);
        auto qb = add_quantize_op(*mm, "quantizelinear", b, scale, zero);
        auto da =
            add_quantize_op(*mm, "dequantizelinear", qa, qa->inputs().at(1), qa->inputs().at(2));
        auto db =
            add_quantize_op(*mm, "dequantizelinear", qb, qb->inputs().at(1), qb->inputs().at(2));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), da, db);
        mm->add_return({dot});
        return p;
    }
    std::string section() const { return "gemm"; }
};
