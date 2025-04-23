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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>

template <migraphx::shape::type_t DType>
struct test_multinomial : verify_program<test_multinomial<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm           = p.get_main_module();
        size_t sample_size = 10;
        size_t batch_size  = 2;
        float seed         = 0.0f;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::vector<float> rand_samples(batch_size * sample_size);
        std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
        migraphx::shape rs{DType, {batch_size, sample_size}};
        auto rs_lit = mm->add_literal(migraphx::literal{rs, rand_samples});

        migraphx::shape s{DType, {batch_size, 5}};
        auto input = mm->add_parameter("input", s);

        auto maxes = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {1}}}), input);
        auto mb_maxes = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, 5}}}), maxes);
        auto cdf = mm->add_instruction(migraphx::make_op("sub"), input, mb_maxes);
        cdf      = mm->add_instruction(migraphx::make_op("exp"), cdf);
        cdf      = mm->add_instruction(
            migraphx::make_op("prefix_scan_sum", {{"axis", 1}, {"exclusive", false}}), cdf);

        mm->add_instruction(migraphx::make_op("multinomial"), cdf, rs_lit);
        return p;
    }
};

template struct test_multinomial<migraphx::shape::float_type>;
template struct test_multinomial<migraphx::shape::half_type>;
// TODO bf16 accumulates more rounding errors in exp() and prefix_scan_sum, which is slightly
// shifting the cdf values and causing off-by-one errors.
//  template struct test_multinomial<migraphx::shape::bf16_type>;
// TODO This fails, need to figure out why
//  template struct test_multinomial<migraphx::shape::fp8e4m3fnuz_type>;
//  template struct test_multinomial<migraphx::shape::fp8e5m2fnuz_type>;
//  template struct test_multinomial<migraphx::shape::fp8e4m3fn_type>;
//  template struct test_multinomial<migraphx::shape::fp8e5m2_type>;
