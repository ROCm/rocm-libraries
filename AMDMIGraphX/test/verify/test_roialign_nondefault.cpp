/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/op/common.hpp>

struct test_roialign_nondefault : verify_program<test_roialign_nondefault>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape x_s{migraphx::shape::float_type, {5, 4, 10, 10}};

        migraphx::shape roi_s{migraphx::shape::float_type, {5, 4}};

        migraphx::shape ind_s{migraphx::shape::int64_type, {5}};
        std::vector<int64_t> ind_vec = {0, 2, 3, 4, 1};

        auto x   = mm->add_parameter("x", x_s);
        auto roi = mm->add_parameter("roi", roi_s);
        auto ind = mm->add_literal(migraphx::literal(ind_s, ind_vec));
        auto r   = mm->add_instruction(
            migraphx::make_op("roialign",
                              {{"coordinate_transformation_mode", "output_half_pixel"},
                               {"mode", migraphx::op::pooling_mode::max},
                               {"spatial_scale", 1.0},
                               {"output_height", 5},
                               {"output_width", 5},
                               {"sampling_ratio", 2}}),
            x,
            roi,
            ind);
        mm->add_return({r});

        return p;
    }
};
