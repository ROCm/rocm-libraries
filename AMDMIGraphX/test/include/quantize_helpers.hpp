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
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

#ifndef MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP
#define MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP

inline migraphx::instruction_ref broadcast_scale(migraphx::module& m,
                                                 migraphx::instruction_ref scale,
                                                 const std::vector<std::size_t>& out_lens,
                                                 std::size_t axis)
{
    if(scale->get_shape().lens() == out_lens)
        return scale;

    migraphx::instruction_ref scale_mb;
    auto scale_lens = scale->get_shape().lens();
    if(scale_lens.front() == 1 and scale_lens.size() == 1)
        scale_mb =
            m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), scale);
    else
        scale_mb = m.add_instruction(
            migraphx::make_op("broadcast", {{"axis", axis}, {"out_lens", out_lens}}), scale);
    return scale_mb;
}

inline migraphx::instruction_ref broadcast_shift(migraphx::module& m,
                                                 migraphx::instruction_ref shift,
                                                 const std::vector<std::size_t>& out_lens)
{
    if(shift->get_shape().lens() == out_lens)
        return shift;
    return m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", out_lens}}), shift);
}

inline migraphx::instruction_ref add_scale_mul(migraphx::module& m,
                                               migraphx::instruction_ref scale1,
                                               migraphx::instruction_ref scale2,
                                               std::size_t axis1,
                                               std::size_t axis2,
                                               const std::vector<std::size_t>& out_lens)
{
    auto scale1_mb = broadcast_scale(m, scale1, out_lens, axis1);
    auto scale2_mb = broadcast_scale(m, scale2, out_lens, axis2);
    return m.add_instruction(migraphx::make_op("mul"), scale1_mb, scale2_mb);
}

inline migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                                 const std::string& name,
                                                 migraphx::instruction_ref x,
                                                 migraphx::instruction_ref scale,
                                                 migraphx::instruction_ref shift,
                                                 std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    auto shift_mb = broadcast_shift(m, shift, lens);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb, shift_mb);
}

inline migraphx::instruction_ref add_quantize_op(migraphx::module& m,
                                                 const std::string& name,
                                                 migraphx::instruction_ref x,
                                                 migraphx::instruction_ref scale,
                                                 std::size_t q_axis = 1)
{
    auto lens     = x->get_shape().lens();
    auto scale_mb = broadcast_scale(m, scale, lens, q_axis);
    return m.add_instruction(migraphx::make_op(name), x, scale_mb);
}

#endif // MIGRAPHX_GUARD_TEST_INCLUDE_QUANTIZE_HELPERS_HPP
