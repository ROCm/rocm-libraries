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
#include <migraphx/op/argmax.hpp>
#include <migraphx/op/argmin.hpp>

template <class T, migraphx::shape::type_t DType, int Axis, bool LastIndex, int NonStdShape>
struct test_arg_ops : verify_program<test_arg_ops<T, DType, Axis, LastIndex, NonStdShape>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{DType, {2, 1, 4, 1025}};
        auto param = mm->add_parameter("data", s);
        switch(NonStdShape)
        {
        case 0:
            param = mm->add_instruction(
                migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), param);
            break;
        case 1:
            param = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4, 1025}}}), param);
            break;
        case 2:
            param = mm->add_instruction(
                migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {3}}}), param);
            break;
        default: break;
        }
        mm->add_instruction(T{Axis, LastIndex}, param);
        return p;
    }

    std::string section() const { return "reduce"; }
};
// transpose argmax tests
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, false, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, false, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, false, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, false, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, false, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, true, 0>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, false, 0>;
// transpose argmin tests
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, false, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, false, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, false, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, false, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, false, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, true, 0>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, false, 0>;
// broadcast argmax tests
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, false, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, false, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, false, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, false, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, false, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, true, 1>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, false, 1>;
// broadcast argmin tests
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, false, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, false, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, false, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, false, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, false, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, true, 1>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, false, 1>;
// slice argmax tests
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, false, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, false, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, false, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, false, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, false, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, true, 2>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, false, 2>;
// slice argmin tests
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, false, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, false, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, false, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, false, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, false, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, true, 2>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, false, 2>;
// default case, standard shape argmax tests
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 0, false, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 1, false, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 2, false, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, 3, false, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -1, false, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, true, 3>;
template struct test_arg_ops<migraphx::op::argmax, migraphx::shape::float_type, -2, false, 3>;
// default case, standard shape argmin tests
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 0, false, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 1, false, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 2, false, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, 3, false, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -3, false, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, true, 3>;
template struct test_arg_ops<migraphx::op::argmin, migraphx::shape::float_type, -4, false, 3>;
