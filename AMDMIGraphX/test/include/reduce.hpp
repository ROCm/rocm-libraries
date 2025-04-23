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
#ifndef MIGRAPHX_GUARD_TEST_INCLUDE_REDUCE_HPP
#define MIGRAPHX_GUARD_TEST_INCLUDE_REDUCE_HPP

#include <test.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <utility>

inline bool all_instructions_are_local(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [&](const auto& ins) {
        return std::all_of(ins.inputs().begin(), ins.inputs().end(), [&](auto input) {
            return m.has_instruction(input);
        });
    });
}

inline void auto_add_return(migraphx::module_ref m, migraphx::instruction_ref ins)
{
    m->add_return({ins});
}

inline void auto_add_return(migraphx::module_ref m, std::vector<migraphx::instruction_ref> inss)
{
    m->add_return(std::move(inss));
}

template <class F>
migraphx::module_ref add_reduce_module(migraphx::program& p,
                                       const std::string& name,
                                       std::vector<migraphx::instruction_ref> inputs,
                                       const std::vector<int64_t>& axes,
                                       const F& f)
{
    auto* rm = p.create_module(name);
    rm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(params), [&](auto input) {
        return rm->add_parameter(
            "x" + std::to_string(params.size()),
            migraphx::shape{input->get_shape().type(), input->get_shape().lens()});
    });
    auto r = f(rm, params, axes);
    auto_add_return(rm, r);
    EXPECT(all_instructions_are_local(*rm));
    return rm;
}

template <class F>
migraphx::instruction_ref add_reduce(migraphx::program& p,
                                     const std::string& name,
                                     const std::vector<migraphx::instruction_ref>& inputs,
                                     const std::vector<int64_t>& axes,
                                     const F& f)
{
    auto* mm = p.get_main_module();
    auto rm  = add_reduce_module(p, name, inputs, axes, std::move(f));
    return mm->add_instruction(migraphx::make_op("fused_reduce", {{"axes", axes}}), inputs, {rm});
}

template <class F>
migraphx::instruction_ref add_reduce(migraphx::program& p,
                                     const std::string& name,
                                     const std::vector<migraphx::instruction_ref>& inputs,
                                     const std::vector<int64_t>& axes,
                                     const std::string& assign,
                                     const F& f)
{
    auto* mm = p.get_main_module();
    auto rm  = add_reduce_module(p, name, inputs, axes, std::move(f));
    return mm->add_instruction(
        migraphx::make_op("split_fused_reduce", {{"axes", axes}, {"assign", assign}}),
        inputs,
        {rm});
}

inline auto squared()
{
    return [](auto* pm, const auto& inputs) {
        return pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[0]);
    };
}

inline auto single_reduce(const std::string& name)
{
    return [=](auto* rm, const auto& inputs, const auto& axes) {
        return rm->add_instruction(migraphx::make_op(name, {{"axes", axes}}), inputs);
    };
}
#endif // MIGRAPHX_GUARD_TEST_INCLUDE_REDUCE_HPP
