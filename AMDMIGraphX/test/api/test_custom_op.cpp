/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <algorithm>
#include <cmath>
#include <exception>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <stdexcept>
#include "test.hpp"

struct sigmoid_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "sigmoid_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context, migraphx::shape, migraphx::arguments inputs) const override
    {
        auto* output_ptr = reinterpret_cast<float*>(inputs[1].data());
        auto input_vec   = inputs[0].as_vector<float>();
        std::transform(input_vec.begin(), input_vec.end(), output_ptr, [](auto x) {
            return 1.f / (1.f + std::exp(-x));
        });
        return inputs[1];
    }

    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("op must have two inputs");
        }
        if(inputs[0].lengths().size() != 1)
        {
            throw std::runtime_error("input arg must be a vector or scalar");
        }
        if(inputs[0].type() != migraphx_shape_float_type)
        {
            throw std::runtime_error("input arg must be of type float");
        }
        if(inputs[0] != inputs[1])
        {
            throw std::runtime_error("input arg and buffer allocation must be of same shape");
        }
        return inputs.back();
    }
};

TEST_CASE(register_custom_op)
{
    sigmoid_custom_op sigmoid_op;
    migraphx::register_experimental_custom_op(sigmoid_op);
    auto op = migraphx::operation("sigmoid_custom_op");
    EXPECT(op.name() == "sigmoid_custom_op");
}

TEST_CASE(run_sigmoid_custom_op)
{
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {12}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto alloc         = m.add_allocation(s);
    auto custom_kernel = m.add_instruction(migraphx::operation("sigmoid_custom_op"), {x, alloc});
    p.compile(migraphx::target("ref"));
    // run program
    migraphx::program_parameters pp;
    auto param_shapes            = p.get_parameter_shapes();
    migraphx::argument input_arg = migraphx::argument::generate(param_shapes["x"]);
    pp.add("x", input_arg);
    auto results         = p.eval(pp);
    auto result          = results[0];
    auto expected_result = input_arg.as_vector<float>();
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto y) { return 1.f / (1.f + std::exp(-y)); });
    EXPECT(bool{result == migraphx::argument(s, expected_result.data())});
}

extern "C" MIGRAPHX_C_EXPORT void migraphx_test_private_disable_exception_catch(bool);

TEST_CASE(run_sigmoid_with_incorrect_shape)
{
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {12}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    migraphx_test_private_disable_exception_catch(true);
    EXPECT(test::throws<std::exception>(
        [&] { m.add_instruction(migraphx::operation("sigmoid_custom_op"), {x}); },
        "Error in compute_shape of: sigmoid_custom_op: op must have two inputs"));
}

struct identity_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "identity_custom_op"; }
    virtual migraphx::argument
    compute(migraphx::context, migraphx::shape, migraphx::arguments inputs) const override
    {
        return inputs[0];
    }

    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 1)
        {
            throw std::runtime_error("Identity op must have only one input");
        }
        return inputs.back();
    }

    virtual std::vector<size_t> output_alias(migraphx::shapes) const override { return {0, 1}; }
};

TEST_CASE(run_custom_op_with_invalid_output_alias)
{
    identity_custom_op i_op;
    migraphx::register_experimental_custom_op(i_op);
    auto op = migraphx::operation("identity_custom_op");
    EXPECT(op.name() == "identity_custom_op");

    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {12}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto i_ins         = m.add_instruction(migraphx::operation("identity_custom_op"), {x});
    migraphx_test_private_disable_exception_catch(true);
    EXPECT(test::throws<std::exception>(
        [&] { p.compile(migraphx::target("ref")); },
        "Currently, CustomOps in MIGraphX only supports one output_alias"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
