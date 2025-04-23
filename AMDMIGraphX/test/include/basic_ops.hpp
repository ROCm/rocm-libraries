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
#include <migraphx/program.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/shape.hpp>

struct sum_op
{
    std::string name() const { return "sum"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct sum_std_op
{
    std::string name() const { return "sum_std"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x + y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        for(auto&& input : inputs)
        {
            if(not input.standard())
                MIGRAPHX_THROW("Not standard shape");
        }
        if(inputs.at(0) != inputs.at(1))
        {
            MIGRAPHX_THROW("Invalid shapes");
        }
        return inputs.front();
    }
};

struct minus_op
{
    std::string name() const { return "minus"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        migraphx::argument result;
        if(args.size() != 2)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape() != args[1].get_shape())
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().size() != 1)
            MIGRAPHX_THROW("Wrong args");
        if(args[0].get_shape().lens().front() != 1)
            MIGRAPHX_THROW("Wrong args");

        args[0].visit_at([&](auto x) {
            args[1].visit_at([&](auto y) { result = migraphx::literal{x - y}.get_argument(); });
        });
        return result;
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
};

struct pass_op
{
    std::string name() const { return "pass"; }
    migraphx::argument compute(const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>& s) const { return s.empty() ? -1 : 0; }
};

struct non_const_pass_op
{
    std::string name() const { return "pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>& s) const { return s.empty() ? -1 : 0; }
};

struct mod_pass_op
{
    std::string name() const { return "mod_pass"; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs,
                                  std::vector<migraphx::module_ref> mods) const
    {
        if(not mods.empty())
        {
            auto out_shapes = mods[0]->get_output_shapes();
            return out_shapes[0];
        }
        if(not inputs.empty())
        {
            return inputs.front();
        }

        return {};
    }

    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct unary_pass_op
{
    std::string name() const { return "unary_pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        if(inputs.size() != 1)
            MIGRAPHX_THROW("Wrong inputs");
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct pass_standard_op
{
    std::string name() const { return "pass"; }
    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        if(args.empty())
            return {};
        return args.front();
    }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        for(auto&& input : inputs)
        {
            if(not input.standard())
                throw std::runtime_error("Not standard shape");
        }
        if(inputs.empty())
            return {};
        return inputs.front();
    }
    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct nop
{
    std::string name() const { return "nop"; }
    migraphx::argument compute(const migraphx::shape&, const std::vector<migraphx::argument>&) const
    {
        return {};
    }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const { return {}; }
};

struct tuple_op
{
    std::string name() const { return "tuple_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        return migraphx::shape(inputs);
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>& input_args) const
    {
        return input_args;
    }
};

inline migraphx::literal get_2x2(int base = 0)
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 2}},
                             {base + 1, base + 2, base + 3, base + 4}};
}

inline migraphx::literal get_2x2_transposed()
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 2}, {1, 2}}, {1, 2, 3, 4}};
}

inline migraphx::literal get_2()
{
    return migraphx::literal{{migraphx::shape::float_type, {2}}, {1, 2}};
}

inline migraphx::literal get_2_broadcasted()
{
    return migraphx::literal{{migraphx::shape::float_type, {2, 1}, {1, 0}}, {1, 2}};
}
