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
#include <iostream>
#include <vector>
#include <cmath>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/run_loop.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/op/loop.hpp>
#include <basic_ops.hpp>
#include "test.hpp"

struct copy_op
{
    std::string name() const { return "copy"; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs.front();
    }

    migraphx::argument
    compute(migraphx::context&, const migraphx::shape&, std::vector<migraphx::argument> args) const
    {
        visit_all(args[0], args[1])([&](auto input, auto output) {
            std::copy(input.begin(), input.end(), output.begin());
        });

        return args[1];
    }

    int output_alias(const std::vector<migraphx::shape>&) const { return 0; }
};

struct test_loop_op
{
    int64_t max_iterations = 10;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.max_iterations, "max_iterations"));
    }

    std::string name() const { return "test_loop_op"; }

    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs,
                                  std::vector<migraphx::module_ref> mods) const
    {
        migraphx::check_shapes{inputs, *this}.standard();
        if(mods.size() != 1)
        {
            MIGRAPHX_THROW("LOOP: operator should have one submodule.");
        }

        const auto& mod     = mods.front();
        auto mod_out_shapes = mod->get_output_shapes();
        auto dep_param_num  = inputs.size() - 2;

        // first item of the mod output shapes is condition used in loop,
        // which is not needed to compute output shape
        mod_out_shapes.erase(mod_out_shapes.begin());
        std::vector<migraphx::shape> ins_out_shapes(mod_out_shapes.begin(),
                                                    mod_out_shapes.begin() + dep_param_num);
        mod_out_shapes.erase(mod_out_shapes.begin(), mod_out_shapes.begin() + dep_param_num);
        for(const auto& out_s : mod_out_shapes)
        {
            auto lens = out_s.lens();
            lens.insert(lens.begin(), max_iterations);
            ins_out_shapes.push_back({out_s.type(), lens});
        }

        return migraphx::shape(ins_out_shapes);
    }

    struct test_loop : public migraphx::op::loop::ref_loop
    {
        test_loop(int64_t iter_num) { max_iterations = iter_num; }

        std::unordered_map<std::string, int> get_output_params(const migraphx::module& m) const
        {
            auto get_output_index = [](const std::string& name) {
                std::string out_prefix = "#output_";
                auto loc               = name.find(out_prefix);
                if(loc != std::string::npos)
                {
                    return std::stoi(name.substr(loc + out_prefix.size()));
                }

                return -1;
            };

            const auto& param_names = m.get_parameter_names();
            std::unordered_map<std::string, int> result;
            for(const auto& name : param_names)
            {
                auto index = get_output_index(name);
                if(index == -1)
                    continue;
                result[name] = index;
            }

            return result;
        }
    };

    migraphx::argument
    compute(migraphx::context& ctx,
            const migraphx::shape& out_shape,
            const std::vector<migraphx::argument>& args,
            const std::vector<migraphx::module_ref>& mods,
            const std::function<std::vector<migraphx::argument>(
                migraphx::module_ref&, const std::unordered_map<std::string, migraphx::argument>&)>&
                run) const
    {
        // wrap up the arguments vector, so ref and gpu impl are the same
        auto cpy_args = args;
        bool in_cond  = args.at(1).at<bool>();
        bool cond     = in_cond;
        int64_t iter  = 0;
        // insert iter and cond used in the loop
        auto s_cond = args.at(1).get_shape();
        auto s_iter = args.at(0).get_shape();
        cpy_args.push_back({s_iter, &iter});
        cpy_args.push_back({s_cond, &cond});
        cpy_args.insert(cpy_args.end(), args.begin() + 2, args.end());
        // add cond and mod outputs to the argument list
        cpy_args.push_back(migraphx::argument(s_cond));
        cpy_args.push_back(migraphx::argument(out_shape));
        // run loop
        return run_loop(test_loop{max_iterations}, {}, ctx, cpy_args, mods, run);
    }
};

static auto create_program(int64_t max_loop_iterations = 10)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape si{migraphx::shape::int64_type};
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::shape sc{migraphx::shape::bool_type};

    auto in_iter = mm->add_parameter("iter_num", si);
    auto in_cond = mm->add_parameter("ccond", sc);
    auto in_val  = mm->add_parameter("val", s);

    auto* body = p.create_module("loop_module");
    auto iter  = body->add_parameter("#loop_module_in_0", si);
    body->add_parameter("#loop_module_in_1", sc);
    auto in_v               = body->add_parameter("#loop_module_in_2", s);
    std::vector<int64_t> vd = {3};
    auto l                  = body->add_literal(migraphx::literal(si, vd));
    auto ad                 = body->add_instruction(migraphx::make_op("add"), iter, l);
    auto val                = body->add_instruction(migraphx::make_op("add"), in_v, ad);
    auto eq                 = body->add_instruction(migraphx::make_op("equal"), iter, l);
    auto beq                = body->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), eq);
    auto neq                     = body->add_instruction(migraphx::make_op("not"), beq);
    std::string out_param_prefix = "loop_module:#output_";
    auto out0  = body->add_parameter(out_param_prefix + std::to_string(0), neq->get_shape());
    auto r_neq = body->add_instruction(copy_op{}, neq, out0);
    auto out2  = body->add_parameter(out_param_prefix + std::to_string(2), val->get_shape());
    auto r_val = body->add_instruction(copy_op{}, val, out2);
    body->add_return({r_neq, r_val, r_val});

    auto rl =
        mm->add_instruction(test_loop_op{max_loop_iterations}, {in_iter, in_cond, in_val}, {body});
    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rl);
    auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rl);
    mm->add_return({r0, r1});

    return p;
};

static auto run_prog(migraphx::program p, int64_t iter_num, bool cond, int64_t ini_val)
{
    migraphx::shape si{migraphx::shape::int64_type};
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::shape sc{migraphx::shape::bool_type};

    p.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pp;
    pp["iter_num"] = migraphx::argument(si, &iter_num);
    pp["ccond"]    = migraphx::argument(sc, &cond);
    pp["val"]      = migraphx::argument(s, &ini_val);
    auto rets      = p.eval(pp);

    std::vector<std::vector<int64_t>> res;
    for(auto& arg : rets)
    {
        std::vector<int64_t> vec;
        arg.visit([&](auto v) { vec.assign(v.begin(), v.end()); });
        res.push_back(vec);
    }

    return res;
}

TEST_CASE(loop_test1)
{
    auto p                         = create_program();
    auto ress                      = run_prog(p, 10, true, 1);
    std::vector<int64_t> gold_last = {19};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13, 19, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test2)
{
    auto p                         = create_program(12);
    auto ress                      = run_prog(p, 4, true, 1);
    std::vector<int64_t> gold_last = {19};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13, 19, 0, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test3)
{
    auto p                         = create_program(3);
    auto ress                      = run_prog(p, 3, true, 1);
    std::vector<int64_t> gold_last = {13};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {4, 8, 13};
    EXPECT(ress.back() == gold_concat);
}

TEST_CASE(loop_test4)
{
    auto p                         = create_program(20);
    auto ress                      = run_prog(p, 5, true, 2);
    std::vector<int64_t> gold_last = {20};
    EXPECT(ress.front() == gold_last);
    std::vector<int64_t> gold_concat = {5, 9, 14, 20, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0,  0,  0, 0, 0, 0, 0, 0};
    EXPECT(ress.back() == gold_concat);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
