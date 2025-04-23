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
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>
#include <sstream>
#include <migraphx/make_op.hpp>

#include <basic_ops.hpp>
#include <pointwise.hpp>
#include <test.hpp>

static migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(sum_op{}, x, y);
    auto one = mm->add_literal(1);
    mm->add_instruction(sum_op{}, sum, one);

    return p;
}

TEST_CASE(calc_implict_deps)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
    migraphx::shape ys{migraphx::shape::float_type, {3, 3}};
    std::vector<float> datax = {1, 2, 3, 4, 5, 6};
    std::vector<float> datay = {8, 7, 6, 5, 4, 3, 2, 1, 0};

    auto lx   = mm->add_literal(migraphx::literal(xs, datax));
    auto ly   = mm->add_literal(migraphx::literal(ys, datay));
    auto cond = mm->add_parameter("cond", cond_s);
    auto x1   = mm->add_parameter("x1", xs);
    auto x2   = mm->add_parameter("x2", xs);
    auto y2   = mm->add_parameter("y2", ys);

    auto* then_mod = p.create_module("If_5_if");
    auto l1        = then_mod->add_literal(migraphx::literal(ys, datay));
    auto a1        = then_mod->add_instruction(migraphx::make_op("add"), x1, lx);
    then_mod->add_return({a1, l1});

    auto* then_mod1 = p.create_module("If_6_if");
    auto l11        = then_mod1->add_literal(migraphx::literal(ys, datay));
    auto a11        = then_mod1->add_instruction(migraphx::make_op("add"), x2, lx);
    then_mod1->add_return({a11, l11});

    auto* else_mod1 = p.create_module("If_6_else");
    auto l21        = else_mod1->add_literal(migraphx::literal(xs, datax));
    auto a21        = else_mod1->add_instruction(migraphx::make_op("mul"), y2, ly);
    else_mod1->add_return({l21, a21});

    auto* else_mod = p.create_module("If_5_else");
    auto l2        = else_mod->add_literal(migraphx::literal(ys, datay));
    auto a2 = else_mod->add_instruction(migraphx::make_op("if"), {cond}, {then_mod1, else_mod1});
    auto a3 = else_mod->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), a2);
    else_mod->add_return({a3, l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    auto implicit_deps = mm->calc_implicit_deps();
    EXPECT(migraphx::contains(implicit_deps, ret));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x1));
    EXPECT(migraphx::contains(implicit_deps.at(ret), x2));
    EXPECT(migraphx::contains(implicit_deps.at(ret), y2));
    EXPECT(migraphx::contains(implicit_deps.at(ret), lx));
    EXPECT(migraphx::contains(implicit_deps.at(ret), ly));
    // test for sorting
    p.sort();
    auto ret_inputs = ret->inputs();
    ret_inputs.insert(ret_inputs.end(), implicit_deps.at(ret).begin(), implicit_deps.at(ret).end());
    EXPECT(std::all_of(ret_inputs.begin(), ret_inputs.end(), [&](const auto i) {
        return std::distance(mm->begin(), i) < std::distance(mm->begin(), ret);
    }));
}

TEST_CASE(module_annotate)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();
    EXPECT(*mm1 == *mm2);

    std::stringstream ss1;
    mm1->annotate(ss1, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    std::stringstream ss2;
    mm2->annotate(ss2, [](auto ins) { std::cout << ins->name() << "_1" << std::endl; });

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(module_ins_clear)
{
    migraphx::program p1 = create_program();
    migraphx::program p2;

    p2 = p1;

    EXPECT(p1 == p2);
}

TEST_CASE(module_name)
{
    migraphx::module m1("name");
    EXPECT(m1.name() == "name");

    auto m2 = m1; // NOLINT
    EXPECT(m2.name() == "name");
    migraphx::module m3;
    m3 = m1;
    EXPECT(m3.name() == "name");
}

TEST_CASE(module_name_main)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    EXPECT(mm->name() == "main");
}

TEST_CASE(module_print_cpp)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->print_cpp(ss1);

    std::stringstream ss2;
    mm2->print_cpp(ss2);

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(module_print_graph)
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->print_graph(ss1, true);

    std::stringstream ss2;
    mm2->print_graph(ss2, true);

    EXPECT(ss1.str() == ss2.str());
}

TEST_CASE(program_module_assign)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});

    auto* else_smod = p.create_module("else_smod");
    auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});

    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);
    auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
    mm->add_return({ret});

    migraphx::program p1 = p;

    EXPECT(p == p1);
}

TEST_CASE(program_module_replace)
{
    auto create_program = [](bool use_if) {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
        auto x = mm->add_parameter("x", sd);

        std::vector<float> one(sd.elements(), 1);
        std::vector<float> two(sd.elements(), 2);

        auto* then_smod = p.create_module("then_smod");
        auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
        auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
        then_smod->add_return({r1});

        auto* else_smod = p.create_module("else_smod");
        auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
        auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
        else_smod->add_return({r2});

        migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
        auto cond = mm->add_parameter("cond", s_cond);

        migraphx::instruction_ref ret{};

        if(use_if)
        {
            ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
        }
        else
        {
            ret = mm->add_instruction(mod_pass_op{}, {cond}, {then_smod, else_smod});
        }

        mm->add_return({ret});

        return p;
    };

    migraphx::program p1 = create_program(false);
    migraphx::program p2 = create_program(true);
    EXPECT(p1 != p2);

    auto* m1               = p1.get_main_module();
    auto ins_pass          = std::prev(std::prev(m1->end()));
    const auto& inputs     = ins_pass->inputs();
    const auto& mod_inputs = ins_pass->module_inputs();
    m1->replace_instruction(ins_pass, migraphx::make_op("if"), inputs, mod_inputs);

    EXPECT(p1 == p2);
}

TEST_CASE(submodule_copy)
{
    migraphx::module mm("main");
    auto x = mm.add_parameter("x", {migraphx::shape::int64_type});

    migraphx::module sm("sub");
    sm.add_instruction(migraphx::make_op("sin"), x);

    mm.add_instruction(migraphx::make_op("if"), {x}, {&sm, &sm});

    auto mm2 = mm;

    EXPECT(mm == mm2);
    EXPECT(mm.get_sub_modules() == mm2.get_sub_modules());
}

TEST_CASE(parameter_name_order)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module mm("main");
    auto x1 = mm.add_parameter("x1", s);
    auto x2 = mm.add_parameter("x2", s);
    auto x3 = mm.add_parameter("x3", s);
    auto x4 = mm.add_parameter("x4", s);

    std::vector<std::string> param_names = {"x1", "x2", "x3", "x4"};
    auto sum1                            = mm.add_instruction(migraphx::make_op("add"), x1, x2);
    auto sum2                            = mm.add_instruction(migraphx::make_op("add"), x3, x4);
    auto r                               = mm.add_instruction(migraphx::make_op("mul"), sum1, sum2);
    mm.add_return({r});

    auto names = mm.get_parameter_names();
    EXPECT(param_names == names);

    auto m1     = mm;
    auto names1 = m1.get_parameter_names();
    EXPECT(param_names == names1);
}

struct map_ins
{
    using type = std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref>;
    map_ins(std::initializer_list<type::value_type> x) : m(x) {}

    operator type*() { return &m; }

    type m;
};

TEST_CASE(insert_instructions_module)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1   = m1.add_parameter("x1", s);
    auto sqrt = m1.add_instruction(migraphx::make_op("sqrt"), {x1});
    m1.add_instruction(migraphx::make_op("add"), {sqrt, x1});

    migraphx::module m2("m2");
    auto x2 = m2.add_parameter("x2", s);
    m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.insert_instructions(sqrt, &m2, map_ins{{x2, x1}});

    EXPECT(std::prev(sqrt)->name() == "sqrt");
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_module)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2 = m2.add_parameter("x2", s);
    m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions(&m2, map_ins{{x2, x1}});

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_range)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2    = m2.add_parameter("x2", s);
    auto sqrt2 = m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions(sqrt2, m2.end(), map_ins{{x2, x1}});
    EXPECT(std::any_of(
        m1.begin(), m1.end(), [&](auto&& ins) { return migraphx::contains(ins.inputs(), x1); }));

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

TEST_CASE(add_instructions_vector)
{
    migraphx::shape s{migraphx::shape::int32_type, {1}};
    migraphx::module m1("m1");
    auto x1 = m1.add_parameter("x1", s);
    m1.add_instruction(migraphx::make_op("sqrt"), {x1});

    migraphx::module m2("m2");
    auto x2    = m2.add_parameter("x2", s);
    auto sqrt2 = m2.add_instruction(migraphx::make_op("sqrt"), {x2});

    m1.add_instructions({sqrt2}, map_ins{{x2, x1}});
    EXPECT(std::any_of(
        m1.begin(), m1.end(), [&](auto&& ins) { return migraphx::contains(ins.inputs(), x1); }));

    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "sqrt"; }) ==
           2);
    EXPECT(std::count_if(m1.begin(), m1.end(), [](auto&& ins) { return ins.name() == "@param"; }) ==
           1);
    EXPECT(contains(m1.get_parameter_shapes(), "x1"));
    EXPECT(not contains(m1.get_parameter_shapes(), "x2"));
}

struct check_for_pass_op
{
    bool* found = nullptr;
    std::string name() const { return "check_for_pass_op"; }
    void apply(migraphx::module& m) const
    {
        *found |= std::any_of(m.begin(), m.end(), [](auto&& ins) { return ins.name() == "pass"; });
    }
};

TEST_CASE(module_bypass)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    sub->set_bypass();
    sub->add_instruction(pass_op{});
    mm->add_instruction(mod_pass_op{}, {}, {sub});
    bool found = false;
    migraphx::run_passes(p, {check_for_pass_op{&found}});
    EXPECT(not found);
}

TEST_CASE(module_without_bypass)
{
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    sub->add_instruction(pass_op{});
    mm->add_instruction(mod_pass_op{}, {}, {sub});
    bool found = false;
    migraphx::run_passes(p, {check_for_pass_op{&found}});
    EXPECT(found);
}

TEST_CASE(multiple_module_dependency)
{
    // Test when an instruction from a submodule depends on previous module
    migraphx::program p;
    auto* mm  = p.get_main_module();
    auto* sub = p.create_module("sub");
    auto l1   = mm->add_literal(migraphx::literal(3));
    // second same literal to make sure instruction_ref is being compared, rather than the
    // instructions
    sub->add_literal(migraphx::literal(3));
    sub->add_instruction(sum_op{}, l1, l1);
    EXPECT((sub->validate() == sub->end()));
}

TEST_CASE(module_split2)
{
    migraphx::shape s{migraphx::shape::float_type, {1}};
    migraphx::module input_m;
    std::vector<migraphx::instruction_ref> inputs;
    {
        auto x1  = input_m.add_parameter("x1", s);
        auto x2  = input_m.add_parameter("x2", s);
        auto x3  = input_m.add_parameter("x3", s);
        auto sx1 = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2 = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        auto sx3 = input_m.add_instruction(migraphx::make_op("sqrt"), x3);
        inputs   = {sx1, sx2, sx3};
    }
    migraphx::module m;
    std::vector<migraphx::instruction_ref> splits;
    {
        auto x1  = m.add_parameter("x1", s);
        auto x2  = m.add_parameter("x2", s);
        auto x3  = m.add_parameter("x3", s);
        auto add = m.add_instruction(migraphx::make_op("add"), x1, x2);
        auto mul = m.add_instruction(migraphx::make_op("mul"), add, x3);
        m.add_return({mul});
        splits.push_back(add);
    }
    auto mods = m.split(inputs, splits);

    migraphx::module m1;
    {
        auto x1  = m1.add_parameter("x1", s);
        auto x2  = m1.add_parameter("x2", s);
        auto add = m1.add_instruction(migraphx::make_op("add"), x1, x2);
        m1.add_return({add});
    }
    migraphx::module m2;
    {
        auto x0  = m2.add_parameter("x0", s);
        auto x1  = m2.add_parameter("x1", s);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), x0, x1);
        m2.add_return({mul});
    }
    EXPECT(mods[0].mod.sort() == m1.sort());
    EXPECT(mods[1].mod.sort() == m2.sort());

    EXPECT(bool{mods[0].inputs[0] == inputs[0]});
    EXPECT(bool{mods[0].inputs[1] == inputs[1]});

    EXPECT(bool{mods[1].inputs[0] == splits.front()});
    EXPECT(bool{mods[1].inputs[1] == inputs[2]});
}

TEST_CASE(module_split_2_dot_ins)
{
    std::vector<migraphx::instruction_ref> inputs;
    std::vector<migraphx::instruction_ref> mod_0_expected_inputs;
    std::vector<migraphx::instruction_ref> mod_1_expected_inputs;
    migraphx::shape s1 = migraphx::shape{migraphx::shape::float_type, {2, 5}};
    migraphx::shape s2 = migraphx::shape{migraphx::shape::float_type, {5, 5}};
    migraphx::module input_m;
    {
        auto x1               = input_m.add_parameter("x1", s1);
        auto x2               = input_m.add_parameter("x2", s1);
        auto x3               = input_m.add_parameter("x3", s1);
        auto x4               = input_m.add_parameter("x4", s1);
        auto y0               = input_m.add_parameter("y0", s1);
        auto y1               = input_m.add_parameter("y1", s2);
        auto sx1              = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2              = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        auto sx3              = input_m.add_instruction(migraphx::make_op("sqrt"), x3);
        auto sx4              = input_m.add_instruction(migraphx::make_op("sqrt"), x4);
        auto sy0              = input_m.add_instruction(migraphx::make_op("sqrt"), y0);
        auto sy1              = input_m.add_instruction(migraphx::make_op("sqrt"), y1);
        inputs                = {sx1, sx2, sx3, sx4, sy0, sy1};
        mod_0_expected_inputs = {sy0, sy1};
        mod_1_expected_inputs = {sx4, sx3, sx2, sx1};
    }
    std::vector<migraphx::instruction_ref> splits;
    migraphx::module m1;
    {
        // params --> {x1, x2, x3, x4, y0, y1} binds to input args {sx1, sx2, sx3, sx4, sy0, sy1}
        auto m1_y0 = m1.add_parameter("y0", s1);
        auto m1_y1 = m1.add_parameter("y1", s2);
        auto m1_x1 = m1.add_parameter("x1", s1);
        auto m1_x2 = m1.add_parameter("x2", s1);
        auto m1_x3 = m1.add_parameter("x3", s1);
        auto m1_x4 = m1.add_parameter("x4", s1);
        // m1_dot = dot(y0, y1) --> dot(sy0, sy1)
        auto m1_dot = m1.add_instruction(migraphx::make_op("dot"), m1_y0, m1_y1);
        // m1_add = add(x0, m1_dot)  --> add(sx1, m1_dot)
        auto m1_add_1 = m1.add_instruction(migraphx::make_op("add"), m1_x1, m1_dot);
        // m1_add_2 = add(x2, x3) --> add(sx2, sx3)
        auto m1_add_2 = m1.add_instruction(migraphx::make_op("add"), m1_x2, m1_x3);
        // m1_sub = sub(x4, m1_relu_2) --> sub(sx4, m1_add_2)
        auto m1_sub = m1.add_instruction(migraphx::make_op("sub"), m1_x4, m1_add_2);
        // m1_mul = mul(m1_sub, m1_add_1)
        auto m1_mul = m1.add_instruction(migraphx::make_op("mul"), m1_sub, m1_add_1);
        m1.add_return({m1_mul});
        splits.push_back(m1_dot);
    }

    migraphx::module mod_0;
    {
        auto mod_0_y1 = mod_0.add_parameter("y1", s2);
        auto mod_0_y0 = mod_0.add_parameter("y0", s1);
        // mod_0_dot(y0, y1) --> dot(sy0, sy1)
        auto mod_0_dot = mod_0.add_instruction(migraphx::make_op("dot"), mod_0_y0, mod_0_y1);
        mod_0.add_return({mod_0_dot});
    }
    migraphx::module mod_1;
    {
        // expected input args are {dot_ins, sx4, sx3, sx2, sx1}
        auto mod_1_x0 = mod_1.add_parameter("x0", s1);
        auto mod_1_x1 = mod_1.add_parameter("x1", s1);
        auto mod_1_x2 = mod_1.add_parameter("x2", s1);
        auto mod_1_x3 = mod_1.add_parameter("x3", s1);
        auto mod_1_x4 = mod_1.add_parameter("x4", s1);
        // m1_add = add(x4, m1_dot)  --> add(sx1, m1_dot)
        auto m1_add = mod_1.add_instruction(migraphx::make_op("add"), mod_1_x4, mod_1_x0);
        // m1_add_2 = add(x3, x2) --> add(sx2, sx3)
        auto m1_add_2 = mod_1.add_instruction(migraphx::make_op("add"), mod_1_x3, mod_1_x2);
        // m1_sub = sub(x1, m1_relu_2) --> sub(sx4, m1_add_2)
        auto m1_sub = mod_1.add_instruction(migraphx::make_op("sub"), mod_1_x1, m1_add_2);
        // m1_mul = mul(m1_sub, m1_add)
        auto m1_mul = mod_1.add_instruction(migraphx::make_op("mul"), m1_sub, m1_add);
        mod_1.add_return({m1_mul});
    }
    auto mods = m1.split(inputs, splits);
    EXPECT(bool{mods[0].mod.sort() == mod_0.sort()});
    const auto mod_0_inputs = mods[0].inputs;
    EXPECT(bool{mod_0_inputs[0] == mod_0_expected_inputs[0]});
    EXPECT(bool{mod_0_inputs[1] == mod_0_expected_inputs[1]});
    const auto mod_1_inputs = mods[1].inputs;
    // first input arg should be the split instruction
    EXPECT(bool{mods[1].mod.sort() == mod_1.sort()});
    EXPECT(bool{mod_1_inputs[0] == splits.front()});
    EXPECT(bool{mod_1_inputs[1] == mod_1_expected_inputs[0]});
    EXPECT(bool{mod_1_inputs[2] == mod_1_expected_inputs[1]});
    EXPECT(bool{mod_1_inputs[3] == mod_1_expected_inputs[2]});
    EXPECT(bool{mod_1_inputs[4] == mod_1_expected_inputs[3]});
}

TEST_CASE(module_split3)
{
    migraphx::shape s{migraphx::shape::float_type, {1}};
    migraphx::module input_m;
    std::vector<migraphx::instruction_ref> inputs;
    {
        auto x1  = input_m.add_parameter("x1", s);
        auto x2  = input_m.add_parameter("x2", s);
        auto sx1 = input_m.add_instruction(migraphx::make_op("sqrt"), x1);
        auto sx2 = input_m.add_instruction(migraphx::make_op("sqrt"), x2);
        inputs   = {sx1, sx2};
    }
    migraphx::module m;
    std::vector<migraphx::instruction_ref> splits1;
    std::vector<migraphx::instruction_ref> splits2;
    {
        auto x1   = m.add_parameter("x1", s);
        auto x2   = m.add_parameter("x2", s);
        auto mul  = m.add_instruction(migraphx::make_op("mul"), x1, x2);
        auto sqrt = m.add_instruction(migraphx::make_op("sqrt"), mul);
        auto add  = m.add_instruction(migraphx::make_op("add"), sqrt, mul);
        m.add_return({add});
        splits1.push_back(mul);
        splits2.push_back(sqrt);
    }
    auto mods = m.split(inputs, splits1, splits2);

    migraphx::module m1;
    {
        auto x1  = m1.add_parameter("x1", s);
        auto x2  = m1.add_parameter("x2", s);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), x1, x2);
        m1.add_return({mul});
    }
    migraphx::module m2;
    {
        auto x0   = m2.add_parameter("x0", s);
        auto sqrt = m2.add_instruction(migraphx::make_op("sqrt"), x0);
        m2.add_return({sqrt});
    }
    migraphx::module m3;
    {
        auto x0  = m3.add_parameter("x0", s);
        auto x1  = m3.add_parameter("x1", s);
        auto add = m3.add_instruction(migraphx::make_op("add"), x0, x1);
        m3.add_return({add});
    }
    EXPECT(mods[0].mod.sort() == m1.sort());
    EXPECT(mods[1].mod.sort() == m2.sort());
    EXPECT(mods[2].mod.sort() == m3.sort());

    EXPECT(bool{mods[0].inputs[0] == inputs[0]});
    EXPECT(bool{mods[0].inputs[1] == inputs[1]});

    EXPECT(bool{mods[1].inputs[0] == splits1.front()});

    EXPECT(bool{mods[2].inputs[0] == splits2.front()});
    EXPECT(bool{mods[2].inputs[1] == splits1.front()});
}

TEST_CASE(fuse_module)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto add = add_pointwise(p, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto mul = add_pointwise(p, "main:pointwise1", {add, z}, single_pointwise("mul"));

        std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
        auto rins    = m1.fuse(*add->module_inputs().front(), add->inputs(), &map_ins).front();
        map_ins[add] = rins;
        auto ret     = m1.fuse(*mul->module_inputs().front(), mul->inputs(), &map_ins);
        m1.add_return(ret);
    }
    migraphx::module m2;
    {
        auto x   = m2.add_parameter("x0", s);
        auto y   = m2.add_parameter("x1", s);
        auto z   = m2.add_parameter("x2", s);
        auto add = m2.add_instruction(migraphx::make_op("add"), x, y);
        auto mul = m2.add_instruction(migraphx::make_op("mul"), add, z);
        m2.add_return({mul});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(get_inputs)
{
    // Test a use case for get_inputs
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto z   = mm->add_parameter("z", s);
    auto add = add_pointwise(p, "main:pointwise0", {x, z}, single_pointwise("add"));
    auto mul = add_pointwise(p, "main:pointwise1", {add, y}, single_pointwise("mul"));

    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
    auto rins    = m1.fuse(*add->module_inputs().front(), add->inputs(), &map_ins).front();
    map_ins[add] = rins;
    auto ret     = m1.fuse(*mul->module_inputs().front(), mul->inputs(), &map_ins);
    m1.add_return(ret);

    // After using the fuse methods above, map_ins contains the following mappings:
    // - instruction in mm -> instruction in m1
    // - instruction in add -> instruction in m1
    // - instruction in mul -> instruction in m1

    // create a map of instructions in m1 to instruction in mm (all the parameters will
    // map to some instruction in mm)
    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_m1_to_mm;
    for(auto const& [i1, i2] : map_ins)
    {
        if(contains(*mm, i1))
            map_m1_to_mm[i2] = i1;
    }
    // get_inputs should return the instructions from mm in the correct order that should
    // be the new inputs to m1
    auto inputs = m1.get_inputs(map_m1_to_mm);

    EXPECT(inputs.size() == 3);
    EXPECT(bool{inputs[0] == x});
    EXPECT(bool{inputs[1] == z});
    EXPECT(bool{inputs[2] == y});
}

TEST_CASE(add_params)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::module m1;
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", s);
    auto y   = mm->add_parameter("y", s);
    auto z   = mm->add_parameter("z", s);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, y);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), add, z);

    // use case: add and mul are to be used as input parameters to a new submodule m1
    std::unordered_map<migraphx::instruction_ref, migraphx::instruction_ref> map_ins;
    m1.add_params({mul, add}, &map_ins);

    migraphx::module m2;
    m2.add_parameter("x0", mul->get_shape());
    m2.add_parameter("x1", add->get_shape());

    // m1 should have parameters x0 and x1 with the shapes of mul and add outputs, respectively
    EXPECT(m1 == m2);
    // map_ins should contain a mapping: mul (in mm) -> x0 (in m1)
    EXPECT(bool{m1.get_parameter("x0") == map_ins[mul]});
    // map_ins should contain a mapping: add (in mm) -> x1 (in m1)
    EXPECT(bool{m1.get_parameter("x1") == map_ins[add]});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
