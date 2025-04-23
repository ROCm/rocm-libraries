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
#include <migraphx/generate.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/param_utils.hpp>
#include <basic_ops.hpp>
#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>
#include <utility>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION);
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION);

struct non_mlir_op
{
    std::string name() const { return "non_mlir_op"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>& inputs) const
    {
        migraphx::check_shapes{inputs, *this}.has(1);
        return inputs.at(0);
    }
};

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p, {migraphx::gpu::fuse_mlir{.enable_extra = true}, migraphx::dead_code_elimination{}});
}

template <class F>
static migraphx::instruction_ref add_mlir(migraphx::program& p,
                                          const std::string& name,
                                          std::vector<migraphx::instruction_ref> inputs,
                                          std::vector<std::string> arg_names,
                                          const F& f)
{
    assert(inputs.size() == arg_names.size() and "One interior parameter name given per input.");
    auto* mm = p.get_main_module();
    auto* pm = p.create_module(name);
    pm->set_bypass();
    std::vector<migraphx::instruction_ref> params;
    for(size_t i = 0, e = inputs.size(); i < e; ++i)
    {
        params.push_back(pm->add_parameter(arg_names[i], inputs[i]->get_shape().as_standard()));
    }
    auto values = f(pm, params);
    auto root   = std::get<0>(values);
    auto r      = std::get<1>(values);
    auto_add_return(pm, r);
    return mm->add_instruction(
        migraphx::make_op("gpu::mlir_op", {{"op", migraphx::to_value(root)}}), inputs, {pm});
}

template <class F>
static migraphx::instruction_ref add_mlir(migraphx::program& p,
                                          const std::string& name,
                                          std::vector<migraphx::instruction_ref> inputs,
                                          const F& f)
{
    std::vector<std::string> arg_names;
    migraphx::transform(migraphx::range(inputs.size()), std::back_inserter(arg_names), [&](auto i) {
        return migraphx::param_name(i);
    });
    return add_mlir(p, name, std::move(inputs), std::move(arg_names), std::move(f));
}

TEST_CASE(dot_reshapes_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 3}});
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_sq = mm->add_instruction(migraphx::make_op("squeeze"), dot_trans);
        auto add    = add_pointwise(p1, "main:pointwise0", {dot_sq, x}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s);
        auto b     = mm->add_parameter("b", s);
        auto x     = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 3}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_rsp = pm->add_instruction(migraphx::make_op("squeeze"), dot_trans);
                auto add     = pm->add_instruction(migraphx::make_op("add"), dot_rsp, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_add)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto add = add_pointwise(p1, "main:pointwise0", {dot, x}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s);
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_transpose_reshape_add)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 6, 6}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 6}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s1);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto xtranspose =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), x);
        auto xreshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s1.lens()}}), xtranspose);
        auto add = add_pointwise(p1, "main:pointwise0", {dot, xreshape}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s1);
        auto x   = mm->add_parameter("x", s1);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto xtranspose = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {1, 0, 2}}}), inputs[2]);
                auto xreshape = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", s1.lens()}}), xtranspose);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, xreshape);
                return std::make_tuple(dot->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_broadcast_mul)
{
    migraphx::shape os{migraphx::shape::float_type, {4, 56, 122, 122}};
    migraphx::shape is{migraphx::shape::float_type, {4, 14, 1, 1}};
    migraphx::shape ws{migraphx::shape::float_type, {56, 14, 1, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", is);
        auto y     = mm->add_parameter("y", os);
        auto w     = mm->add_parameter("w", ws);
        auto conv  = mm->add_instruction(migraphx::make_op("convolution"), x, w);
        auto convb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", os.lens()}}), conv);
        auto mul = add_pointwise(p1, "main:pointwise0", {convb, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", is);
        auto y    = mm->add_parameter("y", os);
        auto w    = mm->add_parameter("w", ws);
        auto conv = add_mlir(
            p2, "mlir_convolution0", {x, w}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto c =
                    pm->add_instruction(migraphx::make_op("convolution"), inputs[0], inputs[1]);
                return std::make_tuple(c->get_operator(), c);
            });
        auto convb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", os.lens()}}), conv);
        auto mul = add_pointwise(p2, "main:pointwise0", {convb, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(multi_use_dot_trans_add_pooling_sub)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto dot       = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);
        auto add = add_pointwise(p1, "main:pointwise0", {dot_trans, x}, single_pointwise("add"));
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {0, 0, 0, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                add);
        auto sub = add_pointwise(p1, "main:pointwise1", {dot, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), dot);

                auto add = pm->add_instruction(migraphx::make_op("add"), dot_trans, inputs[2]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{add, dot});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {0, 0, 0, 1}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto sub = add_pointwise(p2, "main:pointwise1", {dot, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_multi_use_trans_add_pooling_sub)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_unsq =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
        auto add = add_pointwise(p1, "main:pointwise0", {dot_unsq, x}, single_pointwise("add"));
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                add);
        auto sub =
            add_pointwise(p1, "main:pointwise1", {dot_unsq, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5, 4}});
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto dot_trans = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
                auto dot_unsq = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot_unsq, inputs[2]);
                return std::make_tuple(dot->get_operator(),
                                       std::vector<migraphx::instruction_ref>{add, dot});
            });
        auto fused_dot_add =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto pooling =
            mm->add_instruction(migraphx::make_op("pooling",
                                                  {{"mode", migraphx::op::pooling_mode::lpnorm},
                                                   {"padding", {1, 0, 0, 0}},
                                                   {"stride", {1, 1}},
                                                   {"lengths", {2, 1}},
                                                   {"lp_order", 2}}),
                                fused_dot_add);
        auto dot = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto dot_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dot);
        auto dot_reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 5, 4}}}), dot_trans);
        auto sub =
            add_pointwise(p2, "main:pointwise1", {dot_reshape, pooling}, single_pointwise("sub"));
        mm->add_return({sub});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dot_dot_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s2);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, c);
        auto add  = add_pointwise(p1, "main:pointwise0", {dot1, dot2}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s2);
        auto dot1 =
            add_mlir(p2, "mlir_dot0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto dot2 =
            add_mlir(p2, "mlir_dot1", {dot1, c}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto add = add_pointwise(p2, "main:pointwise0", {dot1, dot2}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(dot_dot_pointwise_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 5, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s1);
        auto b    = mm->add_parameter("b", s2);
        auto c    = mm->add_parameter("c", s2);
        auto x    = mm->add_parameter("d", s1);
        auto dot1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto dot2 = mm->add_instruction(migraphx::make_op("dot"), dot1, c);
        auto add1 = add_pointwise(p1, "main:pointwise0", {dot2, x}, single_pointwise("add"));
        auto add2 = add_pointwise(p1, "main:pointwise1", {dot1, add1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("a", s1);
        auto b   = mm->add_parameter("b", s2);
        auto c   = mm->add_parameter("c", s2);
        auto x   = mm->add_parameter("d", s1);
        auto dot1 =
            add_mlir(p2, "mlir_dot0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                return std::make_tuple(dot->get_operator(), dot);
            });
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {dot1, c, x}, [=](auto* pm, const auto& inputs) {
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], inputs[1]);
                auto add = pm->add_instruction(migraphx::make_op("add"), dot, inputs[2]);
                return std::make_tuple(dot->get_operator(), add);
            });
        auto add2 = add_pointwise(p2, "main:pointwise1", {dot1, fused}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(add_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), add, b);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:mlir_dot0",
                     {x, y, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto add =
                             pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                         auto dot = pm->add_instruction(migraphx::make_op("dot"), add, inputs[2]);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(relu_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto b    = mm->add_parameter("b", s);
        auto x    = mm->add_parameter("x", s);
        auto relu = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("relu"));
        auto dot  = mm->add_instruction(migraphx::make_op("dot"), relu, b);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto b   = mm->add_parameter("b", s);
        auto x   = mm->add_parameter("x", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:mlir_dot0",
                     {x, b},
                     {"x0", "x1"},
                     [=](auto* pm, const auto& inputs) {
                         auto relu = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                         auto dot  = pm->add_instruction(migraphx::make_op("dot"), relu, inputs[1]);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(relu_relu_dot)
{
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto relux = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("relu"));
        auto reluy = add_pointwise(p1, "main:pointwise1", {y}, single_pointwise("relu"));
        auto dot   = mm->add_instruction(migraphx::make_op("dot"), relux, reluy);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto fused =
            add_mlir(p2,
                     "main:pointwise0:main:pointwise1:mlir_dot0",
                     {x, y},
                     {"x0", "x1"},
                     [=](auto* pm, const auto& inputs) {
                         auto relux = pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
                         auto reluy = pm->add_instruction(migraphx::make_op("relu"), inputs[1]);
                         auto dot   = pm->add_instruction(migraphx::make_op("dot"), relux, reluy);
                         return std::make_tuple(dot->get_operator(), dot);
                     });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(dequantizelinear_dot)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();

        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = add_pointwise(
            p1, "main:pointwise0", {y, scale, zp}, single_pointwise("dequantizelinear"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, dq);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::int8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::int8_type, {2, 2, 2}}));

        auto fused = add_mlir(
            p2,
            "main:pointwise0:mlir_dot0",
            {y, scalelit, zplit, x},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto unsqueeze1 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[1]);
                auto broadcast1 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
                auto reshape1 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
                auto scale = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape1);

                auto unsqueeze2 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[2]);
                auto broadcast2 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
                auto reshape2 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
                auto zp = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape2);

                auto dq = pm->add_instruction(
                    migraphx::make_op("dequantizelinear"), inputs[0], scale, zp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[3], dq);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unsigned_dequantizelinear_dot)
{
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();

        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::uint8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::uint8_type, {2, 2, 2}}));

        auto unsqueeze1 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scalelit);
        auto broadcast1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
        auto scale = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape1);

        auto unsqueeze2 =
            mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), zplit);
        auto broadcast2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
        auto reshape2 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
        auto zp = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}), reshape2);

        auto dq = add_pointwise(
            p1, "main:pointwise0", {y, scale, zp}, single_pointwise("dequantizelinear"));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, dq);
        mm->add_return({dot});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 3, 5}});
        auto y   = mm->add_parameter("y", migraphx::shape{migraphx::shape::uint8_type, {2, 5, 2}});
        auto scalelit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::float_type, {2, 2, 2}}));
        auto zplit =
            mm->add_literal(migraphx::generate_literal({migraphx::shape::uint8_type, {2, 2, 2}}));

        auto fused = add_mlir(
            p2,
            "main:pointwise0:mlir_dot0",
            {y, scalelit, zplit, x},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto unsqueeze1 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[1]);
                auto broadcast1 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze1);
                auto reshape1 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast1);
                auto scale = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape1);

                auto unsqueeze2 =
                    pm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), inputs[2]);
                auto broadcast2 = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 3, 2}}}), unsqueeze2);
                auto reshape2 = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 6, 2}}}), broadcast2);
                auto zp = pm->add_instruction(
                    migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {5}}}),
                    reshape2);

                auto dq = pm->add_instruction(
                    migraphx::make_op("dequantizelinear"), inputs[0], scale, zp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[3], dq);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_dot)
{
    migraphx::program p1;
    {
        auto* m   = p1.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto w    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_w);
        auto dot  = m->add_instruction(migraphx::make_op("quant_dot"), x, w); // w: {1,8,4,4}
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m   = p2.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 4}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});

        auto fused = add_mlir(
            p2, "int4:mlir_quant_dot0", {x, pk_w}, {"x1", "x2"}, [=](auto* pm, const auto& inputs) {
                auto unpk_w = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[1]);
                auto q = pm->add_instruction(migraphx::make_op("quant_dot"), inputs[0], unpk_w);
                return std::make_tuple(q->get_operator(), q);
            });
        m->add_return({fused});
    }

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_dot_2)
{
    migraphx::program p1;
    {
        auto* m = p1.get_main_module();

        auto pk_x = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto x    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_x); // {1,8,4,4}

        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto w    = m->add_instruction(migraphx::make_op("unpack_int4"), pk_w); // {1,8,4,4}

        auto dot = m->add_instruction(migraphx::make_op("quant_dot"), x, w);
        m->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* m   = p2.get_main_module();
        auto x    = m->add_parameter("x", {migraphx::shape::int8_type, {1, 8, 4, 2}});
        auto pk_w = m->add_parameter("wt_int4", {migraphx::shape::int8_type, {1, 8, 4, 2}});

        auto fused = add_mlir(
            p2, "int4:mlir_quant_dot0", {x, pk_w}, {"x1", "x2"}, [=](auto* pm, const auto& inputs) {
                auto unpk_x = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[0]);
                auto unpk_w = pm->add_instruction(migraphx::make_op("unpack_int4"), inputs[1]);
                auto q      = pm->add_instruction(migraphx::make_op("quant_dot"), unpk_x, unpk_w);
                return std::make_tuple(q->get_operator(), q);
            });
        m->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(int_quant_dot_abs)
{
    migraphx::shape s_a{migraphx::shape::int8_type, {5, 4}};
    migraphx::shape s_b{migraphx::shape::int8_type, {4, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("a", s_a);
        auto b   = mm->add_parameter("b", s_b);
        auto dot = mm->add_instruction(migraphx::make_op("quant_dot"), a, b);
        auto abs = add_pointwise(p1, "main:pointwise0", {dot}, single_pointwise("abs"));
        mm->add_return({abs});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("a", s_a);
        auto b     = mm->add_parameter("b", s_b);
        auto fused =
            add_mlir(p2, "mlir_main:pointwise0", {a, b}, [=](auto* pm, const auto& inputs) {
                auto dot =
                    pm->add_instruction(migraphx::make_op("quant_dot"), inputs[0], inputs[1]);
                auto abs = pm->add_instruction(migraphx::make_op("abs"), dot);
                return std::make_tuple(dot->get_operator(), abs);
            });
        mm->add_return({fused});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(int_quant_dot_tanh_fails)
{
    migraphx::shape s_a{migraphx::shape::int8_type, {5, 4}};
    migraphx::shape s_b{migraphx::shape::int8_type, {4, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto a    = mm->add_parameter("a", s_a);
        auto b    = mm->add_parameter("b", s_b);
        auto dot  = mm->add_instruction(migraphx::make_op("quant_dot"), a, b);
        auto tanh = add_pointwise(p1, "main:pointwise0", {dot}, single_pointwise("tanh"));
        mm->add_return({tanh});
    }
    // This pass should not fuse as int32_t tanh isn't supported.
    run_pass(p1);
    auto* mm = p1.get_main_module();
    bool has_pointwise =
        std::any_of(mm->begin(), mm->end(), [&](const auto& i) { return i.name() == "pointwise"; });
    EXPECT(has_pointwise);
}

TEST_CASE(conv_split_reduce)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        mm->add_return({var, mean});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean});
                     });
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        mm->add_return({var, mean});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add_split_reduce_multi_use)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), mean);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), var);
        auto norm = add_pointwise(
            p1, "main:pointwise2", {add, mean_mb, var_mb}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        mm->add_return({norm});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w   = mm->add_parameter("w", s_w);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean, add});
                     });
        auto cba  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}), mean);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}), var);
        auto norm = add_pointwise(
            p2, "main:pointwise2", {cba, mean_mb, var_mb}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        mm->add_return({norm});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(conv_add_split_reduce_multi_use_conv)
{
    migraphx::shape s_x{migraphx::shape::float_type, {2, 4, 64, 64}};
    migraphx::shape s_w1{migraphx::shape::float_type, {320, 4, 3, 3}};
    migraphx::shape s_w2{migraphx::shape::float_type, {320, 320, 3, 3}};
    migraphx::shape s_b{migraphx::shape::float_type, {32}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w1  = mm->add_parameter("w1", s_w1);
        auto w2  = mm->add_parameter("w2", s_w2);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto mb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}), b);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w1);
        auto reshape = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
        auto add = add_pointwise(p1, "main:pointwise0", {reshape, mb}, single_pointwise("add"));
        auto mean_var = add_reduce(
            p1,
            "main:split_reduce0",
            {add},
            {2, 3, 4},
            "assign_add",
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx    = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return {rsum2, rsum1};
            });
        auto var =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), mean_var);
        auto mean =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), mean_var);
        auto mean_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), mean);
        auto mean_rsp = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mean_mb);
        auto var_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", add->get_shape().lens()}}), var);
        auto var_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), var_mb);
        auto add_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), add);
        auto norm = add_pointwise(
            p1, "main:pointwise2", {add_rsp, mean_rsp, var_rsp}, [=](auto* pm, const auto& inputs) {
                auto sub =
                    pm->add_instruction(migraphx::make_op("sub"), inputs.at(0), inputs.at(1));
                return pm->add_instruction(migraphx::make_op("div"), sub, inputs.at(2));
            });
        auto conv_2 = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), norm, w2);
        mm->add_return({conv_2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s_x);
        auto w1  = mm->add_parameter("w1", s_w1);
        auto w2  = mm->add_parameter("w2", s_w2);
        auto b   = mm->add_literal(migraphx::generate_literal(s_b));
        auto fused =
            add_mlir(p2,
                     "mlir_main:pointwise0_main:split_reduce0",
                     {x, w1, b},
                     {"x0", "x1", "x2"},
                     [=](auto* pm, const auto& inputs) {
                         auto conv = pm->add_instruction(
                             migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                             inputs[0],
                             inputs[1]);
                         auto reshape = pm->add_instruction(
                             migraphx::make_op("reshape", {{"dims", {2, 32, 10, 64, 64}}}), conv);
                         auto mb = pm->add_instruction(
                             migraphx::make_op("broadcast",
                                               {{"axis", 1}, {"out_lens", {2, 32, 10, 64, 64}}}),
                             inputs[2]);
                         auto add  = pm->add_instruction(migraphx::make_op("add"), reshape, mb);
                         auto mul  = pm->add_instruction(migraphx::make_op("mul"), add, add);
                         auto mean = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), add);
                         auto var = pm->add_instruction(
                             migraphx::make_op("reduce_sum", {{"axes", {2, 3, 4}}}), mul);
                         return std::make_tuple(
                             migraphx::make_op("gpu::mlir_op",
                                               {{"op", migraphx::to_value(conv->get_operator())}}),
                             std::vector<migraphx::instruction_ref>{var, mean, add});
                     });
        auto cba  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), fused);
        auto var  = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), fused);
        auto mean = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), fused);
        auto input_fused_conv = add_mlir(
            p2,
            "main:pointwise2:mlir_convolution1",
            {cba, mean, var, w2},
            {"x0", "x1", "x2", "x3"},
            [=](auto* pm, const auto& inputs) {
                auto mean_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}),
                    inputs.at(1));
                auto mean_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), mean_mb);
                auto var_mb = pm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", cba->get_shape().lens()}}),
                    inputs.at(2));
                auto var_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), var_mb);
                auto cba_rsp = pm->add_instruction(
                    migraphx::make_op("reshape", {{"dims", {2, 320, 64, 64}}}), inputs.at(0));
                auto sub  = pm->add_instruction(migraphx::make_op("sub"), cba_rsp, mean_rsp);
                auto div  = pm->add_instruction(migraphx::make_op("div"), sub, var_rsp);
                auto conv = pm->add_instruction(
                    migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}),
                    div,
                    inputs.at(3));
                return std::make_tuple(conv->get_operator(), conv);
            });
        mm->add_return({input_fused_conv});
    }
    if(not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_REDUCE_FUSION{}) or
       not migraphx::enabled(MIGRAPHX_ENABLE_MLIR_INPUT_FUSION{}))
        return;
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(gemm_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    // Original program graph (excluding shape ops): dot -> softmax -> dot
    // When compile pipeline reaches fuse_mlir pass, this has been rewritten
    // as dot -> fused_reduce -> dot, where the fused_reduce module contains
    // the base softmax operations. p1 constructs such a graph
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b = mm->add_instruction(migraphx::make_op("contiguous"), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        b1 = mm->add_instruction(migraphx::make_op("contiguous"), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);

        auto pw_reduce = add_reduce(
            p1,
            "main:fused_reduce0",
            {gemm1},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto rmax = rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}),
                                                inputs[0]);
                rmax      = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);

                auto pw =
                    add_pointwise(p1,
                                  rm,
                                  "main:pointwise0",
                                  {inputs[0], rmax},
                                  [](auto* pm, const auto& pw_inputs) {
                                      auto sub = pm->add_instruction(
                                          migraphx::make_op("sub"), pw_inputs[0], pw_inputs[1]);
                                      return pm->add_instruction(migraphx::make_op("exp"), sub);
                                  });
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw);
                rsum = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);

                return {
                    add_pointwise(p1, rm, "main:pointwise2", {pw, rsum}, single_pointwise("div"))};
            });

        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), pw_reduce, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1);

    // dot -> fused_reduce -> dot should be fused into a single module by the
    // find_mlir_standalone_attention_op matcher, with the fused_reduce module
    // unrolled into softmax (required for mlir during compile ops)
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto a     = mm->add_parameter("1", s1);
        auto b     = mm->add_parameter("2", s1);
        auto b1    = mm->add_parameter("3", s1);
        auto fused = add_mlir(
            p2,
            "mlir_attn_main:fused_reduce0",
            {a, b, b1},
            {"x0", "x1", "x2"},
            [=](auto* pm, const auto& inputs) {
                auto fb = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                fb         = pm->add_instruction(migraphx::make_op("contiguous"), fb);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto smax = pm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), gemm1);

                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[2]);
                fb1        = pm->add_instruction(migraphx::make_op("contiguous"), fb1);
                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), smax, fb1);
                return std::make_tuple(gemm2->get_operator(), gemm2);
            });
        mm->add_return({fused});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(gemm_pw_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();

    // Original program graph (excluding shape ops): dot -> pointwise -> softmax -> dot
    // Here fused_reduce contains pointwise + softmax all in one module.
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b = mm->add_instruction(migraphx::make_op("contiguous"), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        b1 = mm->add_instruction(migraphx::make_op("contiguous"), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);

        auto pw_reduce = add_reduce(
            p1,
            "main:fused_reduce0",
            {gemm1, eight, select, ten},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto pw = add_pointwise(
                    p1, rm, "main:pointwise0", inputs, [](auto* pm, const auto& pw_inputs) {
                        auto mul = pm->add_instruction(
                            migraphx::make_op("mul"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(
                            migraphx::make_op("where"), pw_inputs[2], mul, pw_inputs[3]);
                    });
                auto rmax =
                    rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), pw);
                rmax = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);

                auto pw2 = add_pointwise(
                    p1, rm, "main:pointwise2", {pw, rmax}, [](auto* pm, const auto& pw_inputs) {
                        auto sub = pm->add_instruction(
                            migraphx::make_op("sub"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(migraphx::make_op("exp"), sub);
                    });
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                rsum = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);

                return {
                    add_pointwise(p1, rm, "main:pointwise4", {pw2, rsum}, single_pointwise("div"))};
            });

        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), pw_reduce, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1);

    // Same result as gemm_softmax_gemm, but here the fused_reduce is unrolled into
    // pointwise ops + softmax
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        auto fused = add_mlir(
            p2,
            "mlir_attn_main:fused_reduce0",
            {a, b, eight, select, ten, b1},
            {"x0", "x1", "x2", "x3", "x4", "x5"},
            [=](auto* pm, const auto& inputs) {
                auto fb = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                fb         = pm->add_instruction(migraphx::make_op("contiguous"), fb);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto mul   = pm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    pm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto smax = pm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);

                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[5]);
                fb1        = pm->add_instruction(migraphx::make_op("contiguous"), fb1);
                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), smax, fb1);
                return std::make_tuple(gemm2->get_operator(), gemm2);
            });
        mm->add_return({fused});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(gemm_pw_softmax_gemm_pw)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    migraphx::shape s2{migraphx::shape::bool_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        auto c      = mm->add_parameter("5", s1);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b = mm->add_instruction(migraphx::make_op("contiguous"), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        b1 = mm->add_instruction(migraphx::make_op("contiguous"), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);

        auto pw_reduce = add_reduce(
            p1,
            "main:fused_reduce0",
            {gemm1, eight, select, ten},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto pw = add_pointwise(
                    p1, rm, "main:pointwise0", inputs, [](auto* pm, const auto& pw_inputs) {
                        auto mul = pm->add_instruction(
                            migraphx::make_op("mul"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(
                            migraphx::make_op("where"), pw_inputs[2], mul, pw_inputs[3]);
                    });
                auto rmax =
                    rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), pw);
                rmax = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);

                auto pw2 = add_pointwise(
                    p1, rm, "main:pointwise2", {pw, rmax}, [](auto* pm, const auto& pw_inputs) {
                        auto sub = pm->add_instruction(
                            migraphx::make_op("sub"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(migraphx::make_op("exp"), sub);
                    });
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                rsum = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);

                return {
                    add_pointwise(p1, rm, "main:pointwise4", {pw2, rsum}, single_pointwise("div"))};
            });

        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), pw_reduce, b1);
        auto trailing_pw =
            add_pointwise(p1, mm, "main:pointwise5", {gemm2, c}, single_pointwise("add"));
        mm->add_return({trailing_pw});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto a      = mm->add_parameter("1", s1);
        auto b      = mm->add_parameter("2", s1);
        auto b1     = mm->add_parameter("3", s1);
        auto select = mm->add_parameter("4", s2);
        auto c      = mm->add_parameter("5", s1);
        std::vector<float> eights(s1_elements, 0.125);
        std::vector<float> tens(s1_elements, 10);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto ten   = mm->add_literal(migraphx::literal{s1, tens});
        auto fused = add_mlir(
            p2,
            "mlir_attn_main:fused_reduce0",
            {a, b, eight, select, ten, b1, c},
            {"x0", "x1", "x2", "x3", "x4", "x5", "x6"},
            [=](auto* pm, const auto& inputs) {
                auto fb = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                fb         = pm->add_instruction(migraphx::make_op("contiguous"), fb);
                auto gemm1 = pm->add_instruction(migraphx::make_op("dot"), inputs[0], fb);
                auto mul   = pm->add_instruction(migraphx::make_op("mul"), gemm1, inputs[2]);
                auto where =
                    pm->add_instruction(migraphx::make_op("where"), inputs[3], mul, inputs[4]);
                auto smax = pm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);

                auto fb1 = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[5]);
                fb1        = pm->add_instruction(migraphx::make_op("contiguous"), fb1);
                auto gemm2 = pm->add_instruction(migraphx::make_op("dot"), smax, fb1);
                auto add   = pm->add_instruction(migraphx::make_op("add"), gemm2, inputs[6]);
                return std::make_tuple(gemm2->get_operator(), add);
            });
        mm->add_return({fused});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(gemm_invalid_pw_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::half_type, {1, 12, 256, 256}};
    auto s1_elements = s1.elements();

    // Original program graph (excluding shape ops): dot -> pointwise -> softmax -> dot
    // Here fused_reduce contains pointwise + softmax all in one module.
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        std::vector<float> eights(s1_elements, 0.125);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), b);
        b = mm->add_instruction(migraphx::make_op("contiguous"), b);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}),
                                 b1);
        b1 = mm->add_instruction(migraphx::make_op("contiguous"), b1);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);

        auto pw_reduce = add_reduce(
            p1,
            "main:fused_reduce0",
            {gemm1, eight},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto pw = add_pointwise(
                    p1, rm, "main:pointwise0", inputs, [](auto* pm, const auto& pw_inputs) {
                        auto mul = pm->add_instruction(
                            migraphx::make_op("mul"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(non_mlir_op{}, mul);
                    });
                auto rmax =
                    rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), pw);
                rmax = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);

                auto pw2 = add_pointwise(
                    p1, rm, "main:pointwise2", {pw, rmax}, [](auto* pm, const auto& pw_inputs) {
                        auto sub = pm->add_instruction(
                            migraphx::make_op("sub"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(migraphx::make_op("exp"), sub);
                    });
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                rsum = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);

                return {
                    add_pointwise(p1, rm, "main:pointwise4", {pw2, rsum}, single_pointwise("div"))};
            });

        auto gemm2 = mm->add_instruction(migraphx::make_op("dot"), pw_reduce, b1);
        mm->add_return({gemm2});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto a   = mm->add_parameter("1", s1);
        auto b   = mm->add_parameter("2", s1);
        auto b1  = mm->add_parameter("3", s1);
        std::vector<float> eights(s1_elements, 0.125);
        auto eight = mm->add_literal(migraphx::literal{s1, eights});
        auto gemm1 =
            add_mlir(p2, "mlir_dot0", {a, b}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto tp = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                auto ct  = pm->add_instruction(migraphx::make_op("contiguous"), tp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], ct);
                return std::make_tuple(dot->get_operator(), dot);
            });

        auto pw_reduce = add_reduce(
            p2,
            "main:fused_reduce0",
            {gemm1, eight},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto pw = add_pointwise(
                    p2, rm, "main:pointwise0", inputs, [](auto* pm, const auto& pw_inputs) {
                        auto mul = pm->add_instruction(
                            migraphx::make_op("mul"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(non_mlir_op{}, mul);
                    });
                auto rmax =
                    rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), pw);
                rmax = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rmax);

                auto pw2 = add_pointwise(
                    p2, rm, "main:pointwise2", {pw, rmax}, [](auto* pm, const auto& pw_inputs) {
                        auto sub = pm->add_instruction(
                            migraphx::make_op("sub"), pw_inputs[0], pw_inputs[1]);
                        return pm->add_instruction(migraphx::make_op("exp"), sub);
                    });
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                rsum = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);

                return {
                    add_pointwise(p2, rm, "main:pointwise4", {pw2, rsum}, single_pointwise("div"))};
            });

        auto gemm2 = add_mlir(
            p2, "mlir_dot1", {pw_reduce, b1}, {"y0", "y1"}, [=](auto* pm, const auto& inputs) {
                auto tp = pm->add_instruction(
                    migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), inputs[1]);
                auto ct  = pm->add_instruction(migraphx::make_op("contiguous"), tp);
                auto dot = pm->add_instruction(migraphx::make_op("dot"), inputs[0], ct);
                return std::make_tuple(dot->get_operator(), dot);
            });
        mm->add_return({gemm2});
    }

    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[])
{
    if(migraphx::gpu::mlir_enabled())
    {
        test::run(argc, argv);
    }
    return 0;
}
