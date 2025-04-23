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

#include <onnx_test.hpp>

TEST_CASE(group_query_attention_defaults_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape qkvs{migraphx::shape::half_type, {1, 1, 12288}};
    migraphx::shape pkvs{migraphx::shape::half_type, {1, 32, 4096, 128}};
    migraphx::shape kvs{migraphx::shape::float_type, {1}};
    migraphx::shape consts{migraphx::shape::int32_type, {1}};
    migraphx::shape cs{migraphx::shape::half_type, {4096, 64}};
    migraphx::shape outs{migraphx::shape::half_type, {1, 1, 4096}};

    std::vector<float> cs_data(cs.elements(), 1.0);
    auto slk   = mm->add_literal(migraphx::literal{consts, {1}});
    auto tsl   = mm->add_literal(migraphx::literal{consts, {2}});
    auto cc    = mm->add_literal(migraphx::literal{cs, cs_data});
    auto sc    = mm->add_literal(migraphx::literal{cs, cs_data});
    auto qkv   = mm->add_parameter("qkv", qkvs);
    auto key   = mm->add_parameter("key", kvs);
    auto value = mm->add_parameter("value", kvs);
    auto pk    = mm->add_parameter("past_key_values_key", pkvs);
    auto pv    = mm->add_parameter("past_key_values_value", pkvs);

    auto gqa = mm->add_instruction(migraphx::make_op("group_query_attention",
                                                     {{"do_rotary", 0},
                                                      {"kv_num_heads", 0},
                                                      {"local_window_size", -1},
                                                      {"num_heads", 1},
                                                      {"rotary_interleaved", 0},
                                                      {"scale", 0.0}}),
                                   qkv,
                                   key,
                                   value,
                                   pk,
                                   pv,
                                   slk,
                                   tsl,
                                   cc,
                                   sc);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), gqa);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), gqa);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), gqa);

    auto prog = optimize_onnx("group_query_attention_defaults_test.onnx");
    EXPECT(p == prog);
}
