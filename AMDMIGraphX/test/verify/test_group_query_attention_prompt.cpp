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

struct test_group_query_attention_prompt : verify_program<test_group_query_attention_prompt>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> query_lens{1, 10, 12288};
        std::vector<size_t> kv_lens{1, 32, 4096, 128};
        std::vector<size_t> slk_lens{1, 1};
        std::vector<size_t> tsl_lens{1, 1};
        std::vector<size_t> cs_cache_lens{4096, 64};
        auto dtype = migraphx::shape::half_type;
        migraphx::shape query_s{dtype, query_lens};
        migraphx::shape kv_s{dtype, kv_lens};
        migraphx::shape slk_s{migraphx::shape::int64_type, slk_lens};
        migraphx::shape tsl_s{migraphx::shape::int64_type, tsl_lens};
        migraphx::shape cs_cache_s{dtype, cs_cache_lens};
        auto query = mm->add_parameter("query", query_s);
        std::vector<int> slk_vec(slk_s.elements(), 10);
        std::vector<int> tsl_vec(tsl_s.elements(), 11);
        std::vector<float> cs_min_vec(cs_cache_s.elements(), -1.0);
        std::vector<float> cs_max_vec(cs_cache_s.elements(), 1.0);
        auto k_cache   = mm->add_parameter("k_cache", kv_s);
        auto v_cache   = mm->add_parameter("v_cache", kv_s);
        auto slk       = mm->add_literal(slk_s, slk_vec);
        auto tsl       = mm->add_literal(tsl_s, tsl_vec);
        auto key       = mm->add_literal(0.0f);
        auto value     = mm->add_literal(0.0f);
        auto cs_min    = mm->add_literal(cs_cache_s, cs_min_vec);
        auto cs_max    = mm->add_literal(cs_cache_s, cs_max_vec);
        auto cos_cache = mm->add_parameter("cos_cache", cs_cache_s);
        auto sin_cache = mm->add_parameter("sin_cache", cs_cache_s);
        cos_cache      = mm->add_instruction(migraphx::make_op("clip"), cos_cache, cs_min, cs_max);
        sin_cache      = mm->add_instruction(migraphx::make_op("clip"), sin_cache, cs_min, cs_max);
        auto r         = mm->add_instruction(migraphx::make_op("group_query_attention",
                                                               {{"do_rotary", 1},
                                                                {"kv_num_heads", 32},
                                                                {"local_window_size", -1},
                                                                {"num_heads", 32},
                                                                {"rotary_interleaved", 0}}),
                                     query,
                                     key,
                                     value,
                                     k_cache,
                                     v_cache,
                                     slk,
                                     tsl,
                                     cos_cache,
                                     sin_cache);
        auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), r);
        auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), r);
        auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), r);
        mm->add_return({r0, r1, r2});

        return p;
    }
};
