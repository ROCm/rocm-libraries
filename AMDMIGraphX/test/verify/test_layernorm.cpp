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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <layernorm.hpp>

struct test_layernorm : verify_program<test_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 2, 5};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm2 : verify_program<test_layernorm2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 4, 24};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_large : verify_program<test_layernorm_large>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 32, 262144};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_fp16 : verify_program<test_layernorm_fp16>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::half_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_bf16 : verify_program<test_layernorm_bf16>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::bf16_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_fp8_1 : verify_program<test_layernorm_fp8_1>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::fp8e4m3fnuz_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_fp8_2 : verify_program<test_layernorm_fp8_2>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::fp8e5m2fnuz_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_fp8_3 : verify_program<test_layernorm_fp8_3>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::fp8e4m3fn_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_fp8_4 : verify_program<test_layernorm_fp8_4>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 24, 64};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::fp8e5m2_type, dims});
        add_layernorm(*mm, x, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_eps : verify_program<test_layernorm_eps>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 2, 5};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_layernorm(*mm, x, dims, 1e-5f);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_triadd : verify_program<test_layernorm_triadd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 4, 24};
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y    = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z    = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        add_layernorm(*mm, add2, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_layernorm_triadd_large : verify_program<test_layernorm_triadd_large>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 384, 1024};
        auto x    = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        auto y    = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, dims});
        auto z    = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, dims});
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        add_layernorm(*mm, add2, dims);
        return p;
    }

    std::string section() const { return "reduce"; }
};

struct test_add_layernorm_add_gemm_nonstd : verify_program<test_add_layernorm_add_gemm_nonstd>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto s =
            migraphx::shape::from_permutation(migraphx::shape::float_type, {8, 1, 16}, {1, 2, 0});
        auto x = mm->add_parameter("x", s);
        auto y = mm->add_parameter("y", s);
        auto z = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, {8, 16, 64}});
        auto add           = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto layernorm_ins = add_layernorm(*mm, add, s.lens());
        mm->add_instruction(migraphx::make_op("dot"), layernorm_ins, z);
        return p;
    }
    std::string section() const { return "gemm"; }
};

struct test_pw_layernorm : verify_program<test_pw_layernorm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm                 = p.get_main_module();
        std::vector<size_t> dims = {1, 9, 6};
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, dims});
        add_pointwise_layernorm(*mm, x, dims);
        return p;
    }
};
