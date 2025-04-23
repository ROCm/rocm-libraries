/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP
#define MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP

#include <migraphx/operation.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>

// NOLINTNEXTLINE
#define MIGRAPHX_GPU_TEST_PRECOMPILE(...)                                \
    struct test_compiler : migraphx::gpu::compiler<test_compiler>        \
    {                                                                    \
        std::vector<std::string> names() const { return {__VA_ARGS__}; } \
                                                                         \
        template <class... Ts>                                           \
        migraphx::operation compile_op(Ts&&...) const                    \
        {                                                                \
            MIGRAPHX_THROW("Not compilable");                            \
        }                                                                \
                                                                         \
        template <class... Ts>                                           \
        migraphx::gpu::compiler_replace compile(Ts&&...) const           \
        {                                                                \
            MIGRAPHX_THROW("Not compilable");                            \
        }                                                                \
    };

inline migraphx::operation make_precompile_op(migraphx::rank<0>, const migraphx::operation& op)
{
    return migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(op)}});
}

inline migraphx::operation make_precompile_op(migraphx::rank<1>, const std::string& name)
{
    return make_precompile_op(migraphx::rank<0>{}, migraphx::make_op(name));
}

template <class T>
auto make_precompile_op(const T& x)
{
    return make_precompile_op(migraphx::rank<1>{}, x);
}

#endif // MIGRAPHX_GUARD_TEST_GPU_MAKE_PRECOMPILE_OP_HPP
