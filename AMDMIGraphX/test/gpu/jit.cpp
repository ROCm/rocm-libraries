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
#include <test.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx_kernels.hpp>

// NOLINTNEXTLINE
const std::string write_2s = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void write(char* data) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    data[num] = 2;
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string add_2s_binary = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void add_2(char* x, char* y) 
{
    int num = threadIdx.x + blockDim.x * blockIdx.x;
    y[num] = x[num] + 2;
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string simple_pointwise_increment = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <args.hpp>

using namespace migraphx;

extern "C" {
__global__ void kernel(void* x, void* y) 
{
    make_tensors()(x, y)([](auto xt, auto yt) __device__ {
        auto idx = make_index();
        const auto stride = idx.nglobal();
        for(index_int i = idx.global; i < xt.get_shape().elements(); i += stride)
        {
            yt[i] = xt[i] + 1;
        }
    });
}
    
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string check_define = R"__migraphx__(

#ifndef __DEFINE__
#error __DEFINE__ was not defined
#endif

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string unused_param = R"__migraphx__(

extern "C" {
__global__ void kernel(void* x, void* y) 
{}
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string incorrect_program = R"__migraphx__(

extern "C" {
__global__ void kernel(void* x) 
{
    x += y;
}
}

int main() {}

)__migraphx__";

// NOLINTNEXTLINE
const std::string math_template = R"__migraphx__(
#include <migraphx/kernels/pointwise.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/types.hpp>

namespace migraphx {

template <class T>
struct test_implicit_conversion_op
{
    T x;

    template <index_int N, class U>
    constexpr operator vec<U, N>() const
    {
        if constexpr(vec_size<T>() == 0)
        {
            return x;
        }
        else
        {
            static_assert(vec_size<T>() == N, "Vector mismatch size");
            return __builtin_convertvector(x, vec<U, N>);
        }
    }

    template <class U>
    constexpr operator U() const
    {
        static_assert(is_same<T, bool>{} or not is_same<U, float>{} or is_same<T, U>{});
        return static_cast<U>(x);
    }
};

template <class T>
constexpr test_implicit_conversion_op<T> test_implicit_conversion(T x)
{
    return {x};
}


extern "C" {
__global__ void kernel(${type}* p) 
{
    auto x = *p;
    *p = migraphx::test_implicit_conversion(migraphx::${invoke});
    (void)(1.f + migraphx::vec_at(migraphx::${invoke}, 0));
}
}
}

int main() {}

)__migraphx__";

static migraphx::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, content};
}

TEST_CASE(simple_compile_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", write_2s)}, {}, migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::argument input{{migraphx::shape::int8_type, {5}}};
    auto ginput = migraphx::gpu::to_gpu(input);
    migraphx::gpu::kernel k{binaries.front(), "write"};
    k.launch(nullptr, input.get_shape().elements(), 1024)(ginput.cast<std::int8_t>());
    auto output = migraphx::gpu::from_gpu(ginput);

    EXPECT(output != input);
    auto data = output.get<std::int8_t>();
    EXPECT(migraphx::all_of(data, [](auto x) { return x == 2; }));
}

static auto check_target(const std::string& arch)
{
    auto define  = "__" + arch + "__";
    auto content = migraphx::replace_string(check_define, "__DEFINE__", define);
    return migraphx::gpu::compile_hip_src({make_src_file("main.cpp", content)}, {}, arch);
}

TEST_CASE(compile_target)
{
    EXPECT(not check_target("gfx900").empty());
    EXPECT(not check_target("gfx906").empty());
}

TEST_CASE(compile_errors)
{
    EXPECT(test::throws([&] {
        migraphx::gpu::compile_hip_src(
            {make_src_file("main.cpp", incorrect_program)}, {}, migraphx::gpu::get_device_name());
    }));
}

TEST_CASE(compile_warnings)
{
    auto compile = [](const std::vector<std::string>& params) {
        return migraphx::gpu::compile_hip_src(
            {make_src_file("main.cpp", unused_param)}, params, migraphx::gpu::get_device_name());
    };

    EXPECT(not compile({}).empty());
    EXPECT(not compile({"-Wunused-parameter", "-Wno-error"}).empty());
    EXPECT(not compile({"-Wno-unused-parameter", "-Werror"}).empty());
#ifdef MIGRAPHX_USE_HIPRTC
    EXPECT(test::throws([&] { compile({"-Werror=unused-parameter"}); }));
    EXPECT(test::throws([&] { compile({"-Wunused-parameter", "-Werror"}); }));
#else
    EXPECT(test::throws([&] { compile({"-Werror=unused-parameter"}); }));
    EXPECT(test::throws([&] { compile({"-Wunused-parameter", "-Werror"}); }));
#endif
}

TEST_CASE(has_flags)
{
    EXPECT(migraphx::gpu::hip_has_flags({"--std=c++17"}));
    EXPECT(not migraphx::gpu::hip_has_flags({"--non-existent-flag-to-test-in-migraphx"}));
    EXPECT(migraphx::gpu::hip_has_flags({"-Wunused-parameter"}));
    EXPECT(not migraphx::gpu::hip_has_flags(
        {"-Wnon-existent-warnings-flag-to-test-in-migraphx", "-Werror"}));
}

TEST_CASE(code_object_hip)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("main.cpp", add_2s_binary)}, {}, migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::shape input{migraphx::shape::int8_type, {5}};

    std::vector<migraphx::shape> expected_inputs = {input, input};
    auto co                                      = migraphx::make_op("gpu::code_object",
                                                                     {{"code_object", migraphx::value::binary{binaries.front()}},
                                                                      {"symbol_name", "add_2"},
                                                                      {"global", input.elements()},
                                                                      {"local", 1024},
                                                                      {"expected_inputs", migraphx::to_value(expected_inputs)},
                                                                      {"output", migraphx::to_value(input)}});

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 2; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    migraphx::compile_options options;
    p.compile(migraphx::make_target("gpu"), options);

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_code_object_hip)
{
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};
    migraphx::gpu::hip_compile_options options;
    migraphx::gpu::context ctx;
    options.global = 256 * 1024;
    options.local  = 1024;
    options.inputs = {input, input};
    options.output = input;

    auto co = migraphx::gpu::compile_hip_code_object(ctx, simple_pointwise_increment, options);

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 1; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_pointwise)
{
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};

    migraphx::gpu::context ctx;
    auto co = migraphx::gpu::compile_op(
        "pointwise", ctx, {input, input}, {{"lambda", "[](auto x) { return make_tuple(x + 1); }"}});

    migraphx::program p;
    auto* mm            = p.get_main_module();
    auto input_literal  = migraphx::generate_literal(input);
    auto output_literal = migraphx::transform(input_literal, [](auto x) { return x + 1; });
    auto x              = mm->add_literal(input_literal);
    auto y              = mm->add_parameter("output", input);
    mm->add_instruction(co, x, y);
    p.compile(migraphx::make_target("gpu"), migraphx::compile_options{});

    auto result =
        migraphx::gpu::from_gpu(p.eval({{"output", migraphx::gpu::allocate_gpu(input)}}).front());

    EXPECT(result == output_literal.get_argument());
}

TEST_CASE(compile_math)
{
    std::vector<std::string> math_invoke = {
        // clang-format off
        "abs(x)",
        "acos(x)",
        "acosh(x)",
        "asin(x)",
        "asinh(x)",
        "atan(x)",
        "atanh(x)",
        "ceil(x)",
        "cos(x)",
        "cosh(x)",
        "erf(x)",
        "exp(x)",
        "floor(x)",
        "fmod(x, x)",
        "isnan(x)",
        "log(x)",
        "max(x, x)",
        "min(x, x)",
        "pow(x, 0)",
        "pow(x, x)",
        "remainder(x,x)",
        "round(x)",
        "rsqrt(x)",
        "sin(x)",
        "sinh(x)",
        "sqrt(x)",
        "tan(x)",
        "tanh(x)",
        "where(true, x, x)",
        // clang-format on
    };
    std::vector<std::string> data_types;
    auto vec_sizes = {2, 4, 6};
    for(auto&& t : migraphx::shape::types())
    {
        if(contains({migraphx::shape::bool_type, migraphx::shape::tuple_type}, t))
            continue;
        auto name = migraphx::shape::cpp_type(t);
        if(contains({migraphx::shape::half_type, migraphx::shape::bf16_type}, t))
            name.insert(0, "migraphx::");
        data_types.push_back(name);
        // fp8 doesn't have vectorization support yet, therefore skip it for now.
        std::set<migraphx::shape::type_t> fp8_types = {migraphx::shape::fp8e4m3fnuz_type,
                                                       migraphx::shape::fp8e5m2fnuz_type,
                                                       migraphx::shape::fp8e4m3fn_type,
                                                       migraphx::shape::fp8e5m2_type};
        if(not contains(fp8_types, t))
        {
            migraphx::transform(vec_sizes, std::back_inserter(data_types), [&](auto i) {
                return "migraphx::vec<" + name + ", " + std::to_string(i) + ">";
            });
        }
    }
    migraphx::shape input{migraphx::shape::float_type, {5, 2}};
    migraphx::gpu::hip_compile_options options;
    migraphx::gpu::context ctx;
    options.global = 1024;
    options.local  = 1024;
    options.inputs = {input};
    options.output = input;
    migraphx::par_for(math_invoke.size() * data_types.size(), 1, [&](auto i) {
        const auto& t      = data_types[i % data_types.size()];
        const auto& invoke = math_invoke[i / data_types.size()];
        auto src = migraphx::interpolate_string(math_template, {{"type", t}, {"invoke", invoke}});
        auto co  = migraphx::gpu::compile_hip_code_object(ctx, src, options);
        (void)co;
    });
}

// NOLINTNEXTLINE
const std::string assert_template = R"__migraphx__(
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/types.hpp>
using namespace migraphx;
extern "C" {
__global__ void kernel(void*) 
{
    static_assert(numeric_max<${type}>() == ${max}, "");
    static_assert(numeric_lowest<${type}>() == ${min}, "");
}
}

int main() {}

)__migraphx__";

TEST_CASE(assert_type_min_max)
{
    std::vector<std::string> data_types;
    migraphx::gpu::hip_compile_options options;
    migraphx::gpu::context ctx;
    for(auto&& t : migraphx::shape::types())
    {
        if(contains(
               {
                   migraphx::shape::bool_type,
                   migraphx::shape::tuple_type,
               },
               t))
            continue;
        auto name = migraphx::shape::cpp_type(t);
        if(contains({migraphx::shape::half_type, migraphx::shape::bf16_type}, t))
            name.insert(0, "migraphx::");
        migraphx::shape::visit(t, [&](auto as) {
            std::string min = "";
            std::string max = "";
            // Note 9223372036854775808 is a constant literal that is outside the range of long
            // long type For the same reason, 18446744073709551616 needs postfix ULL to be parsed
            // correctly
            if(t == migraphx::shape::int64_type)
            {
                min = "(" + std::to_string(as.min() + 1) + "LL - 1)";
                max = std::to_string(as.max());
            }
            else if(t == migraphx::shape::uint64_type)
            {
                min = std::to_string(as.min());
                max = std::to_string(as.max()) + "ULL";
            }
            else
            {
                min = std::to_string(as.min());
                max = std::to_string(as.max());
            }
            auto src = migraphx::interpolate_string(assert_template,
                                                    {{"type", name}, {"max", max}, {"min", min}});
            migraphx::shape input{migraphx::shape::float_type, {5, 2}};
            options.global = 1024;
            options.local  = 1024;
            options.inputs = {input};
            options.output = input;
            options.emplace_param("-Wno-float-equal");

            auto co = migraphx::gpu::compile_hip_code_object(ctx, src, options);
        });
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
