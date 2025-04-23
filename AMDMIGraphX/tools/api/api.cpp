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
#include <migraphx/execution_environment.hpp>
#include <migraphx/migraphx.h>
#include <migraphx/rank.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/program.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/tf.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/load_save.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_op.hpp>
#include <migraphx/json.hpp>
#include <migraphx/convert_to_json.hpp>
#include <array>
#include <algorithm>
#include <cstdarg>

namespace migraphx {

#ifdef MIGRAPHX_BUILD_TESTING
static thread_local bool disable_exception_catch = false; // NOLINT

extern "C" MIGRAPHX_C_EXPORT void migraphx_test_private_disable_exception_catch(bool b)
{
    disable_exception_catch = b;
}
#endif

template <class F>
migraphx_status try_(F f, bool output = true) // NOLINT
{
#ifdef MIGRAPHX_BUILD_TESTING
    if(disable_exception_catch)
    {
        f();
    }
    else
    {
#endif
        try
        {
            f();
        }
        catch(const migraphx::exception& ex)
        {
            if(output)
                std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
            if(ex.error > 0)
                return migraphx_status(ex.error);
            else
                return migraphx_status_unknown_error;
        }
        catch(const std::exception& ex)
        {
            if(output)
                std::cerr << "MIGraphX Error: " << ex.what() << std::endl;
            return migraphx_status_unknown_error;
        }
        catch(...)
        {
            return migraphx_status_unknown_error;
        }
#ifdef MIGRAPHX_BUILD_TESTING
    }
#endif
    return migraphx_status_success;
}

static shape::type_t to_shape_type(migraphx_shape_datatype_t t)
{
    switch(t)
    {
    case migraphx_shape_tuple_type: return shape::tuple_type;
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case migraphx_shape_##x: return shape::x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

static migraphx_shape_datatype_t to_shape_type(shape::type_t t)
{
    switch(t)
    {
    case shape::tuple_type: return migraphx_shape_tuple_type;
#define MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT(x, y) \
    case shape::x: return migraphx_shape_##x;
        MIGRAPHX_SHAPE_VISIT_TYPES(MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT)
#undef MIGRAPHX_DETAIL_SHAPE_CASE_CONVERT
    }
    MIGRAPHX_THROW(migraphx_status_bad_param, "Unknown type");
}

template <class T>
static auto to_obj_vector(const T* x, std::size_t n)
{
    std::vector<decltype((*x)->object)> result;
    std::transform(x, x + n, std::back_inserter(result), [&](auto&& y) { return y->object; });
    return result;
}

template <class T, class U>
static auto to_objptr_vector(const U* x, std::size_t n)
{
    std::vector<T> result;
    std::transform(
        x, x + n, std::back_inserter(result), [&](auto&& y) { return std::addressof(y->object); });
    return result;
}

static target get_target(const std::string& name) { return make_target(name); }

static void set_offload_copy(compile_options& options, bool value) { options.offload_copy = value; }

static void set_fast_math(compile_options& options, bool value) { options.fast_math = value; }

static void set_exhaustive_tune_flag(compile_options& options, bool value)
{
    options.exhaustive_tune = value;
}

static void set_file_format(file_options& options, const char* format) { options.format = format; }

static void set_default_dim_value(onnx_options& options, size_t value)
{
    options.default_dim_value = value;
}

static void set_default_dyn_dim_value(onnx_options& options, const shape::dynamic_dimension& dd)
{
    options.default_dyn_dim_value = dd;
}

static void set_default_loop_iterations(onnx_options& options, int64_t value)
{
    options.max_loop_iterations = value;
}

static void set_external_data_path(onnx_options& options, const char* external_data_path)
{
    options.external_data_path = std::string(external_data_path);
}

static void set_limit_loop_iterations(onnx_options& options, int64_t value)
{
    options.limit_max_iterations = value;
}

static void set_nhwc(tf_options& options, bool is_nhwc) { options.is_nhwc = is_nhwc; }

static void set_default_dim_value(tf_options& options, size_t value) { options.batch_size = value; }

static void
set_input_parameter_shape(onnx_options& options, const char* name, std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

static void set_dyn_input_parameter_shape(onnx_options& options,
                                          const char* name,
                                          std::vector<shape::dynamic_dimension> dyn_dims)
{
    options.map_dyn_input_dims[std::string(name)] = std::move(dyn_dims);
}

static void
set_input_parameter_shape(tf_options& options, const char* name, std::vector<std::size_t> dims)
{
    options.map_input_dims[std::string(name)] = std::move(dims);
}

static void set_output_names(tf_options& options, std::vector<const char*> names)
{
    options.output_node_names = std::vector<std::string>(names.begin(), names.end());
}

static std::vector<argument>
run_async(program& p, const parameter_map& params, void* s, std::string_view name)
{
    execution_environment exec_env{any_ptr(s, name), true};
    return p.eval(params, exec_env);
}

template <class Value>
static std::vector<const char*> get_names(const std::unordered_map<std::string, Value>& m)
{
    std::vector<const char*> result;
    std::transform(
        m.begin(), m.end(), std::back_inserter(result), [](auto&& p) { return p.first.c_str(); });
    return result;
}

template <class T>
static std::set<T> make_set(const T* x, std::size_t n)
{
    return {x, x + n};
}

static void quantize_fp16_with_op_names(program& prog, std::vector<std::string>& names)
{
    if(names.empty())
    {
        names = {"all"};
    }

    migraphx::quantize_fp16(prog, names);
}

struct quantize_int8_options
{
    std::vector<parameter_map> calibration   = {};
    std::unordered_set<std::string> op_names = {};
};

static void add_op_name(quantize_int8_options& options, const char* name)
{
    options.op_names.insert(name);
}

static void add_calibration_data(quantize_int8_options& options, parameter_map& data)
{
    options.calibration.push_back(data);
}

static void quantize_int8_wrap(program& prog, const target& t, quantize_int8_options& options)
{
    if(options.op_names.empty())
    {
        options.op_names = {"dot", "convolution"};
    }

    migraphx::quantize_int8(prog, t, options.calibration, options.op_names);
}

struct quantize_fp8_options
{
    std::vector<parameter_map> calibration = {};
};

static void add_calibration_data(quantize_fp8_options& options, parameter_map& data)
{
    options.calibration.push_back(data);
}

static void quantize_fp8_wrap(program& prog, const target& t, quantize_fp8_options& options)
{
    migraphx::quantize_fp8(prog, t, options.calibration);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
#endif

static operation create_op(const char* name, const char* attributes, va_list vlist)
{
    std::string sattributes = attributes == nullptr ? "" : attributes;
    std::vector<char> buffer(sattributes.size() * 2);
    std::vsnprintf(buffer.data(), buffer.size(), sattributes.c_str(), vlist);
    value v = value::object{};
    if(attributes != nullptr)
    {
        v = from_json_string(convert_to_json(std::string(buffer.data())));
    }
    auto op = make_op(name, v);

    return op;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <class T>
static bool equal(const T& x, const T& y)
{
    return x == y;
}

static std::vector<argument> run(program& p, const parameter_map& params) { return p.eval(params); }

static std::vector<shape> get_output_shapes(program& p) { return p.get_output_shapes(); }

static void print_program(const program& p) { std::cout << p << std::endl; }

static void print_module(const module& m) { std::cout << m << std::endl; }

static migraphx::instruction_ref add_allocation(module& m, const migraphx::shape& s)
{
    return m.add_instruction(migraphx::make_op("allocate", {{"shape", migraphx::to_value(s)}}), {});
}

struct experimental_custom_op
{
    std::string name;
    experimental_custom_op() = default;

    experimental_custom_op(std::string pname) : name(std::move(pname)) {}
};

template <class CustomOp>
struct custom_operation
{

    template <class Self, class F>
    static auto reflect(Self&, F)
    {
        return pack();
    }

    value attributes() const
    {
        return {{"custom_op", true}, {"target", op.runs_on_offload_target() ? "gpu" : "cpu"}};
    }

    CustomOp op;
    std::string name() const { return op.xobject.name; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        return op.compute_shape(std::move(inputs));
    }

    // TODO: Compute method with module_args
    argument
    compute(migraphx::context ctx, migraphx::shape output_shape, std::vector<argument> inputs) const
    {
        return op.compute(std::move(ctx), std::move(output_shape), std::move(inputs));
    }

    std::ptrdiff_t output_alias(std::vector<shape> inputs) const
    {
        auto alias_vec = op.output_alias(std::move(inputs));
        // TODO: For now, only support one output alias
        if(alias_vec.empty())
        {
            return -1;
        }
        if(alias_vec.size() > 1)
        {
            MIGRAPHX_THROW("Currently, CustomOps in MIGraphX only supports one output_alias");
        }
        return alias_vec.front();
    }

    bool runs_on_offload_target() const { return op.runs_on_offload_target(); }
};

template <class CustomOp>
static void register_custom_op(const CustomOp& op)
{
    register_op(custom_operation<CustomOp>{op});
}

static migraphx::context get_context(const program& p) { return p.get_context(); }

} // namespace migraphx

<% generate_c_api_body() %>
