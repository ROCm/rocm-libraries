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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_OPERAND_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <migraphx/reflect.hpp>
#include <migraphx/dyn_output.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/streamutils.hpp>
#include <migraphx/normalize_attributes.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/auto_any_cast.hpp>
#include <migraphx/lifetime.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct context;

#ifdef DOXYGEN

/// The operation interface represents an action an instruction will perform. All
/// operation classes must be CopyConstructible.
struct operation
{
    /// A unique name identifying the operation
    std::string name() const;
    /// An optional method that can be used to finalize the operator before running
    void finalize(context& ctx);
    /// This is used to compute the resulting shape from an operation. If an
    /// operation cannot be run with input shapes, then it should throw an
    /// exception.
    shape compute_shape(const std::vector<shape>& input) const;
    /**
     * @brief This performs the operation's computation.
     *
     * This method can be optional when the operation is only used as a placeholder to be lowered
     * later on.
     *
     * @param ctx This is the context created by the `target` during compilation. Implementations
     * can use the target's `context` class rather than the `context` interface class.
     * @param output Equivalent to running `compute_shape` with each `shape` of the `argument`.
     * For a fixed shape, the returned argument will have the same shape as `output`.
     * For a dynamic shape, the returned `argument` will be a fixed shape within the bounds
     * set in the dynamic shape `output`.
     * @param input This is the `argument` result from the previous instruction's computation.
     * @return Return an `argument` of the result computation. The `shape` of `argument` should be
     * the same the `output` shape.
     */
    argument compute(context& ctx, const shape& output, const std::vector<argument>& input) const;
    /// An optional method to return which argument the output will alias. If
    /// there is no aliased output then -1 can be returned.
    std::ptrdiff_t output_alias(const std::vector<shape>& input) const;
    /// An optional stream operator to print the operation. When this is not
    /// implemented, it will just print the operation's name.
    friend std::ostream& operator<<(std::ostream& os, const operation& op);
};

/// Returns true if operation does not require a context to run compute
bool is_context_free(const operation& x);
/// Returns true if operation needs normalization before running compute
bool need_normalization(const operation& x);
/// Returns true if the operation has a finalize method
bool has_finalize(const operation& x);

#else

namespace detail {

namespace operation_operators {

template <class T>
auto operator<<(std::ostream& os, const T& x) -> decltype(os << x.name())
{
    os << x.name();
    char delim = '[';
    reflect_each(x, [&](auto&& y, auto name) {
        os << delim;
        os << name << "=";
        stream_write_value(os, y);
        delim = ',';
    });
    if(delim == ',')
        os << "]";
    return os;
}

template <class T, class U>
auto operator==(const T& x, const U& y) -> decltype(x.name() == y.name())
{
    static_assert(is_reflectable<T>{} or sizeof(T) <= 1,
                  "Missing equality operator or reflect method.");
    if(x.name() != y.name())
        return false;
    const auto& yy = any_cast<T>(y);
    return reflect_tie(x) == reflect_tie(yy);
}

} // namespace operation_operators

template <class T>
auto compute_shape_op(rank<3>,
                      const T& x,
                      const std::vector<shape>& inputs) -> decltype(x.compute_shape(inputs))
{
    return x.compute_shape(inputs);
}

template <class T>
auto compute_shape_op(rank<2>, const T& x, const std::vector<shape>& inputs)
    -> decltype(x.normalize_compute_shape(inputs))
{
    if(inputs.empty())
        MIGRAPHX_THROW("At least one input is required for " + x.name());
    dependent_type<operation, T> y = x;
    normalize_attributes(y, inputs[0]);
    return any_cast<T>(y).normalize_compute_shape(inputs);
}

template <class T>
auto compute_shape_op(rank<1>,
                      const T& x,
                      const std::vector<shape>& inputs) -> decltype(x.compute_shape(inputs, {}))
{
    return x.compute_shape(inputs, {});
}

template <class T>
shape compute_shape_op(rank<0>, const T& x, const std::vector<shape>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Shape not computable: " + name);
}

template <class T>
shape compute_shape_op(const T& x, const std::vector<shape>& inputs)
{
    return compute_shape_op(rank<3>{}, x, inputs);
}

template <class T>
auto mod_compute_shape_op(rank<1>,
                          const T& x,
                          const std::vector<shape>& inputs,
                          const std::vector<module_ref>& mod_args)
    -> decltype(x.compute_shape(inputs, mod_args))
{
    return x.compute_shape(inputs, mod_args);
}

template <class T>
shape mod_compute_shape_op(rank<0>,
                           const T& x,
                           const std::vector<shape>& inputs,
                           const std::vector<module_ref>& mod_args)
{
    if(mod_args.empty())
        return compute_shape_op(x, inputs);
    std::string name = x.name();
    MIGRAPHX_THROW("Shape not computable: " + name);
}

template <class T>
shape mod_compute_shape_op(const T& x,
                           const std::vector<shape>& inputs,
                           const std::vector<module_ref>& mod_args)
{
    return mod_compute_shape_op(rank<1>{}, x, inputs, mod_args);
}

template <class T>
auto compute_op(rank<1>,
                const T& x,
                context& ctx,
                const shape& output_shape,
                const std::vector<argument>& input)
    -> decltype(x.compute(auto_any_cast(ctx),
                          make_compute_output_shape(pack(x, output_shape, input)),
                          input))
{
    return x.compute(
        auto_any_cast(ctx), make_compute_output_shape(pack(x, output_shape, input)), input);
}

template <class T>
argument compute_op(rank<0>, const T& x, context&, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument
compute_op(const T& x, context& ctx, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
auto compute_op(rank<1>, const T& x, const shape& output_shape, const std::vector<argument>& input)
    -> decltype(x.compute(make_compute_output_shape(pack(x, output_shape, input)), input))
{
    return x.compute(make_compute_output_shape(pack(x, output_shape, input)), input);
}

template <class T>
argument compute_op(rank<0>, const T& x, const shape&, const std::vector<argument>&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument compute_op(const T& x, const shape& output_shape, const std::vector<argument>& input)
{
    return compute_op(rank<1>{}, x, output_shape, input);
}

template <class T, class F>
auto compute_op(rank<1>,
                const T& x,
                const shape& output,
                const std::vector<argument>& inputs,
                const std::vector<module_ref>& module_args,
                F f) -> decltype(x.compute(make_compute_output_shape(pack(x, output, inputs)),
                                           inputs,
                                           module_args,
                                           std::move(f)))
{
    return x.compute(
        make_compute_output_shape(pack(x, output, inputs)), inputs, module_args, std::move(f));
}

template <class T, class F>
argument compute_op(rank<0>,
                    const T& x,
                    const shape& output,
                    const std::vector<argument>& inputs,
                    const std::vector<module_ref>& module_args,
                    F) // NOLINT
{
    if(module_args.empty())
        return compute_op(x, output, inputs);
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T, class F>
argument compute_op(const T& x,
                    const shape& output,
                    const std::vector<argument>& inputs,
                    const std::vector<module_ref>& module_args,
                    F f)
{
    return compute_op(rank<1>{}, x, output, inputs, module_args, std::move(f));
}

template <class T, class F>
auto compute_op(rank<4>,
                const T& x,
                context& ctx,
                const shape& output,
                const std::vector<argument>& inputs,
                const std::vector<module_ref>& module_args,
                F f) // NOLINT
    -> decltype(x.compute(auto_any_cast(ctx),
                          make_compute_output_shape(pack(x, output, inputs)),
                          inputs,
                          module_args,
                          std::move(f)))
{
    return x.compute(auto_any_cast(ctx),
                     make_compute_output_shape(pack(x, output, inputs)),
                     inputs,
                     module_args,
                     std::move(f));
}

template <class T, class F>
auto compute_op(rank<3>,
                const T& x,
                context&,
                const shape& output,
                const std::vector<argument>& inputs,
                const std::vector<module_ref>& module_args,
                F f) // NOLINT
    -> decltype(x.compute(
        make_compute_output_shape(pack(x, output, inputs)), inputs, module_args, std::move(f)))
{
    return x.compute(
        make_compute_output_shape(pack(x, output, inputs)), inputs, module_args, std::move(f));
}

template <class T, class F>
auto compute_op(rank<2>,
                const T& x,
                context&,
                const shape& output,
                const std::vector<argument>& inputs,
                const std::vector<module_ref>&,
                F) // NOLINT
    -> decltype(x.compute(make_compute_output_shape(pack(x, output, inputs)), inputs))
{
    return x.compute(make_compute_output_shape(pack(x, output, inputs)), inputs);
}

template <class T, class F>
auto compute_op(rank<1>,
                const T& x,
                context& ctx,
                const shape& output,
                const std::vector<argument>& inputs,
                const std::vector<module_ref>&,
                F) // NOLINT
    -> decltype(x.compute(auto_any_cast(ctx),
                          make_compute_output_shape(pack(x, output, inputs)),
                          inputs))
{
    return x.compute(
        auto_any_cast(ctx), make_compute_output_shape(pack(x, output, inputs)), inputs);
}

template <class T, class F>
argument compute_op(rank<0>,
                    const T& x,
                    context&,
                    const shape&,
                    const std::vector<argument>&,
                    const std::vector<module_ref>&,
                    F) // NOLINT
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T, class F>
argument compute_op(const T& x,
                    context& ctx,
                    const shape& output,
                    const std::vector<argument>& inputs,
                    const std::vector<module_ref>& module_args,
                    F f)
{
    return compute_op(rank<4>{}, x, ctx, output, inputs, module_args, std::move(f));
}

template <class T>
auto is_context_free_op(rank<1>,
                        const T& x,
                        const shape& output_shape,
                        const std::vector<argument>& input)
    -> decltype(x.compute(make_compute_output_shape(pack(x, output_shape, input)), input),
                std::true_type{});

template <class T>
auto is_context_free_op(rank<0>, const T&, const shape&, const std::vector<argument>&)
    -> std::false_type;

template <class T>
auto is_context_free_op(const T& x)
    -> decltype(is_context_free_op(
        rank<1>{}, x, std::declval<const shape&>(), std::declval<std::vector<argument>>()))
{
    return {};
}

template <class T>
auto need_normalization_op(rank<1>, const T& x, const std::vector<shape>& inputs)
    -> decltype(x.normalize_compute_shape(inputs), std::true_type{});

template <class T>
auto need_normalization_op(rank<0>, const T&, const std::vector<shape>&) -> std::false_type;

template <class T>
auto need_normalization_op(const T& x)
    -> decltype(need_normalization_op(rank<1>{}, x, std::declval<std::vector<shape>>()))
{
    return {};
}

template <class T>
std::ptrdiff_t output_alias_op(const T&, const std::vector<shape>&)
{
    return -1;
}

template <class T>
auto finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), void())
{
    x.finalize(auto_any_cast(ctx), output_shape, input);
}

template <class T>
void finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
{
}

template <class T>
void finalize_op(T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
{
    finalize_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
auto has_finalize_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.finalize(auto_any_cast(ctx), output_shape, input), std::true_type{});

template <class T>
auto has_finalize_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
    -> std::false_type;

template <class T>
auto has_finalize_op(const T&) -> decltype(has_finalize_op(rank<1>{},
                                                           std::declval<T&>(),
                                                           std::declval<context&>(),
                                                           std::declval<const shape&>(),
                                                           std::declval<std::vector<shape>>()))
{
    return {};
}

template <class T>
auto compile_op(
    rank<1>, T& x, context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(x.compile(auto_any_cast(ctx), output_shape, input))
{
    return x.compile(auto_any_cast(ctx), output_shape, input);
}

template <class T>
value compile_op(rank<0>, T&, context&, const shape&, const std::vector<shape>&)
{
    return value::object{};
}

template <class T>
value compile_op(const T& x,
                 context& ctx,
                 const shape& output_shape,
                 const std::vector<shape>& input)
{
    return compile_op(rank<1>{}, x, ctx, output_shape, input);
}

template <class T>
value attributes_op(const T&)
{
    return value::object{};
}

template <class T>
value to_value_op(const T& x)
{
    return migraphx::to_value(x);
}

template <class T>
void from_value_op(T& x, const value& v)
{
    if(not(v.is_object() or (v.empty() and v.is_array())))
        MIGRAPHX_THROW("Value is not an object");
    return migraphx::from_value(v, x);
}

template <class T>
lifetime get_lifetime_op(const T&)
{
    return lifetime::local;
}

} // namespace detail

<%
 interface(
     'operation',
     virtual('name', returns = 'std::string', const = True),
     virtual(
         'is_context_free', returns = 'bool', const = True, default = 'detail::is_context_free_op'),
     virtual('need_normalization',
             returns = 'bool',
             const   = True,
             default = 'detail::need_normalization_op'),
     virtual('has_finalize', returns = 'bool', const = True, default = 'detail::has_finalize_op'),
     virtual(
         'get_lifetime', returns = 'lifetime', const = True, default = 'detail::get_lifetime_op'),
     virtual('output_alias',
             returns = 'std::ptrdiff_t',
             input   = 'const std::vector<shape>&',
             const   = True,
             default = 'detail::output_alias_op'),
     virtual('compile',
             returns = 'value',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<shape>&',
             default = 'detail::compile_op'),
     virtual('finalize',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<shape>&',
             default = 'detail::finalize_op'),
     virtual('compute_shape',
             returns = 'shape',
             input   = 'const std::vector<shape>&',
             const   = True,
             default = 'detail::compute_shape_op'),
     virtual('compute_shape',
             returns  = 'shape',
             inputs   = 'const std::vector<shape>&',
             mod_args = 'const std::vector<module_ref>&',
             const    = True,
             default  = 'detail::mod_compute_shape_op'),
     virtual('compute',
             returns = 'argument',
             ctx     = 'context&',
             output  = 'const shape&',
             input   = 'const std::vector<argument>&',
             const   = True,
             default = 'detail::compute_op'),
     virtual('compute',
             returns = 'argument',
             output  = 'const shape&',
             input   = 'const std::vector<argument>&',
             const   = True,
             default = 'detail::compute_op'),
     virtual(
         'compute',
         returns     = 'argument',
         output      = 'const shape&',
         input       = 'const std::vector<argument>&',
         module_args = 'const std::vector<module_ref>&',
         run =
             'std::function<std::vector<argument>(module_ref&, const std::unordered_map<std::string, argument>&)>',
         const   = True,
         default = 'detail::compute_op'),
     virtual(
         'compute',
         returns     = 'argument',
         ctx         = 'context&',
         output      = 'const shape&',
         input       = 'const std::vector<argument>&',
         module_args = 'const std::vector<module_ref>&',
         run =
             'std::function<std::vector<argument>(module_ref&, const std::unordered_map<std::string, argument>&)>',
         const   = True,
         default = 'detail::compute_op'),
     virtual('to_value', returns = 'value', const = True, default = 'detail::to_value_op'),
     virtual('from_value', v = 'const value&', default = 'detail::from_value_op'),
     virtual('attributes', returns = 'value', const = True, default = 'detail::attributes_op'),
     friend('operator<<',
            returns = 'std::ostream &',
            os      = 'std::ostream &',
            op      = 'const operation &',
            using   = 'migraphx::detail::operation_operators::operator<<'),
     friend('operator==',
            returns = 'bool',
            x       = 'const operation &',
            y       = 'const operation &',
            using   = 'migraphx::detail::operation_operators::operator==')) %>

    inline bool operator!=(const operation& x, const operation& y)
{
    return not(x == y);
}

inline value
compile(operation& op, context& ctx, const shape& output_shape, const std::vector<shape>& input)
{
    return op.compile(ctx, output_shape, input);
}
template <class Context>
inline value
compile(operation& op, Context& ctx, const shape& output_shape, const std::vector<shape>& input)
{
    dependent_type<context, Context> ctx2 = std::ref(ctx);
    return compile(op, ctx2, output_shape, input);
}
template <class T, class Context>
inline auto compile(T& op, Context& ctx, const shape& output_shape, const std::vector<shape>& input)
    -> decltype(op.compile(ctx, ctx, output_shape, input))
{
    return op.compile(ctx, ctx, output_shape, input);
}
inline shape compute_shape(const operation& op, const std::vector<shape>& inputs)
{
    return op.compute_shape(inputs);
}

template <class T>
inline auto compute_shape(const T& op,
                          const std::vector<shape>& inputs) -> decltype(op.compute_shape(inputs))
{
    return op.compute_shape(inputs);
}

template <class T>
inline auto compute_shape(const T& op, const std::vector<shape>& inputs)
    -> decltype(op.normalize_compute_shape(inputs))
{
    return detail::compute_shape_op(op, inputs);
}

inline shape compute_shape(const operation& op,
                           const std::vector<shape>& inputs,
                           const std::vector<module_ref>& mod_args)
{
    return op.compute_shape(inputs, mod_args);
}

template <class T>
inline auto compute_shape(const T& op,
                          const std::vector<shape>& inputs,
                          const std::vector<module_ref>& mod_args)
    -> decltype(op.compute_shape(inputs, mod_args))
{
    return op.compute_shape(inputs, mod_args);
}

template <class T>
inline auto compute_shape(const T& op,
                          const std::vector<shape>& inputs,
                          const std::vector<module_ref>& mod_args)
    -> decltype(op.normalize_compute_shape(inputs, mod_args))
{
    return detail::compute_shape_op(op, inputs, mod_args);
}

inline bool is_context_free(const operation& op) { return op.is_context_free(); }

template <class T>
bool is_context_free(const T& x)
{
    return detail::is_context_free_op(x);
}

inline bool need_normalization(const operation& op) { return op.need_normalization(); }

template <class T>
bool need_normalization(const T& x)
{
    return detail::need_normalization_op(x);
}

inline bool has_finalize(const operation& op) { return op.has_finalize(); }

template <class T>
bool has_finalize(const T& x)
{
    return detail::has_finalize_op(x);
}

MIGRAPHX_EXPORT void migraphx_to_value(value& v, const operation& op);
MIGRAPHX_EXPORT void migraphx_from_value(const value& v, operation& op);

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
