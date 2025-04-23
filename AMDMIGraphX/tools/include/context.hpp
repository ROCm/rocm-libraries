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
#ifndef MIGRAPHX_GUARD_CONTEXT_HPP
#define MIGRAPHX_GUARD_CONTEXT_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/any_ptr.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// A context is used to store internal data for a `target`. A context is
/// constructed by a target during compilation and passed to the operations
/// during `eval`.
struct context
{
    /// Wait for any tasks in the context to complete
    void finish() const;
};

#else

template <class T>
value to_value_context(const T&)
{
    return value{};
}

template <class T>
void from_value_context(T&, const value&)
{
}

template <class T>
any_ptr get_queue_context(T&)
{
    return {};
}

template <class T>
void wait_for_context(T&, any_ptr)
{
}

template <class T>
void finish_on_context(T&, any_ptr){}

<%
 interface('context',
           virtual('to_value', returns = 'value', const = True, default = 'to_value_context'),
           virtual('from_value', v = 'const value&', default = 'from_value_context'),
           virtual('get_queue', returns = 'any_ptr', default = 'get_queue_context'),
           virtual('wait_for', queue = 'any_ptr', returns = 'void', default = 'wait_for_context'),
           virtual('finish_on', queue = 'any_ptr', returns = 'void', default = 'finish_on_context'),
           virtual('finish', returns = 'void', const = True)) %>

    inline void migraphx_to_value(value& v, const context& ctx)
{
    v = ctx.to_value();
}

inline void migraphx_from_value(const value& v, context& ctx) { ctx.from_value(v); }

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
