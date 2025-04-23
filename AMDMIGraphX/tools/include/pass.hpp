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
#ifndef MIGRAPHX_GUARD_PASS_HPP
#define MIGRAPHX_GUARD_PASS_HPP

#include <cassert>
#include <string>
#include <memory>
#include <type_traits>
#include <utility>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/rank.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;
struct module;
struct module_pass_manager;

#ifdef DOXYGEN

/// An interface for applying a transformation to the instructions in a
/// `program`
struct pass
{
    /// A unique name used to identify the pass
    std::string name() const;
    /// Run the pass on the module
    void apply(module_pass_manager& mpm) const;
    void apply(module& m) const;
    /// Run the pass on the program
    void apply(program& p) const;
};

#else

MIGRAPHX_EXPORT module& get_module(module_pass_manager& mpm);

namespace detail {

template <class T>
auto module_pass_manager_apply(rank<1>, const T& x, module_pass_manager& mpm)
    -> decltype(x.apply(get_module(mpm)))
{
    return x.apply(get_module(mpm));
}

template <class T>
void module_pass_manager_apply(rank<0>, const T&, module_pass_manager&)
{
}

template <class T>
void module_pass_manager_apply(const T& x, module_pass_manager& mpm)
{
    module_pass_manager_apply(rank<1>{}, x, mpm);
}

} // namespace detail

<%
interface('pass',
    virtual('name', returns='std::string', const=True),
    virtual('apply', returns='void', mpm='module_pass_manager &', const=True, default='migraphx::detail::module_pass_manager_apply'),
    virtual('apply', returns='void', p='program &', const=True, default='migraphx::nop')
)
%>

/// Used in the targets to enable/disable compiler passes
MIGRAPHX_EXPORT pass enable_pass(bool enabled, pass p);

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
