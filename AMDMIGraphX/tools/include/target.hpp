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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_TARGET_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <migraphx/context.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/config.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/module_ref.hpp>
#include <migraphx/support_metric.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/supported_segments.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct value;

#ifdef DOXYGEN

/// An interface for a compilation target
struct target
{
    /// A unique name used to identify the target
    std::string name() const;
    /**
     * @brief The transformation pass to be run during compilation.
     *
     * @param ctx This is the target-dependent context that is created by `get_context`
     * @param options Compiling options passed in by the user
     * @return The passes to be ran
     */
    std::vector<pass> get_passes(context& ctx, const compile_options& options) const;
    /**
     * @brief Construct a context for the target.
     * @return The context to be used during compilation and execution.
     */
    context get_context() const;
    /**
     * @brief Get the ranges of instructions that are supported on a target
     * @param module Module to check for supported instructions
     * @param metric Used to define how the quality of the support should be measured
     * @return the supported segments of the graph
     */
    supported_segments target_is_supported(T&, const_module_ref mod, support_metric metric) const;
    /**
     * @brief copy an argument to the current target.
     *
     * @param arg Input argument to be copied to the target
     * @return Argument in the target.
     */
    argument copy_to(const argument& arg) const;
    /**
     * @brief copy an argument from the current target.
     *
     * @param arg Input argument to be copied from the target
     * @return Argument in the host.
     */
    argument copy_from(const argument& arg) const;
    /**
     * @brief Allocate an argument based on the input shape
     *
     * @param s Shape of the argument to be allocated in the target
     * @return Allocated argument in the target.
     */
    argument allocate(const shape& s) const;
};

#else

template <class T>
argument target_allocate(T& x, const shape&)
{
    std::string name = x.name();
    MIGRAPHX_THROW("Not computable: " + name);
}

template <class T>
argument copy_to_target(T&, const argument& arg)
{
    return arg;
}

template <class T>
argument copy_from_target(T&, const argument& arg)
{
    return arg;
}

template <class T>
supported_segments target_find_supported(T&, const_module_ref, support_metric)
{
    return {};
}

<%
 interface('target',
           virtual('name', returns = 'std::string', const = True),
           virtual('get_passes',
                   ctx     = 'context&',
                   options = 'const compile_options&',
                   returns = 'std::vector<pass>',
                   const   = True),
           virtual('get_context', returns = 'context', const = True),
           virtual('find_supported',
                   returns = 'supported_segments',
                   mod     = 'const_module_ref',
                   m       = 'support_metric',
                   const   = True,
                   default = 'target_find_supported'),
           virtual('copy_to',
                   returns = 'argument',
                   input   = 'const argument&',
                   const   = True,
                   default = 'copy_to_target'),
           virtual('copy_from',
                   returns = 'argument',
                   input   = 'const argument&',
                   const   = True,
                   default = 'copy_from_target'),
           virtual('allocate',
                   s       = 'const shape&',
                   returns = 'argument',
                   const   = True,
                   default = 'target_allocate')) %>

#endif

void migraphx_to_value(value& v, const target& t);
void migraphx_from_value(const value& v, target& t);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
