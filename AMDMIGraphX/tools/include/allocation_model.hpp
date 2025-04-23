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
#ifndef MIGRAPHX_GUARD_ALLOCATION_MODEL_HPP
#define MIGRAPHX_GUARD_ALLOCATION_MODEL_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/operation.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

#ifdef DOXYGEN

/// An interface for target-dependent allocation
struct allocation_model
{
    /// A name of the target-dependent allocate operator
    std::string name() const;
    /// A name of the target-dependent copy operator
    std::string copy() const;
    /// Create an allocation operator for the given shape
    operation allocate(const shape& s) const;
    /// Create a preallocated operator for the given shape
    operation preallocate(const shape& s, const std::string& id) const;
    /// Check if outputs are to be inserted
    bool needs_out_params() const;
};

#else

<%
interface('allocation_model',
    virtual('name', returns='std::string', const=True),
    virtual('copy', returns='std::string', const=True),
    virtual('allocate', s='const shape&', returns='operation', const=True),
    virtual('preallocate', s='const shape&', id='std::string', returns='operation', const=True),
    virtual('needs_out_params', returns='bool', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
