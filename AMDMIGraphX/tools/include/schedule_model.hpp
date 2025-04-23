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
#ifndef MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP
#define MIGRAPHX_GUARD_SCHEDULE_MODEL_HPP

#include <cassert>
#include <string>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct operation;

#ifdef DOXYGEN

/// An interface for target-dependent model for the scheduler
struct schedule_model
{
    /// Get the number of concurrent instruction allowed
    std::size_t concurrency() const;
    /// Schedule a concurrent instruction
    void sched(module& m, instruction_ref ins, std::size_t n) const;
    // Insert necessary waits before an instruction
    void wait(module& m, instruction_ref ins, std::size_t wait_id) const;
    // Insert necessary records after an instruction
    void record(module& m, instruction_ref ins, std::size_t wait_id) const;
    /// Compute weights for an operation
    std::size_t weight(const operation& op) const;
};

#else

<%
interface('schedule_model',
    virtual('concurrency', returns='std::size_t', const=True),
    virtual('sched', m='module&', ins='instruction_ref', n='std::size_t', const=True),
    virtual('wait', m='module&', ins='instruction_ref', wait_id='std::size_t', const=True),
    virtual('record', m='module&', ins='instruction_ref', wait_id='std::size_t', const=True),
    virtual('weight', returns='std::size_t', op='const operation&', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
