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
#ifndef MIGRAPHX_GUARD_STREAM_MODEL_HPP
#define MIGRAPHX_GUARD_STREAM_MODEL_HPP

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

#ifdef DOXYGEN

/// An interface for target-dependent model for the scheduler
struct stream_model
{
    /// Get the number of streams used in the program
    std::size_t get_nstream() const;
    /// Get stream for instruction
    std::size_t get_stream(instruction_ref ins) const;
    /// Get unique event id for instruction
    std::size_t get_event_id(instruction_ref ins) const;
    /// Returns true if instruction has a stream assignment
    bool has_stream(instruction_ref ins) const;
    /// Returns true if the instruction records the event
    bool is_record(instruction_ref ins) const;
    /// Returns true if the instruction wait on the event
    bool is_wait(instruction_ref ins) const;
};

#else

<%
interface('stream_model',
    virtual('get_nstream', returns='std::size_t', const=True),
    virtual('get_stream', ins='instruction_ref', returns='std::size_t', const=True),
    virtual('get_event_id', ins='instruction_ref', returns='std::size_t', const=True),
    virtual('has_stream', ins='instruction_ref', returns='bool', const=True),
    virtual('is_record', ins='instruction_ref', returns='bool', const=True),
    virtual('is_wait', ins='instruction_ref', returns='bool', const=True)
)
%>

#endif

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
