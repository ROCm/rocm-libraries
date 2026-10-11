/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 AIwork4me
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#pragma once

/// \file Freestanding std::initializer_list for MIOpen runtime-compiled
/// kernels (clang lowers braced-init-lists to field stores on this exact
/// layout — verified by Phase-3 canary G57-7 on gfx1151/hiprtc 7.14).
/// Selected by tensor_view.hpp only when no real <initializer_list> is
/// reachable; see miopen_freestanding_type_traits.hpp for the rationale.

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#error "miopen_freestanding_initializer_list.hpp is for runtime-compiled kernels only"
#endif

#if !defined(__clang__)
#error "miopen_freestanding_initializer_list.hpp relies on clang initializer_list lowering"
#endif

namespace std {

template <class E>
class initializer_list
{
    const E* __begin_;
    __SIZE_TYPE__ __size_;

    constexpr initializer_list(const E* b, __SIZE_TYPE__ s) noexcept : __begin_(b), __size_(s) {}

public:
    constexpr initializer_list() noexcept : __begin_(nullptr), __size_(0) {}

    constexpr const E* begin() const noexcept { return __begin_; }
    constexpr const E* end() const noexcept { return __begin_ + __size_; }
    constexpr __SIZE_TYPE__ size() const noexcept { return __size_; }
};

} // namespace std
