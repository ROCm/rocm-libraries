/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>

// Feature detection for std::span
#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#include <span>
#define STINKYTOFU_HAS_STD_SPAN 1
#else
#define STINKYTOFU_HAS_STD_SPAN 0
#endif

namespace stinkytofu {

#if STINKYTOFU_HAS_STD_SPAN

// Use standard std::span when available
template <typename T>
using span = std::span<T>;

#else

// ========================================================================
// Fallback implementation of span (minimal subset for our use case)
// ========================================================================

/// @brief Lightweight non-owning view over a contiguous sequence
///
/// This is a minimal implementation providing only the features we need.
/// When C++20 std::span is available, it's used instead.
///
/// Key differences from std::span:
/// - No dynamic extent support (always fixed at construction)
/// - No subspan operations
/// - Simplified iterator support
template <typename T>
class span {
   public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;

    // ====================================================================
    // Constructors
    // ====================================================================

    /// Default constructor - empty span
    constexpr span() noexcept : data_(nullptr), size_(0) {}

    /// Construct from pointer and size
    constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}

    /// Construct from C array
    template <std::size_t N>
    constexpr span(element_type (&arr)[N]) noexcept : data_(arr), size_(N) {}

    /// Construct from const C array (for span<const T>)
    template <std::size_t N, typename U = T, typename = std::enable_if_t<std::is_const<U>::value> >
    constexpr span(std::remove_const_t<element_type> (&arr)[N]) noexcept : data_(arr), size_(N) {}

    // ====================================================================
    // Element access
    // ====================================================================

    constexpr reference operator[](size_type idx) const noexcept {
        return data_[idx];
    }

    constexpr reference front() const noexcept {
        return data_[0];
    }

    constexpr reference back() const noexcept {
        return data_[size_ - 1];
    }

    constexpr pointer data() const noexcept {
        return data_;
    }

    // ====================================================================
    // Iterators
    // ====================================================================

    constexpr iterator begin() const noexcept {
        return data_;
    }

    constexpr iterator end() const noexcept {
        return data_ + size_;
    }

    constexpr const_iterator cbegin() const noexcept {
        return data_;
    }

    constexpr const_iterator cend() const noexcept {
        return data_ + size_;
    }

    // ====================================================================
    // Capacity
    // ====================================================================

    constexpr size_type size() const noexcept {
        return size_;
    }

    constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(element_type);
    }

    constexpr bool empty() const noexcept {
        return size_ == 0;
    }

   private:
    pointer data_;
    size_type size_;
};

// ========================================================================
// Deduction guides (C++17)
// ========================================================================

template <typename T, std::size_t N>
span(T (&)[N]) -> span<T>;

template <typename T, std::size_t N>
span(const T (&)[N]) -> span<const T>;

#endif  // STINKYTOFU_HAS_STD_SPAN

}  // namespace stinkytofu
