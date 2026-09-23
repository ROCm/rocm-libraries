/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <cassert>

// isa<X>(ptr): true if ptr can be dynamically cast to X*
template <typename To, typename From>
inline bool isa(const From* ptr) {
    return To::classof(ptr);
}

// cast<X>(ptr): dynamically cast ptr to X*. Asserts if !isa<X>(ptr).
template <typename To, typename From>
inline To* cast(From* ptr) {
    assert(isa<To>(ptr) && "cast<Ty>() argument of incompatible type!");
    return static_cast<To*>(ptr);
}

template <typename To, typename From>
inline const To* cast(const From* ptr) {
    assert(isa<To>(ptr) && "cast<Ty>() argument of incompatible type!");
    return static_cast<const To*>(ptr);
}

// dyn_cast<X>(ptr): dynamically cast ptr to X* if isa<X>(ptr), else nullptr.
template <class To, class From>
inline To* dyn_cast(From* ptr) {
    return isa<To>(ptr) ? cast<To>(ptr) : nullptr;
}

template <class To, class From>
inline const To* dyn_cast(const From* ptr) {
    return isa<To>(ptr) ? cast<To>(ptr) : nullptr;
}
