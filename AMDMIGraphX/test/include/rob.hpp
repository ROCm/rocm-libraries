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
#ifndef MIGRAPHX_GUARD_ROB_HPP
#define MIGRAPHX_GUARD_ROB_HPP

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

// Used to access private member variables
template <class Tag>
struct stowed
{
    // NOLINTNEXTLINE
    static typename Tag::type value;
};
template <class Tag>
// NOLINTNEXTLINE
typename Tag::type stowed<Tag>::value;

template <class Tag, typename Tag::type X>
struct stow_private
{
    stow_private() noexcept { stowed<Tag>::value = X; }
    // NOLINTNEXTLINE
    static stow_private instance;
};
template <class Tag, typename Tag::type X>
// NOLINTNEXTLINE
stow_private<Tag, X> stow_private<Tag, X>::instance;

template <class C, class T>
struct mem_data_ptr
{
    using type = T C::*;
};

// NOLINTNEXTLINE
#define MIGRAPHX_ROB(name, Type, C, mem)               \
    struct name##_tag : mem_data_ptr<C, Type>          \
    {                                                  \
    };                                                 \
    template struct stow_private<name##_tag, &C::mem>; \
    template <class T>                                 \
    static auto& name(T && x)                          \
    {                                                  \
        return x.*stowed<name##_tag>::value;           \
    }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif
