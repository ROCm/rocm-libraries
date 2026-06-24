/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#ifndef ROCM_GUARD_ROCM_ASSERT_HPP
#define ROCM_GUARD_ROCM_ASSERT_HPP

#include <rocm/config.hpp>
#include <rocm/stdint.hpp>

#ifndef __HIPCC_RTC__
#include <cstdlib>
#include <cstdio>
#endif

namespace rocm {
inline namespace ROCM_INLINE_NS {

// Workaround hip's broken abort on device code
#ifdef __HIP_DEVICE_COMPILE__
// NOLINTNEXTLINE
#define ROCM_HIP_NORETURN
#else
// NOLINTNEXTLINE
#define ROCM_HIP_NORETURN [[noreturn]]
#endif

namespace debug {
struct swallow
{
    template <class... Ts>
    constexpr swallow(Ts&&...)
    {
    }
};

template <size_t N>
struct print_buffer
{
    char buffer[N + 1] = {0};
    char* pos          = buffer;

    constexpr void append(char c)
    {
        if(c == 0)
            return;
        if(pos < buffer + N)
        {
            *pos = c;
            pos++;
        }
    }
    static constexpr void reverse(char* first, char* last)
    {
        if(first == last)
            return;
        last--;
        while(first < last)
        {
            char tmp = *first;
            *first   = *last;
            *last    = tmp;
            first++;
            last--;
        }
    }

    template <class T, class = decltype(T{} % 10, -T{})>
    constexpr void append(T i)
    {
        if(i < 0)
        {
            append('-');
            i = -i;
        }
        if(i == 0)
        {
            append('0');
            return;
        }
        char* start = pos;
        while(i != 0)
        {
            char c = (i % 10) + '0';
            append(c);
            i = i / 10;
        }
        reverse(start, pos);
    }

    constexpr void append(const char* str)
    {
        if(str == nullptr)
            return;
        int i = 512;
        while(*str != 0 and i > 0)
        {
            append(*str);
            str++;
            i--;
        }
    }

    template <size_t M>
    constexpr void append(const char (&array)[M])
    {
        for(int i = 0; i < M; i++)
            append(array[i]);
    }
};

template <class... Ts>
ROCM_HIP_HOST_DEVICE void print(const Ts&... xs)
{
    print_buffer<1024> buffer;
    swallow{(buffer.append(xs), 0)...};
    printf("%s", buffer.buffer);
}

} // namespace debug

// noreturn cannot be used on this function because abort in hip is broken
template <class T1, class T2, class T3, class T4>
ROCM_HIP_NORETURN inline ROCM_HIP_HOST_DEVICE void
assert_fail(const T1& assertion, const T2& file, const T3& line, const T4& function)
{
    // printf is broken on hip with more than one argument, so use a simple print functions instead
    debug::print(file, ":", line, ": ", function, ": assertion '", assertion, "' failed.\n");
    // printf("%s:%s: %s: assertion '%s' failed.\n", file, line, function, assertion);
    abort();
}

// NOLINTNEXTLINE
#define ROCM_ASSERT_FAIL(cond, ...)                         \
    ((cond) ? void(0) : [](auto&&... private_migraphx_xs) { \
        assert_fail(private_migraphx_xs...);                \
    }(__VA_ARGS__))

// NOLINTNEXTLINE
#define ROCM_CHECK(cond) ROCM_ASSERT_FAIL(cond, #cond, __FILE__, __LINE__, __PRETTY_FUNCTION__)

#ifdef ROCM_DEBUG
// NOLINTNEXTLINE
#define ROCM_ASSERT ROCM_CHECK
#define ROCM_ASSUME ROCM_CHECK
#define ROCM_UNREACHABLE() ROCM_ASSERT(false)
#else
// NOLINTNEXTLINE
#define ROCM_ASSUME __builtin_assume
#define ROCM_UNREACHABLE __builtin_unreachable
#define ROCM_ASSERT(cond)
#endif

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_ASSERT_HPP
