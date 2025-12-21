// Copyright (C) 2024, 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_SCOPE_EXIT_H
#define ROCFFT_SCOPE_EXIT_H

#include <utility>

// A simple scope_exit class that executes a function when it goes out of scope.
// This is a replacement for boost::scope_exit.
template <typename Func>
class scope_exit
{
public:
    explicit scope_exit(Func&& func)
        : func_(std::forward<Func>(func))
        , active_(true)
    {
    }

    scope_exit(scope_exit&& other) noexcept
        : func_(std::move(other.func_))
        , active_(other.active_)
    {
        other.active_ = false;
    }

    ~scope_exit()
    {
        if(active_)
        {
            func_();
        }
    }

    scope_exit(const scope_exit&)            = delete;
    scope_exit& operator=(const scope_exit&) = delete;
    scope_exit& operator=(scope_exit&&)      = delete;

    void release()
    {
        active_ = false;
    }

private:
    Func func_;
    bool active_;
};

// Helper function to create a scope_exit with type deduction
template <typename Func>
scope_exit<Func> make_scope_exit(Func&& func)
{
    return scope_exit<Func>(std::forward<Func>(func));
}

// Macro to simplify usage - captures all variables by value like BOOST_SCOPE_EXIT_ALL(=)
// Usage: SCOPE_EXIT { cleanup_code; };
// Note: This captures by reference for more flexibility; use [=] in the lambda if copy is needed
#define ROCFFT_SCOPE_EXIT_CONCAT_(a, b) a##b
#define ROCFFT_SCOPE_EXIT_CONCAT(a, b)  ROCFFT_SCOPE_EXIT_CONCAT_(a, b)
#define ROCFFT_SCOPE_EXIT               auto ROCFFT_SCOPE_EXIT_CONCAT(_scope_exit_, __LINE__) = make_scope_exit

#endif // ROCFFT_SCOPE_EXIT_H
