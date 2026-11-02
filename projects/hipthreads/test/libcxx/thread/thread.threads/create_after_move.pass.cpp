// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// UNSUPPORTED: no-threads

// <thread>

// class wthread

// wthread(wthread&& t);

// Regression test for https://github.com/ROCm/rocm-libraries/issues/13626
// The host move constructor previously did not count the new hip::wthread as a live object. Once the moved-to and
// moved-from objects were destroyed, the scheduler kernel was never launched again, so join() on any hip::wthread
// created afterwards never returned.

#include <hip/thread>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

struct F
{
    __device__ void operator()() const {}
};

int main(int, char**)
{
    {
        hip::wthread t0 = support::make_test_thread(F());
        hip::wthread t1 = ::std::move(t0);
        t1.join();
    }
    // t0 and t1 are destroyed. A hip::wthread created after that must still run.
    hip::wthread t2 = support::make_test_thread(F());
    t2.join();

    return 0;
}
