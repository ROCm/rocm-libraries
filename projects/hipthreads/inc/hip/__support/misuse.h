// -*- C++ -*-

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef __LIBHIPTHREADS___SUPPORT_MISUSE_H__
#define __LIBHIPTHREADS___SUPPORT_MISUSE_H__

#include "hip/hip_runtime.h"
#include <cstdio>

// Device-side detection of library misuse (e.g. destroying a still-joinable
// hip::wthread, recursive-locking a non-recursive mutex, unlocking a mutex the
// caller does not own). These conditions used to be guarded by assert(), which
// is compiled out under NDEBUG, so in release builds misuse silently became
// undefined behavior (UB / livelock / state corruption). The hook below detects
// misuse in *every* build type.
//
// This lives in a header (rather than src/hip/thread.cxx) because most misuse
// sites are in header-only code (the mutex / lock primitives). A header-only
// consumer compiled with -DNDEBUG would strip an assert no matter how the
// library archive was built, so the detection itself must be header-visible and
// independent of NDEBUG. The state and handler are `inline __device__` so they
// merge to a single instance across translation units under -fgpu-rdc (ODR),
// exactly like a normal inline variable/function.

namespace cuda {

// Test hook. In normal use both stay at their defaults and misuse aborts via
// __builtin_trap(). A test can set __hipthreads_misuse_nonfatal = true so that
// misuse is *recorded* (incrementing __hipthreads_misuse_count) and the
// offending operation is skipped instead of trapping, letting the test observe
// that detection happened and then exit cleanly.
inline __device__ bool __hipthreads_misuse_nonfatal = false;
inline __device__ unsigned int __hipthreads_misuse_count = 0;

// Single device-side path for library misuse. In normal use this never returns
// (it aborts via __builtin_trap()). In test/nonfatal mode it records the misuse
// and returns, and the call site is responsible for skipping the offending
// operation (see the __HIPTHREADS_REPORT_MISUSE call sites).
//
// `what` is a full, self-contained description supplied by the call site.
// Device code cannot use fprintf/stderr (stderr is a __host__ symbol), so the
// message goes to printf (the HIP device-printf stream). Call sites should use
// the __HIPTHREADS_REPORT_MISUSE macro below so __FILE__/__LINE__/__func__ are
// captured automatically at the point of misuse.
inline __device__ void __hipthreads_report_misuse(const char *what, const char *file, unsigned int line, const char *func) {
    if (__hipthreads_misuse_nonfatal) {
        atomicAdd(&__hipthreads_misuse_count, 1u);
        return;
    }
    // \033[1;31m = bold red, \033[0m = reset, so the fatal message stands out.
    printf("\033[1;31m"
           "hipThreads FATAL: %s\n"
           "  at %s:%u in %s"
           "\033[0m\n",
           what, file, line, func);
    __builtin_trap();
}

} // namespace cuda

// Wrap __hipthreads_report_misuse so each call site bakes in its own source
// location. The do/while(false) makes the macro a single statement that
// requires a trailing semicolon. Internal-only; not part of the public API.
#define __HIPTHREADS_REPORT_MISUSE(what)                                                                                \
    do {                                                                                                                \
        ::cuda::__hipthreads_report_misuse((what), __FILE__, __LINE__, __func__);                                       \
    } while (false)

// Convenience wrapper for the common "invariant must hold, otherwise it's
// misuse" pattern. Like assert(), `cond` is the condition that must be TRUE;
// when it is false, the misuse is reported and (in nonfatal mode, where the
// report returns) the enclosing function returns early. It therefore only fits
// sites whose nonfatal recovery is a bare `return;` -- sites that need a
// different recovery (e.g. `return *this;`, `detach()`, `break`) must use
// __HIPTHREADS_REPORT_MISUSE directly. Internal-only; not part of the public API.
#define __HIPTHREADS_ASSERT(cond, what)                                                                                 \
    do {                                                                                                                \
        if (!(cond)) [[unlikely]] {                                                                                     \
            __HIPTHREADS_REPORT_MISUSE(what);                                                                           \
            return;                                                                                                     \
        }                                                                                                              \
    } while (false)

#endif // __LIBHIPTHREADS___SUPPORT_MISUSE_H__
