/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <rpp.h>

#include <cstdio>

// In a HOST build nothing puts RPP_BACKEND_HIP on the command line, so its
// value reaches this file only through the generated rpp_backend.h -- the same
// view a downstream application has. That is where a `#ifdef RPP_BACKEND_HIP`
// guard wrongly takes the HIP branch, so compiling this file is itself the
// regression check.

#ifndef RPP_BACKEND_HIP
#error "RPP_BACKEND_HIP is undefined -- rpp.h did not include the generated rpp_backend.h"
#endif

// rpp_backend.h is generated with #cmakedefine01, so the macro is always
// defined and carries the backend in its value. Guards must use #if, not #ifdef.
static_assert(RPP_BACKEND_HIP == 0 || RPP_BACKEND_HIP == 1,
              "RPP_BACKEND_HIP must expand to 0 or 1");

static_assert(RPP_BACKEND_HIP == EXPECTED_RPP_BACKEND_HIP,
              "RPP_BACKEND_HIP does not match the backend RPP was configured with");

int main() {
    rppHandle_t handle = nullptr;

    if (rppCreate(&handle, 1, 0, nullptr, RPP_HOST_BACKEND) != rppStatusSuccess ||
        handle == nullptr) {
        std::printf("FAIL: rppCreate() with RPP_HOST_BACKEND did not succeed\n");
        return 1;
    }

    if (rppDestroy(handle, RPP_HOST_BACKEND) != rppStatusSuccess) {
        std::printf("FAIL: rppDestroy() with RPP_HOST_BACKEND did not succeed\n");
        return 1;
    }

    std::printf("PASS: RPP_BACKEND_HIP=%d, HOST handle created and destroyed\n", RPP_BACKEND_HIP);
    return 0;
}
