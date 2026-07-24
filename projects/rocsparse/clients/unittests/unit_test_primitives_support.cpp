/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

//
// Link-seam support for unit_test_primitives.cpp.
//
// The rocsparse::primitives translation units under test are compiled directly
// into the rocsparse-unit-test-device binary (their explicit instantiations are
// hidden in librocsparse and cannot be linked from the .so). Those TUs use the
// RETURN_IF_HIP_ERROR macro, which -- only on a HIP error path -- references two
// small diagnostic helpers that are likewise hidden in the library:
//
//   * rocsparse::get_rocsparse_status_for_hip_status(hipError_t)
//   * rocsparse::error_message(rocsparse_status, const char*, ...)
//
// Rather than dragging the entire library/src/common/rocsparse_handle.cpp (which
// also defines the public handle C API and pulls a large dependency chain) into
// this test binary, we provide the minimal definitions here so the primitive
// code links. This is a standard unit-test "link seam"; it does not stub or fake
// any primitive under test -- the primitives run their real rocprim code. These
// helpers are only invoked when a HIP call fails (never on the passing paths),
// and error_message is a no-op unless the library's verbose debug env var is set,
// matching the library default.
//
// get_rocsparse_status_for_hip_status mirrors the library mapping so that, if a
// primitive ever does hit a HIP error during a test, the returned status is
// still meaningful.
//

#include "rocsparse_message.hpp" // declarations of warning_message / error_message

#include <hip/hip_runtime.h>

namespace rocsparse
{
    rocsparse_status get_rocsparse_status_for_hip_status(hipError_t status);

    rocsparse_status get_rocsparse_status_for_hip_status(hipError_t status)
    {
        switch(status)
        {
        case hipSuccess:
            return rocsparse_status_success;

        case hipErrorMemoryAllocation:
        case hipErrorLaunchOutOfResources:
            return rocsparse_status_memory_error;

        case hipErrorInvalidDevicePointer:
            return rocsparse_status_invalid_pointer;

        case hipErrorInvalidDevice:
        case hipErrorInvalidResourceHandle:
            return rocsparse_status_invalid_handle;

        case hipErrorInvalidValue:
            return rocsparse_status_internal_error;

        case hipErrorNoDevice:
        case hipErrorUnknown:
        default:
            return rocsparse_status_internal_error;
        }
    }

    void warning_message(const char*, const char*, const char*, int) { }

    void error_message(rocsparse_status, const char*, const char*, const char*, int) { }
}

//
// Compile-in seam for the segmented radix sort primitives.
//
// unit_test_primitives.cpp exercises rocsparse::primitives::segmented_radix_sort_keys
// and segmented_radix_sort_pairs, but their translation units are not listed in
// ROCSPARSE_UNIT_TEST_PRIMITIVE_SOURCES (CMakeLists.txt) and their explicit
// instantiations are hidden in librocsparse. Rather than modify the build, we
// pull those two .cpp files in here directly -- the same "compile the library
// .cpp into the test target" technique already used for the other primitives via
// the CMake source list. Their only extra dependencies (rocprim, the static
// inline rocsparse::clz, and the plain-hip rocsparse_hipMemcpyAsync macro in a
// non-debug/non-memstat build) resolve without any additional library object.
//
#include "../../library/src/primitives/rocsparse_segmented_radix_sort_keys.cpp"
#include "../../library/src/primitives/rocsparse_segmented_radix_sort_pairs.cpp"
