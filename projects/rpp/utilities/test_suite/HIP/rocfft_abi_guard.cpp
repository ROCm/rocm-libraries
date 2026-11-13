/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

/*! \file rocfft_abi_guard.cpp
 *  \brief Compile-time check that RPP's vendored rocFFT declarations still match rocFFT.
 *
 *  RPP loads rocFFT at runtime instead of linking it, so it reproduces the handful of rocFFT
 *  declarations it needs in src/modules/rocfft_dlopen/ rather than including
 *  <rocfft/rocfft.h>. Nothing in a normal RPP build compares those copies to the originals,
 *  and both kinds of copy can drift silently:
 *
 *    - the enums, because rocFFT spells its enumerators without explicit values, so RPP's
 *      constants are positional. Insert an enumerator upstream and RPP keeps compiling while
 *      quietly asking for the wrong transform type or array layout.
 *    - the prototypes, because a dlsym'd address is cast to whatever RPP declared. A dropped
 *      or transposed parameter is never diagnosed anywhere; the call simply goes out wrong.
 *
 *  This file is the backstop for both. It is the one place that includes the real rocFFT
 *  header alongside RPP's copies and static_asserts them against each other, so that drift
 *  breaks this build instead of a user's spectrogram. It needs the rocFFT development
 *  headers, and is therefore built only when they are present - it is a test-suite check, not
 *  a dependency of the library.
 *
 *  Scope: enumerator values, enum sizes, the opaque handle types, and the full signature of
 *  every entry point RPP calls.
 */

#include <rocfft/rocfft.h>
// For rocfft_version_*, which rocfft.h does not pull in. Reporting the version from macros
// rather than rocfft_get_version_string() keeps this a pure compile-time check that needs no
// link against librocfft.
#include <rocfft/rocfft-version.h>

#include <cstdio>
#include <type_traits>

// RPP's copy of the enums and handles, kept in a namespace so the two can coexist in one
// translation unit. The header declares types only and includes nothing, which is what makes
// this safe; see the warning at the top of rocfft_vendored_types.h.
namespace rpp_vendored {
#include "rocfft_vendored_types.h"
}  // namespace rpp_vendored

// RPP's prototypes, at global scope, where the type names in them resolve to the real rocFFT
// declarations from <rocfft/rocfft.h> above rather than to the vendored ones. Instantiating
// them against real types is the whole point: a vendored instantiation could never compare
// equal to anything, because rpp_vendored::rocfft_status is a distinct C++ type from
// ::rocfft_status whatever its values are, so is_same would be false for reasons that say
// nothing about drift. Rebuilding the parameter lists out of real rocFFT types removes that
// difference and leaves only the shape to compare - which is what the asserts below do.
//
// Include at global scope only. Inside namespace rpp_vendored these same names would still
// find the real rocFFT types by enclosing-scope lookup unless rocfft_vendored_types.h came
// first, so a vendored instantiation would compile either way and quietly mean whichever the
// include order made it mean. Nothing here needs one, so there is none.
// extern "C" to match the linkage the real header declares its functions with.
extern "C" {
#include "rocfft_vendored_prototypes.h"
}

// Every enumerator RPP reproduces, checked against the real header. A failure here means
// rocFFT's enumerators moved and src/modules/rocfft_dlopen/rocfft_vendored_types.h must be
// updated to match - do not silence it.
#define RPP_ASSERT_ROCFFT_ENUM(name)                                                \
    static_assert(static_cast<int>(::name) == static_cast<int>(rpp_vendored::name), \
                  "rocFFT enumerator " #name                                        \
                  " no longer matches RPP's vendored copy; "                        \
                  "update src/modules/rocfft_dlopen/rocfft_vendored_types.h")

RPP_ASSERT_ROCFFT_ENUM(rocfft_status_success);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_failure);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_arg_value);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_dimensions);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_array_type);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_strides);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_distance);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_offset);
RPP_ASSERT_ROCFFT_ENUM(rocfft_status_invalid_work_buffer);

RPP_ASSERT_ROCFFT_ENUM(rocfft_transform_type_complex_forward);
RPP_ASSERT_ROCFFT_ENUM(rocfft_transform_type_complex_inverse);
RPP_ASSERT_ROCFFT_ENUM(rocfft_transform_type_real_forward);
RPP_ASSERT_ROCFFT_ENUM(rocfft_transform_type_real_inverse);

RPP_ASSERT_ROCFFT_ENUM(rocfft_precision_single);
RPP_ASSERT_ROCFFT_ENUM(rocfft_precision_double);
RPP_ASSERT_ROCFFT_ENUM(rocfft_precision_half);

RPP_ASSERT_ROCFFT_ENUM(rocfft_placement_inplace);
RPP_ASSERT_ROCFFT_ENUM(rocfft_placement_notinplace);

RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_complex_interleaved);
RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_complex_planar);
RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_real);
RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_hermitian_interleaved);
RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_hermitian_planar);
RPP_ASSERT_ROCFFT_ENUM(rocfft_array_type_unset);

#undef RPP_ASSERT_ROCFFT_ENUM

// The enumerators above are compared as int, which would not notice rocFFT widening an enum
// past int or giving it a fixed underlying type of a different size - either would change how
// the values are passed.
#define RPP_ASSERT_ROCFFT_ENUM_SIZE(type)                       \
    static_assert(sizeof(::type) == sizeof(rpp_vendored::type), \
                  "rocFFT enum " #type                          \
                  " changed size; update "                      \
                  "src/modules/rocfft_dlopen/rocfft_vendored_types.h")

RPP_ASSERT_ROCFFT_ENUM_SIZE(rocfft_status);
RPP_ASSERT_ROCFFT_ENUM_SIZE(rocfft_transform_type);
RPP_ASSERT_ROCFFT_ENUM_SIZE(rocfft_precision);
RPP_ASSERT_ROCFFT_ENUM_SIZE(rocfft_result_placement);
RPP_ASSERT_ROCFFT_ENUM_SIZE(rocfft_array_type);

#undef RPP_ASSERT_ROCFFT_ENUM_SIZE

// The three handles RPP passes through. These are pointers to incomplete types on both sides,
// so the check is that they are still opaque pointers rather than something RPP would have to
// know the layout of.
static_assert(std::is_pointer<::rocfft_plan>::value &&
                  std::is_pointer<rpp_vendored::rocfft_plan>::value,
              "rocfft_plan is no longer an opaque pointer");
static_assert(std::is_pointer<::rocfft_plan_description>::value &&
                  std::is_pointer<rpp_vendored::rocfft_plan_description>::value,
              "rocfft_plan_description is no longer an opaque pointer");
static_assert(std::is_pointer<::rocfft_execution_info>::value &&
                  std::is_pointer<rpp_vendored::rocfft_execution_info>::value,
              "rocfft_execution_info is no longer an opaque pointer");

// Every entry point RPP resolves with dlsym, with the signature it is called through compared
// against the real one. This catches what the enum checks cannot: a wrong parameter count or
// order, a missing pointer or const, a changed return type. A failure here means rocFFT
// changed an API RPP calls, and that RPP is calling it with the wrong arguments.
#define RPP_ASSERT_ROCFFT_SIGNATURE(name)                                   \
    static_assert(std::is_same<decltype(&::name), ::fp_##name>::value,      \
                  "rocFFT's " #name                                         \
                  " no longer has the signature RPP calls it with; update " \
                  "src/modules/rocfft_dlopen/rocfft_vendored_prototypes.h")

RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_setup);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_cleanup);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_create);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_destroy);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_get_work_buffer_size);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_description_create);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_description_destroy);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_plan_description_set_data_layout);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_execution_info_create);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_execution_info_destroy);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_execution_info_set_stream);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_execution_info_set_work_buffer);
RPP_ASSERT_ROCFFT_SIGNATURE(rocfft_execute);

#undef RPP_ASSERT_ROCFFT_SIGNATURE

int main() {
    // Reaching main() means every assertion above held at compile time; there is nothing left
    // to check at runtime. Printing the rocFFT the check ran against makes the result useful
    // in a CTest log, where "passed" on its own would not say which header was compared.
    std::printf(
        "rocfft_abi_guard: RPP's vendored rocFFT enums and prototypes match rocFFT "
        "%d.%d.%d\n",
        rocfft_version_major, rocfft_version_minor, rocfft_version_patch);
    return 0;
}
