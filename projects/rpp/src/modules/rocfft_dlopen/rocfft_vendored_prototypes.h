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

/*! \file rocfft_vendored_prototypes.h
 *  \brief The rocFFT entry points RPP calls, as function pointer types.
 *
 *  These reproduce the prototypes in <rocfft/rocfft.h> so that RPP can bind rocFFT at runtime
 *  without its headers. See rocfft_runtime.hpp for the rationale.
 *
 *  This file is deliberately NOT include-guarded. It names its parameter types but never
 *  defines them, so the types it produces depend on which declarations of those names are in
 *  scope where it is included, and it has two inclusion sites that differ in exactly that:
 *
 *    - rocfft_runtime.hpp includes it after rocfft_vendored_types.h and inside namespace
 *      rpp::rocfft_dlopen, where the names resolve to RPP's vendored types. The pointers RPP
 *      actually calls rocFFT through come from that inclusion.
 *    - utilities/test_suite/HIP/rocfft_abi_guard.cpp includes it at global scope, below
 *      <rocfft/rocfft.h>, where the same names instead resolve to the real rocFFT
 *      declarations. That rebuilds the signatures below out of real rocFFT types, which can
 *      then be compared to the real rocFFT functions with static_assert. Instantiating them
 *      against the vendored types and comparing that would prove nothing, because a vendored
 *      enum is a distinct C++ type from the rocFFT one however identical its values are.
 *
 *  Because the names bind to whatever is in scope, an inclusion in the wrong place compiles
 *  just as happily and silently means something else. Keep the guard's inclusion at global
 *  scope, and keep rocfft_runtime.hpp's below rocfft_vendored_types.h.
 *
 *  That check covers what a transcription can get wrong and the enum guard cannot see:
 *  parameter count, parameter order, pointer and const qualification, and return type. A
 *  mistake in any of them would otherwise reach a dlsym'd pointer uncaught, since nothing
 *  else compares these declarations to rocFFT's.
 *
 *  Keep every declaration here a typedef, and keep the file free of #includes - both
 *  inclusion sites depend on both properties. Having no #includes means it must be included
 *  somewhere size_t is already declared; <stddef.h> is enough, and is all <rocfft/rocfft.h>
 *  itself relies on.
 */

typedef rocfft_status (*fp_rocfft_setup)();
typedef rocfft_status (*fp_rocfft_cleanup)();
typedef rocfft_status (*fp_rocfft_plan_create)(rocfft_plan* plan, rocfft_result_placement placement,
                                               rocfft_transform_type transform_type,
                                               rocfft_precision precision, size_t dimensions,
                                               const size_t* lengths, size_t number_of_transforms,
                                               const rocfft_plan_description description);
typedef rocfft_status (*fp_rocfft_plan_destroy)(rocfft_plan plan);
typedef rocfft_status (*fp_rocfft_plan_get_work_buffer_size)(const rocfft_plan plan,
                                                             size_t* size_in_bytes);
typedef rocfft_status (*fp_rocfft_plan_description_create)(rocfft_plan_description* description);
typedef rocfft_status (*fp_rocfft_plan_description_destroy)(rocfft_plan_description description);
typedef rocfft_status (*fp_rocfft_plan_description_set_data_layout)(
    rocfft_plan_description description, const rocfft_array_type in_array_type,
    const rocfft_array_type out_array_type, const size_t* in_offsets, const size_t* out_offsets,
    const size_t in_strides_size, const size_t* in_strides, const size_t in_distance,
    const size_t out_strides_size, const size_t* out_strides, const size_t out_distance);
typedef rocfft_status (*fp_rocfft_execution_info_create)(rocfft_execution_info* info);
typedef rocfft_status (*fp_rocfft_execution_info_destroy)(rocfft_execution_info info);
typedef rocfft_status (*fp_rocfft_execution_info_set_stream)(rocfft_execution_info info,
                                                             void* stream);
typedef rocfft_status (*fp_rocfft_execution_info_set_work_buffer)(rocfft_execution_info info,
                                                                  void* work_buffer,
                                                                  const size_t size_in_bytes);
typedef rocfft_status (*fp_rocfft_execute)(const rocfft_plan plan, void* in_buffer[],
                                           void* out_buffer[], rocfft_execution_info info);
