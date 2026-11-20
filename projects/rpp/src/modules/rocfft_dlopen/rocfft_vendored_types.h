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

/*! \file rocfft_vendored_types.h
 *  \brief The subset of <rocfft/rocfft.h> types that RPP reproduces for its dlopen binding.
 *
 *  Included by rocfft_runtime.hpp so that RPP compiles without the rocFFT headers. See
 *  that file for the rationale.
 *
 *  WARNING: rocFFT declares these enumerators WITHOUT explicit values, so every constant
 *  below is positional - inserting an enumerator upstream would silently change what these
 *  names mean here. utilities/test_suite/HIP/rocfft_abi_guard.cpp includes this file
 *  alongside the real <rocfft/rocfft.h> and static_asserts the two against each other, so
 *  that such drift fails a build rather than corrupting a transform at runtime. Keep this
 *  file self-contained and free of #includes: the guard includes it inside a namespace to
 *  keep the two sets of declarations apart, which is only safe while that holds.
 */

#ifndef RPP_ROCFFT_DLOPEN_ROCFFT_VENDORED_TYPES_H
#define RPP_ROCFFT_DLOPEN_ROCFFT_VENDORED_TYPES_H

typedef enum rocfft_status_e {
    rocfft_status_success = 0,
    rocfft_status_failure = 1,
    rocfft_status_invalid_arg_value = 2,
    rocfft_status_invalid_dimensions = 3,
    rocfft_status_invalid_array_type = 4,
    rocfft_status_invalid_strides = 5,
    rocfft_status_invalid_distance = 6,
    rocfft_status_invalid_offset = 7,
    rocfft_status_invalid_work_buffer = 8,
} rocfft_status;

typedef enum rocfft_transform_type_e {
    rocfft_transform_type_complex_forward = 0,
    rocfft_transform_type_complex_inverse = 1,
    rocfft_transform_type_real_forward = 2,
    rocfft_transform_type_real_inverse = 3,
} rocfft_transform_type;

typedef enum rocfft_precision_e {
    rocfft_precision_single = 0,
    rocfft_precision_double = 1,
    rocfft_precision_half = 2,
} rocfft_precision;

typedef enum rocfft_result_placement_e {
    rocfft_placement_inplace = 0,
    rocfft_placement_notinplace = 1,
} rocfft_result_placement;

typedef enum rocfft_array_type_e {
    rocfft_array_type_complex_interleaved = 0,
    rocfft_array_type_complex_planar = 1,
    rocfft_array_type_real = 2,
    rocfft_array_type_hermitian_interleaved = 3,
    rocfft_array_type_hermitian_planar = 4,
    rocfft_array_type_unset = 5,
} rocfft_array_type;

typedef struct rocfft_plan_t* rocfft_plan;
typedef struct rocfft_plan_description_t* rocfft_plan_description;
typedef struct rocfft_execution_info_t* rocfft_execution_info;

#endif  // RPP_ROCFFT_DLOPEN_ROCFFT_VENDORED_TYPES_H
