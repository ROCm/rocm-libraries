// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../../../shared/hip_object_wrapper.h"

// RAII for rocFFT types

typedef hip_object_wrapper_t<rocfft_plan,
                             rocfft_plan_create,
                             rocfft_plan_destroy,
                             rocfft_status_success>
    rocfft_plan_wrapper_t;

typedef hip_object_wrapper_t<rocfft_plan_description,
                             rocfft_plan_description_create,
                             rocfft_plan_description_destroy,
                             rocfft_status_success>
    rocfft_plan_description_wrapper_t;

typedef hip_object_wrapper_t<rocfft_execution_info,
                             rocfft_execution_info_create,
                             rocfft_execution_info_destroy,
                             rocfft_status_success>
    rocfft_execution_info_wrapper_t;

typedef hip_object_wrapper_t<rocfft_field,
                             rocfft_field_create,
                             rocfft_field_destroy,
                             rocfft_status_success>
    rocfft_field_wrapper_t;

typedef hip_object_wrapper_t<rocfft_brick,
                             rocfft_brick_create,
                             rocfft_brick_destroy,
                             rocfft_status_success>
    rocfft_brick_wrapper_t;
