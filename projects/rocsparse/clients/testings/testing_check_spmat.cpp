/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

#include <testing_check_spmat_dispatch.hpp>

template <typename I, typename J, typename T>
void testing_check_spmat_bad_arg(const Arguments& arg)
{
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle            handle = local_handle;
    rocsparse_check_spmat_stage stage  = rocsparse_check_spmat_stage_compute;
    rocsparse_spmat_descr       mat    = (rocsparse_spmat_descr)0x4;

    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {4, 5};

#define PARAMS handle, mat, data_status, stage, buffer_size, temp_buffer
    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_check_spmat, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_check_spmat, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_check_spmat, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    {
        size_t* buffer_size = nullptr;
        void*   temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_check_spmat, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    EXPECT_ROCSPARSE_STATUS(
        rocsparse_check_spmat(handle, mat, data_status, stage, nullptr, nullptr),
        rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename T>
void testing_check_spmat(const Arguments& arg)
{
    switch(arg.formatA)
    {
    case rocsparse_format_coo:
    {
        testing_check_spmat_dispatch<rocsparse_format_coo, I, I, T>::testing_check_spmat(arg);
        return;
    }

    case rocsparse_format_csr:
    {
        testing_check_spmat_dispatch<rocsparse_format_csr, I, J, T>::testing_check_spmat(arg);
        return;
    }

    case rocsparse_format_csc:
    {
        testing_check_spmat_dispatch<rocsparse_format_csc, I, J, T>::testing_check_spmat(arg);
        return;
    }

    case rocsparse_format_ell:
    {
        testing_check_spmat_dispatch<rocsparse_format_ell, I, I, T>::testing_check_spmat(arg);
        return;
    }

    case rocsparse_format_bsr:
    {
        testing_check_spmat_dispatch<rocsparse_format_bsr, I, J, T>::testing_check_spmat(arg);
        return;
    }

    case rocsparse_format_coo_aos:
    case rocsparse_format_bell:
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        return;
    }
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                  \
    template void testing_check_spmat_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_check_spmat<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
void testing_check_spmat_extra(const Arguments& arg) {}
