/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing_sddmm_dispatch.hpp"

template <typename I, typename J, typename T>
void testing_sddmm_bad_arg(const Arguments& arg)
{

    switch(arg.formatA)
    {
    case rocsparse_format_coo:
    {
        testing_sddmm_dispatch<rocsparse_format_coo, I, I, T>::testing_sddmm_bad_arg(arg);
        return;
    }

    case rocsparse_format_coo_aos:
    {
        testing_sddmm_dispatch<rocsparse_format_coo_aos, I, I, T>::testing_sddmm_bad_arg(arg);
        return;
    }

    case rocsparse_format_csr:
    {
        testing_sddmm_dispatch<rocsparse_format_csr, I, J, T>::testing_sddmm_bad_arg(arg);
        return;
    }

    case rocsparse_format_csc:
    {
        testing_sddmm_dispatch<rocsparse_format_csc, I, J, T>::testing_sddmm_bad_arg(arg);
        return;
    }
    case rocsparse_format_ell:
    {
        testing_sddmm_dispatch<rocsparse_format_ell, I, I, T>::testing_sddmm_bad_arg(arg);
        return;
    }
    case rocsparse_format_bell:
    {
        std::cerr << "testing_sddmm not_implemented for bell format." << std::endl;
        exit(1);
        return;
    }
    case rocsparse_format_bsr:
    {
        std::cerr << "testing_sddmm not_implemented for bsr format." << std::endl;
        exit(1);
        return;
    }
    }
}

template <typename I, typename J, typename T>
void testing_sddmm(const Arguments& arg)
{

    switch(arg.formatA)
    {

    case rocsparse_format_coo:
    {
        testing_sddmm_dispatch<rocsparse_format_coo, I, I, T>::testing_sddmm(arg);
        return;
    }

    case rocsparse_format_csr:
    {
        testing_sddmm_dispatch<rocsparse_format_csr, I, J, T>::testing_sddmm(arg);
        return;
    }

    case rocsparse_format_coo_aos:
    {
        testing_sddmm_dispatch<rocsparse_format_coo_aos, I, I, T>::testing_sddmm(arg);
        return;
    }

    case rocsparse_format_csc:
    {
        testing_sddmm_dispatch<rocsparse_format_csc, I, J, T>::testing_sddmm(arg);
        return;
    }

    case rocsparse_format_ell:
    {
        testing_sddmm_dispatch<rocsparse_format_ell, I, I, T>::testing_sddmm(arg);
        return;
    }

    case rocsparse_format_bell:
    {
        std::cerr << "rocsparse_status_not_implemented" << std::endl;
        exit(1);
        return;
    }

    case rocsparse_format_bsr:
    {
        std::cerr << "rocsparse_status_not_implemented" << std::endl;
        exit(1);
        return;
    }
    }
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                            \
    template void testing_sddmm_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_sddmm<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_sddmm_extra(const Arguments& arg) {}
