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

template <typename T>
void testing_dnvec_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_dnvec_descr local_descr;
    int64_t               local_size      = safe_size;
    rocsparse_datatype    local_data_type = get_datatype<T>();

    {
        rocsparse_dnvec_descr* descr     = &local_descr;
        int64_t                size      = local_size;
        void*                  values    = (void*)0x4;
        rocsparse_datatype     data_type = local_data_type;

#define PARAMS_CREATE descr, size, values, data_type
        bad_arg_analysis(rocsparse_create_dnvec_descr, PARAMS_CREATE);
#undef PARAMS_CREATE
        // Check valid descriptor creations
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnvec_descr(descr, 0, nullptr, data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(local_descr),
                                rocsparse_status_success);
        // rocsparse_destroy_dnvec_descr
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(nullptr),
                                rocsparse_status_invalid_pointer);

        // Create valid descriptor
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnvec_descr(descr, size, values, data_type),
                                rocsparse_status_success);
    }

    {
        rocsparse_dnvec_descr descr     = local_descr;
        int64_t*              size      = &local_size;
        void**                values    = (void**)0x4;
        rocsparse_datatype*   data_type = &local_data_type;
#define PARAMS_GET descr, size, values, data_type
        bad_arg_analysis(rocsparse_dnvec_get, PARAMS_GET);
#undef PARAMS_GET
    }

    // Destroy valid descriptor
    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnvec_descr(local_descr), rocsparse_status_success);
}

template <typename T>
void testing_dnvec_descr(const Arguments& arg)
{
}

#define INSTANTIATE(TTYPE)                                                  \
    template void testing_dnvec_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_dnvec_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_dnvec_descr_extra(const Arguments& arg) {}
