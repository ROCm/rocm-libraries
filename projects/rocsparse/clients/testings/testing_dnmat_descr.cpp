/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_dnmat_descr_bad_arg(const Arguments& arg)
{
    static const size_t   safe_size = 100;
    rocsparse_dnmat_descr local_descr{};

    int64_t            local_rows         = safe_size;
    int64_t            local_cols         = safe_size;
    int64_t            local_ld           = safe_size;
    rocsparse_order    local_order        = rocsparse_order_column;
    rocsparse_datatype local_data_type    = get_datatype<T>();
    rocsparse_int      local_batch_count  = safe_size;
    int64_t            local_batch_stride = safe_size;

    {
        rocsparse_dnmat_descr* descr     = &local_descr;
        int64_t                rows      = local_rows;
        int64_t                cols      = local_cols;
        int64_t                ld        = local_ld;
        void*                  values    = (void*)0x4;
        rocsparse_order        order     = local_order;
        rocsparse_datatype     data_type = local_data_type;

#define PARAMS_CREATE descr, rows, cols, ld, values, data_type, order
        bad_arg_analysis(rocsparse_create_dnmat_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

        // rocsparse_destroy_dnmat_descr_ex
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(nullptr),
                                rocsparse_status_invalid_pointer);

        // Check valid descriptor creations
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_dnmat_descr(descr, 0, cols, ld, nullptr, data_type, order),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(*descr), rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_dnmat_descr(descr, rows, 0, ld, nullptr, data_type, order),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(*descr), rocsparse_status_success);
    }

    {
        int64_t*            rows      = &local_rows;
        int64_t*            cols      = &local_cols;
        int64_t*            ld        = &local_ld;
        void**              values    = (void**)0x4;
        rocsparse_order*    order     = &local_order;
        rocsparse_datatype* data_type = &local_data_type;

        // Create valid descriptor
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_dnmat_descr(&local_descr,
                                                             local_rows,
                                                             local_cols,
                                                             local_ld,
                                                             (void*)0x4,
                                                             local_data_type,
                                                             local_order),
                                rocsparse_status_success);
        rocsparse_dnmat_descr descr = local_descr;

#define PARAMS_GET descr, rows, cols, ld, values, data_type, order
        bad_arg_analysis(rocsparse_dnmat_get, PARAMS_GET);
#undef PARAMS_GET

        rocsparse_int* batch_count  = &local_batch_count;
        int64_t*       batch_stride = &local_batch_stride;

#define PARAMS_GET descr, batch_count, batch_stride
        bad_arg_analysis(rocsparse_dnmat_get_strided_batch, PARAMS_GET);
#undef PARAMS_GET

        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_dnmat_descr(descr), rocsparse_status_success);
    }
}

template <typename T>
void testing_dnmat_descr(const Arguments& arg)
{
}

#define INSTANTIATE(TTYPE)                                                  \
    template void testing_dnmat_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_dnmat_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_dnmat_descr_extra(const Arguments& arg) {}
