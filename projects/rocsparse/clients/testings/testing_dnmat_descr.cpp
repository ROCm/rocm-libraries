/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;
    rocsparse_error        p_error[1];
    //
    // rocsparse_dnmat_create
    //
    {
        rocsparse_dnmat_descr* p_descr    = (rocsparse_dnmat_descr*)0x4;
        int64_t                rows       = 4;
        int64_t                cols       = 7;
        int64_t                ld         = std::max(rows, cols);
        rocsparse_order        order      = rocsparse_order_column;
        const void*            const_data = (const void*)0x4;
        void*                  data       = (void*)0x4;
        rocsparse_datatype     data_type  = rocsparse_datatype_f32_r;

#define PARAMS_CREATE handle, p_descr, data_type, order, rows, cols, ld, const_data, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 3;
            const int32_t            args_to_exclude[nargs_to_exclude] = {6, 8, 9};
            select_bad_arg_analysis(
                rocsparse_dnmat_create, nargs_to_exclude, args_to_exclude, PARAMS_CREATE);
        }
#undef PARAMS_CREATE
    }

    //
    // rocsparse_dnmat_create_batched
    //
    {
        rocsparse_dnmat_descr* p_descr       = (rocsparse_dnmat_descr*)0x4;
        int64_t                rows          = 4;
        int64_t                cols          = 7;
        int64_t                ld            = std::max(rows, cols);
        const void*            const_data    = (const void*)0x4;
        void*                  data          = (void*)0x4;
        rocsparse_datatype     data_type     = local_data_type;
        rocsparse_batchtype    batch_type    = rocsparse_batchtype_pointerarray;
        rocsparse_batchstorage batch_storage = rocsparse_batchstorage_soa;
        int64_t                batch_count   = 1;
        int64_t                batch_dist    = 1;
        rocsparse_order        order         = rocsparse_order_row;
#define PARAMS_CREATE_BATCHED                                                                  \
    handle, p_descr, data_type, order, rows, cols, ld, batch_type, batch_storage, batch_count, \
        batch_dist, const_data, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 4;
            const int32_t            args_to_exclude[nargs_to_exclude] = {6, 10, 12, 13};
            select_bad_arg_analysis(rocsparse_dnmat_create_batched,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_CREATE_BATCHED);
        }
#undef PARAMS_CREATE_BATCHED
    }

    //
    // rocsparse_dnmat_get_data
    //
    {
        rocsparse_dnmat_descr descr  = (rocsparse_dnmat_descr)0x4;
        void**                p_data = (void**)0x4;

#define PARAMS_GET_DATA handle, descr, p_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 1;
            const int32_t            args_to_exclude[nargs_to_exclude] = {3};
            select_bad_arg_analysis(
                rocsparse_dnmat_get_data, nargs_to_exclude, args_to_exclude, PARAMS_GET_DATA);
        }
#undef PARAMS_GET_DATA
    }

    //
    // rocsparse_dnmat_get_const_data
    //
    {
        rocsparse_dnmat_descr descr        = (rocsparse_dnmat_descr)0x4;
        const void**          p_const_data = (const void**)0x4;

#define PARAMS_GET_CONST_DATA handle, descr, p_const_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 1;
            const int32_t            args_to_exclude[nargs_to_exclude] = {3};
            select_bad_arg_analysis(rocsparse_dnmat_get_const_data,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_GET_CONST_DATA);
        }
#undef PARAMS_GET_CONST_DATA
    }

    //
    // rocsparse_dnmat_set_data
    //
    {
        rocsparse_dnmat_descr descr = (rocsparse_dnmat_descr)0x4;
        void*                 data  = (void*)0x4;

#define PARAMS_SET_DATA handle, descr, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {2, 3};
            select_bad_arg_analysis(
                rocsparse_dnmat_set_data, nargs_to_exclude, args_to_exclude, PARAMS_SET_DATA);
        }
#undef PARAMS_SET_DATA
    }

    //
    // rocsparse_dnmat_set_const_data
    //
    {
        rocsparse_dnmat_descr descr      = (rocsparse_dnmat_descr)0x4;
        const void*           const_data = (const void*)0x4;

#define PARAMS_SET_CONST_DATA handle, descr, const_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {2, 3};
            select_bad_arg_analysis(rocsparse_dnmat_set_const_data,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_SET_CONST_DATA);
        }
#undef PARAMS_SET_CONST_DATA
    }

    //
    // rocsparse_dnmat_get_prop
    //
    {
        rocsparse_dnmat_descr descr               = (rocsparse_dnmat_descr)0x4;
        rocsparse_dnmat_prop  prop                = rocsparse_dnmat_prop_rows;
        void*                 p_value             = (void*)0x4;
        size_t                value_size_in_bytes = sizeof(int64_t);

#define PARAMS_GET_PROP handle, descr, prop, p_value, value_size_in_bytes, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
            select_bad_arg_analysis(
                rocsparse_dnmat_get_prop, nargs_to_exclude, args_to_exclude, PARAMS_GET_PROP);
        }
#undef PARAMS_GET_PROP
    }
    {
        rocsparse_dnmat_descr descr               = (rocsparse_dnmat_descr)0x4;
        rocsparse_dnmat_prop  prop                = rocsparse_dnmat_prop_rows;
        void*                 p_const_value       = (void*)0x4;
        size_t                value_size_in_bytes = sizeof(int64_t);

#define PARAMS_SET_PROP handle, descr, prop, p_const_value, value_size_in_bytes, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
            select_bad_arg_analysis(
                rocsparse_dnmat_set_prop, nargs_to_exclude, args_to_exclude, PARAMS_SET_PROP);
        }
#undef PARAMS_SET_PROP
    }

    //
    // rocsparse_dnmat_destroy
    //
    {
        rocsparse_dnmat_descr descr = (rocsparse_dnmat_descr)0x4;
#define PARAMS_DESTROY handle, descr, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {1, 2};
            select_bad_arg_analysis(
                rocsparse_dnmat_destroy, nargs_to_exclude, args_to_exclude, PARAMS_DESTROY);
        }
#undef PARAMS_DESTROY
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
