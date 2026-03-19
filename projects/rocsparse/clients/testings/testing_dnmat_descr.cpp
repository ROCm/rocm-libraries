/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_enum.hpp"
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
    rocsparse_local_handle handle;
    rocsparse_dnmat_descr  descr{};
    rocsparse_order        order    = rocsparse_order_row;
    rocsparse_datatype     datatype = rocsparse_datatype_f32_r;
    rocsparse_error        p_error[1]{};
    int64_t                M             = 10;
    int64_t                N             = 4;
    int64_t                ld            = M + 3;
    const void*            const_data    = (const void*)0x4;
    void*                  data          = (void*)0x4;
    int64_t                batch_count   = 3;
    int64_t                batch_dist    = 100;
    rocsparse_batchtype    batch_type    = rocsparse_batchtype_strided;
    rocsparse_batchstorage batch_storage = rocsparse_batchstorage_soa;

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_descr_create_batch(handle,
                                                             &descr,
                                                             datatype,
                                                             order,
                                                             M,
                                                             N,
                                                             ld,
                                                             batch_type,
                                                             batch_storage,
                                                             batch_count,
                                                             batch_dist,
                                                             const_data,
                                                             data,
                                                             p_error));

    {
        void* inject_data = (void*)0x4;
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_data(handle, descr, inject_data, p_error));
        void* fetch_data;
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_data(handle, descr, &fetch_data, p_error));
        ASSERT_EQ(fetch_data, inject_data);
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_data(handle, descr, data, p_error));
    }

    {
        const void* inject_const_data = (const void*)0x4;
        CHECK_ROCSPARSE_ERROR(
            rocsparse_dnmat_set_const_data(handle, descr, inject_const_data, p_error));
        const void* fetch_data;
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_const_data(handle, descr, &fetch_data, p_error));
        ASSERT_EQ(fetch_data, inject_const_data);
        CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_const_data(handle, descr, const_data, p_error));
    }

    for(rocsparse_dnmat_prop prop : rocsparse_dnmat_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_dnmat_prop_rows:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(M, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_cols:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(N, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_ld:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(ld, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batch_count:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_count, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batch_dist:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_dist, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batchtype:
        {
            rocsparse_batchtype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_type, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_order:
        {
            rocsparse_order prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(order, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batchstorage:
        {
            rocsparse_batchstorage prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_storage, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_datatype:
        {
            rocsparse_datatype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(datatype, prop_value);
            break;
        }
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_descr_destroy(handle, descr, p_error));

    //
    // Create.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_descr_create(
        handle, &descr, datatype, order, M, N, ld, const_data, data, p_error));

    for(rocsparse_dnmat_prop prop : rocsparse_dnmat_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_dnmat_prop_cols:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(N, prop_value);
            break;
        }
        case rocsparse_dnmat_prop_rows:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(M, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_ld:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(ld, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batch_count:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(1, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batch_dist:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(0, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batchtype:
        {
            rocsparse_batchtype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(rocsparse_batchtype_strided, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_order:
        {
            rocsparse_order prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(order, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_batchstorage:
        {
            rocsparse_batchstorage prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(rocsparse_batchstorage_soa, prop_value);
            break;
        }

        case rocsparse_dnmat_prop_datatype:
        {
            rocsparse_datatype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(datatype, prop_value);
            break;
        }
        }
    }

    for(rocsparse_dnmat_prop prop : rocsparse_dnmat_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_dnmat_prop_rows:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_cols:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_ld:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_batch_count:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_batch_dist:
        {
            for(int64_t set_prop_value : {0, 1, 2})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_batchtype:
        {
            for(rocsparse_batchtype set_prop_value : rocsparse_batchtype_t::values)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_batchtype prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_order:
        {
            for(rocsparse_order set_prop_value : rocsparse_order_t::values)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_order prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_batchstorage:
        {
            for(rocsparse_batchstorage set_prop_value : rocsparse_batchstorage_t::values)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_batchstorage prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_dnmat_prop_datatype:
        {
            for(rocsparse_datatype set_prop_value :
                {rocsparse_datatype_f32_r, rocsparse_datatype_f64_r})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_datatype prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_dnmat_descr_destroy(handle, descr, p_error));
}

#define INSTANTIATE(TTYPE)                                                  \
    template void testing_dnmat_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_dnmat_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_dnmat_descr_extra(const Arguments& arg) {}
