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

#include "testing.hpp"

static void testing_idvec_descr_bad_arg(const Arguments& arg)
{
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;
    rocsparse_error        p_error[1];

    //
    // rocsparse_idvec_create
    //
    {
        rocsparse_idvec_descr* p_descr    = (rocsparse_idvec_descr*)0x4;
        int64_t                size       = 10;
        int64_t                inc        = 1;
        const void*            const_data = (const void*)0x4;
        void*                  data       = (void*)0x4;
        rocsparse_indextype    indextype  = rocsparse_indextype_i32;
        rocsparse_index_base   base       = rocsparse_index_base_zero;
#define PARAMS_CREATE handle, p_descr, indextype, base, size, inc, const_data, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 3;
            const int32_t            args_to_exclude[nargs_to_exclude] = {5, 7, 8};
            select_bad_arg_analysis(
                rocsparse_idvec_create, nargs_to_exclude, args_to_exclude, PARAMS_CREATE);
        }
#undef PARAMS_CREATE
    }

    //
    // rocsparse_idvec_create_batched
    //
    {
        rocsparse_idvec_descr* p_descr       = (rocsparse_idvec_descr*)0x4;
        rocsparse_indextype    indextype     = rocsparse_indextype_i64;
        rocsparse_index_base   base          = rocsparse_index_base_zero;
        int64_t                size          = 10;
        int64_t                inc           = 1;
        rocsparse_batchtype    batch_type    = rocsparse_batchtype_strided;
        rocsparse_batchstorage batch_storage = rocsparse_batchstorage_soa;
        int64_t                batch_count   = 1;
        int64_t                batch_dist    = 0;
        const void*            const_data    = (const void*)0x4;
        void*                  data          = (void*)0x4;
#define PARAMS_CREATE_BATCHED                                                            \
    handle, p_descr, indextype, base, size, inc, batch_type, batch_storage, batch_count, \
        batch_dist, const_data, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 4;
            const int32_t            args_to_exclude[nargs_to_exclude] = {5, 9, 11, 12};
            select_bad_arg_analysis(rocsparse_idvec_create_batched,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_CREATE_BATCHED);
        }
#undef PARAMS_CREATE_BATCHED
    }

    //
    // rocsparse_idvec_get_data
    //
    {
        rocsparse_idvec_descr descr  = (rocsparse_idvec_descr)0x4;
        void**                p_data = (void**)0x4;

#define PARAMS_GET_DATA handle, descr, p_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 1;
            const int32_t            args_to_exclude[nargs_to_exclude] = {3};
            select_bad_arg_analysis(
                rocsparse_idvec_get_data, nargs_to_exclude, args_to_exclude, PARAMS_GET_DATA);
        }
#undef PARAMS_GET_DATA
    }

    //
    // rocsparse_idvec_get_const_data
    //
    {
        rocsparse_idvec_descr descr        = (rocsparse_idvec_descr)0x4;
        const void**          p_const_data = (const void**)0x4;

#define PARAMS_GET_CONST_DATA handle, descr, p_const_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 1;
            const int32_t            args_to_exclude[nargs_to_exclude] = {3};
            select_bad_arg_analysis(rocsparse_idvec_get_const_data,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_GET_CONST_DATA);
        }
#undef PARAMS_GET_CONST_DATA
    }

    //
    // rocsparse_idvec_set_data
    //
    {
        rocsparse_idvec_descr descr = (rocsparse_idvec_descr)0x4;
        void*                 data  = (void*)0x4;

#define PARAMS_SET_DATA handle, descr, data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {2, 3};
            select_bad_arg_analysis(
                rocsparse_idvec_set_data, nargs_to_exclude, args_to_exclude, PARAMS_SET_DATA);
        }
#undef PARAMS_SET_DATA
    }

    //
    // rocsparse_idvec_set_const_data
    //
    {
        rocsparse_idvec_descr descr      = (rocsparse_idvec_descr)0x4;
        const void*           const_data = (const void*)0x4;

#define PARAMS_SET_CONST_DATA handle, descr, const_data, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {2, 3};
            select_bad_arg_analysis(rocsparse_idvec_set_const_data,
                                    nargs_to_exclude,
                                    args_to_exclude,
                                    PARAMS_SET_CONST_DATA);
        }
#undef PARAMS_SET_CONST_DATA
    }

    //
    // rocsparse_idvec_get_prop
    //
    {
        rocsparse_idvec_descr descr               = (rocsparse_idvec_descr)0x4;
        rocsparse_idvec_prop  prop                = rocsparse_idvec_prop_size;
        void*                 p_value             = (void*)0x4;
        size_t                value_size_in_bytes = sizeof(int64_t);

#define PARAMS_GET_PROP handle, descr, prop, p_value, value_size_in_bytes, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
            select_bad_arg_analysis(
                rocsparse_idvec_get_prop, nargs_to_exclude, args_to_exclude, PARAMS_GET_PROP);
        }
#undef PARAMS_GET_PROP
    }

    {
        rocsparse_idvec_descr descr               = (rocsparse_idvec_descr)0x4;
        rocsparse_idvec_prop  prop                = rocsparse_idvec_prop_size;
        void*                 p_value             = (void*)0x4;
        size_t                value_size_in_bytes = sizeof(int64_t);

#define PARAMS_SET_PROP handle, descr, prop, p_value, value_size_in_bytes, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {4, 5};
            select_bad_arg_analysis(
                rocsparse_idvec_set_prop, nargs_to_exclude, args_to_exclude, PARAMS_SET_PROP);
        }
#undef PARAMS_SET_PROP
    }

    //
    // rocsparse_idvec_destroy
    //
    {
        rocsparse_idvec_descr descr = (rocsparse_idvec_descr)0x4;
#define PARAMS_DESTROY handle, descr, p_error
        {
            static constexpr int32_t nargs_to_exclude                  = 2;
            const int32_t            args_to_exclude[nargs_to_exclude] = {1, 2};
            select_bad_arg_analysis(
                rocsparse_idvec_destroy, nargs_to_exclude, args_to_exclude, PARAMS_DESTROY);
        }
#undef PARAMS_DESTROY
    }
}

template <typename T>
void testing_idvec_descr_bad_arg(const Arguments& arg)
{
    testing_idvec_descr_bad_arg(arg);
}

#include "rocsparse_enum.hpp"

void testing_idvec_descr_extra(const Arguments& arg)
{
    rocsparse_local_handle handle;
    rocsparse_idvec_descr  descr{};
    rocsparse_indextype    indextype = rocsparse_indextype_i32;
    rocsparse_error        p_error[1]{};
    int64_t                size          = 10;
    int64_t                sizelm        = sizeof(float);
    int64_t                inc           = 1;
    const void*            const_data    = (const void*)0x4;
    void*                  data          = (void*)0x4;
    int64_t                batch_count   = 3;
    int64_t                batch_dist    = 100;
    rocsparse_batchtype    batch_type    = rocsparse_batchtype_strided;
    rocsparse_batchstorage batch_storage = rocsparse_batchstorage_soa;
    rocsparse_index_base   base          = rocsparse_index_base_one;

    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_create_batched(handle,
                                                         &descr,
                                                         indextype,
                                                         base,
                                                         size,
                                                         inc,
                                                         batch_type,
                                                         batch_storage,
                                                         batch_count,
                                                         batch_dist,
                                                         const_data,
                                                         data,
                                                         p_error));

    for(rocsparse_idvec_prop prop : rocsparse_idvec_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_idvec_prop_size:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(size, prop_value);
            break;
        }

        case rocsparse_idvec_prop_size_in_bytes:
        {
            size_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(size * sizelm, prop_value);
            break;
        }

        case rocsparse_idvec_prop_inc:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(inc, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batch_count:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_count, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batch_dist:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_dist, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batchtype:
        {
            rocsparse_batchtype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_type, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batchstorage:
        {
            rocsparse_batchstorage prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(batch_storage, prop_value);
            break;
        }

        case rocsparse_idvec_prop_indextype:
        {
            rocsparse_indextype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(indextype, prop_value);
            break;
        }

        case rocsparse_idvec_prop_base:
        {
            rocsparse_index_base prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(base, prop_value);
            break;
        }
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_destroy(handle, descr, p_error));

    //
    // Create.
    //
    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_create(
        handle, &descr, indextype, base, size, inc, const_data, data, p_error));

    for(rocsparse_idvec_prop prop : rocsparse_idvec_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_idvec_prop_size:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(size, prop_value);
            break;
        }

        case rocsparse_idvec_prop_size_in_bytes:
        {
            size_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(size * sizelm, prop_value);
            break;
        }

        case rocsparse_idvec_prop_inc:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(inc, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batch_count:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(1, prop_value);
            break;
        }

        case rocsparse_idvec_prop_base:
        {
            rocsparse_index_base prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(base, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batch_dist:
        {
            int64_t prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(0, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batchtype:
        {
            rocsparse_batchtype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(rocsparse_batchtype_strided, prop_value);
            break;
        }

        case rocsparse_idvec_prop_batchstorage:
        {
            rocsparse_batchstorage prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(rocsparse_batchstorage_soa, prop_value);
            break;
        }

        case rocsparse_idvec_prop_indextype:
        {
            rocsparse_indextype prop_value;
            CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
            ASSERT_EQ(indextype, prop_value);
            break;
        }
        }
    }

    for(rocsparse_idvec_prop prop : rocsparse_idvec_prop_t::values)
    {
        switch(prop)
        {
        case rocsparse_idvec_prop_size:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_size_in_bytes:
        {
            break;
        }

        case rocsparse_idvec_prop_inc:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_batch_count:
        {
            for(int64_t set_prop_value : {0, 1, 2, 4})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_batch_dist:
        {
            for(int64_t set_prop_value : {0, 1, 2})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                int64_t prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_batchtype:
        {
            for(rocsparse_batchtype set_prop_value : rocsparse_batchtype_t::values)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_batchtype prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_batchstorage:
        {
            for(rocsparse_batchstorage set_prop_value : rocsparse_batchstorage_t::values)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_batchstorage prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }

        case rocsparse_idvec_prop_indextype:
        {
            for(rocsparse_indextype set_prop_value :
                {rocsparse_indextype_i32, rocsparse_indextype_i64})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_indextype prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }
        case rocsparse_idvec_prop_base:
        {
            for(rocsparse_index_base set_prop_value :
                {rocsparse_index_base_zero, rocsparse_index_base_one})
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_set_prop(
                    handle, descr, prop, &set_prop_value, sizeof(set_prop_value), p_error));
                rocsparse_indextype prop_value;
                CHECK_ROCSPARSE_ERROR(rocsparse_idvec_get_prop(
                    handle, descr, prop, &prop_value, sizeof(prop_value), p_error));
                ASSERT_EQ(set_prop_value, prop_value);
            }
            break;
        }
        }
    }

    CHECK_ROCSPARSE_ERROR(rocsparse_idvec_destroy(handle, descr, p_error));
}

template <typename T>
void testing_idvec_descr(const Arguments& arg)
{
}

#define INSTANTIATE(TTYPE)                                                  \
    template void testing_idvec_descr_bad_arg<TTYPE>(const Arguments& arg); \
    template void testing_idvec_descr<TTYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
