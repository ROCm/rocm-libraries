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

template <typename I, typename T>
void testing_spvec_descr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_spvec_descr local_descr{};
    int64_t               local_size     = safe_size;
    int64_t               local_nnz      = safe_size;
    rocsparse_index_base  local_idx_base = rocsparse_index_base_zero;

    rocsparse_indextype local_idx_type  = get_indextype<I>();
    rocsparse_datatype  local_data_type = get_datatype<T>();

    {
        rocsparse_spvec_descr* descr     = &local_descr;
        int64_t                size      = local_size;
        int64_t                nnz       = local_nnz;
        void*                  values    = (void*)0x4;
        void*                  indices   = (void*)0x4;
        rocsparse_index_base   idx_base  = local_idx_base;
        rocsparse_indextype    idx_type  = local_idx_type;
        rocsparse_datatype     data_type = local_data_type;

#define PARAMS_CREATE descr, size, nnz, indices, values, idx_type, idx_base, data_type
        bad_arg_analysis(rocsparse_create_spvec_descr, PARAMS_CREATE);
#undef PARAMS_CREATE

        // size < nnz
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_spvec_descr(
                descr, size, (size + 1), indices, values, idx_type, idx_base, data_type),
            rocsparse_status_invalid_size);

        // rocsparse_destroy_spvec_descr_ex
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(nullptr),
                                rocsparse_status_invalid_pointer);

        // Check valid descriptor creations
        EXPECT_ROCSPARSE_STATUS(rocsparse_create_spvec_descr(
                                    descr, 0, 0, nullptr, nullptr, idx_type, idx_base, data_type),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(local_descr),
                                rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_spvec_descr(
                descr, size, 0, nullptr, nullptr, idx_type, idx_base, data_type),
            rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(local_descr),
                                rocsparse_status_success);

        // Create valid descriptor
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_create_spvec_descr(
                descr, size, nnz, indices, values, idx_type, idx_base, data_type),
            rocsparse_status_success);
    }

    {
        rocsparse_spvec_descr descr     = local_descr;
        int64_t*              size      = &local_size;
        int64_t*              nnz       = &local_nnz;
        void**                values    = (void**)0x4;
        void**                indices   = (void**)0x4;
        rocsparse_index_base* idx_base  = &local_idx_base;
        rocsparse_indextype*  idx_type  = &local_idx_type;
        rocsparse_datatype*   data_type = &local_data_type;

#define PARAMS_GET descr, size, nnz, indices, values, idx_type, idx_base, data_type
        bad_arg_analysis(rocsparse_spvec_get, PARAMS_GET);
#undef PARAMS_GET

#define PARAMS_GET_INDEX_BASE descr, idx_base
        bad_arg_analysis(rocsparse_spvec_get_index_base, PARAMS_GET_INDEX_BASE);
#undef PARAMS_GET_INDEX_BASE

#define PARAMS_GET_VALUES descr, values
        bad_arg_analysis(rocsparse_spvec_get_values, PARAMS_GET_VALUES);
#undef PARAMS_GET_VALUES
    }

    EXPECT_ROCSPARSE_STATUS(rocsparse_destroy_spvec_descr(local_descr), rocsparse_status_success);
}

template <typename I, typename T>
void testing_spvec_descr(const Arguments& arg)
{
}

#define INSTANTIATE(ITYPE, TTYPE)                                                  \
    template void testing_spvec_descr_bad_arg<ITYPE, TTYPE>(const Arguments& arg); \
    template void testing_spvec_descr<ITYPE, TTYPE>(const Arguments& arg)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
void testing_spvec_descr_extra(const Arguments& arg) {}
