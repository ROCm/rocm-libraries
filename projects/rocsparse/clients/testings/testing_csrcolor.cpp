/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_enum.hpp"

template <typename T>
void testing_csrcolor_bad_arg(const Arguments& arg)
{

    rocsparse_local_handle    local_handle;
    rocsparse_local_mat_descr local_descr;
    rocsparse_local_mat_info  local_info;

    rocsparse_handle          handle            = local_handle;
    rocsparse_int             m                 = 4;
    rocsparse_int             nnz               = 4;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  csr_val           = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind       = (const rocsparse_int*)0x4;
    const floating_data_t<T>* fraction_to_color = (const floating_data_t<T>*)0x4;
    rocsparse_int*            ncolors           = (rocsparse_int*)0x4;
    rocsparse_int*            coloring          = (rocsparse_int*)0x4;
    rocsparse_int*            reordering        = nullptr;
    rocsparse_mat_info        info              = local_info;

#define PARAMS                                                                            \
    handle, m, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, fraction_to_color, ncolors, \
        coloring, reordering, info

    { //
        // Exclude reordering since this is an optional argument.
        //
        static constexpr int nargs_to_exclude                  = 1;
        static constexpr int args_to_exclude[nargs_to_exclude] = {10};

        select_bad_arg_analysis(rocsparse_csrcolor<T>,
                                nargs_to_exclude,
                                args_to_exclude,
                                handle,
                                m,
                                nnz,
                                descr,
                                csr_val,
                                csr_row_ptr,
                                csr_col_ind,
                                fraction_to_color,
                                ncolors,
                                coloring,
                                reordering,
                                info);
    }

    //
    // Not implemented cases.
    //
    for(auto val : rocsparse_matrix_type_t::values)
    {
        if(val != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, val));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(PARAMS),
                                    rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);

#undef PARAMS
}

template <typename T>
void testing_csrcolor(const Arguments& arg)
{
    //
    // Create the matrix factory.
    //
    rocsparse_matrix_factory<T> matrix_factory(arg);

    //
    // Get dimensions.
    //
    rocsparse_int            M                 = arg.M;
    rocsparse_index_base     csr_base          = arg.baseA;
    const floating_data_t<T> fraction_to_color = static_cast<floating_data_t<T>>(arg.percentage);

    //
    // Local variables.
    //
    rocsparse_local_handle    handle;
    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_info  mat_info;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    rocsparse_int ncolor;

    //
    // Argument sanity check before allocating invalid memory
    //
    if(M == 0)
    {
        EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(handle,
                                                      0,
                                                      7,
                                                      csr_descr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr),
                                rocsparse_status_success);

        EXPECT_ROCSPARSE_STATUS(rocsparse_csrcolor<T>(handle,
                                                      7,
                                                      0,
                                                      csr_descr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr),
                                rocsparse_status_success);
        return;
    }

    //
    // Init a CSR symmetric matrix.
    //
    host_csr_matrix<T> hA;
    {
        host_csr_matrix<T> nonsymA;
        matrix_factory.init_csr(nonsymA, M, M, csr_base);
        CHECK_ROCSPARSE_ERROR(rocsparse_matrix_utils::host_csrsym(nonsymA, hA));
    }

    //
    // Allocate device memory and transfer data from host.
    //
    device_csr_matrix<T>               dA(hA);
    device_dense_vector<rocsparse_int> dcoloring(hA.m);
    device_dense_vector<rocsparse_int> dreordering(hA.m);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(rocsparse_csrcolor<T>(handle,
                                                    dA.m,
                                                    dA.nnz,
                                                    csr_descr,
                                                    dA.val,
                                                    dA.ptr,
                                                    dA.ind,
                                                    &fraction_to_color,
                                                    &ncolor,
                                                    dcoloring,
                                                    dreordering,
                                                    mat_info));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dcoloring", dcoloring, "dreordering", dreordering);
        }

        //
        // CHECK CONSISTENCY: COUNT NUMBER OF COLORS IN HCOLORING
        //
        host_dense_vector<rocsparse_int> hcoloring(dcoloring);

        //
        // CHECK CONSISTENCY: CHECK ANY COLOR NOT BEING SHARED BY TWO ADJACENT NODES.
        //
        for(rocsparse_int i = 0; i < M; ++i)
        {
            auto icolor = hcoloring[i];
            for(rocsparse_int at = hA.ptr[i] - hA.base; at < hA.ptr[i + 1] - hA.base; ++at)
            {
                auto j = hA.ind[at] - hA.base;
                if(i != j)
                {
                    auto jcolor = hcoloring[j];
                    EXPECT_ROCSPARSE_STATUS((icolor != jcolor) ? rocsparse_status_success
                                                               : rocsparse_status_internal_error,
                                            rocsparse_status_success);
                }
            }
        }

        //
        // Verification of the number of colors by counting them.
        //

        //
        // Check if colors are contiguous
        //
        {
            rocsparse_int max_value = 0;
            for(rocsparse_int i = 0; i < hcoloring.size(); ++i)
            {
                //
                // Check value is well defined.
                //
                EXPECT_ROCSPARSE_STATUS((hcoloring[i] >= 0 && hcoloring[i] < M)
                                            ? rocsparse_status_success
                                            : rocsparse_status_internal_error,
                                        rocsparse_status_success);

                //
                // Calculate maximum value.
                //
                if(hcoloring[i] > max_value)
                {
                    max_value = hcoloring[i];
                }
            }
            ++max_value;

            bool* marker = new bool[max_value];
            for(rocsparse_int i = 0; i < max_value; ++i)
            {
                marker[i] = false;
            }

            for(rocsparse_int i = 0; i < hcoloring.size(); ++i)
            {
                marker[hcoloring[i]] = true;
            }

            for(rocsparse_int i = 0; i < max_value; ++i)
            {
                EXPECT_ROCSPARSE_STATUS(marker[i] ? rocsparse_status_success
                                                  : rocsparse_status_internal_error,
                                        rocsparse_status_success);
            }

            delete[] marker;

            //
            // Compare the number of colors.
            //
            unit_check_scalar(max_value, ncolor);
        }

        if(dreordering)
        {
            //
            // Need to verify this is a valid permutation array..
            //
            host_dense_vector<rocsparse_int> hreordering(dreordering);
            host_dense_vector<rocsparse_int> cache(M);
            for(rocsparse_int i = 0; i < M; ++i)
            {
                cache[i] = 0;
            }

            for(rocsparse_int i = 0; i < M; ++i)
            {
                EXPECT_ROCSPARSE_STATUS((hreordering[i] >= 0 && hreordering[i] < M)
                                            ? rocsparse_status_success
                                            : rocsparse_status_internal_error,
                                        rocsparse_status_success);

                cache[hreordering[i]] = 1;
            }

            for(rocsparse_int i = 0; i < M; ++i)
            {
                EXPECT_ROCSPARSE_STATUS((cache[i] != 0) ? rocsparse_status_success
                                                        : rocsparse_status_internal_error,
                                        rocsparse_status_success);
            }
        }
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csrcolor<T>,
                                                                      handle,
                                                                      dA.m,
                                                                      dA.nnz,
                                                                      csr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      &fraction_to_color,
                                                                      &ncolor,
                                                                      dcoloring,
                                                                      dreordering,
                                                                      mat_info);

        display_timing_info(display_key_t::M,
                            dA.m,
                            display_key_t::nnz,
                            dA.nnz,
                            "frac",
                            fraction_to_color,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                               \
    template void testing_csrcolor_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrcolor<TYPE>(const Arguments& arg)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrcolor_extra(const Arguments& arg) {}
