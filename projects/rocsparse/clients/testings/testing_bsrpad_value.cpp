/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <limits>

template <typename T>
void testing_bsrpad_value_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nnzb        = safe_size;
    rocsparse_int             block_dim   = 1;
    T                         value       = 1;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    T*                        bsr_val     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_col_ind = (rocsparse_int*)0x4;

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {5};

#define PARAMS handle, m, mb, nnzb, block_dim, value, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind

    select_bad_arg_analysis(rocsparse_bsrpad_value<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    // m > block_dim * mb
    mb = 3;
    m  = block_dim * mb + 1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

    // m <= block_dim * (mb - 1)
    mb = 3;
    m  = block_dim * (mb - 1);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

    // block_dim == 0
    block_dim = 0;
    mb        = 3;
    m         = block_dim * mb;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

    // Shape relation (mb - 1) * block_dim < m <= mb * block_dim near the rocsparse_int limits,
    // including shapes whose products overflow 32 bits (GitHub issue #15913).
    // nnzb = 0 so that a valid shape returns before any array is dereferenced.
    {
        // Exact in int64_t for these inputs, which are all <= INT32_MAX.
        auto shape_status = [](int64_t m_, int64_t mb_, int64_t block_dim_) {
            return ((mb_ - 1) * block_dim_ < m_ && m_ <= mb_ * block_dim_)
                       ? rocsparse_status_success
                       : rocsparse_status_invalid_size;
        };

        struct shape_case
        {
            rocsparse_int    m, mb, block_dim;
            rocsparse_status expected;
        };

        constexpr rocsparse_int imax = std::numeric_limits<int32_t>::max();
        constexpr auto          ok   = rocsparse_status_success;
        constexpr auto          bad  = rocsparse_status_invalid_size;

        const shape_case cases[] = {
            {1, 1, 1, ok},
            {1, 2, 2, bad},
            {1, 3, 1431655766, bad},
            {4, 4, 1073741825, bad},
            {1, imax, imax, bad},
            {3, 1073741825, 4, bad},
            {imax, 2, 1073741824, ok},
            {1073741825, 2, 1073741824, ok},
            {1073741824, 2, 1073741824, bad},
            {imax, 3, 1073741823, ok},
            {imax, 3, 1073741824, bad},
            {imax - 1, 2, 1073741823, ok},
            {1, 1, imax, ok},
            {imax, 1, imax, ok},
            {1, 2, imax, bad},
            {0, 0, 1, ok},
            {0, 0, imax, ok},
            {0, 1, 1, bad},
            {1, 0, 1, bad},
            {1, 1, 1431655766, ok},
            {0, 1, 5, bad},
        };

        nnzb = 0;
        for(const auto& c : cases)
        {
            m         = c.m;
            mb        = c.mb;
            block_dim = c.block_dim;
#ifdef GOOGLE_TEST
            SCOPED_TRACE(testing::Message()
                         << "m=" << m << " mb=" << mb << " block_dim=" << block_dim);
#endif
            EXPECT_ROCSPARSE_STATUS(shape_status(m, mb, block_dim), c.expected);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), c.expected);
        }

        // An invalid pointer is reported before an invalid shape.
        m           = 1;
        mb          = 3;
        block_dim   = 1431655766;
        bsr_row_ptr = nullptr;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS),
                                rocsparse_status_invalid_pointer);
        bsr_row_ptr = (rocsparse_int*)0x4;

        // 3 * block_dim = 2^64 + 2 overflows a 64-bit rocsparse_int.
        if(sizeof(rocsparse_int) == sizeof(int64_t))
        {
            const int64_t big = 6148914691236517206;
            m                 = 1;
            mb                = 3;
            block_dim         = static_cast<rocsparse_int>(big);
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS),
                                    rocsparse_status_invalid_size);
        }
    }

#undef PARAMS
}

template <typename T>
void testing_bsrpad_value(const Arguments& arg)
{
    static constexpr bool       toint     = false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, toint, full_rank);

    rocsparse_int          M         = arg.M;
    rocsparse_int          block_dim = arg.block_dim;
    rocsparse_index_base   base      = arg.baseA;
    rocsparse_direction    direction = arg.direction;
    rocsparse_storage_mode storage   = arg.storage;
    T                      value     = 1;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr bsr_descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, base));

    // Set matrix storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, storage));

    // Allocate device memory
    device_gebsr_matrix<T> dbsr;

    {
        host_csr_matrix<T> hcsrA;

        // Generate a temporary sorted csr matrix to get the correct dimensions
        hcsrA.define(M, M, 0, base);
        matrix_factory.init_csr(hcsrA.ptr,
                                hcsrA.ind,
                                hcsrA.val,
                                hcsrA.m,
                                hcsrA.n,
                                hcsrA.nnz,
                                hcsrA.base,
                                rocsparse_matrix_type_general,
                                rocsparse_fill_mode_lower,
                                rocsparse_storage_mode_sorted);
        M = hcsrA.m;

        device_csr_matrix<T> dcsrA(hcsrA);

        rocsparse_matrix_utils::convert(
            dcsrA, direction, block_dim, base, rocsparse_storage_mode_sorted, dbsr);

        switch(storage)
        {
        case rocsparse_storage_mode_unsorted:
        {
            host_gebsr_matrix<T> hbsr(dbsr);
            rocsparse_matrix_utils::host_gebsrunsort<T>(
                hbsr.ptr.data(), hbsr.ind.data(), hbsr.mb, hbsr.base);
            dbsr(hbsr);
            break;
        }
        case rocsparse_storage_mode_sorted:
        {
            break;
        }
        }

        Mb = dbsr.mb;
    }

    if(arg.unit_check)
    {
        // Allocate host memory for BSR matrix
        host_gebsr_matrix<T> hbsr(dbsr);

        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrpad_value<T>(
            handle, M, Mb, dbsr.nnzb, block_dim, value, bsr_descr, dbsr.val, dbsr.ptr, dbsr.ind));

        host_gebsr_matrix<T> hbsrC(dbsr);

        // CPU bsrpad_value
        host_bsrpad_value<T>(
            M, Mb, hbsr.nnzb, block_dim, value, hbsr.val, hbsr.ptr, hbsr.ind, hbsr.base);

        hbsrC.unit_check(hbsr);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_bsrpad_value<T>,
                                                                      handle,
                                                                      M,
                                                                      Mb,
                                                                      dbsr.nnzb,
                                                                      block_dim,
                                                                      value,
                                                                      bsr_descr,
                                                                      dbsr.val,
                                                                      dbsr.ptr,
                                                                      dbsr.ind);

        double gbyte_count = 0;
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::Mb,
                            Mb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                   \
    template void testing_bsrpad_value_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrpad_value<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrpad_value_extra(const Arguments& arg)
{
    // The padded diagonal entry of each case lies past offset INT32_MAX of bsr_val, so its
    // offset must be computed in 64 bits. All blocks sit in the last block row, the last one
    // on the diagonal, and only the last row of that block is padded, so the padded entry is
    // the last entry of bsr_val. bsr_val is zero-filled on the device and only the padded
    // entry and its predecessor are read back.
    using T = float;

    struct pad_case
    {
        rocsparse_int m, mb, nnzb, block_dim;
    };

    const pad_case cases[] = {
        // One block: i * block_dim = 2147534622 exceeds INT32_MAX; i * block_dim + i = 2147580963.
        {46341, 1, 1, 46342},
        // 513 blocks: block_index * block_dim * block_dim + i * block_dim + i = 2151677951.
        {std::numeric_limits<int32_t>::max(), 1 << 20, 513, 2048},
    };

    rocsparse_local_handle    handle(arg);
    rocsparse_local_mat_descr bsr_descr;

    const T value = static_cast<T>(1);

    for(const auto& c : cases)
    {
#ifdef GOOGLE_TEST
        SCOPED_TRACE(testing::Message() << "m=" << c.m << " mb=" << c.mb << " nnzb=" << c.nnzb
                                        << " block_dim=" << c.block_dim);
#endif
        std::vector<rocsparse_int> hbsr_row_ptr(c.mb + 1, 0);
        std::vector<rocsparse_int> hbsr_col_ind(c.nnzb);
        hbsr_row_ptr[c.mb] = c.nnzb;
        for(rocsparse_int j = 0; j < c.nnzb; ++j)
        {
            hbsr_col_ind[j] = c.mb - c.nnzb + j;
        }

        const int64_t nval = static_cast<int64_t>(c.nnzb) * c.block_dim * c.block_dim;
        const int64_t pad  = nval - 1;

        device_vector<rocsparse_int> dbsr_row_ptr(c.mb + 1);
        device_vector<rocsparse_int> dbsr_col_ind(c.nnzb);
        device_vector<T>             dbsr_val(nval);

        CHECK_HIP_ERROR(hipMemcpy(dbsr_row_ptr,
                                  hbsr_row_ptr.data(),
                                  sizeof(rocsparse_int) * (c.mb + 1),
                                  hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dbsr_col_ind,
                                  hbsr_col_ind.data(),
                                  sizeof(rocsparse_int) * c.nnzb,
                                  hipMemcpyHostToDevice));

        for(const auto storage : {rocsparse_storage_mode_sorted, rocsparse_storage_mode_unsorted})
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, storage));
            CHECK_HIP_ERROR(hipMemset(dbsr_val, 0, sizeof(T) * nval));

            CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrpad_value<T>(handle,
                                                                     c.m,
                                                                     c.mb,
                                                                     c.nnzb,
                                                                     c.block_dim,
                                                                     value,
                                                                     bsr_descr,
                                                                     dbsr_val,
                                                                     dbsr_row_ptr,
                                                                     dbsr_col_ind));
            CHECK_HIP_ERROR(hipDeviceSynchronize());

            T hval[2];
            CHECK_HIP_ERROR(hipMemcpy(
                hval, static_cast<T*>(dbsr_val) + pad - 1, sizeof(T) * 2, hipMemcpyDeviceToHost));

            unit_check_scalar<T>(static_cast<T>(0), hval[0]);
            unit_check_scalar<T>(value, hval[1]);
        }
    }
}
