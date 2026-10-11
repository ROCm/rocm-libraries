/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_csr2hyb_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat local_hyb;

    rocsparse_handle          handle         = local_handle;
    rocsparse_int             m              = safe_size;
    rocsparse_int             n              = safe_size;
    const rocsparse_mat_descr descr          = local_descr;
    const T*                  csr_val        = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr    = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind    = (const rocsparse_int*)0x4;
    rocsparse_hyb_mat         hyb            = local_hyb;
    rocsparse_hyb_partition   partition_type = rocsparse_hyb_partition_auto;

    int           nargs_to_exclude   = 1;
    const int     args_to_exclude[1] = {8};
    rocsparse_int user_ell_width     = 0;

#define PARAMS \
    handle, m, n, descr, csr_val, csr_row_ptr, csr_col_ind, hyb, user_ell_width, partition_type
    select_bad_arg_analysis(rocsparse_csr2hyb<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(PARAMS), rocsparse_status_requires_sorted_storage);
#undef PARAMS
}

template <typename T>
void testing_csr2hyb(const Arguments& arg)
{

    // Sample matrix
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M              = arg.M;
    rocsparse_int               N              = arg.N;
    rocsparse_index_base        base           = arg.baseA;
    rocsparse_hyb_partition     part           = arg.part;
    rocsparse_int               user_ell_width = arg.algo;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create hyb matrix
    rocsparse_local_hyb_mat hyb;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;
    host_vector<rocsparse_int> hhyb_ell_col_ind_gold;
    host_vector<T>             hhyb_ell_val_gold;
    host_vector<rocsparse_int> hhyb_coo_row_ind_gold;
    host_vector<rocsparse_int> hhyb_coo_col_ind_gold;
    host_vector<T>             hhyb_coo_val_gold;

    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, M, N, nnz, base);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Use a user supplied ELL width.
    if(part == rocsparse_hyb_partition_user)
    {
        // ELL width -33 means we take a reasonable pre-computed width
        if(user_ell_width == -33)
        {
            user_ell_width = (M == 0) ? 0 : nnz / M;
        }

        // Test invalid user_ell_width
        rocsparse_int max_allowed = (M == 0) ? 0 : ((2 * nnz - 1) / M + 1);

        if(user_ell_width > max_allowed)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(handle,
                                                         M,
                                                         N,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         hyb,
                                                         user_ell_width,
                                                         part),
                                    (M == 0 || N == 0) ? rocsparse_status_success
                                                       : rocsparse_status_invalid_value);

            return;
        }
    }

    // Max ELL width, no COO part
    if(part == rocsparse_hyb_partition_max)
    {
        // Compute max ELL width
        rocsparse_int ell_max_width = 0;
        for(rocsparse_int i = 0; i < M; ++i)
        {
            ell_max_width = std::max(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i], ell_max_width);
        }

        rocsparse_int max_allowed = (M == 0) ? 0 : ((2 * nnz - 1) / M + 1);

        if(ell_max_width > max_allowed)
        {
            EXPECT_ROCSPARSE_STATUS(rocsparse_csr2hyb<T>(handle,
                                                         M,
                                                         N,
                                                         descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         hyb,
                                                         user_ell_width,
                                                         part),
                                    (M == 0 || N == 0) ? rocsparse_status_success
                                                       : rocsparse_status_invalid_value);

            return;
        }
    }

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(
            handle, M, N, descr, dcsr_val, dcsr_row_ptr, dcsr_col_ind, hyb, user_ell_width, part));

        // Copy output to host
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        int64_t       ell_nnz = dhyb->ell_nnz;
        rocsparse_int coo_nnz = dhyb->coo_nnz;

        host_vector<rocsparse_int> hhyb_ell_col_ind(ell_nnz);
        host_vector<T>             hhyb_ell_val(ell_nnz);
        host_vector<rocsparse_int> hhyb_coo_row_ind(coo_nnz);
        host_vector<rocsparse_int> hhyb_coo_col_ind(coo_nnz);
        host_vector<T>             hhyb_coo_val(coo_nnz);

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(hhyb_ell_col_ind,
                                  dhyb->ell_col_ind,
                                  sizeof(rocsparse_int) * ell_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hhyb_ell_val, dhyb->ell_val, sizeof(T) * ell_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_row_ind,
                                  dhyb->coo_row_ind,
                                  sizeof(rocsparse_int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hhyb_coo_col_ind,
                                  dhyb->coo_col_ind,
                                  sizeof(rocsparse_int) * coo_nnz,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hhyb_coo_val, dhyb->coo_val, sizeof(T) * coo_nnz, hipMemcpyDeviceToHost));

        // CPU csr2hyb
        rocsparse_int ell_width_gold = user_ell_width;
        int64_t       ell_nnz_gold;
        rocsparse_int coo_nnz_gold;

        host_csr_to_hyb<T>(M,
                           N,
                           nnz,
                           hcsr_row_ptr,
                           hcsr_col_ind,
                           hcsr_val,
                           hhyb_ell_col_ind_gold,
                           hhyb_ell_val_gold,
                           ell_width_gold,
                           ell_nnz_gold,
                           hhyb_coo_row_ind_gold,
                           hhyb_coo_col_ind_gold,
                           hhyb_coo_val_gold,
                           coo_nnz_gold,
                           part,
                           base);

        unit_check_scalar<rocsparse_int>(M, dhyb->m);
        unit_check_scalar<rocsparse_int>(N, dhyb->n);
        unit_check_scalar<rocsparse_int>(ell_width_gold, dhyb->ell_width);
        unit_check_scalar<int64_t>(ell_nnz_gold, dhyb->ell_nnz);
        unit_check_scalar<rocsparse_int>(coo_nnz_gold, dhyb->coo_nnz);
        hhyb_ell_col_ind_gold.unit_check(hhyb_ell_col_ind);
        hhyb_ell_val_gold.unit_check(hhyb_ell_val);
        hhyb_coo_row_ind_gold.unit_check(hhyb_coo_row_ind);
        hhyb_coo_col_ind_gold.unit_check(hhyb_coo_col_ind);
        hhyb_coo_val_gold.unit_check(hhyb_coo_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2hyb<T>,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      descr,
                                                                      dcsr_val,
                                                                      dcsr_row_ptr,
                                                                      dcsr_col_ind,
                                                                      hyb,
                                                                      user_ell_width,
                                                                      part);

        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        int64_t       ell_nnz = dhyb->ell_nnz;
        rocsparse_int coo_nnz = dhyb->coo_nnz;

        double gbyte_count = csr2hyb_gbyte_count<T>(M, nnz, ell_nnz, coo_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::ell_nnz,
                            ell_nnz,
                            display_key_t::coo_nnz,
                            coo_nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2hyb_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2hyb<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
// Builds an M x N CSR matrix whose first n_full rows are fully dense (N nonzeros
// each), followed by one row holding partial_len < N leading nonzeros, the rest
// empty. With rocsparse_hyb_partition_max: ell_width = N, no COO part, and
// ell_nnz = (int64_t)N * M. The checks read back ELL slots beyond INT32_MAX.
static void testing_csr2hyb_extra_large_ell(const Arguments& arg,
                                            rocsparse_int    M,
                                            rocsparse_int    N,
                                            rocsparse_int    n_full,
                                            rocsparse_int    partial_len,
                                            bool             gate_fits_int32)
{
    const rocsparse_index_base base = rocsparse_index_base_zero;

    const int64_t expected_ell_nnz = static_cast<int64_t>(N) * static_cast<int64_t>(M);
    const int64_t csr_nnz = static_cast<int64_t>(N) * static_cast<int64_t>(n_full) + partial_len;
    const int64_t gate_numerator = 2 * (csr_nnz - 1);

    // unit_check_* only instantiates integer checks for 32-bit types, so the
    // 64-bit guards are asserted as booleans (they hold => 1) to stay valid in
    // both the test and bench builds.
    unit_check_scalar<int32_t>(partial_len < N ? 1 : 0, 1);
    unit_check_scalar<int32_t>(expected_ell_nnz > static_cast<int64_t>(INT32_MAX) ? 1 : 0, 1);
    unit_check_scalar<int32_t>(csr_nnz < static_cast<int64_t>(INT32_MAX) ? 1 : 0, 1);
    unit_check_scalar<int32_t>(gate_numerator / M + 1 >= N ? 1 : 0, 1);
    unit_check_scalar<int32_t>(
        (gate_numerator <= static_cast<int64_t>(INT32_MAX)) == gate_fits_int32 ? 1 : 0, 1);

    rocsparse_local_handle    handle(arg);
    rocsparse_local_mat_descr descr;
    rocsparse_local_hyb_mat   hyb;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // The row-pointer array is small, so it uses host_dense_vector to keep its
    // footprint traceable; the two csr_nnz-sized arrays stay pageable
    // host_vector to avoid pinning ~8.6 GB.
    host_dense_vector<rocsparse_int> hcsr_row_ptr(M + 1);
    host_vector<rocsparse_int>       hcsr_col_ind(csr_nnz);
    host_vector<float>               hcsr_val(csr_nnz);

    int64_t acc = 0;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        hcsr_row_ptr[i] = static_cast<rocsparse_int>(acc + base);

        const rocsparse_int row_nnz = (i < n_full) ? N : ((i == n_full) ? partial_len : 0);
        for(rocsparse_int j = 0; j < row_nnz; ++j)
        {
            hcsr_col_ind[acc + j] = j + base;
            hcsr_val[acc + j]     = 1.0f;
        }
        acc += row_nnz;
    }
    hcsr_row_ptr[M] = static_cast<rocsparse_int>(acc + base);

    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
    device_vector<float>         dcsr_val(csr_nnz);

    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind, sizeof(rocsparse_int) * csr_nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val, sizeof(float) * csr_nnz, hipMemcpyHostToDevice));

    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<float>(handle,
                                                   M,
                                                   N,
                                                   descr,
                                                   dcsr_val,
                                                   dcsr_row_ptr,
                                                   dcsr_col_ind,
                                                   hyb,
                                                   0,
                                                   rocsparse_hyb_partition_max));

    rocsparse_hyb_mat ptr  = hyb;
    test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

    unit_check_scalar(N, dhyb->ell_width);
    unit_check_scalar<int32_t>(dhyb->ell_nnz > static_cast<int64_t>(INT32_MAX) ? 1 : 0, 1);
    unit_check_scalar<int32_t>(expected_ell_nnz == dhyb->ell_nnz ? 1 : 0, 1);

    // Probe #1: the last physical ELL slot (index ell_nnz - 1). With
    // ELL_IND(i, el, m, width) = el * m + i that slot is (row M - 1, ELL column
    // N - 1). Row M - 1 is empty, so it must hold the padding sentinel (-1).
    rocsparse_int last_col = 0;
    CHECK_HIP_ERROR(hipMemcpy(&last_col,
                              dhyb->ell_col_ind + (expected_ell_nnz - 1),
                              sizeof(rocsparse_int),
                              hipMemcpyDeviceToHost));
    unit_check_scalar<rocsparse_int>(-1, last_col);

    // Probe #2: a real entry past INT32_MAX. Row n_full - 1, ELL column N - 1,
    // idx = (N - 1) * M + (n_full - 1), which stores CSR column N - 1.
    const int64_t probe_idx
        = static_cast<int64_t>(N - 1) * static_cast<int64_t>(M) + static_cast<int64_t>(n_full - 1);
    unit_check_scalar<int32_t>(probe_idx > static_cast<int64_t>(INT32_MAX) ? 1 : 0, 1);
    rocsparse_int probe_col = -2;
    CHECK_HIP_ERROR(hipMemcpy(
        &probe_col, dhyb->ell_col_ind + probe_idx, sizeof(rocsparse_int), hipMemcpyDeviceToHost));
    unit_check_scalar<rocsparse_int>(N - 1 + base, probe_col);
}

void testing_csr2hyb_extra(const Arguments& arg)
{
    // Regression for two 32-bit overflows in rocsparse_csr2hyb on matrices whose
    // padded ELL part exceeds INT32_MAX elements:
    //
    //  (a) ell_nnz = ell_width * m was a 32-bit product. Past INT32_MAX it wrapped
    //      negative, so the ELL arrays were never allocated while the fill kernel
    //      still wrote ell_width * m entries.
    //  (b) The width gate max_row_nnz = 2 * (csr_nnz - 1) / m + 1 was evaluated in
    //      32-bit. Once csr_nnz > 2^30 it went negative and every ell_width was
    //      rejected with rocsparse_status_invalid_value.
    //
    // The gate bounds ell_width * m <= 2 * (csr_nnz - 1) + m, so crossing INT32_MAX
    // needs csr_nnz close to 2^30. csr_row_ptr is 32-bit in the default build, so
    // csr_nnz itself must stay below 2^31; roughly half the rows are dense and the
    // ELL padding pushes ell_width * m past the boundary.
    //
    // Each case needs ~26 GB (CSR input + padded ELL output), so the memory guard
    // in test_csr2hyb.yaml filters this test out on smaller cards.
    const rocsparse_int M = 53688;
    const rocsparse_int N = 40000;

    // Case (a) alone: csr_nnz = 40000 * 26843 + 20000 = 1,073,740,000 <= 2^30, so
    // 2 * (csr_nnz - 1) = 2,147,479,998 fits in 32-bit and the gate gives
    // max_row_nnz = 40000. Only the ell_nnz product (2,147,520,000) overflows.
    testing_csr2hyb_extra_large_ell(arg, M, N, 26843, 20000, true);

    // Cases (a) and (b): csr_nnz = 40000 * 26845 = 1,073,800,000 > 2^30, so the
    // 32-bit gate would also overflow. Probe #2 lands at 39999 * 53688 + 26844
    // = 2,147,493,156.
    testing_csr2hyb_extra_large_ell(arg, M, N, 26845, 0, false);
}
