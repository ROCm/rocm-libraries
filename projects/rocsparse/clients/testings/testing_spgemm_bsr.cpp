/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "auto_testing_bad_arg.hpp"
#include "testing.hpp"

template <typename T, typename I = rocsparse_int, typename J = rocsparse_int>
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr         descr,
                                            device_gebsr_matrix<T, I, J>& bsr_matrix)
{
    return rocsparse_bsr_set_pointers(descr, bsr_matrix.ptr, bsr_matrix.ind, bsr_matrix.val);
}

//
//
//
template <typename I, typename J, typename T>
void testing_spgemm_bsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle handle    = local_handle;
    J                mb        = safe_size;
    J                nb        = safe_size;
    J                kb        = safe_size;
    I                nnzb_A    = safe_size;
    I                nnzb_B    = safe_size;
    I                nnzb_C    = safe_size;
    I                nnzb_D    = safe_size;
    J                block_dim = safe_size;

    void* bsr_row_ptr_A = (void*)0x4;
    void* bsr_col_ind_A = (void*)0x4;
    void* bsr_val_A     = (void*)0x4;
    void* bsr_row_ptr_B = (void*)0x4;
    void* bsr_col_ind_B = (void*)0x4;
    void* bsr_val_B     = (void*)0x4;
    void* bsr_row_ptr_C = (void*)0x4;
    void* bsr_col_ind_C = (void*)0x4;
    void* bsr_val_C     = (void*)0x4;
    void* bsr_row_ptr_D = (void*)0x4;
    void* bsr_col_ind_D = (void*)0x4;
    void* bsr_val_D     = (void*)0x4;

    rocsparse_direction    dir     = rocsparse_direction_row;
    rocsparse_operation    trans_A = rocsparse_operation_none;
    rocsparse_operation    trans_B = rocsparse_operation_none;
    rocsparse_index_base   base    = rocsparse_index_base_zero;
    rocsparse_spgemm_alg   alg     = rocsparse_spgemm_alg_default;
    rocsparse_spgemm_stage stage   = rocsparse_spgemm_stage_compute;

    // Index and data type
    rocsparse_indextype itype        = get_indextype<I>();
    rocsparse_indextype jtype        = get_indextype<J>();
    rocsparse_datatype  compute_type = get_datatype<T>();

    // SpGEMM structures
    rocsparse_local_spmat local_A(mb,
                                  kb,
                                  nnzb_A,
                                  dir,
                                  block_dim,
                                  bsr_row_ptr_A,
                                  bsr_col_ind_A,
                                  bsr_val_A,
                                  itype,
                                  jtype,
                                  base,
                                  compute_type,
                                  rocsparse_format_bsr);
    rocsparse_local_spmat local_B(kb,
                                  nb,
                                  nnzb_B,
                                  dir,
                                  block_dim,
                                  bsr_row_ptr_B,
                                  bsr_col_ind_B,
                                  bsr_val_B,
                                  itype,
                                  jtype,
                                  base,
                                  compute_type,
                                  rocsparse_format_bsr);
    rocsparse_local_spmat local_C(mb,
                                  nb,
                                  nnzb_C,
                                  dir,
                                  block_dim,
                                  bsr_row_ptr_C,
                                  bsr_col_ind_C,
                                  bsr_val_C,
                                  itype,
                                  jtype,
                                  base,
                                  compute_type,
                                  rocsparse_format_bsr);
    rocsparse_local_spmat local_D(mb,
                                  nb,
                                  nnzb_D,
                                  dir,
                                  block_dim,
                                  bsr_row_ptr_D,
                                  bsr_col_ind_D,
                                  bsr_val_D,
                                  itype,
                                  jtype,
                                  base,
                                  compute_type,
                                  rocsparse_format_bsr);

    rocsparse_spmat_descr A = local_A;
    rocsparse_spmat_descr B = local_B;
    rocsparse_spmat_descr C = local_C;
    rocsparse_spmat_descr D = local_D;

    int       nargs_to_exclude   = 4;
    const int args_to_exclude[4] = {3, 6, 12, 13};

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

#define PARAMS                                                                                \
    handle, trans_A, trans_B, alpha, A, B, beta, D, C, compute_type, alg, stage, buffer_size, \
        temp_buffer
    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################
    {
        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)nullptr;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################
    {
        const T* alpha = (const T*)0x4;
        const T* beta  = (const T*)nullptr;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################
    {
        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)0x4;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################
    {
        const T* alpha = (const T*)0x4;
        const T* beta  = (const T*)0x4;

        size_t* buffer_size = (size_t*)0x4;
        void*   temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = (size_t*)0x4;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = (void*)0x4;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);

        buffer_size = nullptr;
        temp_buffer = nullptr;
        select_bad_arg_analysis(rocsparse_spgemm, nargs_to_exclude, args_to_exclude, PARAMS);
    }
#undef PARAMS

    const T* alpha = (const T*)0x4;
    const T* beta  = (const T*)0x4;
    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans_A,
                                             trans_B,
                                             alpha,
                                             A,
                                             B,
                                             beta,
                                             D,
                                             C,
                                             compute_type,
                                             alg,
                                             rocsparse_spgemm_stage_buffer_size,
                                             nullptr,
                                             nullptr),
                            rocsparse_status_invalid_pointer);
}

template <typename I, typename J, typename T>
void testing_spgemm_bsr(const Arguments& arg)
{
    J                     M         = arg.M;
    J                     N         = arg.N;
    J                     K         = arg.K;
    J                     block_dim = arg.block_dim;
    rocsparse_direction   dir       = arg.direction;
    rocsparse_operation   trans_A   = arg.transA;
    rocsparse_operation   trans_B   = arg.transB;
    rocsparse_index_base  base_A    = arg.baseA;
    rocsparse_index_base  base_B    = arg.baseB;
    rocsparse_index_base  base_C    = arg.baseC;
    rocsparse_index_base  base_D    = arg.baseD;
    rocsparse_spgemm_alg  alg       = arg.spgemm_alg;
    static constexpr bool full_rank = false;

    J Mb = (M + block_dim - 1) / block_dim;
    J Nb = (N + block_dim - 1) / block_dim;
    J Kb = (K + block_dim - 1) / block_dim;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

    // -99 means nullptr
    T* h_alpha_ptr = (h_alpha == (T)-99) ? nullptr : &h_alpha;
    T* h_beta_ptr  = (h_beta == (T)-99) ? nullptr : &h_beta;
    T* d_alpha_ptr = (h_alpha == (T)-99) ? nullptr : d_alpha;
    T* d_beta_ptr  = (h_beta == (T)-99) ? nullptr : d_beta;

    // data type
    rocsparse_datatype compute_type = get_datatype<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    if((Mb <= 0 || Nb <= 0 || Kb <= 0 || block_dim <= 0))
    {
        return;
    }

    // Declare host matrices.
    host_gebsr_matrix<T, I, J> hA;

    rocsparse_matrix_factory<T, I, J> matrix_factory(arg, arg.timing ? false : true, full_rank);
    rocsparse_matrix_factory_random<T, I, J> rf(full_rank);

    hA.define(dir, Mb, Kb, 0, block_dim, block_dim, base_A);
    matrix_factory.init_gebsr(hA, Mb, Kb, block_dim, block_dim, base_A);

    M = Mb * hA.row_block_dim;
    K = Kb * hA.col_block_dim;

    host_gebsr_matrix<T, I, J> hB, hC, hD;
    hB.define(dir, Kb, Nb, 0, block_dim, block_dim, base_B);
    hC.define(dir, Mb, Nb, 0, block_dim, block_dim, base_C);
    hD.define(dir, Mb, Nb, 0, block_dim, block_dim, base_D);

    rf.init_gebsr(hB.ptr,
                  hB.ind,
                  hB.val,
                  hB.dir,
                  hB.mb,
                  hB.nb,
                  hB.nnzb,
                  hB.row_block_dim,
                  hB.col_block_dim,
                  hB.base,
                  rocsparse_matrix_type_general,
                  rocsparse_fill_mode_lower,
                  rocsparse_storage_mode_sorted);
    rf.init_gebsr(hD.ptr,
                  hD.ind,
                  hD.val,
                  hD.dir,
                  hD.mb,
                  hD.nb,
                  hD.nnzb,
                  hD.row_block_dim,
                  hD.col_block_dim,
                  hD.base,
                  rocsparse_matrix_type_general,
                  rocsparse_fill_mode_lower,
                  rocsparse_storage_mode_sorted);

    device_gebsr_matrix<T, I, J> dA(hA);
    device_gebsr_matrix<T, I, J> dB(hB);
    device_gebsr_matrix<T, I, J> dC(hC);
    device_gebsr_matrix<T, I, J> dD(hD);

    // Declare local spmat.
    rocsparse_local_spmat A(dA), B(dB), C(dC), D(dD);

    size_t buffer_size;
    void*  dbuffer = nullptr;

    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans_A,
                                             trans_B,
                                             h_alpha_ptr,
                                             A,
                                             B,
                                             h_beta_ptr,
                                             D,
                                             C,
                                             compute_type,
                                             alg,
                                             rocsparse_spgemm_stage_buffer_size,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_success);

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                             trans_A,
                                             trans_B,
                                             h_alpha_ptr,
                                             A,
                                             B,
                                             h_beta_ptr,
                                             D,
                                             C,
                                             compute_type,
                                             alg,
                                             rocsparse_spgemm_stage_nnz,
                                             &buffer_size,
                                             dbuffer),
                            rocsparse_status_success);

    // Update memory
    int64_t rows_C;
    int64_t cols_C;
    int64_t nnzb_C;
    CHECK_ROCSPARSE_ERROR(rocsparse_spmat_get_size(C, &rows_C, &cols_C, &nnzb_C));

    if(nnzb_C > 0)
    {
        dC.define(dC.dir, dC.mb, dC.nb, nnzb_C, dC.row_block_dim, dC.col_block_dim, dC.base);
        CHECK_ROCSPARSE_ERROR(rocsparse_bsr_set_pointers(C, dC));
    }

    if(arg.unit_check)
    {
        // Compute C on host.
        I hC_nnzb = 0;
        host_bsrgemm_nnzb<T, I, J>(hA.mb,
                                   hC.nb,
                                   hA.nb,
                                   hA.row_block_dim,
                                   h_alpha_ptr,
                                   hA.ptr,
                                   hA.ind,
                                   hB.ptr,
                                   hB.ind,
                                   h_beta_ptr,
                                   hD.ptr,
                                   hD.ind,
                                   hC.ptr,
                                   &hC_nnzb,
                                   hA.base,
                                   hB.base,
                                   hC.base,
                                   hD.base);

        hC.define(hC.dir, hC.mb, hC.nb, hC_nnzb, hC.row_block_dim, hC.col_block_dim, hC.base);

        host_bsrgemm<T, I, J>(dir,
                              hA.mb,
                              hC.nb,
                              hA.nb,
                              hA.row_block_dim,
                              h_alpha_ptr,
                              hA.ptr,
                              hA.ind,
                              hA.val,
                              hB.ptr,
                              hB.ind,
                              hB.val,
                              h_beta_ptr,
                              hD.ptr,
                              hD.ind,
                              hD.val,
                              hC.ptr,
                              hC.ind,
                              hC.val,
                              hA.base,
                              hB.base,
                              hC.base,
                              hD.base);

        // Compute C (host pointer)
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 h_alpha_ptr,
                                                 A,
                                                 B,
                                                 h_beta_ptr,
                                                 D,
                                                 C,
                                                 compute_type,
                                                 alg,
                                                 rocsparse_spgemm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer),
                                rocsparse_status_success);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC pointer mode host", dC);
        }

        // Check
        hC.near_check(dC);

        // Compute C (device pointer)
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        EXPECT_ROCSPARSE_STATUS(rocsparse_spgemm(handle,
                                                 trans_A,
                                                 trans_B,
                                                 d_alpha_ptr,
                                                 A,
                                                 B,
                                                 d_beta_ptr,
                                                 D,
                                                 C,
                                                 compute_type,
                                                 alg,
                                                 rocsparse_spgemm_stage_compute,
                                                 &buffer_size,
                                                 dbuffer),
                                rocsparse_status_success);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC pointer mode device", dC);
        }

        // Check
        hC.near_check(dC);
    }

    if(arg.timing)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_analysis_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spgemm,
                                               handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               compute_type,
                                               alg,
                                               rocsparse_spgemm_stage_nnz,
                                               &buffer_size,
                                               dbuffer);

        const double gpu_solve_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spgemm,
                                               handle,
                                               trans_A,
                                               trans_B,
                                               h_alpha_ptr,
                                               A,
                                               B,
                                               h_beta_ptr,
                                               D,
                                               C,
                                               compute_type,
                                               alg,
                                               rocsparse_spgemm_stage_compute,
                                               &buffer_size,
                                               dbuffer);

        double gflop_count = bsrgemm_gflop_count<T, I, J>(
            Mb, hA.row_block_dim, h_alpha_ptr, hA.ptr, hA.ind, hB.ptr, h_beta_ptr, hD.ptr, hA.base);
        double gbyte_count = bsrgemm_gbyte_count<T, I, J>(Mb,
                                                          Nb,
                                                          Kb,
                                                          hA.row_block_dim,
                                                          hA.nnzb,
                                                          hB.nnzb,
                                                          hC.nnzb,
                                                          hD.nnzb,
                                                          h_alpha_ptr,
                                                          h_beta_ptr);

        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);
        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);

        display_timing_info(display_key_t::trans_A,
                            rocsparse_operation2string(trans_A),
                            display_key_t::trans_B,
                            rocsparse_operation2string(trans_B),
                            display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::Kb,
                            Kb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnzb_A,
                            dA.nnzb,
                            display_key_t::nnzb_B,
                            dB.nnzb,
                            display_key_t::nnzb_C,
                            dC.nnzb,
                            display_key_t::nnzb_D,
                            dD.nnzb,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::beta,
                            h_beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::analysis_time_ms,
                            get_gpu_time_msec(gpu_analysis_time_used),
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_spgemm_bsr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spgemm_bsr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_spgemm_bsr_extra(const Arguments& arg) {}
