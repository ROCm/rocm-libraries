/*! \file */
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

//
//
//
template <typename T>
void testing_bsrgemm_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;
    T                   h_alpha   = 0.6;
    T                   h_beta    = 0.2;

    host_dense_vector<rocsparse_int> hbsr_row_ptr_C(safe_size + 1);
    hbsr_row_ptr_C[0] = 0;
    hbsr_row_ptr_C[1] = 1;
    device_dense_vector<rocsparse_int> dbsr_row_ptr_C(hbsr_row_ptr_C);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;
    // Create matrix descriptors
    rocsparse_local_mat_descr local_descr_A;
    rocsparse_local_mat_descr local_descr_B;
    rocsparse_local_mat_descr local_descr_C;
    rocsparse_local_mat_descr local_descr_D;

    // Create info desciptor
    rocsparse_local_mat_info local_info_C;

    rocsparse_handle    handle      = local_handle;
    rocsparse_direction dir         = rocsparse_direction_row;
    rocsparse_operation trans_A     = rocsparse_operation_none;
    rocsparse_operation trans_B     = rocsparse_operation_none;
    rocsparse_int       mb          = safe_size;
    rocsparse_int       nb          = safe_size;
    rocsparse_int       kb          = safe_size;
    rocsparse_mat_info  info_C      = local_info_C;
    size_t*             buffer_size = (size_t*)0x4;
    void*               temp_buffer = (void*)0x4;

    // C matrix
    const rocsparse_mat_descr descr_C       = local_descr_C;
    T*                        bsr_val_C     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr_C = (rocsparse_int*)dbsr_row_ptr_C;
    rocsparse_int*            bsr_col_ind_C = (rocsparse_int*)0x4;
    rocsparse_int*            nnzb_C        = (rocsparse_int*)0x4;

#define PARAMS_BUFFER_SIZE                                                                       \
    handle, dir, trans_A, trans_B, mb, nb, kb, block_dim, alpha, descr_A, nnzb_A, bsr_row_ptr_A, \
        bsr_col_ind_A, descr_B, nnzb_B, bsr_row_ptr_B, bsr_col_ind_B, beta, descr_D, nnzb_D,     \
        bsr_row_ptr_D, bsr_col_ind_D, info_C, buffer_size

#define PARAMS_NNZB                                                                       \
    handle, dir, trans_A, trans_B, mb, nb, kb, block_dim, descr_A, nnzb_A, bsr_row_ptr_A, \
        bsr_col_ind_A, descr_B, nnzb_B, bsr_row_ptr_B, bsr_col_ind_B, descr_D, nnzb_D,    \
        bsr_row_ptr_D, bsr_col_ind_D, descr_C, bsr_row_ptr_C, nnzb_C, info_C, temp_buffer

#define PARAMS                                                                                  \
    handle, dir, trans_A, trans_B, mb, nb, kb, block_dim, alpha, descr_A, nnzb_A, bsr_val_A,    \
        bsr_row_ptr_A, bsr_col_ind_A, descr_B, nnzb_B, bsr_val_B, bsr_row_ptr_B, bsr_col_ind_B, \
        beta, descr_D, nnzb_D, bsr_val_D, bsr_row_ptr_D, bsr_col_ind_D, descr_C, bsr_val_C,     \
        bsr_row_ptr_C, bsr_col_ind_C, info_C, temp_buffer

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // ###############################################
    // Scenario 1: alpha == nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A == B == D == nullptr
        static constexpr int nargs_to_exclude_buffer_size = 14;
        static constexpr int nargs_to_exclude_nnzb        = 16;
        static constexpr int nargs_to_exclude             = 22;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
        const int args_to_exclude_nnzb[nargs_to_exclude_nnzb]
            = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24};
        const int args_to_exclude[nargs_to_exclude] = {8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30};

        const T* alpha = (const T*)nullptr;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_A        = 0;
        const T*                  bsr_val_A     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)nullptr;
        rocsparse_int             block_dim     = 2;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_B        = 0;
        const T*                  bsr_val_B     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_D        = 0;
        const T*                  bsr_val_D     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_D = (const rocsparse_int*)nullptr;
        select_bad_arg_analysis(rocsparse_bsrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);

        select_bad_arg_analysis(
            rocsparse_bsrgemm_nnzb, nargs_to_exclude_nnzb, args_to_exclude_nnzb, PARAMS_NNZB);

        select_bad_arg_analysis(rocsparse_bsrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 2: alpha != nullptr && beta == nullptr
    // ###############################################
    {
        // In this scenario matrices A != B != nullptr and D == nullptr
        static constexpr int nargs_to_exclude_buffer_size = 6;
        static constexpr int nargs_to_exclude_nnzb        = 5;
        static constexpr int nargs_to_exclude             = 7;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {8, 17, 18, 19, 20, 21};
        const int args_to_exclude_nnzb[nargs_to_exclude_nnzb] = {16, 17, 18, 19, 24};
        const int args_to_exclude[nargs_to_exclude]           = {19, 20, 21, 22, 23, 24, 30};

        const T* alpha = &h_alpha;
        const T* beta  = (const T*)nullptr;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnzb_A        = safe_size;
        const T*                  bsr_val_A     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)0x4;
        rocsparse_int             block_dim     = 2;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnzb_B        = safe_size;
        const T*                  bsr_val_B     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_D        = 0;
        const T*                  bsr_val_D     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_D = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_D = (const rocsparse_int*)nullptr;

        select_bad_arg_analysis(rocsparse_bsrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);
        select_bad_arg_analysis(
            rocsparse_bsrgemm_nnzb, nargs_to_exclude_nnzb, args_to_exclude_nnzb, PARAMS_NNZB);
        select_bad_arg_analysis(rocsparse_bsrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 3: alpha == nullptr && beta != nullptr
    // ###############################################

    {
        // In this scenario matrices A == B == nullptr and D != nullptr
        static constexpr int nargs_to_exclude_buffer_size = 11;
        static constexpr int nargs_to_exclude_nnzb        = 9;
        static constexpr int nargs_to_exclude             = 12;

        const int args_to_exclude_buffer_size[nargs_to_exclude_buffer_size]
            = {6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        const int args_to_exclude_nnzb[nargs_to_exclude_nnzb] = {8, 9, 10, 11, 12, 13, 14, 15, 24};
        const int args_to_exclude[nargs_to_exclude]
            = {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30};

        const T* alpha = (const T*)nullptr;
        const T* beta  = &h_beta;

        // A matrix
        const rocsparse_mat_descr descr_A       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_A        = 0;
        const T*                  bsr_val_A     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)nullptr;
        rocsparse_int             block_dim     = 2;

        // B matrix
        const rocsparse_mat_descr descr_B       = (const rocsparse_mat_descr) nullptr;
        rocsparse_int             nnzb_B        = 0;
        const T*                  bsr_val_B     = (const T*)nullptr;
        const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)nullptr;
        const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)nullptr;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnzb_D        = safe_size;
        const T*                  bsr_val_D     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_D = (const rocsparse_int*)0x4;

        select_bad_arg_analysis(rocsparse_bsrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);
        select_bad_arg_analysis(
            rocsparse_bsrgemm_nnzb, nargs_to_exclude_nnzb, args_to_exclude_nnzb, PARAMS_NNZB);
        select_bad_arg_analysis(rocsparse_bsrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // ###############################################
    // Scenario 4: alpha != nullptr && beta != nullptr
    // ###############################################

    {
        // In this scenario matrices A != B != D != nullptr
        int nargs_to_exclude_buffer_size = 2;
        int nargs_to_exclude             = 2;

        const int args_to_exclude_buffer_size[2] = {8, 17};
        const int args_to_exclude[2]             = {8, 19};

        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;

        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnzb_A        = safe_size;
        const T*                  bsr_val_A     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)0x4;
        rocsparse_int             block_dim     = 2;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnzb_B        = safe_size;
        const T*                  bsr_val_B     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnzb_D        = safe_size;
        const T*                  bsr_val_D     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_D = (const rocsparse_int*)0x4;

        select_bad_arg_analysis(rocsparse_bsrgemm_buffer_size<T>,
                                nargs_to_exclude_buffer_size,
                                args_to_exclude_buffer_size,
                                PARAMS_BUFFER_SIZE);
        bad_arg_analysis(rocsparse_bsrgemm_nnzb, PARAMS_NNZB);
        select_bad_arg_analysis(rocsparse_bsrgemm<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    // Not implemented cases.
    {
        const T* alpha = &h_alpha;
        const T* beta  = &h_beta;
        // A matrix
        const rocsparse_mat_descr descr_A       = local_descr_A;
        rocsparse_int             nnzb_A        = safe_size;
        const T*                  bsr_val_A     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_A = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_A = (const rocsparse_int*)0x4;
        rocsparse_int             block_dim     = safe_size;

        // B matrix
        const rocsparse_mat_descr descr_B       = local_descr_B;
        rocsparse_int             nnzb_B        = safe_size;
        const T*                  bsr_val_B     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_B = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_B = (const rocsparse_int*)0x4;

        // D matrix
        const rocsparse_mat_descr descr_D       = local_descr_D;
        rocsparse_int             nnzb_D        = safe_size;
        const T*                  bsr_val_D     = (const T*)0x4;
        const rocsparse_int*      bsr_row_ptr_D = (const rocsparse_int*)0x4;
        const rocsparse_int*      bsr_col_ind_D = (const rocsparse_int*)0x4;

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_buffer_size<T>(PARAMS_BUFFER_SIZE),
                                    rocsparse_status_not_implemented);
            trans_B = op;
        }

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_nnzb(PARAMS_NNZB),
                                    rocsparse_status_not_implemented);
            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_nnzb(PARAMS_NNZB),
                                    rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_nnzb(PARAMS_NNZB),
                                    rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm_nnzb(PARAMS_NNZB),
                                    rocsparse_status_not_implemented);
            trans_B = op;
        }

        {
            rocsparse_operation op = trans_A;
            trans_A                = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_A = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_A = op;

            op      = trans_B;
            trans_B = rocsparse_operation_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_B = rocsparse_operation_conjugate_transpose;
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrgemm<T>(PARAMS), rocsparse_status_not_implemented);
            trans_B = op;
        }
    }

#undef PARAMS
#undef PARAMS_NNZB
#undef PARAMS_BUFFER_SIZE
}

enum testing_bsrgemm_scenario
{
    testing_bsrgemm_scenario_none,
    testing_bsrgemm_scenario_alpha,
    testing_bsrgemm_scenario_beta,
    testing_bsrgemm_scenario_alpha_and_beta,
};

template <typename T>
void testing_bsrgemm(const Arguments& arg)
{
    rocsparse_int         M         = arg.M;
    rocsparse_int         N         = arg.N;
    rocsparse_int         K         = arg.K;
    rocsparse_int         block_dim = arg.block_dim;
    rocsparse_direction   dir       = arg.direction;
    rocsparse_operation   transA    = arg.transA;
    rocsparse_operation   transB    = arg.transB;
    rocsparse_index_base  baseA     = arg.baseA;
    rocsparse_index_base  baseB     = arg.baseB;
    rocsparse_index_base  baseC     = arg.baseC;
    rocsparse_index_base  baseD     = arg.baseD;
    static constexpr bool full_rank = false;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;
    rocsparse_int Kb = (K + block_dim - 1) / block_dim;

    T v_alpha = arg.get_alpha<T>(), v_beta = arg.get_beta<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr descrA;
    rocsparse_local_mat_descr descrB;
    rocsparse_local_mat_descr descrC;
    rocsparse_local_mat_descr descrD;
    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrC, baseC));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrD, baseD));

    // Create matrix info for C
    rocsparse_local_mat_info info;

    void* dbuffer = nullptr;

#define PARAMS_BUFFER_SIZE(alpha, beta, A, B, C, D, out_buffer_size)                               \
    handle, dir, transA, transB, A.mb, C.nb, A.nb, block_dim, alpha, descrA, A.nnzb, A.ptr, A.ind, \
        descrB, B.nnzb, B.ptr, B.ind, beta, descrD, D.nnzb, D.ptr, D.ind, info, &out_buffer_size

#define PARAMS_NNZB(A, B, C, D, out_nnzb)                                                          \
    handle, dir, transA, transB, A.mb, C.nb, A.nb, block_dim, descrA, A.nnzb, A.ptr, A.ind,        \
        descrB, B.nnzb, B.ptr, B.ind, descrD, D.nnzb, D.ptr, D.ind, descrC, C.ptr, out_nnzb, info, \
        dbuffer

#define PARAMS(alpha, beta, A, B, C, D)                                                            \
    handle, dir, transA, transB, A.mb, C.nb, A.nb, block_dim, alpha, descrA, A.nnzb, A.val, A.ptr, \
        A.ind, descrB, B.nnzb, B.val, B.ptr, B.ind, beta, descrD, D.nnzb, D.val, D.ptr, D.ind,     \
        descrC, C.val, C.ptr, C.ind, info, dbuffer

    // 4 Scenarios need to be tested:

    // Scenario 1: alpha == nullptr && beta == nullptr
    // Scenario 2: alpha != nullptr && beta == nullptr
    // Scenario 3: alpha == nullptr && beta != nullptr
    // Scenario 4: alpha != nullptr && beta != nullptr

    // alpha == -99 means test for alpha == nullptr
    // beta  == -99 means test for beta == nullptr
    testing_bsrgemm_scenario scenario = testing_bsrgemm_scenario_none;
    if(v_alpha != static_cast<T>(-99) && v_beta == static_cast<T>(-99))
    {
        scenario = testing_bsrgemm_scenario_alpha;
    }
    else if(v_alpha == static_cast<T>(-99) && v_beta != static_cast<T>(-99))
    {
        scenario = testing_bsrgemm_scenario_beta;
    }
    else if(v_alpha != static_cast<T>(-99) && v_beta != static_cast<T>(-99))
    {
        scenario = testing_bsrgemm_scenario_alpha_and_beta;
    }

    host_dense_vector<T> h_alpha(0), h_beta(0);
    switch(scenario)
    {
    case testing_bsrgemm_scenario_none:
    {
        break;
    }
    case testing_bsrgemm_scenario_alpha:
    {
        h_alpha.resize(1);
        *h_alpha = v_alpha;
        break;
    }
    case testing_bsrgemm_scenario_beta:
    {
        h_beta.resize(1);
        *h_beta = v_beta;
        break;
    }
    case testing_bsrgemm_scenario_alpha_and_beta:
    {
        h_alpha.resize(1);
        *h_alpha = v_alpha;
        h_beta.resize(1);
        *h_beta = v_beta;
        break;
    }
    }

    // Declare host and device objects.
    host_gebsr_matrix<T>   h_A, h_B, h_C, h_D;
    device_gebsr_matrix<T> d_A;

    // Initialize matrices.
    rocsparse_matrix_factory<T>        matrix_factory(arg, arg.timing ? false : true, full_rank);
    rocsparse_matrix_factory_random<T> rf(full_rank);

    matrix_factory.init_bsr(h_A, d_A, Mb, Kb, baseA);

    M = Mb * d_A.row_block_dim;
    K = Kb * d_A.col_block_dim;

    switch(scenario)
    {
    case testing_bsrgemm_scenario_none:
    {
        break;
    }
    case testing_bsrgemm_scenario_alpha:
    {
        h_B.define(dir, Kb, Nb, 0, block_dim, block_dim, baseB);
        rf.init_gebsr(h_B.ptr,
                      h_B.ind,
                      h_B.val,
                      h_B.dir,
                      h_B.mb,
                      h_B.nb,
                      h_B.nnzb,
                      h_B.row_block_dim,
                      h_B.col_block_dim,
                      h_B.base,
                      rocsparse_matrix_type_general,
                      rocsparse_fill_mode_lower,
                      rocsparse_storage_mode_sorted);

        break;
    }
    case testing_bsrgemm_scenario_beta:
    {
        h_D.define(dir, Mb, Nb, 0, block_dim, block_dim, baseD);
        rf.init_gebsr(h_D.ptr,
                      h_D.ind,
                      h_D.val,
                      h_D.dir,
                      h_D.mb,
                      h_D.nb,
                      h_D.nnzb,
                      h_D.row_block_dim,
                      h_D.col_block_dim,
                      h_D.base,
                      rocsparse_matrix_type_general,
                      rocsparse_fill_mode_lower,
                      rocsparse_storage_mode_sorted);
        break;
    }
    case testing_bsrgemm_scenario_alpha_and_beta:
    {
        h_B.define(dir, Kb, Nb, 0, block_dim, block_dim, baseB);
        rf.init_gebsr(h_B.ptr,
                      h_B.ind,
                      h_B.val,
                      h_B.dir,
                      h_B.mb,
                      h_B.nb,
                      h_B.nnzb,
                      h_B.row_block_dim,
                      h_B.col_block_dim,
                      h_B.base,
                      rocsparse_matrix_type_general,
                      rocsparse_fill_mode_lower,
                      rocsparse_storage_mode_sorted);

        h_D.define(dir, Mb, Nb, 0, block_dim, block_dim, baseD);
        rf.init_gebsr(h_D.ptr,
                      h_D.ind,
                      h_D.val,
                      h_D.dir,
                      h_D.mb,
                      h_D.nb,
                      h_D.nnzb,
                      h_D.row_block_dim,
                      h_D.col_block_dim,
                      h_D.base,
                      rocsparse_matrix_type_general,
                      rocsparse_fill_mode_lower,
                      rocsparse_storage_mode_sorted);
        break;
    }
    }

    h_C.define(dir, Mb, Nb, 0, block_dim, block_dim, baseC);

    // Declare device objects.
    device_gebsr_matrix<T> d_B(h_B), d_C(h_C), d_D(h_D);
    device_dense_vector<T> d_alpha(h_alpha), d_beta(h_beta);

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
    CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm_buffer_size<T>(
        PARAMS_BUFFER_SIZE(h_alpha, h_beta, d_A, d_B, d_C, d_D, buffer_size)));

    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // normalize
        rocsparse_vector_utils<T>::normalize(h_A.val);

        d_A.val.transfer_from(h_A.val);

        // Host calculation
        rocsparse_int out_nnzb;

        host_bsrgemm_nnzb<T, rocsparse_int, rocsparse_int>(h_A.mb,
                                                           h_C.nb,
                                                           h_A.nb,
                                                           h_A.row_block_dim,
                                                           h_alpha,
                                                           h_A.ptr,
                                                           h_A.ind,
                                                           h_B.ptr,
                                                           h_B.ind,
                                                           h_beta,
                                                           h_D.ptr,
                                                           h_D.ind,
                                                           h_C.ptr,
                                                           &out_nnzb,
                                                           h_A.base,
                                                           h_B.base,
                                                           h_C.base,
                                                           h_D.base);

        h_C.define(
            h_C.dir, h_C.mb, h_C.nb, out_nnzb, h_C.row_block_dim, h_C.col_block_dim, h_C.base);

        host_bsrgemm<T, rocsparse_int, rocsparse_int>(dir,
                                                      h_A.mb,
                                                      h_C.nb,
                                                      h_A.nb,
                                                      h_A.row_block_dim,
                                                      h_alpha,
                                                      h_A.ptr,
                                                      h_A.ind,
                                                      h_A.val,
                                                      h_B.ptr,
                                                      h_B.ind,
                                                      h_B.val,
                                                      h_beta,
                                                      h_D.ptr,
                                                      h_D.ind,
                                                      h_D.val,
                                                      h_C.ptr,
                                                      h_C.ind,
                                                      h_C.val,
                                                      h_A.base,
                                                      h_B.base,
                                                      h_C.base,
                                                      h_D.base);

        // GPU with pointer mode host
        host_scalar<rocsparse_int> h_out_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm_nnzb(PARAMS_NNZB(d_A, d_B, d_C, d_D, h_out_nnz)));

        d_C.define(
            d_C.dir, d_C.mb, d_C.nb, *h_out_nnz, d_C.row_block_dim, d_C.col_block_dim, d_C.base);
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm<T>(PARAMS(h_alpha, h_beta, d_A, d_B, d_C, d_D)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode host", d_C);
        }
        h_C.near_check(d_C);

        // GPU with pointer mode host
        device_scalar<rocsparse_int> d_out_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm_nnzb(PARAMS_NNZB(d_A, d_B, d_C, d_D, d_out_nnz)));
        host_scalar<rocsparse_int> h_out_nnz2(d_out_nnz);
        d_C.define(
            d_C.dir, d_C.mb, d_C.nb, *h_out_nnz2, d_C.row_block_dim, d_C.col_block_dim, d_C.base);
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm<T>(PARAMS(d_alpha, d_beta, d_A, d_B, d_C, d_D)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("C pointer mode device", d_C);
        }
        h_C.near_check(d_C);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        // Warm up
        rocsparse_int out_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_bsrgemm_nnzb(PARAMS_NNZB(d_A, d_B, d_C, d_D, &out_nnz)));
        d_C.define(
            d_C.dir, d_C.mb, d_C.nb, out_nnz, d_C.row_block_dim, d_C.col_block_dim, d_C.base);

        const double gpu_solve_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_bsrgemm<T>, PARAMS(h_alpha, h_beta, d_A, d_B, d_C, d_D));

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        double gflop_count = bsrgemm_gflop_count<T, rocsparse_int, rocsparse_int>(
            Mb, d_A.row_block_dim, h_alpha, h_A.ptr, h_A.ind, h_B.ptr, h_beta, h_D.ptr, h_A.base);
        double gbyte_count = bsrgemm_gbyte_count<T, rocsparse_int, rocsparse_int>(
            Mb, Nb, Kb, d_A.row_block_dim, d_A.nnzb, d_B.nnzb, d_C.nnzb, d_D.nnzb, h_alpha, h_beta);

        double gpu_gflops = get_gpu_gflops(gpu_solve_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_solve_time_used, gbyte_count);

        char alpha[32], beta[32];
        sprintf(alpha, "null");
        sprintf(beta, "null");
        if(h_alpha.data() != nullptr)
        {
            std::stringstream ss;
            ss << *h_alpha;
            sprintf(alpha, "%s", ss.str().c_str());
        }

        if(h_beta.data() != nullptr)
        {
            std::stringstream ss;
            ss << *h_beta;
            sprintf(beta, "%s", ss.str().c_str());
        }

        display_timing_info(display_key_t::trans_A,
                            rocsparse_operation2string(transA),
                            display_key_t::trans_B,
                            rocsparse_operation2string(transB),
                            display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::Kb,
                            Kb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnzb_A,
                            d_A.nnzb,
                            display_key_t::nnzb_B,
                            d_B.nnzb,
                            display_key_t::nnzb_C,
                            d_C.nnzb,
                            display_key_t::nnzb_D,
                            d_D.nnzb,
                            display_key_t::alpha,
                            alpha,
                            display_key_t::beta,
                            beta,
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_solve_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));

#undef PARAMS
#undef PARAMS_NNZB
#undef PARAMS_BUFFER_SIZE
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsrgemm_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrgemm<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrgemm_extra(const Arguments& arg) {}
