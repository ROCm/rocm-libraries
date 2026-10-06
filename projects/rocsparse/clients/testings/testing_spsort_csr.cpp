/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <algorithm>
#include <numeric>

namespace
{
    // Sorts the column indices within each row.
    template <typename I, typename J, typename T>
    void host_spsort_csr(host_csr_matrix<T, I, J>& A)
    {
        for(int64_t i = 0; i < A.m; ++i)
        {
            const int64_t start = A.ptr[i] - A.base;
            const int64_t end   = A.ptr[i + 1] - A.base;

            std::vector<int64_t> perm(end - start);
            std::iota(perm.begin(), perm.end(), start);
            std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
                return A.ind[a] < A.ind[b];
            });

            std::vector<J> sorted_ind(end - start);
            std::vector<T> sorted_val(end - start);
            for(int64_t k = 0; k < end - start; ++k)
            {
                sorted_ind[k] = A.ind[perm[k]];
                sorted_val[k] = A.val[perm[k]];
            }

            std::copy(sorted_ind.begin(), sorted_ind.end(), A.ind.data() + start);
            std::copy(sorted_val.begin(), sorted_val.end(), A.val.data() + start);
        }
    }

    // Shuffles the entries within each row.
    template <typename I, typename J, typename T>
    void host_shuffle_csr(host_csr_matrix<T, I, J>& A)
    {
        for(int64_t i = 0; i < A.m; ++i)
        {
            const int64_t start = A.ptr[i] - A.base;
            const int64_t end   = A.ptr[i + 1] - A.base;
            for(int64_t k = start; k < end; ++k)
            {
                const int64_t j = random_generator<int64_t>(start, end - 1);
                std::swap(A.ind[k], A.ind[j]);
                std::swap(A.val[k], A.val[j]);
            }
        }
    }

    void set_spsort_inputs(rocsparse_handle       handle,
                           rocsparse_spsort_descr descr,
                           rocsparse_spsort_alg   alg)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
            handle, descr, rocsparse_spsort_input_alg, &alg, sizeof(alg), nullptr));
    }

    // Runs the analysis and compute stages of the sort of source into target.
    void run_spsort(rocsparse_handle            handle,
                    rocsparse_const_spmat_descr source,
                    rocsparse_spmat_descr       target)
    {
        rocsparse_spsort_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
        set_spsort_inputs(handle, descr, rocsparse_spsort_alg_default);

        for(const rocsparse_spsort_stage stage :
            {rocsparse_spsort_stage_analysis, rocsparse_spsort_stage_compute})
        {
            size_t buffer_size = 0;
            CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
                handle, descr, source, target, stage, &buffer_size, nullptr));
            void* dbuffer = nullptr;
            CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
            CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
                handle, descr, source, target, stage, buffer_size, dbuffer, nullptr));
            CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
    }
}

template <typename I, typename J, typename T>
void testing_spsort_csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    device_vector<I> d_csr_row_ptr(safe_size + 1);
    device_vector<J> d_csr_col_ind(safe_size);
    device_vector<T> d_csr_val(safe_size);

    const rocsparse_spsort_alg alg = rocsparse_spsort_alg_default;

    size_t buffer_size;

    auto make_mat = [&](rocsparse_datatype ttype = get_datatype<T>()) {
        return rocsparse_local_spmat(safe_size,
                                     safe_size,
                                     safe_size,
                                     d_csr_row_ptr,
                                     d_csr_col_ind,
                                     d_csr_val,
                                     get_indextype<I>(),
                                     get_indextype<J>(),
                                     rocsparse_index_base_zero,
                                     ttype);
    };

    // The direction is not needed, and is ignored if it is set.
    {
        rocsparse_local_spmat mat = make_mat();

        auto expect_success = [&](const rocsparse_direction* dir) {
            rocsparse_spsort_descr descr;
            CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
            set_spsort_inputs(handle, descr, alg);
            if(dir != nullptr)
            {
                CHECK_ROCSPARSE_ERROR(rocsparse_spsort_set_input(
                    handle, descr, rocsparse_spsort_input_direction, dir, sizeof(*dir), nullptr));
            }
            CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
                handle, descr, mat, mat, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));
            CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
        };

        const rocsparse_direction row    = rocsparse_direction_row;
        const rocsparse_direction column = rocsparse_direction_column;
        expect_success(nullptr);
        expect_success(&row);
        expect_success(&column);
    }

    // Batch checks.
    {
        rocsparse_spsort_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
        set_spsort_inputs(handle, descr, alg);

        auto expect_batch_status = [&](int64_t          batch_count_A,
                                       int64_t          offsets_batch_stride_A,
                                       int64_t          columns_values_batch_stride_A,
                                       int64_t          batch_count_B,
                                       int64_t          offsets_batch_stride_B,
                                       int64_t          columns_values_batch_stride_B,
                                       rocsparse_status status) {
            rocsparse_local_spmat mat_A = make_mat();
            rocsparse_local_spmat mat_B = make_mat();
            CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_strided_batch(
                mat_A, batch_count_A, offsets_batch_stride_A, columns_values_batch_stride_A));
            CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_strided_batch(
                mat_B, batch_count_B, offsets_batch_stride_B, columns_values_batch_stride_B));
            EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_buffer_size(handle,
                                                                 descr,
                                                                 mat_A,
                                                                 mat_B,
                                                                 rocsparse_spsort_stage_analysis,
                                                                 &buffer_size,
                                                                 nullptr),
                                    status);
            if(status != rocsparse_status_success)
            {
                EXPECT_ROCSPARSE_STATUS(rocsparse_spsort(handle,
                                                         descr,
                                                         mat_A,
                                                         mat_B,
                                                         rocsparse_spsort_stage_analysis,
                                                         0,
                                                         nullptr,
                                                         nullptr),
                                        status);
            }
        };

        // clang-format off
        // A single source matrix with zero batch strides is sorted into every batch.
        expect_batch_status(1, 0, 0, 2, safe_size + 1, safe_size, rocsparse_status_success);
        expect_batch_status(1, 0, 0, 2, 0, safe_size, rocsparse_status_success);

        // Different batch counts.
        expect_batch_status(1, safe_size + 1, 0, 2, safe_size + 1, safe_size, rocsparse_status_invalid_value);
        expect_batch_status(1, 0, safe_size, 2, safe_size + 1, safe_size, rocsparse_status_invalid_value);
        expect_batch_status(2, safe_size + 1, safe_size, 3, safe_size + 1, safe_size, rocsparse_status_invalid_value);
        expect_batch_status(2, safe_size + 1, safe_size, 1, 0, 0, rocsparse_status_invalid_value);

        // Batch strides that make the batches overlap.
        expect_batch_status(2, safe_size + 1, safe_size - 1, 2, safe_size + 1, safe_size, rocsparse_status_invalid_size);
        expect_batch_status(2, safe_size + 1, safe_size, 2, safe_size + 1, safe_size - 1, rocsparse_status_invalid_size);
        expect_batch_status(2, safe_size, safe_size, 2, safe_size + 1, safe_size, rocsparse_status_invalid_size);
        expect_batch_status(2, safe_size + 1, safe_size, 2, safe_size, safe_size, rocsparse_status_invalid_size);
        expect_batch_status(1, 0, 0, 2, safe_size + 1, 0, rocsparse_status_invalid_size);

        // Either matrix may share its row pointer between batches.
        expect_batch_status(2, safe_size + 1, safe_size, 2, 0, safe_size, rocsparse_status_success);
        expect_batch_status(2, 0, safe_size, 2, 0, safe_size, rocsparse_status_success);
        expect_batch_status(2, 0, safe_size, 2, safe_size + 1, safe_size, rocsparse_status_success);
        // clang-format on

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
    }

    // Unsupported data types.
    {
        rocsparse_spsort_descr descr;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
        set_spsort_inputs(handle, descr, alg);

        for(const rocsparse_datatype unsupported_ttype :
            {rocsparse_datatype_u8_r, rocsparse_datatype_i32_r, rocsparse_datatype_u32_r})
        {
            rocsparse_local_spmat mat_A = make_mat(unsupported_ttype);
            rocsparse_local_spmat mat_B = make_mat(unsupported_ttype);
            EXPECT_ROCSPARSE_STATUS(rocsparse_spsort_buffer_size(handle,
                                                                 descr,
                                                                 mat_A,
                                                                 mat_B,
                                                                 rocsparse_spsort_stage_analysis,
                                                                 &buffer_size,
                                                                 nullptr),
                                    rocsparse_status_invalid_value);
            EXPECT_ROCSPARSE_STATUS(rocsparse_spsort(handle,
                                                     descr,
                                                     mat_A,
                                                     mat_B,
                                                     rocsparse_spsort_stage_analysis,
                                                     0,
                                                     nullptr,
                                                     nullptr),
                                    rocsparse_status_invalid_value);
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
    }
}

template <typename I, typename J, typename T>
void testing_spsort_csr(const Arguments& arg)
{
    J                    M    = arg.M;
    J                    N    = arg.N;
    rocsparse_index_base base = arg.baseA;
    rocsparse_spsort_alg alg  = static_cast<rocsparse_spsort_alg>(arg.algo);

    // A batch count of one for A sorts the single matrix A into every batch of B.
    const int64_t batch_count_A = std::max<int64_t>(arg.batch_count_A, 1);
    const int64_t batch_count_B = std::max<int64_t>(arg.batch_count_B, 1);

    rocsparse_local_handle handle(arg);

    rocsparse_matrix_factory<T, I, J> matrix_factory(arg);

    host_csr_matrix<T, I, J> hA_single;
    matrix_factory.init_csr(hA_single, M, N, base);

    const int64_t nnz = hA_single.nnz;
    // A matrix without rows has no row pointer.
    const int64_t offsets_size = (M > 0) ? static_cast<int64_t>(M) + 1 : 0;

    // A is padded between batches to check that its strides are honoured, B is packed. Without
    // a row pointer there is nothing to pad.
    const int64_t offsets_batch_stride_A
        = (batch_count_A > 1 && offsets_size > 0) ? offsets_size + 2 : 0;
    const int64_t offsets_batch_stride_B        = (batch_count_B > 1) ? offsets_size : 0;
    const int64_t columns_values_batch_stride_A = (batch_count_A > 1) ? nnz + 3 : 0;
    const int64_t columns_values_batch_stride_B = (batch_count_B > 1) ? nnz : 0;

    const int64_t size_ptr_A = (batch_count_A - 1) * offsets_batch_stride_A + offsets_size;
    const int64_t size_ptr_B = (batch_count_B - 1) * offsets_batch_stride_B + offsets_size;
    const int64_t size_A     = (batch_count_A - 1) * columns_values_batch_stride_A + nnz;
    const int64_t size_B     = (batch_count_B - 1) * columns_values_batch_stride_B + nnz;

    // The batches are uniform: they all share one shuffled sparsity pattern, and only their
    // values are scaled differently, so that a mix up between batches shows up in the result.
    host_dense_vector<I> hA_ptr(size_ptr_A);
    host_dense_vector<J> hA_ind(size_A);
    host_dense_vector<T> hA_val(size_A);
    host_dense_vector<I> hB_ptr_gold(size_ptr_B);
    host_dense_vector<J> hB_ind_gold(size_B);
    host_dense_vector<T> hB_val_gold(size_B);

    std::fill(hA_ptr.data(), hA_ptr.data() + size_ptr_A, static_cast<I>(-1));
    std::fill(hA_ind.data(), hA_ind.data() + size_A, static_cast<J>(-1));
    std::fill(hA_val.data(), hA_val.data() + size_A, static_cast<T>(-1));

    host_shuffle_csr(hA_single);
    for(int64_t batch = 0; batch < batch_count_B; ++batch)
    {
        const int64_t batch_A = (batch_count_A == 1) ? 0 : batch;

        host_csr_matrix<T, I, J> hA_batch(hA_single);
        for(int64_t k = 0; k < nnz; ++k)
        {
            hA_batch.val[k] = hA_batch.val[k] * static_cast<T>(batch_A + 1);
        }

        host_csr_matrix<T, I, J> hB_batch(hA_batch);
        host_spsort_csr(hB_batch);

        if(batch < batch_count_A)
        {
            std::copy(hA_batch.ptr.data(),
                      hA_batch.ptr.data() + offsets_size,
                      hA_ptr.data() + batch * offsets_batch_stride_A);
            std::copy(hA_batch.ind.data(),
                      hA_batch.ind.data() + nnz,
                      hA_ind.data() + batch * columns_values_batch_stride_A);
            std::copy(hA_batch.val.data(),
                      hA_batch.val.data() + nnz,
                      hA_val.data() + batch * columns_values_batch_stride_A);
        }
        std::copy(hB_batch.ptr.data(),
                  hB_batch.ptr.data() + offsets_size,
                  hB_ptr_gold.data() + batch * offsets_batch_stride_B);
        std::copy(hB_batch.ind.data(),
                  hB_batch.ind.data() + nnz,
                  hB_ind_gold.data() + batch * columns_values_batch_stride_B);
        std::copy(hB_batch.val.data(),
                  hB_batch.val.data() + nnz,
                  hB_val_gold.data() + batch * columns_values_batch_stride_B);
    }

    // The in place sort keeps the padding of A, and the row pointer is unchanged.
    host_dense_vector<J> hA_ind_gold(hA_ind);
    host_dense_vector<T> hA_val_gold(hA_val);
    for(int64_t batch = 0; batch < batch_count_A; ++batch)
    {
        std::copy(hB_ind_gold.data() + batch * columns_values_batch_stride_B,
                  hB_ind_gold.data() + batch * columns_values_batch_stride_B + nnz,
                  hA_ind_gold.data() + batch * columns_values_batch_stride_A);
        std::copy(hB_val_gold.data() + batch * columns_values_batch_stride_B,
                  hB_val_gold.data() + batch * columns_values_batch_stride_B + nnz,
                  hA_val_gold.data() + batch * columns_values_batch_stride_A);
    }

    device_dense_vector<I> dA_ptr(hA_ptr);
    device_dense_vector<J> dA_ind(hA_ind);
    device_dense_vector<T> dA_val(hA_val);
    device_dense_vector<I> dB_ptr(size_ptr_B);
    device_dense_vector<J> dB_ind(size_B);
    device_dense_vector<T> dB_val(size_B);

    rocsparse_local_spmat matA(M,
                               N,
                               nnz,
                               dA_ptr,
                               dA_ind,
                               dA_val,
                               get_indextype<I>(),
                               get_indextype<J>(),
                               base,
                               get_datatype<T>());
    rocsparse_local_spmat matB(M,
                               N,
                               nnz,
                               dB_ptr,
                               dB_ind,
                               dB_val,
                               get_indextype<I>(),
                               get_indextype<J>(),
                               base,
                               get_datatype<T>());
    CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_strided_batch(
        matA, batch_count_A, offsets_batch_stride_A, columns_values_batch_stride_A));
    CHECK_ROCSPARSE_ERROR(rocsparse_csr_set_strided_batch(
        matB, batch_count_B, offsets_batch_stride_B, columns_values_batch_stride_B));

    rocsparse_spsort_descr descr;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_create(handle, &descr, nullptr));
    set_spsort_inputs(handle, descr, alg);

    // Analysis
    size_t buffer_size = 0;
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, matB, rocsparse_spsort_stage_analysis, &buffer_size, nullptr));

    void* dbuffer = nullptr;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort(
        handle, descr, matA, matB, rocsparse_spsort_stage_analysis, buffer_size, dbuffer, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    dbuffer = nullptr;

    // Compute
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(
        handle, descr, matA, matB, rocsparse_spsort_stage_compute, &buffer_size, nullptr));
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Out of place: B holds the sorted matrix and A is left unchanged.
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(handle,
                                               descr,
                                               matA,
                                               matB,
                                               rocsparse_spsort_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr));

        hB_ptr_gold.unit_check(dB_ptr);
        hB_ind_gold.unit_check(dB_ind);
        hB_val_gold.unit_check(dB_val);
        hA_ptr.unit_check(dA_ptr);
        hA_ind.unit_check(dA_ind);
        hA_val.unit_check(dA_val);

        // In place: A is sorted into itself.
        size_t in_place_buffer_size = 0;
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort_buffer_size(handle,
                                                           descr,
                                                           matA,
                                                           matA,
                                                           rocsparse_spsort_stage_compute,
                                                           &in_place_buffer_size,
                                                           nullptr));
        void* in_place_dbuffer = nullptr;
        CHECK_HIP_ERROR(rocsparse_hipMalloc(&in_place_dbuffer, in_place_buffer_size));
        CHECK_ROCSPARSE_ERROR(rocsparse_spsort(handle,
                                               descr,
                                               matA,
                                               matA,
                                               rocsparse_spsort_stage_compute,
                                               in_place_buffer_size,
                                               in_place_dbuffer,
                                               nullptr));
        CHECK_HIP_ERROR(rocsparse_hipFree(in_place_dbuffer));

        hA_ptr.unit_check(dA_ptr);
        hA_ind_gold.unit_check(dA_ind);
        hA_val_gold.unit_check(dA_val);
    }

    if(arg.timing)
    {
        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_spsort,
                                               handle,
                                               descr,
                                               (rocsparse_const_spmat_descr)matA,
                                               (rocsparse_spmat_descr)matB,
                                               rocsparse_spsort_stage_compute,
                                               buffer_size,
                                               dbuffer,
                                               nullptr);

        const double gbyte_count
            = spsort_csr_gbyte_count<I, J, T>(M, nnz, batch_count_B, offsets_batch_stride_B == 0);
        const double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::batch_count_A,
                            batch_count_A,
                            display_key_t::batch_count_B,
                            batch_count_B,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
    CHECK_ROCSPARSE_ERROR(rocsparse_spsort_descr_destroy(handle, descr, nullptr));
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                                 \
    template void testing_spsort_csr_bad_arg<ITYPE, JTYPE, TTYPE>(const Arguments& arg); \
    template void testing_spsort_csr<ITYPE, JTYPE, TTYPE>(const Arguments& arg)

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
void testing_spsort_csr_extra(const Arguments& arg)
{
    rocsparse_local_handle handle(arg);

    // 32-bit offsets with 64-bit column indices, which the test dispatch does not cover.
    {
        const int64_t m   = 3;
        const int64_t n   = 6;
        const int64_t nnz = 7;

        const host_vector<int32_t> hptr      = {1, 4, 4, 8};
        const host_vector<int64_t> hind      = {6, 2, 4, 5, 1, 3, 2};
        const host_vector<float>   hval      = {1, 2, 3, 4, 5, 6, 7};
        const host_vector<int64_t> hind_gold = {2, 4, 6, 1, 2, 3, 5};
        const host_vector<float>   hval_gold = {2, 3, 1, 5, 7, 6, 4};

        for(const bool in_place : {false, true})
        {
            device_vector<int32_t> dA_ptr(hptr);
            device_vector<int64_t> dA_ind(hind);
            device_vector<float>   dA_val(hval);
            device_vector<int32_t> dB_ptr(m + 1);
            device_vector<int64_t> dB_ind(nnz);
            device_vector<float>   dB_val(nnz);

            rocsparse_local_spmat matA(m,
                                       n,
                                       nnz,
                                       dA_ptr,
                                       dA_ind,
                                       dA_val,
                                       rocsparse_indextype_i32,
                                       rocsparse_indextype_i64,
                                       rocsparse_index_base_one,
                                       rocsparse_datatype_f32_r);
            rocsparse_local_spmat matB(m,
                                       n,
                                       nnz,
                                       dB_ptr,
                                       dB_ind,
                                       dB_val,
                                       rocsparse_indextype_i32,
                                       rocsparse_indextype_i64,
                                       rocsparse_index_base_one,
                                       rocsparse_datatype_f32_r);

            run_spsort(handle, matA, in_place ? matA : matB);

            host_vector<int32_t> hptr_result(m + 1);
            host_vector<int64_t> hind_result(nnz);
            host_vector<float>   hval_result(nnz);
            hptr_result.transfer_from(in_place ? dA_ptr : dB_ptr);
            hind_result.transfer_from(in_place ? dA_ind : dB_ind);
            hval_result.transfer_from(in_place ? dA_val : dB_val);
            hptr.unit_check(hptr_result);
            hind_gold.unit_check(hind_result);
            hval_gold.unit_check(hval_result);
        }
    }

    // A matrix without rows: the single offset of the target is the index base, whether the
    // source has an offsets array or not.
    for(const rocsparse_index_base base : {rocsparse_index_base_zero, rocsparse_index_base_one})
    {
        for(const bool source_has_offsets : {false, true})
        {
            const host_vector<int32_t> hptr_gold = {static_cast<int32_t>(base)};

            device_vector<int32_t> dA_ptr(hptr_gold);
            device_vector<int32_t> dB_ptr(host_vector<int32_t>{-1});

            rocsparse_local_spmat matA(0,
                                       5,
                                       0,
                                       source_has_offsets ? (void*)dA_ptr : nullptr,
                                       nullptr,
                                       nullptr,
                                       rocsparse_indextype_i32,
                                       rocsparse_indextype_i32,
                                       base,
                                       rocsparse_datatype_f32_r);
            rocsparse_local_spmat matB(0,
                                       5,
                                       0,
                                       dB_ptr,
                                       nullptr,
                                       nullptr,
                                       rocsparse_indextype_i32,
                                       rocsparse_indextype_i32,
                                       base,
                                       rocsparse_datatype_f32_r);

            run_spsort(handle, matA, matB);

            host_vector<int32_t> hptr_result(1);
            hptr_result.transfer_from(dB_ptr);
            hptr_gold.unit_check(hptr_result);
        }
    }
}
