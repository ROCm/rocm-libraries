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

#include <iostream>
#include <vector>

#include <rocsparse/rocsparse.h>

#define HIP_CHECK(stat)                                                                       \
    {                                                                                         \
        if(stat != hipSuccess)                                                                \
        {                                                                                     \
            std::cerr << "Error: hip error " << stat << " in line " << __LINE__ << std::endl; \
            return -1;                                                                        \
        }                                                                                     \
    }

#define ROCSPARSE_CHECK(stat)                                                         \
    {                                                                                 \
        if(stat != rocsparse_status_success)                                          \
        {                                                                             \
            std::cerr << "Error: rocsparse error " << stat << " in line " << __LINE__ \
                      << std::endl;                                                   \
            return -1;                                                                \
        }                                                                             \
    }

//! [doc example]
#if defined(ROCSPARSE_WITH_DIAGONAL_SOLVE)

template <typename T>
static rocsparse_datatype datatype_of();
template <>
rocsparse_datatype datatype_of<float>()
{
    return rocsparse_datatype_f32_r;
}
template <>
rocsparse_datatype datatype_of<double>()
{
    return rocsparse_datatype_f64_r;
}
template <>
rocsparse_datatype datatype_of<rocsparse_double_complex>()
{
    return rocsparse_datatype_f64_c;
}

template <typename T>
static T make(double re)
{
    return static_cast<T>(re);
}
template <>
rocsparse_double_complex make<rocsparse_double_complex>(double re)
{
    return rocsparse_double_complex(re, 0.0);
}

template <typename T>
static int run_demo(rocsparse_handle handle, const char* name)
{
    std::cout << "== diagonal solve, datatype " << name << " ==" << std::endl;

    constexpr rocsparse_int m   = 3;
    constexpr rocsparse_int nnz = 5;

    const rocsparse_int hptr[m + 1] = {0, 1, 3, 5};
    const rocsparse_int hind[nnz]   = {0, 0, 1, 1, 2};
    const T hval[nnz] = {make<T>(2.0), make<T>(1.0), make<T>(-3.0), make<T>(1.0), make<T>(4.0)};

    rocsparse_int *dptr, *dind;
    T*             dval;
    HIP_CHECK(hipMalloc(&dptr, sizeof(rocsparse_int) * (m + 1)));
    HIP_CHECK(hipMalloc(&dind, sizeof(rocsparse_int) * nnz));
    HIP_CHECK(hipMalloc(&dval, sizeof(T) * nnz));
    HIP_CHECK(hipMemcpy(dptr, hptr, sizeof(rocsparse_int) * (m + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dind, hind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dval, hval, sizeof(T) * nnz, hipMemcpyHostToDevice));

    const rocsparse_datatype dtype = datatype_of<T>();

    rocsparse_spmat_descr matA;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(&matA,
                                               m,
                                               m,
                                               nnz,
                                               dptr,
                                               dind,
                                               dval,
                                               rocsparse_indextype_i32,
                                               rocsparse_indextype_i32,
                                               rocsparse_index_base_zero,
                                               dtype));
    const rocsparse_fill_mode fill = rocsparse_fill_mode_lower;
    const rocsparse_diag_type diag = rocsparse_diag_type_non_unit;
    ROCSPARSE_CHECK(
        rocsparse_spmat_set_attribute(matA, rocsparse_spmat_fill_mode, &fill, sizeof(fill)));
    ROCSPARSE_CHECK(
        rocsparse_spmat_set_attribute(matA, rocsparse_spmat_diag_type, &diag, sizeof(diag)));

    const T                         alpha = make<T>(1.0);
    const rocsparse_operation       op    = rocsparse_operation_none;
    const rocsparse_analysis_policy apol  = rocsparse_analysis_policy_reuse;

    {
        const T hx[m] = {make<T>(1.0), make<T>(1.0), make<T>(1.0)};
        T *     dx, *dy;
        HIP_CHECK(hipMalloc(&dx, sizeof(T) * m));
        HIP_CHECK(hipMalloc(&dy, sizeof(T) * m));
        HIP_CHECK(hipMemcpy(dx, hx, sizeof(T) * m, hipMemcpyHostToDevice));

        rocsparse_dnvec_descr vx, vy;
        ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vx, m, dx, dtype));
        ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vy, m, dy, dtype));

        rocsparse_sptrsv_descr descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&descr));
        const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_operation, &op, sizeof(op), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_scalar_datatype, &dtype, sizeof(dtype), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsv_input_compute_datatype,
                                                   &dtype,
                                                   sizeof(dtype),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_analysis_policy, &apol, sizeof(apol), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_scalar_alpha, &alpha, sizeof(&alpha), nullptr));

        auto stage = [&](rocsparse_sptrsv_stage s) -> int {
            size_t sz;
            void*  buf = nullptr;
            ROCSPARSE_CHECK(
                rocsparse_sptrsv_buffer_size(handle, descr, matA, vx, vy, s, &sz, nullptr));
            HIP_CHECK(hipMalloc(&buf, sz));
            ROCSPARSE_CHECK(rocsparse_sptrsv(handle, descr, matA, vx, vy, s, sz, buf, nullptr));
            HIP_CHECK(hipFree(buf));
            return 0;
        };
        if(stage(rocsparse_sptrsv_stage_analysis) != 0)
        {
            return -1;
        }

        const rocsparse_solve_mode diagonal = rocsparse_solve_mode_diagonal;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsv_input_solve_mode,
                                                   &diagonal,
                                                   sizeof(diagonal),
                                                   nullptr));

        for(rocsparse_diagonal_modifier mod :
            {rocsparse_diagonal_modifier_none, rocsparse_diagonal_modifier_absolute})
        {
            ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                       descr,
                                                       rocsparse_sptrsv_input_diagonal_modifier,
                                                       &mod,
                                                       sizeof(mod),
                                                       nullptr));
            if(stage(rocsparse_sptrsv_stage_compute) != 0)
            {
                return -1;
            }
        }

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(descr));
        ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vx));
        ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vy));
        HIP_CHECK(hipFree(dx));
        HIP_CHECK(hipFree(dy));
    }

    {
        constexpr rocsparse_int nrhs = 2;
        std::vector<T>          hX(static_cast<size_t>(m) * nrhs, make<T>(1.0));
        T *                     dX, *dY;
        HIP_CHECK(hipMalloc(&dX, sizeof(T) * m * nrhs));
        HIP_CHECK(hipMalloc(&dY, sizeof(T) * m * nrhs));
        HIP_CHECK(hipMemcpy(dX, hX.data(), sizeof(T) * m * nrhs, hipMemcpyHostToDevice));

        rocsparse_dnmat_descr mX, mY;
        ROCSPARSE_CHECK(
            rocsparse_create_dnmat_descr(&mX, m, nrhs, m, dX, dtype, rocsparse_order_column));
        ROCSPARSE_CHECK(
            rocsparse_create_dnmat_descr(&mY, m, nrhs, m, dY, dtype, rocsparse_order_column));

        rocsparse_sptrsm_descr descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsm_descr(&descr));
        const rocsparse_sptrsm_alg alg = rocsparse_sptrsm_alg_default;
        const rocsparse_operation  opX = rocsparse_operation_none;
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_operation_A, &op, sizeof(op), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_operation_X, &opX, sizeof(opX), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_scalar_datatype, &dtype, sizeof(dtype), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsm_input_compute_datatype,
                                                   &dtype,
                                                   sizeof(dtype),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_analysis_policy, &apol, sizeof(apol), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(
            handle, descr, rocsparse_sptrsm_input_scalar_alpha, &alpha, sizeof(&alpha), nullptr));

        auto stage = [&](rocsparse_sptrsm_stage s) -> int {
            size_t sz;
            void*  buf = nullptr;
            ROCSPARSE_CHECK(
                rocsparse_sptrsm_buffer_size(handle, descr, matA, mX, mY, s, &sz, nullptr));
            HIP_CHECK(hipMalloc(&buf, sz));
            ROCSPARSE_CHECK(rocsparse_sptrsm(handle, descr, matA, mX, mY, s, sz, buf, nullptr));
            HIP_CHECK(hipFree(buf));
            return 0;
        };
        if(stage(rocsparse_sptrsm_stage_analysis) != 0)
        {
            return -1;
        }

        const rocsparse_solve_mode diagonal = rocsparse_solve_mode_diagonal;
        ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsm_input_solve_mode,
                                                   &diagonal,
                                                   sizeof(diagonal),
                                                   nullptr));

        for(rocsparse_diagonal_modifier mod :
            {rocsparse_diagonal_modifier_none, rocsparse_diagonal_modifier_absolute})
        {
            ROCSPARSE_CHECK(rocsparse_sptrsm_set_input(handle,
                                                       descr,
                                                       rocsparse_sptrsm_input_diagonal_modifier,
                                                       &mod,
                                                       sizeof(mod),
                                                       nullptr));
            if(stage(rocsparse_sptrsm_stage_compute) != 0)
            {
                return -1;
            }
        }

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsm_descr(descr));
        ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mX));
        ROCSPARSE_CHECK(rocsparse_destroy_dnmat_descr(mY));
        HIP_CHECK(hipFree(dX));
        HIP_CHECK(hipFree(dY));
    }

    {
        constexpr rocsparse_int batch_count = 2;
        std::vector<T>          hx(static_cast<size_t>(m) * batch_count, make<T>(1.0));
        T *                     dx, *dy;
        HIP_CHECK(hipMalloc(&dx, sizeof(T) * m * batch_count));
        HIP_CHECK(hipMalloc(&dy, sizeof(T) * m * batch_count));
        HIP_CHECK(hipMemcpy(dx, hx.data(), sizeof(T) * m * batch_count, hipMemcpyHostToDevice));

        rocsparse_dnvec_descr vx, vy;
        ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vx, m, dx, dtype));
        ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&vy, m, dy, dtype));
        ROCSPARSE_CHECK(rocsparse_dnvec_set_strided_batch(vx, batch_count, m));
        ROCSPARSE_CHECK(rocsparse_dnvec_set_strided_batch(vy, batch_count, m));

        rocsparse_sptrsv_descr descr;
        ROCSPARSE_CHECK(rocsparse_create_sptrsv_descr(&descr));
        const rocsparse_sptrsv_alg alg = rocsparse_sptrsv_alg_default;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_alg, &alg, sizeof(alg), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_operation, &op, sizeof(op), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_scalar_datatype, &dtype, sizeof(dtype), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsv_input_compute_datatype,
                                                   &dtype,
                                                   sizeof(dtype),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_analysis_policy, &apol, sizeof(apol), nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_scalar_alpha, &alpha, sizeof(&alpha), nullptr));

        const rocsparse_solve_mode        diagonal = rocsparse_solve_mode_diagonal;
        const rocsparse_diagonal_modifier mod      = rocsparse_diagonal_modifier_absolute;
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(handle,
                                                   descr,
                                                   rocsparse_sptrsv_input_solve_mode,
                                                   &diagonal,
                                                   sizeof(diagonal),
                                                   nullptr));
        ROCSPARSE_CHECK(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_diagonal_modifier, &mod, sizeof(mod), nullptr));

        auto stage = [&](rocsparse_sptrsv_stage s) -> int {
            size_t sz;
            void*  buf = nullptr;
            ROCSPARSE_CHECK(
                rocsparse_sptrsv_buffer_size(handle, descr, matA, vx, vy, s, &sz, nullptr));
            HIP_CHECK(hipMalloc(&buf, sz));
            ROCSPARSE_CHECK(rocsparse_sptrsv(handle, descr, matA, vx, vy, s, sz, buf, nullptr));
            HIP_CHECK(hipFree(buf));
            return 0;
        };
        if(stage(rocsparse_sptrsv_stage_analysis) != 0
           || stage(rocsparse_sptrsv_stage_compute) != 0)
        {
            return -1;
        }

        ROCSPARSE_CHECK(rocsparse_destroy_sptrsv_descr(descr));
        ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vx));
        ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(vy));
        HIP_CHECK(hipFree(dx));
        HIP_CHECK(hipFree(dy));
    }

    ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(matA));
    HIP_CHECK(hipFree(dptr));
    HIP_CHECK(hipFree(dind));
    HIP_CHECK(hipFree(dval));
    return 0;
}

int main()
{
    rocsparse_handle handle;
    ROCSPARSE_CHECK(rocsparse_create_handle(&handle));

    if(run_demo<double>(handle, "f64") != 0
       || run_demo<rocsparse_double_complex>(handle, "c64") != 0)
    {
        rocsparse_destroy_handle(handle);
        return -1;
    }

    ROCSPARSE_CHECK(rocsparse_destroy_handle(handle));
    std::cout << "diagonal solve example done" << std::endl;
    return 0;
}

#else

int main()
{
    std::cout << "rocsparse was built without the diagonal solve "
                 "(BUILD_WITH_DIAGONAL_SOLVE=OFF)"
              << std::endl;
    return 0;
}

#endif
//! [doc example]
