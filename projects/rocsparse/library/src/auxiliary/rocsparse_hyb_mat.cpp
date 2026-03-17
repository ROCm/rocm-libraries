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

#include "rocsparse_spvec_descr.hpp"
#include "rocsparse_utility.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparse_create_hyb_mat is a structure holding the rocsparse HYB
 * matrix. It must be initialized using rocsparse_create_hyb_mat()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, hyb);
    *hyb = new _rocsparse_hyb_mat;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Copy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_copy_hyb_mat(rocsparse_hyb_mat dest, const rocsparse_hyb_mat src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->partition != rocsparse_hyb_partition_auto);
    previously_created |= (dest->ell_nnz != 0);
    previously_created |= (dest->ell_width != 0);
    previously_created |= (dest->ell_col_ind != nullptr);
    previously_created |= (dest->ell_val != nullptr);
    previously_created |= (dest->coo_nnz != 0);
    previously_created |= (dest->coo_row_ind != nullptr);
    previously_created |= (dest->coo_col_ind != nullptr);
    previously_created |= (dest->coo_val != nullptr);
    previously_created |= (dest->data_type_T != rocsparse_datatype_f32_r);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->partition != src->partition);
        invalid |= (dest->ell_width != src->ell_width);
        invalid |= (dest->ell_nnz != src->ell_nnz);
        invalid |= (dest->coo_nnz != src->coo_nnz);
        invalid |= (dest->data_type_T != src->data_type_T);

        if(invalid)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }
    }

    size_t T_size = sizeof(float);
    switch(src->data_type_T)
    {
    case rocsparse_datatype_f16_r:
    {
        T_size = sizeof(_Float16);
        break;
    }
    case rocsparse_datatype_bf16_r:
    {
        T_size = sizeof(rocsparse_bfloat16);
        break;
    }
    case rocsparse_datatype_f32_r:
    {
        T_size = sizeof(float);
        break;
    }
    case rocsparse_datatype_f64_r:
    {
        T_size = sizeof(double);
        break;
    }
    case rocsparse_datatype_f32_c:
    {
        T_size = sizeof(rocsparse_float_complex);
        break;
    }
    case rocsparse_datatype_f64_c:
    {
        T_size = sizeof(rocsparse_double_complex);
        break;
    }
    case rocsparse_datatype_i8_r:
    {
        T_size = sizeof(int8_t);
        break;
    }
    case rocsparse_datatype_u8_r:
    {
        T_size = sizeof(uint8_t);
        break;
    }
    case rocsparse_datatype_i32_r:
    {
        T_size = sizeof(int32_t);
        break;
    }
    case rocsparse_datatype_u32_r:
    {
        T_size = sizeof(uint32_t);
        break;
    }
    }

    if(src->ell_col_ind != nullptr)
    {
        if(dest->ell_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->ell_col_ind, sizeof(rocsparse_int) * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->ell_col_ind,
                                      src->ell_col_ind,
                                      sizeof(rocsparse_int) * src->ell_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->ell_val != nullptr)
    {
        if(dest->ell_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->ell_val, T_size * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->ell_val, src->ell_val, T_size * src->ell_nnz, hipMemcpyDeviceToDevice));
    }

    if(src->coo_row_ind != nullptr)
    {
        if(dest->coo_row_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->coo_row_ind, sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_row_ind,
                                      src->coo_row_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_col_ind != nullptr)
    {
        if(dest->coo_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->coo_col_ind, sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_col_ind,
                                      src->coo_col_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_val != nullptr)
    {
        if(dest->coo_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->coo_val, T_size * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->coo_val, src->coo_val, T_size * src->coo_nnz, hipMemcpyDeviceToDevice));
    }

    dest->m           = src->m;
    dest->n           = src->n;
    dest->partition   = src->partition;
    dest->ell_width   = src->ell_width;
    dest->ell_nnz     = src->ell_nnz;
    dest->coo_nnz     = src->coo_nnz;
    dest->data_type_T = src->data_type_T;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Destroy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, hyb);

    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    RETURN_IF_HIP_ERROR(hipDeviceSynchronize());

    // Clean up ELL part
    if(hyb->ell_col_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->ell_col_ind));
    }
    if(hyb->ell_val != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->ell_val));
    }

    // Clean up COO part
    if(hyb->coo_row_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_row_ind));
    }
    if(hyb->coo_col_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_col_ind));
    }
    if(hyb->coo_val != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_val));
    }

    delete hyb;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#ifdef __cplusplus
}
#endif
