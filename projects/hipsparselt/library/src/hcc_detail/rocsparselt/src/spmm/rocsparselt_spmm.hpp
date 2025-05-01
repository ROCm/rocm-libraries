/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#ifndef ROCSPARSELT_SPMM_HPP
#define ROCSPARSELT_SPMM_HPP

//#include "gemm_tensile.hpp"

#include "handle.h"
#include "hipsparselt_ostream.hpp"
#include "utility.hpp"
#if BUILD_WITH_TENSILE
#include "tensile_host.hpp"
#else
#include "kernel_launcher.hpp"
#endif

template <typename Ti, typename To = Ti, typename Tc = To>
rocsparselt_status spmm_typecasting(const char*                     caller,
                                    const _rocsparselt_handle*      handle,
                                    const _rocsparselt_matmul_plan* plan,
                                    const void*                     alpha,
                                    const void*                     beta,
                                    const void*                     a,
                                    const void*                     b,
                                    const void*                     c,
                                    void*                           d,
                                    void*                           workspace,
                                    hipStream_t*                    streams,
                                    int32_t                         numStreams,
                                    int*                            config_id,
                                    const int                       config_max_id,
                                    const int                       search_iterations)
{
    // check alignment of pointers before casting
    if(!isAligned(a, sizeof(Ti)) || !isAligned(b, sizeof(Ti)) || !isAligned(c, sizeof(Ti))
       || !isAligned(d, sizeof(To)))
    {
        hipsparselt_cerr << "memmory is not aligned" << std::endl;
        return rocsparselt_status_invalid_size;
    }

    RocsparseltContractionProblem<Ti, To, Tc>* problem;

    auto status = ConstructRocSparseLtProblem(
        caller,
        &problem,
        plan->matmul_descr,
        reinterpret_cast<const Tc*>(alpha),
        reinterpret_cast<const Tc*>(beta),
        reinterpret_cast<const Ti*>(a),
        reinterpret_cast<const Ti*>(b),
        reinterpret_cast<const To*>(c),
        (To*)d,
        true,
        workspace,
        plan->alg_selection->config_max_id == 0
            ? 0
            : plan->alg_selection->configs[plan->alg_selection->config_id].max_workspace_bytes,
        streams,
        numStreams);

    if(status != rocsparselt_status_success)
        return status;

    status = runContractionProblem<Ti, To, Tc>(*problem,
#if BUILD_WITH_TENSILE
                                               &plan->alg_selection->configs[0],
#endif
                                               config_id,
                                               config_max_id,
                                               search_iterations);

    delete problem;

    return status;
}

inline rocsparselt_status rocsparselt_spmm_template(const char*                     caller,
                                                    const _rocsparselt_handle*      handle,
                                                    const _rocsparselt_matmul_plan* plan,
                                                    const void*                     alpha,
                                                    const void*                     beta,
                                                    const void*                     a,
                                                    const void*                     b,
                                                    const void*                     c,
                                                    void*                           d,
                                                    void*                           workspace,
                                                    hipStream_t*                    streams,
                                                    int32_t                         numStreams,
                                                    int*                            config_id,
                                                    const int                       config_max_id,
                                                    const int search_iterations)
{
    rocsparselt_status rs_status = rocsparselt_status_not_implemented;

#define EX_TYPECASTING_PARM                                                                   \
    caller, handle, plan, alpha, beta, a, b, c, d, workspace, streams, numStreams, config_id, \
        config_max_id, search_iterations

    hipDataType              a_type       = plan->matmul_descr->matrix_A->type;
    hipDataType              b_type       = plan->matmul_descr->matrix_B->type;
    hipDataType              c_type       = plan->matmul_descr->matrix_C->type;
    hipDataType              d_type       = plan->matmul_descr->matrix_D->type;
    rocsparselt_compute_type compute_type = plan->matmul_descr->compute_type;

    if(a_type == HIP_R_16F && b_type == HIP_R_16F)
    {
        if(c_type == HIP_R_16F && d_type == HIP_R_16F)
        {
            if(compute_type == rocsparselt_compute_f32)
            {
                rs_status = spmm_typecasting<__half, __half, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIP_R_16BF && b_type == HIP_R_16BF)
    {
        if(c_type == HIP_R_16BF && d_type == HIP_R_16BF)
        {
            if(compute_type == rocsparselt_compute_f32)
            {
                rs_status
                    = spmm_typecasting<hip_bfloat16, hip_bfloat16, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIP_R_8I && b_type == HIP_R_8I)
    {
        if(c_type == HIP_R_8I && d_type == HIP_R_8I)
        {
            if(compute_type == rocsparselt_compute_i32)
            {
                rs_status = spmm_typecasting<int8_t, int8_t, float>(EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == HIP_R_16F && d_type == HIP_R_16F)
        {
            if(compute_type == rocsparselt_compute_i32)
            {
                rs_status = spmm_typecasting<int8_t, __half, float>(EX_TYPECASTING_PARM);
            }
        }
        else if(c_type == HIP_R_16BF && d_type == HIP_R_16BF)
        {
            if(compute_type == rocsparselt_compute_i32)
            {
                rs_status = spmm_typecasting<int8_t, hip_bfloat16, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIP_R_8F_E4M3 && b_type == HIP_R_8F_E4M3)
    {
        if(c_type == HIP_R_32F && d_type == HIP_R_32F)
        {
            if(compute_type == rocsparselt_compute_f32)
            {
                rs_status
                    = spmm_typecasting<__hip_fp8_e4m3, float, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else if(a_type == HIP_R_8F_E5M2 && b_type == HIP_R_8F_E5M2)
    {
        if(c_type == HIP_R_32F && d_type == HIP_R_32F)
        {
            if(compute_type == rocsparselt_compute_f32)
            {
                rs_status
                    = spmm_typecasting<__hip_fp8_e5m2, float, float>(EX_TYPECASTING_PARM);
            }
        }
    }
    else
    {
        rs_status = rocsparselt_status_not_implemented;
    }

    return rs_status;
}
#endif
