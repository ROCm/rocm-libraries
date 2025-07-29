/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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

#include "rocsparselt_spmm.hpp"
#include "definitions.h"
#include "handle.h"
#include "status.h"
#include "rocsparselt_spmm_utils.hpp"
#include "utility.hpp"

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_get_workspace(const rocsparselt_handle*      handle,
                                                    const rocsparselt_matmul_plan* plan,
                                                    size_t*                        workspaceSize)

{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(plan == nullptr)
    {
        log_error(_handle, __func__, "plan is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _plan = reinterpret_cast<const _rocsparselt_matmul_plan*>(plan);
    if(!check_is_init_plan(_plan))
    {
        log_error(_handle, __func__, "plan did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(workspaceSize == nullptr)
    {
        log_error(_handle, __func__, "workspaceSize is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    {
        if(_plan->alg_selection->config_max_id == 0)
            *workspaceSize = 0;
        else
        {
            *workspaceSize = _plan->alg_selection->configs[_plan->alg_selection->config_id]
                                 .max_workspace_bytes
                            + _plan->alg_selection->configs[_plan->alg_selection->config_id]
                                 .synchronizer_bytes;

        }
        log_api(_handle, __func__, *workspaceSize);
        return rocsparselt_status_success;
    }
}

rocsparselt_status rocsparselt_matmul_impl(const char*                    caller,
                                           const rocsparselt_handle*      handle,
                                           const rocsparselt_matmul_plan* plan,
                                           const void*                    alpha,
                                           const void*                    d_A,
                                           const void*                    d_B,
                                           const void*                    beta,
                                           const void*                    d_C,
                                           void*                          d_D,
                                           void*                          workspace,
                                           hipStream_t*                   streams,
                                           int32_t                        numStreams,
                                           bool                           search = false)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        hipsparselt_cerr << "handle is a NULL pointer" << std::endl;
        return rocsparselt_status_invalid_handle;
    }
    auto _handle = reinterpret_cast<const _rocsparselt_handle*>(handle);
    if(!check_is_init_handle(_handle))
    {
        hipsparselt_cerr << "handle did not initialized or already destroyed" << std::endl;
        return rocsparselt_status_invalid_handle;
    }

    if(plan == nullptr)
    {
        log_error(_handle, caller, "plan is a NULL pointer");
        return rocsparselt_status_invalid_handle;
    }
    auto _plan = reinterpret_cast<const _rocsparselt_matmul_plan*>(plan);
    if(!check_is_init_plan(_plan))
    {
        log_error(_handle, caller, "plan did not initialized or already destroyed");
        return rocsparselt_status_invalid_handle;
    }

    // Check if pointer is valid
    if(alpha == nullptr)
    {
        log_error(_handle, caller, "alpha is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_A == nullptr)
    {
        log_error(_handle, caller, "d_A is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_B == nullptr)
    {
        log_error(_handle, caller, "d_B is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(beta == nullptr)
    {
        log_error(_handle, caller, "beta is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_C == nullptr)
    {
        log_error(_handle, caller, "d_C is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    if(d_D == nullptr)
    {
        log_error(_handle, caller, "d_D is a NULL pointer");
        return rocsparselt_status_invalid_pointer;
    }

    size_t workspaceSize = 0;
    if(search)
    {
        for(int id = 0; id < _plan->alg_selection->config_max_id; id++)
        {
            workspaceSize = max(workspaceSize, _plan->alg_selection->configs[id].max_workspace_bytes \
                            + _plan->alg_selection->configs[id].synchronizer_bytes);
        }
    }
    else
    {
        if (_plan->alg_selection->config_max_id != 0)
        {
            workspaceSize = _plan->alg_selection->configs[_plan->alg_selection->config_id].max_workspace_bytes \
                            + _plan->alg_selection->configs[_plan->alg_selection->config_id].synchronizer_bytes;
        }
    }

    size_t workspaceSizeIn = 0;
    // Only check the input workspace's allocated size when the input workspace is going to be used.
    if(workspace != nullptr && workspaceSize != 0)
    {
        RETURN_IF_HIP_ERROR(hipMemPtrGetInfo(workspace, &workspaceSizeIn));
        workspaceSizeIn = min(workspaceSizeIn, workspaceSize);
    }

    // If search is enabled, it will search avaiable solutions with input workspace size.
    if(workspaceSizeIn && workspaceSizeIn < workspaceSize)
    {
        if(search)
        {
            std::ostringstream stringStream;
            stringStream << "The parameter number 9 (workspace) required a device memroy with ";
            stringStream << workspaceSize  << " bytes, but current is " << workspaceSizeIn << " bytes.\n";
            stringStream << "Some of the solutions will be skiped during the Search.";
            auto msg = stringStream.str();
            hipsparselt_cout << msg << std::endl;
            log_info(_handle, caller, msg);
        }
        else
        {
            std::ostringstream stringStream;
            stringStream << "The parameter number 9 (workspace) had an illegal value expected a device memroy with ";
            stringStream << workspaceSize  << " bytes, but current is " << workspaceSizeIn << " bytes.";
            auto msg = stringStream.str();
            hipsparselt_cerr << msg << std::endl;
            log_error(_handle, caller, msg);
            return rocsparselt_status_invalid_value;
        }
    }

    if(numStreams < 0)
    {
        hipsparselt_cerr << "The parameter number 11 (numStreams) had an illegal value: "
                         << numStreams << std::endl;
        log_error(_handle, caller, "numStreams should >= 0");
        return rocsparselt_status_invalid_value;
    }
    else if(streams == nullptr && numStreams > 0)
    {
        hipsparselt_cerr << "The parameter number 10 (streams) had an illegal value: nullptr"
                         << std::endl;
        log_error(_handle,
                  caller,
                  "streams should not be a NULL pointer because the numStreams is not 0");
        return rocsparselt_status_invalid_value;
    }

    // algorithm selection
    int config_id         = _plan->alg_selection->config_id;
    int config_max_id     = _plan->alg_selection->config_max_id;
    int search_iterations = search ? _plan->alg_selection->search_iterations : 0; //default

#define EX_PARM                                                                                               \
    caller, _handle, _plan, alpha, beta, d_A, d_B, d_C, d_D, workspace, workspaceSizeIn, streams, numStreams, \
        &config_id, config_max_id, search_iterations

    log_api(_handle,
            caller,
            "plan[in]",
            *_plan,
            "alpha[in]",
            alpha,
            "d_A[in]",
            d_A,
            "d_B[in]",
            d_B,
            "beta[in]",
            beta,
            "d_C[in]",
            d_C,
            "d_D[in]",
            d_D,
            "workspace[in]",
            workspace,
            "workspaceSize[in]",
            workspaceSizeIn,
            "streams[in]",
            streams,
            "numStreams[in]",
            numStreams);

    rocsparselt_status status = rocsparselt_spmm_template(EX_PARM);
    if(search && status == rocsparselt_status_success)
    {
        log_info(_handle, caller, "found the best config_id", config_id);
        _plan->alg_selection->config_id = config_id;
    }
    return status;
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul(const rocsparselt_handle*      handle,
                                      const rocsparselt_matmul_plan* plan,
                                      const void*                    alpha,
                                      const void*                    d_A,
                                      const void*                    d_B,
                                      const void*                    beta,
                                      const void*                    d_C,
                                      void*                          d_D,
                                      void*                          workspace,
                                      hipStream_t*                   streams,
                                      int32_t                        numStreams)

{
    return rocsparselt_matmul_impl(
        __func__, handle, plan, alpha, d_A, d_B, beta, d_C, d_D, workspace, streams, numStreams);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocsparselt_status rocsparselt_matmul_search(const rocsparselt_handle* handle,
                                             rocsparselt_matmul_plan*  plan,
                                             const void*               alpha,
                                             const void*               d_A,
                                             const void*               d_B,
                                             const void*               beta,
                                             const void*               d_C,
                                             void*                     d_D,
                                             void*                     workspace,
                                             hipStream_t*              streams,
                                             int32_t                   numStreams)

{
    return rocsparselt_matmul_impl(__func__,
                                   handle,
                                   plan,
                                   alpha,
                                   d_A,
                                   d_B,
                                   beta,
                                   d_C,
                                   d_D,
                                   workspace,
                                   streams,
                                   numStreams,
                                   true);
}
#ifdef __cplusplus
}
#endif

template <typename Ti, typename To, typename Tc>
rocsparselt_status ConstructRocSparseLtProblem(const char*                                 caller,
                                               RocsparseltContractionProblem<Ti, To, Tc>** prob,
                                               const _rocsparselt_matmul_descr* matmul_descr,
                                               const Tc*                        alpha,
                                               const Tc*                        beta,
                                               const Ti*                        a,
                                               const Ti*                        b,
                                               const To*                        c,
                                               To*                              d,
                                               bool                             strided_batch,
                                               void*                            workspace,
                                               size_t                           workspaceSize,
                                               hipStream_t*                     streams,
                                               int32_t                          numStreams)
{
    std::shared_ptr<Tc> _one = std::make_shared<Tc>(static_cast<Tc>(1));
    if(alpha == nullptr)
        alpha = _one.get();

    if(beta == nullptr)
        beta = _one.get();

    int64_t              metadata_offset;
    const unsigned char* metadata;

    // matrix A
    int64_t offset_a       = 0;
    int64_t batch_stride_a = matmul_descr->matrix_A->batch_stride;
    int     num_batches_a  = matmul_descr->matrix_A->num_batches;
    //int64_t c_k            = matmul_descr->matrix_A->c_k;
    int64_t c_ld = matmul_descr->matrix_A->c_ld;
    int64_t c_n  = matmul_descr->matrix_A->c_n;

    if(matmul_descr->is_sparse_a)
    {

        batch_stride_a = batch_stride_a == 0 ? 0 : c_ld * c_n;

        metadata_offset = rocsparselt_metadata_offset_in_compressed_matrix(
            c_n, c_ld, (batch_stride_a == 0 ? 1 : num_batches_a), matmul_descr->matrix_A->type);
        metadata = (a == nullptr) ? nullptr
                                  : reinterpret_cast<const unsigned char*>(a) + metadata_offset;
    }

    // matrix B
    int64_t offset_b       = 0;
    int64_t batch_stride_b = matmul_descr->matrix_B->batch_stride;
    if(!matmul_descr->is_sparse_a)
    {
        //c_k            = matmul_descr->matrix_B->c_k;
        c_ld           = matmul_descr->matrix_B->c_ld;
        c_n            = matmul_descr->matrix_B->c_n;
        batch_stride_b = batch_stride_b == 0 ? 0 : c_ld * c_n;

        metadata_offset = rocsparselt_metadata_offset_in_compressed_matrix(
            c_n, c_ld, (batch_stride_b == 0 ? 1 : num_batches_a), matmul_descr->matrix_B->type);
        metadata = (b == nullptr) ? nullptr
                                  : reinterpret_cast<const unsigned char*>(b) + metadata_offset;
    }

    // matrix C
    int64_t offset_c = 0;

    // matrix D
    int64_t offset_d = 0;

    // activation
    hipsparselt_activation_type act_type    = hipsparselt_activation_type::none;
    float                       act_args[2] = {0.0f, 0.0f};

    if(matmul_descr->activation == rocsparselt_matmul_activation_relu)
    {
        act_args[0] = matmul_descr->activation_relu_threshold;
        act_args[1] = matmul_descr->activation_relu_upperbound;
        if(act_args[0] == 0 && act_args[1] == std::numeric_limits<float>::infinity())
            act_type = hipsparselt_activation_type::relu;
        else
            act_type = hipsparselt_activation_type::clippedrelu;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_gelu)
    {
        act_type    = hipsparselt_activation_type::gelu;
        act_args[0] = matmul_descr->activation_gelu_scaling;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_abs)
        act_type = hipsparselt_activation_type::abs;
    else if(matmul_descr->activation == rocsparselt_matmul_activation_leakyrelu)
    {
        act_type    = hipsparselt_activation_type::leakyrelu;
        act_args[0] = matmul_descr->activation_leakyrelu_alpha;
    }
    else if(matmul_descr->activation == rocsparselt_matmul_activation_sigmoid)
        act_type = hipsparselt_activation_type::sigmoid;
    else if(matmul_descr->activation == rocsparselt_matmul_activation_tanh)
    {
        act_type    = hipsparselt_activation_type::tanh;
        act_args[0] = matmul_descr->activation_tanh_alpha;
        act_args[1] = matmul_descr->activation_tanh_beta;
    }

    int64_t   _batch_stride_a, _offset_a;
    int64_t   _batch_stride_b, _offset_b;
    const Ti *_a, *_b;

    if(!matmul_descr->_swap_ab)
    {
        _batch_stride_a = batch_stride_a;
        _offset_a       = offset_a;
        _batch_stride_b = batch_stride_b;
        _offset_b       = offset_b;
        _a              = a;
        _b              = b;
    }
    else
    {
        _batch_stride_a = batch_stride_b;
        _offset_a       = offset_b;
        _batch_stride_b = batch_stride_a;
        _offset_b       = offset_a;
        _a              = b;
        _b              = a;
    }

    (*prob) = new RocsparseltContractionProblem<Ti, To, Tc>(matmul_descr->handle,
                                                            matmul_descr->_op_A,
                                                            matmul_descr->_op_B,
                                                            matmul_descr->matrix_D->order,
                                                            matmul_descr->_m,
                                                            matmul_descr->_n,
                                                            matmul_descr->_k,
                                                            alpha,
                                                            _a,
                                                            nullptr,
                                                            matmul_descr->_lda,
                                                            _batch_stride_a,
                                                            _offset_a,
                                                            _b,
                                                            nullptr,
                                                            matmul_descr->_ldb,
                                                            _batch_stride_b,
                                                            _offset_b,
                                                            beta,
                                                            c,
                                                            nullptr,
                                                            matmul_descr->matrix_C->ld,
                                                            matmul_descr->matrix_C->batch_stride,
                                                            offset_c,
                                                            d,
                                                            nullptr,
                                                            matmul_descr->matrix_D->ld,
                                                            matmul_descr->matrix_D->batch_stride,
                                                            offset_d,
                                                            num_batches_a,
                                                            strided_batch,
                                                            matmul_descr->_is_sparse_a,
                                                            metadata,
                                                            act_type,
                                                            act_args[0],
                                                            act_args[1],
                                                            matmul_descr->bias_pointer,
                                                            matmul_descr->bias_stride,
                                                            matmul_descr->bias_type,
                                                            matmul_descr->alpha_vector_scaling,
                                                            workspace,
                                                            workspaceSize,
                                                            streams,
                                                            numStreams);
    return rocsparselt_status_success;
}

#define GENERATE_DEFINITIONS(Ti, To, Tc)                                 \
    template rocsparselt_status ConstructRocSparseLtProblem<Ti, To, Tc>( \
        const char*,                                                     \
        RocsparseltContractionProblem<Ti, To, Tc>**,                     \
        const _rocsparselt_matmul_descr*,                                \
        const Tc*,                                                       \
        const Tc*,                                                       \
        const Ti*,                                                       \
        const Ti*,                                                       \
        const To*,                                                       \
        To*,                                                             \
        bool,                                                            \
        void*,                                                           \
        size_t,                                                          \
        hipStream_t*,                                                    \
        int32_t);

GENERATE_DEFINITIONS(__half, __half, float)
GENERATE_DEFINITIONS(hip_bfloat16, hip_bfloat16, float)
GENERATE_DEFINITIONS(int8_t, int8_t, float)
GENERATE_DEFINITIONS(int8_t, __half, float)
GENERATE_DEFINITIONS(int8_t, hip_bfloat16, float)
GENERATE_DEFINITIONS(__hip_fp8_e4m3, float, float)
GENERATE_DEFINITIONS(__hip_fp8_e5m2, float, float)

#undef GENERATE_DEFINITIONS
