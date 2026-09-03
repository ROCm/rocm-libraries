/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
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

#include "hipblaslt/hipblaslt.h"
#include "UserDrivenTuningParser.hpp"
#include "check_numerics_matrix.hpp"
#include "exceptions.hpp"
#include "handle.h"
#include "hipblaslt/hipblaslt-ext-op.h"
#include "hipblaslt_internal.hpp"

#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocblaslt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "Debug.hpp"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

bool override_path_compare_git_version(OverrideSingleton& override, hipblasLtHandle_t& handle)
{
    char git_version[128];
    hipblasLtGetGitRevision(handle, &git_version[0]);
    static std::string cached_firstline;
    static std::string cached_path;
    static bool        cached = false;
    std::string        firstline;

    if(!cached || cached_path != override.file_path)
    {
        std::ifstream file_read(override.file_path);
        std::getline(file_read, firstline);
        cached_firstline = firstline;
        cached_path      = override.file_path;
        cached           = true;
    }
    else
    {
        firstline = cached_firstline;
    }

    std::string header = "Git Version: ";
    size_t      pos    = firstline.find(header);
    if(pos != std::string::npos)
    {
        std::string file_version = firstline.substr(pos + header.length());
        if(file_version == git_version)
            return true;
    }

    override.env_mode = false;

    return false;
}

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status_ status)
{
    switch(status)
    {
    case rocblaslt_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblaslt_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblaslt_status_not_implemented:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_pointer:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_invalid_size:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblaslt_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    case rocblaslt_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblaslt_status_arch_mismatch:
        return HIPBLAS_STATUS_ARCH_MISMATCH;
    default:
        throw HIPBLAS_STATUS_INVALID_ENUM;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIPBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)              \
    {                                                                  \
        hipblasStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPBLAS_STATUS_SUCCESS)             \
        {                                                              \
            return TMP_STATUS_FOR_CHECK;                               \
        }                                                              \
    }

#define RETURN_IF_ROCBLASLT_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocblaslt_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocblaslt_status_success)            \
        {                                                               \
            return RocBlasLtStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                               \
    }

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtCreate");

    // Check if handle is valid
    if(handle == nullptr)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    int             deviceId;
    hipError_t      err;
    hipblasStatus_t retval = HIPBLAS_STATUS_SUCCESS;
    // TODO: Synchronizer size pass into predicate SynchronizerSizeCheck
    // 1K just for small size now, need to cal corner case if support all situations
    void* d_Synchronizer = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_Synchronizer, 16 * 409600 * sizeof(int)));
    CHECK_HIP_ERROR(hipMemset(d_Synchronizer, 0, sizeof(int) * 16 * 409600));

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocBlasLtStatusToHIPStatus(rocblaslt_create((rocblaslt_handle*)handle));
        (*(rocblaslt_handle*)handle)->Synchronizer = d_Synchronizer;
    }
    rocblaslt::Debug::Instance().markerStop();
    return retval;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtDestroy");
    if(handle != nullptr and (*(rocblaslt_handle)handle).Synchronizer != nullptr)
    {
        CHECK_HIP_ERROR(hipFree((*(rocblaslt_handle)handle).Synchronizer));
    }

    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_destroy((const rocblaslt_handle)handle));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtSetSmCountTarget(hipblasLtHandle_t handle, int32_t smCountTarget)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtSetSmCountTarget");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_set_sm_count_target((rocblaslt_handle)handle, smCountTarget));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetSmCountTarget(hipblasLtHandle_t handle, int32_t* smCountTarget)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtGetSmCountTarget");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_get_sm_count_target((rocblaslt_handle)handle, smCountTarget));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtCheckNumericsDrain(hipblasLtHandle_t handle, uint32_t* first_nan_call_id)
try
{
    if(handle == nullptr)
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    const uint32_t first_nan = hipblaslt_check_numerics_drain_handle((rocblaslt_handle)handle);
    if(first_nan_call_id)
        *first_nan_call_id = first_nan;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutCreate(hipblasLtMatrixLayout_t* matDescr,
                                            hipDataType              valueType,
                                            uint64_t                 rows,
                                            uint64_t                 cols,
                                            int64_t                  ld)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutCreate");
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matrix_layout_create(
        (rocblaslt_matrix_layout*)matDescr, valueType, rows, cols, ld));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t descr)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_destroy((const rocblaslt_matrix_layout)descr));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescCreate(hipblasLtMatmulDesc_t* matmulDesc,
                                          hipblasComputeType_t   computeType,
                                          hipDataType            scaleType)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescCreate");
    char* override = std::getenv("HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32");
    if(override && (computeType == hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_TF32)
       && (std::string(override) != ""))
    {
        switch(std::stoi(std::string(override)))
        {
        case 0:
            computeType = hipblasComputeType_t::HIPBLAS_COMPUTE_32F;
            break;
        case 2:
            computeType = hipblasComputeType_t::HIPBLAS_COMPUTE_32F_FAST_16BF;
            break;
        case 1:
        default:
            break;
        }
    }
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_desc_create(
        (rocblaslt_matmul_desc*)matmulDesc, (rocblaslt_compute_type)computeType, scaleType));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  const void*                      buf,
                                                  size_t                           sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutSetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_set_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(hipblasLtMatrixLayout_t          matLayout,
                                                  hipblasLtMatrixLayoutAttribute_t attr,
                                                  void*                            buf,
                                                  size_t                           sizeInBytes,
                                                  size_t*                          sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixLayoutGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matrix_layout_get_attribute((rocblaslt_matrix_layout)matLayout,
                                              (rocblaslt_matrix_layout_attribute)attr,
                                              buf,
                                              sizeInBytes,
                                              sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t descr)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_destroy((const rocblaslt_matmul_desc)descr));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// Per-call handoff from the RMSNorm producer GEMM (K1) to the cross-tile reduction (K2).
// K1 writes partialBuf through the normal Tensile problem inputs; K2 consumes that buffer plus
// the by-value arguments below. Full RMSNorm reduces and applies immediately, while the
// decomposed flow has K2 write the final row scale into the descriptor below for GEMM2.
struct RmsNormHandoff
{
    void*   partialBuf = nullptr; // f32 partial sums, row-major [M_padded, nTilesN]
    int32_t M          = 0;       // logical rows; padded rows in partialBuf are ignored
    int32_t N          = 0;       // feature dimension reduced by RMSNorm
    int32_t nTilesN    = 0;       // columns of partialBuf, ceil(N / MacroTile1)
    float   invD       = 0.f;     // 1 / N
    float   eps        = 0.f;     // RMSNorm epsilon
};

// Cross-call state for the decomposed RMSNorm flow. The producer reduction materializes the
// finalized rstd here, and the later GEMM2 scale-apply epilogue consumes it. The full flow never
// creates this handle because its reduction applies rstd to D in the same matmul call.
struct hipblasLtFusedEpilogueRMSNormDescriptor
{
    // FP32 rstd, tightly packed [M * batch]. M and batch are implicit in the consumer GEMM2
    // problem, which must match the producer for the decomposed flow.
    void* per_row_scale = nullptr;
    size_t per_row_scale_size = 0;
    // Set when a caller supplies per_row_scale. The producer and consumer run on the same stream,
    // so the consumer can be submitted after the producer without host synchronization.
    bool populated = false;
};

// Definition of the opaque handle declared in hipblaslt.h. Owns the composed list of
// epilogue stages plus their parameters. Attached, non-owning, to a matmul descriptor via
// HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE.
struct hipblasLtFusedEpilogueDescriptor
{
    std::vector<hipblasLtFuseableEpilogue_t> stages;

    // Residual-add parameters. residual_output is optional; if unset, the residual input
    // tensor is updated in place with the post-add residual stream.
    void* residual        = nullptr;
    void* residual_output = nullptr;

    // RMSNorm parameters (shared by the full RMSNorm and partial-RMSNorm-stats stages).
    void* rmsnorm_gamma = nullptr;
    float rmsnorm_eps   = 0.f;
    bool  eps_set       = false;

    // Decomposed-flow handoff descriptor, set on both the producer and consumer handles.
    hipblasLtFusedEpilogueRMSNormDescriptor* rmsnorm_stats = nullptr;

    // Requant parameters and policy.
    void*                              requant_scale        = nullptr;
    void*                              requant_amax         = nullptr;
    hipblasLtRequantScaleComputeMode_t requant_compute_mode = HIPBLASLT_REQUANT_SCALE_STATIC;
    hipblasLtRequantScaleGranularity_t requant_granularity  = HIPBLASLT_REQUANT_SCALE_PER_TENSOR;
    // MX microscaling requant attributes.
    void*       requant_mx_scale       = nullptr;
    int32_t     requant_mx_block_size  = 32;
    hipDataType requant_mx_output_type = HIP_R_8F_E4M3;
    // Optional bf16 dual-store output (PartialRMSStoreBf16D). Null unless set by caller.
    void*       requant_mx_residual_out = nullptr;
};

namespace
{
    // Supported RMSNorm-chain rank. A legal chain is an order-preserving subsequence of the
    // supported order
    //   residual add -> {RMSNorm | partial RMSNorm stats | RMSNorm scale-apply} -> AMax -> requant
    // with each stage appearing at most once. The three normalization stages share rank 1 so
    // that at most one of them can appear in a single chain (full vs decomposed are mutually
    // exclusive). Returns -1 for unrecognized stages or stages reserved for other epilogue
    // families (e.g. SwiGLU).
    int rmsnorm_chain_rank(hipblasLtFuseableEpilogue_t e)
    {
        switch(e)
        {
        case HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD:
            return 0;
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM:
        case HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS:
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY:
            return 1;
        case HIPBLASLT_FUSEABLE_EPILOGUE_AMAX:
            return 2;
        case HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT:
            return 3;
        default:
            return -1;
        }
    }

    // Classify a stage into a chain family so full and decomposed stages cannot be mixed in a
    // single chain (section 4.3): 0 = shared/neutral, 1 = full RMSNorm family, 2 = decomposed.
    // Requant is neutral: full RMSNorm uses it as an output epilogue, while the decomposed
    // producer can use it for the CODA dynamic-quantized handoff.
    int rmsnorm_chain_family(hipblasLtFuseableEpilogue_t e)
    {
        switch(e)
        {
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM:
        case HIPBLASLT_FUSEABLE_EPILOGUE_AMAX:
            return 1;
        case HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS:
        case HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY:
            return 2;
        default:
            return 0;
        }
    }

    bool fused_epilogue_has_stage(const hipblasLtFusedEpilogueDescriptor* d,
                                  hipblasLtFuseableEpilogue_t             e)
    {
        for(auto s : d->stages)
            if(s == e)
                return true;
        return false;
    }

    bool requant_compute_mode_valid(hipblasLtRequantScaleComputeMode_t mode)
    {
        return mode == HIPBLASLT_REQUANT_SCALE_STATIC
               || mode == HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
    }

    bool requant_granularity_valid(hipblasLtRequantScaleGranularity_t granularity)
    {
        return granularity == HIPBLASLT_REQUANT_SCALE_PER_TENSOR
               || granularity == HIPBLASLT_REQUANT_SCALE_PER_ROW
               || granularity == HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
    }
}

// Resolve the opaque fused-epilogue handle into the layer-visible POD declared in
// rocblaslt-types.h. Defined here because the handle definition is private to this
// translation unit; the rocblaslt / TensileLite layers only see the forward declaration.
// The enclosing public API in this file is extern "C"; this helper is declared with C++
// linkage (it lives among C++ types in rocblaslt-types.h), so force C++ linkage here to
// match the declaration.
extern "C++" bool rocblaslt_resolve_fused_epilogue(const hipblasLtFusedEpilogueDescriptor* desc,
                                                   RocblasltFusedEpilogueInfo&             out)
{
    if(desc == nullptr)
        return false;
    out.hasResidualAdd
        = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
    out.hasRMSNorm = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
    out.hasPartialRMSStats
        = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
    out.hasRMSNormScaleApply
        = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);
    out.hasRequant = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
    out.rmsnormGamma   = desc->rmsnorm_gamma;
    out.rmsnormEps     = desc->rmsnorm_eps;
    out.residual       = desc->residual;
    out.residualOutput = desc->residual_output;
    out.requantScale       = desc->requant_scale;
    out.requantAmax        = desc->requant_amax;
    out.requantComputeMode = desc->requant_compute_mode;
    out.requantGranularity = desc->requant_granularity;
    out.requantMxScale       = desc->requant_mx_scale;
    out.requantMxBlockSize   = desc->requant_mx_block_size;
    out.requantMxOutputType  = desc->requant_mx_output_type;
    out.requantMxResidualOut = desc->requant_mx_residual_out;
    if(desc->rmsnorm_stats != nullptr)
    {
        out.perRowScale       = desc->rmsnorm_stats->per_row_scale;
        out.rmsStatsPopulated = desc->rmsnorm_stats->populated;
    }
    return true;
}

hipblasStatus_t hipblasLtFusedEpilogueCreate(hipblasLtFusedEpilogueDescriptor_t* desc)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;
    *desc = new hipblasLtFusedEpilogueDescriptor();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueAdd(hipblasLtFusedEpilogueDescriptor_t desc,
                                          hipblasLtFuseableEpilogue_t        epilogue)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;

    const int rank = rmsnorm_chain_rank(epilogue);
    if(rank < 0)
        return HIPBLAS_STATUS_INVALID_VALUE; // unrecognized or unsupported epilogue

    // Reject duplicates and out-of-order additions: the accumulated chain must stay an
    // order-preserving subsequence of the supported RMSNorm chain.
    if(!desc->stages.empty())
    {
        const int prev_rank = rmsnorm_chain_rank(desc->stages.back());
        if(rank <= prev_rank)
            return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Reject mixing full and decomposed RMSNorm stages in one chain: a single chain
    // uses either HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM or the decomposed producer/consumer
    // stages, never both.
    if(epilogue == HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT
       && fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY))
        return HIPBLAS_STATUS_INVALID_VALUE; // requant is producer/full-flow only

    const int family = rmsnorm_chain_family(epilogue);
    if(family != 0)
    {
        for(auto s : desc->stages)
        {
            const int existing = rmsnorm_chain_family(s);
            if(existing != 0 && existing != family)
                return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }

    desc->stages.push_back(epilogue);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueSetAttribute(hipblasLtFusedEpilogueDescriptor_t desc,
                                                   hipblasLtFusedEpilogueAttribute_t  attr,
                                                   const void*                        value,
                                                   size_t                             sizeInBytes)
try
{
    if(desc == nullptr || value == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;

    switch(attr)
    {
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_gamma, value, sizeof(void*));
        if(desc->rmsnorm_gamma == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS:
        if(sizeInBytes < sizeof(float))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_eps, value, sizeof(float));
        desc->eps_set = true;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->residual, value, sizeof(void*));
        if(desc->residual == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->residual_output, value, sizeof(void*));
        break;
    case HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->rmsnorm_stats, value, sizeof(void*));
        if(desc->rmsnorm_stats == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_scale, value, sizeof(void*));
        if(desc->requant_scale == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_AMAX_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_amax, value, sizeof(void*));
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE:
        if(sizeInBytes < sizeof(hipblasLtRequantScaleComputeMode_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_compute_mode, value, sizeof(hipblasLtRequantScaleComputeMode_t));
        if(!requant_compute_mode_valid(desc->requant_compute_mode))
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY:
        if(sizeInBytes < sizeof(hipblasLtRequantScaleGranularity_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_granularity, value, sizeof(hipblasLtRequantScaleGranularity_t));
        if(!requant_granularity_valid(desc->requant_granularity))
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_mx_scale, value, sizeof(void*));
        if(desc->requant_mx_scale == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE:
        if(sizeInBytes < sizeof(int32_t))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_mx_block_size, value, sizeof(int32_t));
        if(desc->requant_mx_block_size <= 0)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE:
        if(sizeInBytes < sizeof(hipDataType))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_mx_output_type, value, sizeof(hipDataType));
        if(desc->requant_mx_output_type != HIP_R_8F_E4M3)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    case HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_RESIDUAL_OUT_POINTER:
        if(sizeInBytes < sizeof(void*))
            return HIPBLAS_STATUS_INVALID_VALUE;
        memcpy(&desc->requant_mx_residual_out, value, sizeof(void*));
        if(desc->requant_mx_residual_out == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    default:
        return HIPBLAS_STATUS_INVALID_VALUE;
    }
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueDestroy(hipblasLtFusedEpilogueDescriptor_t desc)
try
{
    delete desc;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtFusedEpilogueRMSNormDescriptorCreate(hipblasLtFusedEpilogueRMSNormDescriptor_t* desc)
try
{
    if(desc == nullptr)
        return HIPBLAS_STATUS_INVALID_VALUE;
    *desc = new hipblasLtFusedEpilogueRMSNormDescriptor();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtFusedEpilogueRMSNormDescriptorDestroy(hipblasLtFusedEpilogueRMSNormDescriptor_t desc)
try
{
    delete desc;
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc, void* buffer, size_t sizeInBytes)
{
    if(desc == nullptr || buffer == nullptr || sizeInBytes == 0)
        return HIPBLAS_STATUS_INVALID_VALUE;
    desc->per_row_scale      = buffer;
    desc->per_row_scale_size = sizeInBytes;
    desc->populated          = true;
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                const void*                     buf,
                                                size_t                          sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescSetAttribute");

    // Validate a fused-epilogue handle before it is attached to the matmul descriptor.
    // This is the API-call-time gate for stage-specific required inputs.
    if(matmulAttr == HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE)
    {
        if(buf == nullptr || sizeInBytes < sizeof(void*))
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        hipblasLtFusedEpilogueDescriptor_t fused = nullptr;
        memcpy(&fused, buf, sizeof(void*));
        if(fused != nullptr
           && fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD)
           && fused->residual == nullptr)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        // gamma and eps back both the full RMSNorm stage and the decomposed producer
        // (partial RMSNorm stats) stage.
        if(fused != nullptr
           && (fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM)
               || fused_epilogue_has_stage(fused,
                                           HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)))
        {
            if(fused->rmsnorm_gamma == nullptr || !fused->eps_set)
            {
                rocblaslt::Debug::Instance().markerStop();
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }
        // Both decomposed stages require the opaque RMSNorm handoff descriptor to be set so
        // the producer and consumer calls share the same object.
        if(fused != nullptr
           && (fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)
               || fused_epilogue_has_stage(fused,
                                           HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY))
           && fused->rmsnorm_stats == nullptr)
        {
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(fused != nullptr && fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT))
        {
            const bool mxBlock = fused->requant_granularity == HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
            if(mxBlock)
            {
                // MX requant: needs mxScale pointer, positive block size, and e4m3 output.
                if(fused->requant_mx_scale == nullptr
                   || fused->requant_mx_block_size <= 0
                   || fused->requant_mx_output_type != HIP_R_8F_E4M3)
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
            }
            else
            {
                // Non-MX requant: needs an f32 scale pointer and valid mode/granularity.
                if(fused->requant_scale == nullptr
                   || !requant_compute_mode_valid(fused->requant_compute_mode)
                   || !requant_granularity_valid(fused->requant_granularity))
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                // Decomposed producer requires dynamic per-row scale (non-MX only).
                if(fused_epilogue_has_stage(fused,
                                            HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS)
                   && (fused->requant_compute_mode != HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX
                       || fused->requant_granularity != HIPBLASLT_REQUANT_SCALE_PER_ROW))
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
            }
        }
    }

    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_set_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulDescGetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                void*                           buf,
                                                size_t                          sizeInBytes,
                                                size_t*                         sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_desc_get_attribute((rocblaslt_matmul_desc)matmulDesc,
                                            (rocblaslt_matmul_desc_attributes)matmulAttr,
                                            buf,
                                            sizeInBytes,
                                            sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceCreate");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_create((rocblaslt_matmul_preference*)pref));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceDestroy");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_destroy((const rocblaslt_matmul_preference)pref));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceSetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          const void*                           data,
                                          size_t                                dataSize)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceSetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_set_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  dataSize));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulPreferenceGetAttribute(hipblasLtMatmulPreference_t           pref,
                                          hipblasLtMatmulPreferenceAttributes_t attribute,
                                          void*                                 data,
                                          size_t                                sizeInBytes,
                                          size_t*                               sizeWritten)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulPreferenceGetAttribute");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_matmul_preference_get_attribute((rocblaslt_matmul_preference)pref,
                                                  (rocblaslt_matmul_preference_attributes)attribute,
                                                  data,
                                                  sizeInBytes,
                                                  sizeWritten));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t
    hipblasLtMatmulAlgoGetHeuristic(hipblasLtHandle_t                handle,
                                    hipblasLtMatmulDesc_t            matmulDesc,
                                    hipblasLtMatrixLayout_t          Adesc,
                                    hipblasLtMatrixLayout_t          Bdesc,
                                    hipblasLtMatrixLayout_t          Cdesc,
                                    hipblasLtMatrixLayout_t          Ddesc,
                                    hipblasLtMatmulPreference_t      pref,
                                    int                              requestedAlgoCount,
                                    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                    int*                             returnAlgoCount)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulAlgoGetHeuristic");

    OverrideSingleton& override = OverrideSingleton::getInstance();
    if(override.env_mode)
    {
        bool override_success = override_path_compare_git_version(override, handle);
        if(override_success)
            log_info(__func__, "HIPBLASLT_TUNING_OVERRIDE_FILE is the correct setting.");
        else
            log_error(
                __func__,
                "The hipBLASLt git version and the override file git version are not the same.");
    }

    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_algo_get_heuristic(
        (rocblaslt_handle)handle,
        (rocblaslt_matmul_desc)matmulDesc,
        (rocblaslt_matrix_layout)Adesc,
        (rocblaslt_matrix_layout)Bdesc,
        (rocblaslt_matrix_layout)Cdesc,
        (rocblaslt_matrix_layout)Ddesc,
        (rocblaslt_matmul_preference)pref,
        requestedAlgoCount,
        (rocblaslt_matmul_heuristic_result*)heuristicResultsArray,
        returnAlgoCount));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatmul(hipblasLtHandle_t            handle,
                                hipblasLtMatmulDesc_t        matmul_descr,
                                const void*                  alpha,
                                const void*                  A,
                                hipblasLtMatrixLayout_t      matA,
                                const void*                  B,
                                hipblasLtMatrixLayout_t      matB,
                                const void*                  beta,
                                const void*                  C,
                                hipblasLtMatrixLayout_t      matC,
                                void*                        D,
                                hipblasLtMatrixLayout_t      matD,
                                const hipblasLtMatmulAlgo_t* algo,
                                void*                        workspace,
                                size_t                       workspaceSizeInBytes,
                                hipStream_t                  stream)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmul");
    hipblasStatus_t return_status = HIPBLAS_STATUS_SUCCESS;

    // Fused-epilogue guard. Wired for gfx950:
    //   - full RMSNorm (single-call producer K1 + reduce-and-apply row_div),
    //   - the decomposed producer (PARTIAL_RMSNORM_STATS / K1 + row_rstd reduce-and-return), and
    //   - the decomposed consumer (RMSNORM_SCALE_APPLY / Kernel 3 RstdScale).
    // Any other fused chain has no matching kernel yet, so reject with NOT_SUPPORTED before launch.
    if(auto* desc = (rocblaslt_matmul_desc)matmul_descr)
    {
        if(desc->fused_epilogue != nullptr)
        {
            const auto* fused = desc->fused_epilogue;
            const bool  fullRmsNorm
                = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
            const bool scaleApply
                = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);
            const bool partialStats
                = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
            const bool requant
                = fused_epilogue_has_stage(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
            const bool mxRequant
                = requant
                  && (fused->requant_granularity == HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX);
            if(!fullRmsNorm && !scaleApply && !partialStats && !mxRequant)
            {
                rocblaslt::Debug::Instance().markerStop();
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            // Chained RMSNorm+requant (MX or per-row) supports bf16 A/B, or fp8-e4m3
            // A/B carrying MX block-scale inputs (the MXFP8-input PartialRMS path).
            if(fullRmsNorm && requant)
            {
                const auto* aLayout = (rocblaslt_matrix_layout)matA;
                const auto* bLayout = (rocblaslt_matrix_layout)matB;
                if(aLayout == nullptr || bLayout == nullptr)
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                using ScalingFormat = RocblasltContractionProblem::ScalingFormat;
                const bool aMxScale = desc->scaleAType == ScalingFormat::Block_32_UE8M0
                                      || desc->scaleAType == ScalingFormat::Block_32_UE8M0_32_8_EXT;
                const bool bMxScale = desc->scaleBType == ScalingFormat::Block_32_UE8M0
                                      || desc->scaleBType == ScalingFormat::Block_32_UE8M0_32_8_EXT;
                const bool bothBf16
                    = aLayout->type == HIP_R_16BF && bLayout->type == HIP_R_16BF;
                const bool bothFp8Mx = aLayout->type == HIP_R_8F_E4M3
                                       && bLayout->type == HIP_R_8F_E4M3 && aMxScale && bMxScale;
                if(!bothBf16 && !bothFp8Mx)
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
            }
            // Decomposed flow: validate the caller-owned FP32 [rows*batch] handoff buffer on both
            // sides. The producer's row_rstd reduction writes exactly one float per row, and the
            // consumer reads it through a ScaleAlphaVec SRD bounded to the same row count, so both
            // calls must supply a buffer covering their own D row count. The two stages are
            // mutually exclusive within a chain.
            if(partialStats || scaleApply)
            {
                auto*          dlay  = (rocblaslt_matrix_layout)matD;
                const uint64_t rows  = dlay ? dlay->m : 0;
                const int32_t  batch = dlay ? dlay->batch_count : 1;
                if(dlay == nullptr || rows == 0 || batch <= 0 || fused->rmsnorm_stats == nullptr
                   || !fused->rmsnorm_stats->populated
                   || fused->rmsnorm_stats->per_row_scale == nullptr)
                {
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                const size_t batchCount = static_cast<size_t>(batch);
                const size_t requiredSize
                    = static_cast<size_t>(rows) * batchCount * sizeof(float);
                if(fused->rmsnorm_stats->per_row_scale_size < requiredSize)
                {
                    log_error(__func__,
                              "decomposed RMSNorm handoff buffer is smaller than M * batchCount "
                              "* sizeof(float)");
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
            }
        }
    }

    return_status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul((rocblaslt_handle)handle,
                                                                (rocblaslt_matmul_desc)matmul_descr,
                                                                alpha,
                                                                A,
                                                                (rocblaslt_matrix_layout)matA,
                                                                B,
                                                                (rocblaslt_matrix_layout)matB,
                                                                beta,
                                                                C,
                                                                (rocblaslt_matrix_layout)matC,
                                                                D,
                                                                (rocblaslt_matrix_layout)matD,
                                                                (const rocblaslt_matmul_algo*)algo,
                                                                workspace,
                                                                workspaceSizeInBytes,
                                                                stream));
    rocblaslt::Debug::Instance().markerStop();
    return return_status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtMatrixTransformDescCreate(hipblasLtMatrixTransformDesc_t* transformDesc,
                                                   hipDataType                     scaleType)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescCreate");
    static_assert(sizeof(rocblaslt_matrix_transform_desc)
                      <= sizeof(hipblasLtMatrixTransformDescOpaque_t),
                  "hipblasLtMatrixTransformDescOpaque_t must have enough space");
    rocblaslt_matrix_transform_desc desc;
    desc.scaleType = scaleType;
    *transformDesc = new hipblasLtMatrixTransformDescOpaque_t;
    memcpy((*transformDesc)->data, &desc, sizeof(desc));
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescDestroy");
    if(transformDesc)
        delete transformDesc;
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescSetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             const void*                              buf,
                                             size_t                                   sizeInBytes)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescSetAttribute");
    if(!buf || sizeInBytes != sizeof(int32_t))
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    // all possible values should be int32_t
    assert(sizeInBytes == sizeof(int32_t));
    int32_t value{};
    memcpy(&value, buf, sizeInBytes);

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        desc->scaleType = static_cast<hipDataType>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        desc->pointerMode = static_cast<hipblasLtPointerMode_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        desc->opA = static_cast<hipblasOperation_t>(value);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        desc->opB = static_cast<hipblasOperation_t>(value);
        break;
    }
    default:
        assert(false && "Unknown attribute");
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
        break;
    }
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t
    hipblasLtMatrixTransformDescGetAttribute(hipblasLtMatrixTransformDesc_t           transformDesc,
                                             hipblasLtMatrixTransformDescAttributes_t attr,
                                             void*                                    buf,
                                             size_t                                   sizeInBytes,
                                             size_t*                                  sizeWritten)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransformDescGetAttribute");
    if(!sizeInBytes && !sizeWritten)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes && !sizeWritten)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    if(sizeInBytes != sizeof(int32_t))
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    rocblaslt_matrix_transform_desc* desc
        = reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]);
    int32_t value{};

    switch(attr)
    {
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE:
    {
        value = static_cast<int32_t>(desc->scaleType);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE:
    {
        value = static_cast<int32_t>(desc->pointerMode);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA:
    {
        value = static_cast<int32_t>(desc->opA);
        break;
    }
    case HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB:
    {
        value = static_cast<int32_t>(desc->opB);
        break;
    }
    default:
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
        assert(false && "Unknown attribute");
        break;
    }

    memcpy(buf, &value, sizeInBytes);
    *sizeWritten = sizeof(int32_t);
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t hipblasLtMatrixTransform(hipblasLtHandle_t              lightHandle,
                                         hipblasLtMatrixTransformDesc_t transformDesc,
                                         const void*             alpha, /* host or device pointer */
                                         const void*             A,
                                         hipblasLtMatrixLayout_t Adesc,
                                         const void*             beta, /* host or device pointer */
                                         const void*             B,
                                         hipblasLtMatrixLayout_t Bdesc,
                                         void*                   C,
                                         hipblasLtMatrixLayout_t Cdesc,
                                         hipStream_t             stream)
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatrixTransform");
    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matrix_transform(
        (rocblaslt_handle)lightHandle,
        reinterpret_cast<rocblaslt_matrix_transform_desc*>(&transformDesc->data[0]),
        alpha,
        A,
        (rocblaslt_matrix_layout)Adesc,
        beta,
        B,
        (rocblaslt_matrix_layout)Bdesc,
        C,
        (rocblaslt_matrix_layout)Cdesc,
        stream));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}

// Other Utilities
hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle, int* version)
try
{
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    *version = HIPBLASLT_VERSION_MAJOR * 100000 + HIPBLASLT_VERSION_MINOR * 100
               + HIPBLASLT_VERSION_PATCH;

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle, char* rev)
try
{
    // Get hipBLASLt revision
    if(handle == nullptr)
    {
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(HIPBLASLT_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetArchName(char** archName)
try
{
    *archName        = nullptr;
    std::string arch = rocblaslt_internal_get_arch_name();
    *archName        = (char*)malloc(arch.size() + 1);
    memcpy(*archName, arch.c_str(), arch.size() + 1);
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    if(archName != nullptr)
    {
        free(*archName);
        *archName = nullptr;
    }
    return exception_to_hipblas_status();
}

#ifdef __cplusplus
}
#endif
