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
#include "rocblaslt_mat_utils.hpp"

#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocblaslt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "Debug.hpp"

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

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

hipblasStatus_t refuse_row_order(const char*      caller,
                                 hipblasLtOrder_t orderA,
                                 hipblasLtOrder_t orderB,
                                 hipblasLtOrder_t orderC,
                                 hipblasLtOrder_t orderD)
{
    const hipblasLtOrder_t orders[] = {orderA, orderB, orderC, orderD};
    const char* const      names[]  = {"A", "B", "C", "D"};
    for(int i = 0; i < 4; ++i)
    {
        if(orders[i] == HIPBLASLT_ORDER_ROW)
        {
            log_error(caller,
                      "HIPBLASLT_ORDER_ROW is not supported by this API; hipblasLtMatmul and "
                      "hipblasLtMatrixTransform accept it",
                      "matrix",
                      names[i]);
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
    }
    return HIPBLAS_STATUS_SUCCESS;
}

hipblasStatus_t refuse_row_order(const char*             caller,
                                 rocblaslt_matrix_layout matA,
                                 rocblaslt_matrix_layout matB,
                                 rocblaslt_matrix_layout matC,
                                 rocblaslt_matrix_layout matD)
{
    // A null layout is reported by the caller's own validation, and a grouped layout sends the
    // call to the grouped path, which refuses every order but COL.
    for(auto mat : {matA, matB, matC, matD})
        if(!mat || mat->batch_mode == HIPBLASLT_BATCH_MODE_GROUPED)
            return HIPBLAS_STATUS_SUCCESS;

    return refuse_row_order(caller, matA->order, matB->order, matC->order, matD->order);
}

bool gemm_pointers_missing(int64_t                m,
                           int64_t                n,
                           int64_t                k,
                           int64_t                batchCount,
                           rocblaslt_compute_type computeType,
                           hipDataType            scaleType,
                           hipDataType            typeA,
                           rocblaslt_pointer_mode pointerMode,
                           const void*            alpha,
                           const void*            beta,
                           const void*            A,
                           const void*            B,
                           const void*            C,
                           const void*            D)
{
    // validateMatmulArgs returns early for an empty problem, before it checks pointers.
    if(!m || !batchCount)
        return false;
    if(!beta)
        return true;
    const bool alphaForbidsNull = alphaForbidsNullAB(
        alphaBetaScalarType(scaleType, computeType, typeA), pointerMode, alpha, A, B);
    return n && ((k && alphaForbidsNull) || !C || !D);
}

bool gemm_pointers_missing(rocblaslt_matmul_desc   desc,
                           rocblaslt_matrix_layout matA,
                           rocblaslt_matrix_layout matB,
                           rocblaslt_matrix_layout matC,
                           rocblaslt_matrix_layout matD,
                           const void*             alpha,
                           const void*             beta,
                           const void*             A,
                           const void*             B,
                           const void*             C,
                           const void*             D)
{
    if(!desc || !matA || !matB || !matC || !matD)
        return true;
    return gemm_pointers_missing(matD->m,
                                 matD->n,
                                 desc->op_A == HIPBLAS_OP_N ? matA->n : matA->m,
                                 matA->batch_count,
                                 desc->compute_type,
                                 desc->scale_type,
                                 matA->type,
                                 desc->pointermode,
                                 alpha,
                                 beta,
                                 A,
                                 B,
                                 C,
                                 D);
}

namespace
{
    // The arguments rocblaslt_matmul checks before anything else. A missing one keeps its
    // usual status rather than being reported as an unsupported order.
    bool matmul_arguments_present(hipblasLtHandle_t       handle,
                                  hipblasLtMatmulDesc_t   matmulDesc,
                                  const void*             alpha,
                                  const void*             A,
                                  hipblasLtMatrixLayout_t matA,
                                  const void*             B,
                                  hipblasLtMatrixLayout_t matB,
                                  const void*             beta,
                                  const void*             C,
                                  hipblasLtMatrixLayout_t matC,
                                  void*                   D,
                                  hipblasLtMatrixLayout_t matD,
                                  void*                   workspace,
                                  size_t                  workspaceSizeInBytes)
    {
        if(!handle || !matmulDesc || !matA || !matB || !matC || !matD || !alpha || !beta || !C || !D
           || (!workspace && workspaceSizeInBytes > 0))
            return false;
        auto desc = (rocblaslt_matmul_desc)matmulDesc;
        return !alphaForbidsNullAB(alphaBetaScalarType(desc->scale_type,
                                                       desc->compute_type,
                                                       ((rocblaslt_matrix_layout)matA)->type),
                                   desc->pointermode,
                                   alpha,
                                   A,
                                   B);
    }

    // The arguments rocblaslt_matmul_algo_get_heuristic checks before anything else.
    bool heuristic_arguments_present(hipblasLtHandle_t           handle,
                                     hipblasLtMatmulDesc_t       matmulDesc,
                                     hipblasLtMatmulPreference_t pref,
                                     int                         requestedAlgoCount,
                                     int*                        returnAlgoCount)
    {
        return handle && matmulDesc && pref && returnAlgoCount && requestedAlgoCount >= 1;
    }

    /********************************************************************************
     * Row-major support for hipblasLtMatmul and hipblasLtMatmulAlgoGetHeuristic.
     *
     * The kernels read every matrix as column-major. A row-major r x c matrix with
     * leading dimension ld has the same bytes as a column-major c x r matrix with the
     * same ld, so a row-major A or B becomes its column-major transpose with its op
     * flipped. A row-major D (with C) is computed as D^T = op(B)^T * op(A)^T, which
     * swaps the two operands and m with n. The heuristic and matmul rewrite the same
     * way, so an algo from one is valid for the other.
     *******************************************************************************/
    struct ColumnMajorProblem
    {
        _rocblaslt_matmul_desc   desc;
        _rocblaslt_matrix_layout a, b, c, d;
        // A and B trade places, so the caller passes B's data as A and A's as B.
        bool swapped = false;
    };

    // Only called for HIPBLAS_OP_N and HIPBLAS_OP_T; rewrite_row_major refuses the rest.
    hipblasOperation_t flip(hipblasOperation_t op)
    {
        return op == HIPBLAS_OP_N ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    }

    bool is_n_or_t(hipblasOperation_t op)
    {
        return op == HIPBLAS_OP_N || op == HIPBLAS_OP_T;
    }

    void to_column_major(_rocblaslt_matrix_layout& layout)
    {
        std::swap(layout.m, layout.n);
        layout.order = HIPBLASLT_ORDER_COL;
    }

    // These compute types name A's and B's types separately.
    bool is_mixed_operand_compute(rocblaslt_compute_type type)
    {
        switch(type)
        {
        case rocblaslt_compute_f32_fast_f8bf8_fnuz:
        case rocblaslt_compute_f32_fast_bf8f8_fnuz:
        case rocblaslt_compute_f32_fast_f8bf8:
        case rocblaslt_compute_f32_fast_bf8f8:
            return true;
        default:
            return false;
        }
    }

    bool is_scalar_scale(RocblasltContractionProblem::ScalingFormat format)
    {
        return format == RocblasltContractionProblem::ScalingFormat::None
               || format == RocblasltContractionProblem::ScalingFormat::Scalar;
    }

    // Epilogues that apply one function to each element of D, so they do not care
    // which axis of D is which.
    bool is_elementwise_epilogue(rocblaslt_epilogue epilogue)
    {
        switch(epilogue)
        {
        case ROCBLASLT_EPILOGUE_DEFAULT:
        case ROCBLASLT_EPILOGUE_RELU:
        case ROCBLASLT_EPILOGUE_GELU:
        case ROCBLASLT_EPILOGUE_SIGMOID:
        case ROCBLASLT_EPILOGUE_SWISH_EXT:
        case ROCBLASLT_EPILOGUE_CLAMP_EXT:
            return true;
        default:
            return false;
        }
    }

    hipblasStatus_t refuse_rewrite(const char* caller, const char* reason)
    {
        log_error(caller, "HIPBLASLT_ORDER_ROW is not supported here", "reason", reason);
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    // Sets `rewritten` and fills `out` when any layout is row-major. Leaves `rewritten`
    // false for an all-column-major call, a grouped call (the grouped path refuses
    // row-major itself), a null argument, or HIPBLASLT_OPERATION_INVALID (the normal
    // validation reports both).
    hipblasStatus_t rewrite_row_major(const char*             caller,
                                      rocblaslt_handle        handle,
                                      rocblaslt_matmul_desc   desc,
                                      rocblaslt_matrix_layout matA,
                                      rocblaslt_matrix_layout matB,
                                      rocblaslt_matrix_layout matC,
                                      rocblaslt_matrix_layout matD,
                                      ColumnMajorProblem&     out,
                                      bool&                   rewritten)
    {
        rewritten = false;
        if(!desc || !matA || !matB || !matC || !matD)
            return HIPBLAS_STATUS_SUCCESS;
        for(auto mat : {matA, matB, matC, matD})
            if(mat->batch_mode == HIPBLASLT_BATCH_MODE_GROUPED)
                return HIPBLAS_STATUS_SUCCESS;

        const bool rowA = matA->order == HIPBLASLT_ORDER_ROW;
        const bool rowB = matB->order == HIPBLASLT_ORDER_ROW;
        const bool rowC = matC->order == HIPBLASLT_ORDER_ROW;
        const bool rowD = matD->order == HIPBLASLT_ORDER_ROW;
        if(!rowA && !rowB && !rowC && !rowD)
            return HIPBLAS_STATUS_SUCCESS;

        if(desc->op_A == HIPBLASLT_OPERATION_INVALID || desc->op_B == HIPBLASLT_OPERATION_INVALID)
            return HIPBLAS_STATUS_SUCCESS;
        if(rowC != rowD)
            return refuse_rewrite(caller, "C and D must have the same order");
        if(!is_n_or_t(desc->op_A) || !is_n_or_t(desc->op_B))
            return refuse_rewrite(caller,
                                  "only HIPBLAS_OP_N and HIPBLAS_OP_T have a row-major equivalent");
        if(!is_scalar_scale(desc->scaleAType) || !is_scalar_scale(desc->scaleBType))
            return refuse_rewrite(caller, "block and vector scale tensors have a fixed layout");
        if(matA->type == HIP_R_4I || matB->type == HIP_R_4I)
            return refuse_rewrite(caller, "int4 weights have a fixed layout");
        if(rowD && !is_elementwise_epilogue(desc->epilogue))
            return refuse_rewrite(caller,
                                  "bias, aux and gradient epilogues index a fixed axis of D");
        if(rowD && desc->pointermode == rocblaslt_pointer_mode_alpha_device_vector_beta_host)
            return refuse_rewrite(caller, "an alpha vector holds one value per row of D");
        if(rowD && effective_uniform_summation_order(handle, desc) != 0)
            return refuse_rewrite(caller,
                                  "uniform summation order holds across the M dimension of D, "
                                  "which a row-major D moves to N; disable it to use a "
                                  "row-major D");
        if(rowD
           && (matA->type != matB->type || desc->compute_input_typeA != desc->compute_input_typeB
               || is_mixed_operand_compute(desc->compute_type)))
            return refuse_rewrite(caller,
                                  "a row-major D swaps A and B, so they must have the same type");

        out.desc.copy(*desc);
        out.desc.m_data = desc->m_data;
        out.a           = *matA;
        out.b           = *matB;
        out.c           = *matC;
        out.d           = *matD;
        if(rowA)
        {
            to_column_major(out.a);
            out.desc.op_A = flip(out.desc.op_A);
        }
        if(rowB)
        {
            to_column_major(out.b);
            out.desc.op_B = flip(out.desc.op_B);
        }
        if(rowD)
        {
            to_column_major(out.c);
            to_column_major(out.d);
            std::swap(out.a, out.b);
            const hipblasOperation_t opA = out.desc.op_A;
            out.desc.op_A                = flip(out.desc.op_B);
            out.desc.op_B                = flip(opA);
            // Both scales are scalars here, and they scale the whole product.
            std::swap(out.desc.scaleA, out.desc.scaleB);
            std::swap(out.desc.scaleAType, out.desc.scaleBType);
            out.swapped = true;
        }
        rewritten = true;
        return HIPBLAS_STATUS_SUCCESS;
    }
} // namespace

#if HIPBLASLT_HAS_GEMM_A2A_FUSION
/********************************************************************************
 * Fused epilogue descriptor.
 *
 * Opaque to the caller. Its contents are read only here, in the C-API layer that
 * knows what the stages mean; rocBLASLt carries it as a forward-declared pointer.
 *******************************************************************************/
struct hipblasLtFusedEpilogueDescriptor
{
    std::vector<hipblasLtFuseableEpilogue_t> stages;

    // HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX parameters. The per-rank arrays are
    // sized by the caller's SetAttribute call; that their length is the
    // communicator's world size is checked where the communicator is visible.
    int64_t                           a2a_extent     = 0;
    bool                              a2a_extent_set = false;
    hipblasLtA2ACompletionMode_t      a2a_completion = HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL;
    uint32_t                          comm_channel   = 0;
    std::vector<void*>                a2a_recv_ptrs;
    std::vector<hipblasLtSdmaQueue_t> a2a_queues;
};

namespace
{
    bool fused_epilogue_has_stage(const hipblasLtFusedEpilogueDescriptor* d,
                                  hipblasLtFuseableEpilogue_t             e)
    {
        for(auto s : d->stages)
            if(s == e)
                return true;
        return false;
    }

    bool fused_epilogue_stage_recognized(hipblasLtFuseableEpilogue_t stage)
    {
        switch(stage)
        {
        case HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX:
            return true;
        }
        return false;
    }

    // The collective family holds one stage and composes with nothing. Rejecting a
    // companion here rather than at launch is deliberate: the caller is asking for
    // something the API does not define, so it should learn that at the call that
    // expresses the mistake. Legalizing a combination later turns this into
    // success, which is the compatible direction.
    bool fused_epilogue_stage_composes(const hipblasLtFusedEpilogueDescriptor* desc,
                                       hipblasLtFuseableEpilogue_t             stage)
    {
        if(desc->stages.empty())
            return true;
        if(stage == HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX)
            return false;
        return !fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX);
    }

    // Every stage present must have the parameters it cannot run without. World
    // size is not known here, so the per-rank arrays are checked for content but
    // not length; validate_fused_a2a_launch compares both against it.
    hipblasStatus_t validate_fused_epilogue_contents(const hipblasLtFusedEpilogueDescriptor* desc)
    {
        if(!fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX))
            return HIPBLAS_STATUS_SUCCESS;

        if(!desc->a2a_extent_set || desc->a2a_recv_ptrs.empty() || desc->a2a_queues.empty())
        {
            log_error(__func__, "all-to-all stage is missing a required attribute");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        for(size_t j = 0; j < desc->a2a_recv_ptrs.size(); ++j)
        {
            if(desc->a2a_recv_ptrs[j] == nullptr)
            {
                log_error(__func__, "null all-to-all peer-recv pointer for rank", (int)j);
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }

        for(size_t j = 0; j < desc->a2a_queues.size(); ++j)
        {
            const hipblasLtSdmaQueue_t& q = desc->a2a_queues[j];
            if(q.queueBuf == nullptr || q.rptr == nullptr || q.wptr == nullptr
               || q.doorbell == nullptr)
            {
                log_error(__func__, "incomplete all-to-all SDMA queue for rank", (int)j);
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }

        return HIPBLAS_STATUS_SUCCESS;
    }

    // Checked before a solution is selected and again at launch, this being the
    // first point where both the communicator and D's layout are in hand.
    //
    // The requirements that depend on the selected solution's macro tile -
    // n_shard % MT0 == 0, MT1 < 2^14, MT0/MT1 in {128, 256}, data-parallel only,
    // and no split-K - are enforced by the kernel generator as solution
    // rejections, so they become "no usable algo" rather than a return code here.
    hipblasStatus_t validate_fused_a2a_launch(rocblaslt_handle                        handle,
                                              const hipblasLtFusedEpilogueDescriptor* desc,
                                              rocblaslt_matrix_layout                 matD)
    {
        if(handle == nullptr || matD == nullptr)
            return HIPBLAS_STATUS_INVALID_VALUE;

        // Registration is optional, but it is not a fallback: an all-to-all stage
        // without a communicator is an error, not a silent unfused GEMM.
        const uint32_t world = handle->device_comm_world;
        if(world == 0)
        {
            log_error(__func__, "all-to-all stage requires hipblasLtSetDeviceComm on this handle");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        if(desc->comm_channel >= handle->device_comm_channels)
        {
            log_error(__func__, "comm channel out of range", (int)desc->comm_channel);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        if(desc->a2a_recv_ptrs.size() != world || desc->a2a_queues.size() != world)
        {
            log_error(__func__, "all-to-all per-rank arrays do not have world entries");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        // The receive-buffer layout has no batch axis.
        if(matD->batch_count > 1)
        {
            log_error(__func__, "all-to-all does not support batched GEMM", matD->batch_count);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        const int64_t am = desc->a2a_extent;
        if(am % (int64_t)world != 0)
        {
            log_error(__func__, "all-to-all extent does not divide by world", am);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(am > (int64_t)matD->m)
        {
            log_error(__func__, "all-to-all extent exceeds D's free-0 extent", am);
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        // The copy moves n_shard contiguous elements per token row, so a strided
        // feature axis would ship unrelated data.
        if(matD->order != HIPBLASLT_ORDER_COL)
        {
            log_error(__func__, "all-to-all requires a unit free-0 stride on D");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        if(matD->type != HIP_R_16BF)
        {
            log_error(__func__, "all-to-all supports a BF16 D only", (int)matD->type);
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }

        // Copy widths and pitches are expressed in the descriptor's 16-byte
        // addressing granularity and are not rounded, and its extent and pitch
        // fields are narrow: 14 bits for an extent, 19 for the source pitch.
        constexpr int64_t elemsPer16B = 16 / sizeof(hip_bfloat16);
        const int64_t     shard       = am / (int64_t)world;
        const int64_t     ldd         = matD->ld;
        if(shard % elemsPer16B != 0 || ldd % elemsPer16B != 0)
        {
            log_error(__func__, "all-to-all shard extent and ldd must be 16-byte multiples");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        if(shard / elemsPer16B >= (int64_t{1} << 14) || ldd / elemsPer16B >= (int64_t{1} << 19))
        {
            log_error(__func__, "all-to-all shard extent or ldd exceeds the copy descriptor");
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        return HIPBLAS_STATUS_SUCCESS;
    }

    // Runs the checks a matmul carrying a fused epilogue owes before it is
    // dispatched. A descriptor with no communicating stage passes through.
    hipblasStatus_t validate_fused_epilogue_dispatch(hipblasLtHandle_t       handle,
                                                     hipblasLtMatmulDesc_t   matmulDesc,
                                                     hipblasLtMatrixLayout_t Ddesc)
    {
        if(matmulDesc == nullptr)
            return HIPBLAS_STATUS_SUCCESS;

        const hipblasLtFusedEpilogueDescriptor* desc
            = ((rocblaslt_matmul_desc)matmulDesc)->fused_epilogue;
        if(desc == nullptr
           || !fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX))
            return HIPBLAS_STATUS_SUCCESS;

        const hipblasStatus_t contents = validate_fused_epilogue_contents(desc);
        if(contents != HIPBLAS_STATUS_SUCCESS)
            return contents;

        return validate_fused_a2a_launch(
            (rocblaslt_handle)handle, desc, (rocblaslt_matrix_layout)Ddesc);
    }

    /****************************************************************************
     * Device communicator.
     ***************************************************************************/

    // What each rank contributes to the registration allgather. Opaque to the
    // caller, which must neither interpret nor reorder it. A peer in this process
    // is reached through its raw device pointer, once peer access to its device
    // is enabled, which is what the ordinal is for; one in another process is
    // reached through the IPC handle. All of it travels because which case a peer
    // falls into is only known once the payloads are in hand.
    struct DeviceCommExchange
    {
        uint32_t          magic;
        uint32_t          rank;
        uint32_t          world;
        uint32_t          nChannels;
        int32_t           device;
        uint64_t          pid;
        void*             flags;
        uint32_t          ipcValid;
        hipIpcMemHandle_t ipc;
    };

    constexpr uint32_t kDeviceCommMagic = 0x41324131u; // "1A2A"

    // Undoes whatever hipblasLtSetDeviceComm managed to put on the handle, from
    // any point in it, and leaves the handle unregistered. Each of the three
    // kinds of entry is released the way it was acquired, and a peer in this
    // process is another handle's allocation, so it is left alone.
    void release_device_comm(rocblaslt_handle handle)
    {
        for(uint32_t j = 0; j < HIPBLASLT_DEVICE_COMM_MAX_WORLD; ++j)
        {
            void* region = handle->device_comm_peer_flags[j];
            if(region != nullptr)
            {
                if(j == handle->device_comm_rank)
                    static_cast<void>(hipFree(region));
                else if(handle->device_comm_peer_flags_mapped[j])
                    static_cast<void>(hipIpcCloseMemHandle(region));
            }
            handle->device_comm_peer_flags[j]        = nullptr;
            handle->device_comm_peer_flags_mapped[j] = false;
        }
        handle->device_comm_rank     = 0;
        handle->device_comm_channels = 0;
        handle->device_comm_world    = 0;
    }
}

// The two pointers alias the descriptor's own storage; they stay valid only as long as
// the descriptor does.
bool rocblaslt_resolve_fused_epilogue(const hipblasLtFusedEpilogueDescriptor* desc,
                                      RocblasltFusedEpilogueInfo&             out)
{
    if(desc == nullptr)
        return false;

    out.hasA2APrefix = fused_epilogue_has_stage(desc, HIPBLASLT_FUSEABLE_EPILOGUE_A2A_PREFIX);
    out.a2aSdmaQueues = desc->a2a_queues.empty() ? nullptr : desc->a2a_queues.data();
    out.a2aRecvPtrs = desc->a2a_recv_ptrs.empty() ? nullptr : desc->a2a_recv_ptrs.data();
    out.a2aExtent         = desc->a2a_extent;
    out.a2aCompletionMode = desc->a2a_completion;
    out.commChannel       = desc->comm_channel;
    return true;
}
#endif

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
    // Two flag regions with different shapes: GSU reduction keeps the large
    // per-problem buffer it has always had, Stream-K gets a small one that can
    // afford a private region per stream. Both are allocated here so that no
    // matmul path allocates device memory, which would break hipGraph capture.
    void*            d_Synchronizer = nullptr;
    void*            d_StreamKFlags = nullptr;
    constexpr size_t gsuBytes = _rocblaslt_handle::c_syncGsuTotalElements * sizeof(int);
    constexpr size_t skBytes  = _rocblaslt_handle::c_syncSkTotalElements * sizeof(int);
    CHECK_HIP_ERROR(hipMalloc(&d_Synchronizer, gsuBytes));
    CHECK_HIP_ERROR(hipMemset(d_Synchronizer, 0, gsuBytes));
    if(hipError_t e = hipMalloc(&d_StreamKFlags, skBytes); e != hipSuccess)
    {
        static_cast<void>(hipFree(d_Synchronizer));
        CHECK_HIP_ERROR(e);
    }
    if(hipError_t e = hipMemset(d_StreamKFlags, 0, skBytes); e != hipSuccess)
    {
        static_cast<void>(hipFree(d_StreamKFlags));
        static_cast<void>(hipFree(d_Synchronizer));
        CHECK_HIP_ERROR(e);
    }

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = RocBlasLtStatusToHIPStatus(rocblaslt_create((rocblaslt_handle*)handle));
        (*(rocblaslt_handle*)handle)->Synchronizer = d_Synchronizer;
        (*(rocblaslt_handle*)handle)->StreamKFlags = d_StreamKFlags;
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
    if(handle != nullptr and (*(rocblaslt_handle)handle).StreamKFlags != nullptr)
    {
        CHECK_HIP_ERROR(hipFree((*(rocblaslt_handle)handle).StreamKFlags));
    }
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    if(handle != nullptr)
    {
        release_device_comm((rocblaslt_handle)handle);
    }
#endif

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

hipblasStatus_t hipblasLtSetUniformSummationOrder(hipblasLtHandle_t handle,
                                                  int32_t           uniformSummationOrder)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtSetUniformSummationOrder");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_set_uniform_summation_order((rocblaslt_handle)handle, uniformSummationOrder));
    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtGetUniformSummationOrder(hipblasLtHandle_t handle,
                                                  int32_t*          uniformSummationOrder)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtGetUniformSummationOrder");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_get_uniform_summation_order((rocblaslt_handle)handle, uniformSummationOrder));
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

hipblasStatus_t hipblasLtGroupedMatrixLayoutCreate(hipblasLtMatrixLayout_t* matLayout,
                                                   hipDataType              type,
                                                   int                      groupCount,
                                                   const void*              rowsArray,
                                                   const void*              colsArray,
                                                   const void*              ldArray)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtGroupedMatrixLayoutCreate");
    auto status = RocBlasLtStatusToHIPStatus(
        rocblaslt_grouped_matrix_layout_create((rocblaslt_matrix_layout*)matLayout,
                                               type,
                                               groupCount,
                                               rowsArray,
                                               colsArray,
                                               ldArray));
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

hipblasStatus_t hipblasLtMatmulDescSetAttribute(hipblasLtMatmulDesc_t           matmulDesc,
                                                hipblasLtMatmulDescAttributes_t matmulAttr,
                                                const void*                     buf,
                                                size_t                          sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtMatmulDescSetAttribute");

    // No kernel applies an aux scale, so refuse one rather than leave E unscaled.
    if(matmulAttr == HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER && buf != nullptr
       && sizeInBytes >= sizeof(void*))
    {
        const void* scaleAux = nullptr;
        memcpy(&scaleAux, buf, sizeof(void*));
        if(scaleAux != nullptr)
        {
            log_error(__func__,
                      "MatmulDesc: an epilogue aux scale is not supported and is not applied",
                      "attribute",
                      "HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",
                      "scale_aux",
                      scaleAux,
                      "size_in_bytes",
                      sizeInBytes);
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_NOT_SUPPORTED;
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

#if HIPBLASLT_HAS_GEMM_A2A_FUSION
hipblasStatus_t hipblasLtFusedEpilogueCreate(hipblasLtFusedEpilogueDescriptor_t* desc)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtFusedEpilogueCreate");
    if(desc == nullptr)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    *desc = new hipblasLtFusedEpilogueDescriptor();
    rocblaslt::Debug::Instance().markerStop();
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
    rocblaslt::Debug::Instance().markerStart("hipblasLtFusedEpilogueAdd");
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(desc == nullptr || !fused_epilogue_stage_recognized(epilogue))
    {
        status = HIPBLAS_STATUS_INVALID_VALUE;
    }
    else if(fused_epilogue_has_stage(desc, epilogue)
            || !fused_epilogue_stage_composes(desc, epilogue))
    {
        log_error(__func__, "epilogue stage does not compose with the stages already added");
        status = HIPBLAS_STATUS_INVALID_VALUE;
    }
    else
    {
        desc->stages.push_back(epilogue);
    }

    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueSetAttribute(hipblasLtFusedEpilogueDescriptor_t desc,
                                                   hipblasLtFusedEpilogueAttribute_t  attr,
                                                   const void*                        buf,
                                                   size_t                             sizeInBytes)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtFusedEpilogueSetAttribute");
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    if(desc == nullptr || buf == nullptr)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    switch(attr)
    {
    case HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_EXTENT:
    {
        if(sizeInBytes < sizeof(int64_t))
        {
            log_error(__func__, "invalid all-to-all extent buf size", sizeInBytes);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        int64_t am = 0;
        memcpy(&am, buf, sizeof(am));
        if(am <= 0)
        {
            log_error(__func__, "all-to-all extent must be positive", (int)am);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        desc->a2a_extent     = am;
        desc->a2a_extent_set = true;
        break;
    }
    case HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_COMPLETION_MODE:
    {
        if(sizeInBytes < sizeof(hipblasLtA2ACompletionMode_t))
        {
            log_error(__func__, "invalid all-to-all completion mode buf size", sizeInBytes);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        hipblasLtA2ACompletionMode_t mode = HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL;
        memcpy(&mode, buf, sizeof(mode));
        if(mode != HIPBLASLT_A2A_COMPLETION_IN_KERNEL_FULL)
        {
            log_error(__func__, "unsupported all-to-all completion mode", (int)mode);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        desc->a2a_completion = mode;
        break;
    }
    case HIPBLASLT_FUSED_EPILOGUE_COMM_CHANNEL:
    {
        if(sizeInBytes < sizeof(uint32_t))
        {
            log_error(__func__, "invalid comm channel buf size", sizeInBytes);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        // The upper bound is the communicator's nChannels, which lives on the
        // library handle and is not in hand here, so it is checked where the
        // handle is: before a solution is selected, and again at launch.
        memcpy(&desc->comm_channel, buf, sizeof(desc->comm_channel));
        break;
    }
    case HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_RECV_PTRS:
    {
        if(sizeInBytes == 0 || sizeInBytes % sizeof(void*) != 0
           || sizeInBytes / sizeof(void*) > HIPBLASLT_DEVICE_COMM_MAX_WORLD)
        {
            log_error(__func__, "invalid all-to-all peer-recv array size", sizeInBytes);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        const size_t count = sizeInBytes / sizeof(void*);
        desc->a2a_recv_ptrs.assign(count, nullptr);
        memcpy(desc->a2a_recv_ptrs.data(), buf, sizeInBytes);
        break;
    }
    case HIPBLASLT_FUSED_EPILOGUE_A2A_PREFIX_SDMA_QUEUES:
    {
        if(sizeInBytes == 0 || sizeInBytes % sizeof(hipblasLtSdmaQueue_t) != 0
           || sizeInBytes / sizeof(hipblasLtSdmaQueue_t) > HIPBLASLT_DEVICE_COMM_MAX_WORLD)
        {
            log_error(__func__, "invalid all-to-all SDMA queue array size", sizeInBytes);
            status = HIPBLAS_STATUS_INVALID_VALUE;
            break;
        }
        const size_t count = sizeInBytes / sizeof(hipblasLtSdmaQueue_t);
        desc->a2a_queues.assign(count, hipblasLtSdmaQueue_t{});
        memcpy(desc->a2a_queues.data(), buf, sizeInBytes);
        break;
    }
    default:
        log_error(__func__, "invalid fused epilogue attribute", (int)attr);
        status = HIPBLAS_STATUS_INVALID_VALUE;
        break;
    }

    rocblaslt::Debug::Instance().markerStop();
    return status;
}
catch(...)
{
    return exception_to_hipblas_status();
}

hipblasStatus_t hipblasLtFusedEpilogueDestroy(hipblasLtFusedEpilogueDescriptor_t desc)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtFusedEpilogueDestroy");
    delete desc;
    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}

// TODO(#11940): Move communicator and peer-transport setup to RCCL or a shared
// transport layer; keep the fused GEMM handoff in hipBLASLt.
hipblasStatus_t hipblasLtSetDeviceComm(hipblasLtHandle_t              handle,
                                       uint32_t                       rank,
                                       uint32_t                       world,
                                       uint32_t                       nChannels,
                                       hipblasLtDeviceCommAllgatherFn allgather,
                                       void*                          userData)
try
{
    rocblaslt::Debug::Instance().markerStart("hipblasLtSetDeviceComm");

    if(handle == nullptr)
    {
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    }

    rocblaslt_handle h = (rocblaslt_handle)handle;

    if(world < 1 || world > HIPBLASLT_DEVICE_COMM_MAX_WORLD || rank >= world || nChannels == 0
       || allgather == nullptr)
    {
        log_error(__func__, "invalid communicator arguments; world", (int)world);
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Exactly once per handle, matching arguments or not. That is what makes world
    // immutable, and world participates in solution selection.
    if(h->device_comm_world != 0)
    {
        log_error(__func__, "this handle already carries a communicator");
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_INVALID_VALUE;
    }

    // Travels in the payload: a peer in this process is reached from here only
    // once peer access to its device is enabled, and that call names the ordinal.
    int myDevice = 0;
    if(hipGetDevice(&myDevice) != hipSuccess)
    {
        log_error(__func__, "could not resolve this rank's device ordinal");
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    // The kernel owns the flag region's layout, so its size is asked for rather
    // than recomputed here; that keeps one definition of it.
    void*        flags = nullptr;
    const size_t bytes = (size_t)nChannels * rocblaslt_device_comm_flag_block_bytes();
    // Fine-grained, because a peer's copy engine updates these lines.
    if(hipExtMallocWithFlags(&flags, bytes, hipDeviceMallocFinegrained) != hipSuccess
       || flags == nullptr)
    {
        log_error(__func__, "could not allocate the communicator's flag regions");
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }
    if(hipMemset(flags, 0, bytes) != hipSuccess)
    {
        static_cast<void>(hipFree(flags));
        rocblaslt::Debug::Instance().markerStop();
        return HIPBLAS_STATUS_ALLOC_FAILED;
    }

    DeviceCommExchange mine{};
    mine.magic     = kDeviceCommMagic;
    mine.rank      = rank;
    mine.world     = world;
    mine.nChannels = nChannels;
    mine.device    = myDevice;
    mine.pid       = (uint64_t)getpid();
    mine.flags     = flags;
    // Only needed by a peer in another process; a failure here is reported when
    // such a peer is actually found, not before.
    mine.ipcValid = (hipIpcGetMemHandle(&mine.ipc, flags) == hipSuccess) ? 1u : 0u;

    std::vector<DeviceCommExchange> all(world);
    hipblasStatus_t                 status
        = allgather(userData, &mine, all.data(), sizeof(DeviceCommExchange));
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        static_cast<void>(hipFree(flags));
        rocblaslt::Debug::Instance().markerStop();
        return status;
    }

    // The handle owns the allocation from here on: it is this rank's own entry,
    // and rank is what release_device_comm looks at to tell it from the peers.
    h->device_comm_rank             = rank;
    h->device_comm_peer_flags[rank] = flags;

    for(uint32_t j = 0; j < world; ++j)
    {
        const DeviceCommExchange& peer = all[j];
        if(peer.magic != kDeviceCommMagic || peer.rank != j || peer.world != world
           || peer.nChannels != nChannels || peer.flags == nullptr)
        {
            log_error(__func__, "ranks disagree about the communicator; rank", (int)j);
            release_device_comm(h);
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }

        if(j == rank || peer.pid == mine.pid)
        {
            // A pointer from this process is addressable as it stands only on
            // this device; reaching another one takes peer access, or the copy
            // engine faults on a region it was handed. Already-enabled is a
            // success, since in one process the W handles enable each other
            // pairwise and either direction may be established first.
            if(j != rank && peer.device != mine.device)
            {
                const hipError_t peerAccess = hipDeviceEnablePeerAccess(peer.device, 0);
                if(peerAccess != hipSuccess && peerAccess != hipErrorPeerAccessAlreadyEnabled)
                {
                    log_error(__func__, "cannot reach a peer's device from this one; rank", (int)j);
                    release_device_comm(h);
                    rocblaslt::Debug::Instance().markerStop();
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
            }
            h->device_comm_peer_flags[j]        = peer.flags;
            h->device_comm_peer_flags_mapped[j] = false;
            continue;
        }

        void* mapped = nullptr;
        if(!peer.ipcValid
           || hipIpcOpenMemHandle(&mapped, peer.ipc, hipIpcMemLazyEnablePeerAccess) != hipSuccess)
        {
            log_error(__func__, "cannot map a peer's flag region into this process; rank", (int)j);
            release_device_comm(h);
            rocblaslt::Debug::Instance().markerStop();
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        h->device_comm_peer_flags[j]        = mapped;
        h->device_comm_peer_flags_mapped[j] = true;
    }

    h->device_comm_channels = nChannels;
    // Last, and what makes the handle registered: every peer's region is in hand.
    h->device_comm_world = world;

    rocblaslt::Debug::Instance().markerStop();
    return HIPBLAS_STATUS_SUCCESS;
}
catch(...)
{
    return exception_to_hipblas_status();
}
#endif

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

#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    // Everything a communicating stage can be judged on before a solution exists
    // is judged here: an all-to-all's extent participates in tile selection, so a
    // request that cannot be served must not come back as a usable algo.
    hipblasStatus_t fused_status
        = validate_fused_epilogue_dispatch(handle, matmulDesc, Ddesc);
    if(fused_status != HIPBLAS_STATUS_SUCCESS)
    {
        rocblaslt::Debug::Instance().markerStop();
        return fused_status;
    }
#endif

    auto               desc = (rocblaslt_matmul_desc)matmulDesc;
    auto               matA = (rocblaslt_matrix_layout)Adesc;
    auto               matB = (rocblaslt_matrix_layout)Bdesc;
    auto               matC = (rocblaslt_matrix_layout)Cdesc;
    auto               matD = (rocblaslt_matrix_layout)Ddesc;
    ColumnMajorProblem columnMajor;
    bool               rewritten = false;
    if(auto order_status
       = heuristic_arguments_present(handle, matmulDesc, pref, requestedAlgoCount, returnAlgoCount)
             ? rewrite_row_major(__func__,
                                 (rocblaslt_handle)handle,
                                 desc,
                                 matA,
                                 matB,
                                 matC,
                                 matD,
                                 columnMajor,
                                 rewritten)
             : HIPBLAS_STATUS_SUCCESS;
       order_status != HIPBLAS_STATUS_SUCCESS)
    {
        *returnAlgoCount = 0;
        rocblaslt::Debug::Instance().markerStop();
        return order_status;
    }
    if(rewritten)
    {
        desc = &columnMajor.desc;
        matA = &columnMajor.a;
        matB = &columnMajor.b;
        matC = &columnMajor.c;
        matD = &columnMajor.d;
    }

    auto status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul_algo_get_heuristic(
        (rocblaslt_handle)handle,
        desc,
        matA,
        matB,
        matC,
        matD,
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

#if HIPBLASLT_HAS_GEMM_A2A_FUSION
    return_status = validate_fused_epilogue_dispatch(handle, matmul_descr, matD);
    if(return_status != HIPBLAS_STATUS_SUCCESS)
    {
        rocblaslt::Debug::Instance().markerStop();
        return return_status;
    }
#endif

    auto               desc    = (rocblaslt_matmul_desc)matmul_descr;
    auto               layoutA = (rocblaslt_matrix_layout)matA;
    auto               layoutB = (rocblaslt_matrix_layout)matB;
    auto               layoutC = (rocblaslt_matrix_layout)matC;
    auto               layoutD = (rocblaslt_matrix_layout)matD;
    ColumnMajorProblem columnMajor;
    bool               rewritten = false;
    if(matmul_arguments_present(handle,
                                matmul_descr,
                                alpha,
                                A,
                                matA,
                                B,
                                matB,
                                beta,
                                C,
                                matC,
                                D,
                                matD,
                                workspace,
                                workspaceSizeInBytes))
        return_status = rewrite_row_major(__func__,
                                          (rocblaslt_handle)handle,
                                          desc,
                                          layoutA,
                                          layoutB,
                                          layoutC,
                                          layoutD,
                                          columnMajor,
                                          rewritten);
    if(return_status != HIPBLAS_STATUS_SUCCESS)
    {
        rocblaslt::Debug::Instance().markerStop();
        return return_status;
    }
    if(rewritten)
    {
        desc    = &columnMajor.desc;
        layoutA = &columnMajor.a;
        layoutB = &columnMajor.b;
        layoutC = &columnMajor.c;
        layoutD = &columnMajor.d;
        if(columnMajor.swapped)
            std::swap(A, B);
    }

    return_status = RocBlasLtStatusToHIPStatus(rocblaslt_matmul((rocblaslt_handle)handle,
                                                                desc,
                                                                alpha,
                                                                A,
                                                                layoutA,
                                                                B,
                                                                layoutB,
                                                                beta,
                                                                C,
                                                                layoutC,
                                                                D,
                                                                layoutD,
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
