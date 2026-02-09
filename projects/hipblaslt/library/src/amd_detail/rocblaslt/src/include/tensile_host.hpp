/*! \file */
/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

/*********************************************************
 * Declaration of the rocBLAS<->Tensile interface layer. *
 *********************************************************/

#pragma once

/*****************************************************************************
 * WARNING: Tensile-specific data types, functions and macros should only be *
 * referenced from tensile_host.cpp. This header file defines the interface  *
 * that the rest of rocBLAS uses to access Tensile. If another Tensile       *
 * feature needs to be accessed, the API for accessing it should be defined  *
 * in this file, without referencing any Tensile-specific identifiers here.  *
 *****************************************************************************/

#include "handle.h"
//#include "tuple_helper.hpp"
#include "UserDrivenTuningParser.hpp"
#include "utility.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/DataTypes.hpp>
#include <atomic>

// Return the value category for a value, as a double precision value, such
// such as whether it's 0, 1, -1 or some other value. Tensile uses a double
// precision value to express the category of beta. This function is to
// convert complex or other types to a double representing the category.
template <typename T>
constexpr double value_category(const T& beta)
{
    return beta == T(0) ? 0.0 : beta == T(1) ? 1.0 : beta == T(-1) ? -1.0 : 2.0;
}

void initTensileGemmData(rocblaslt_handle       handle,
                         rocblaslt::RocGemmType gemmType,
                         hipblasOperation_t     opA,
                         hipblasOperation_t     opB,
                         hipDataType            typeA,
                         hipDataType            typeB,
                         hipDataType            typeC,
                         hipDataType            typeD,
                         rocblaslt_compute_type typeCompute,
                         size_t                 maxWorkspaceBytes,
                         std::shared_ptr<void>& gemmData);

/*******************************************************************************
 * runContractionProblem() solves a RocblasltContractionProblem *
 *******************************************************************************/
rocblaslt_status runContractionProblem(rocblaslt_handle                   handle,
                                       const rocblaslt_matmul_algo*       algo,
                                       RocblasltContractionProblem const& problem,
                                       std::shared_ptr<void>              gemmData);

rocblaslt_status gemmCreate(RocblasltContractionProblem const& problem,
                            std::shared_ptr<void>&             gemmData,
                            size_t&                            gemmCount);

rocblaslt_status groupedGemmCreate(std::vector<RocblasltContractionProblem>& probs,
                                   std::shared_ptr<void>&                    gemmData,
                                   size_t&                                   gemmCount);

template <typename Tuning>
rocblaslt_status makeArgument(rocblaslt_handle             handle,
                              const rocblaslt::RocGemmType gemmType,
                              const rocblaslt_matmul_algo& algo,
                              const Tuning*                tuning,
                              void*                        workspace,
                              size_t                       workspaceSizeInBytes,
                              bool                         useUserArgs,
                              hipStream_t                  stream,
                              std::shared_ptr<void>        gemmData);

// Run gemm only, without creating args, problems,...
rocblaslt_status runKernelFromInvocation(rocblaslt_handle       handle,
                                         rocblaslt::RocGemmType gemmType,
                                         std::shared_ptr<void>  gemmData,
                                         hipStream_t            stream,
                                         hipEvent_t             start = nullptr,
                                         hipEvent_t             stop  = nullptr);

rocblaslt_status getDeviceUserArgumentsValuesFromContractionProblem(rocblaslt_handle       handle,
                                                                    rocblaslt::RocGemmType gemmType,
                                                                    std::shared_ptr<void>  gemmData,
                                                                    void* hostDeviceUserArgs);

rocblaslt_status runKernelFromNewDeviceUserArguments(rocblaslt_handle       handle,
                                                     rocblaslt::RocGemmType gemmType,
                                                     std::shared_ptr<void>  gemmData,
                                                     void*                  deviceUserArgs,
                                                     hipStream_t            stream);

rocblaslt_status runKernelFromDeviceUserArguments(rocblaslt_handle             handle,
                                                  rocblaslt::RocGemmType       gemmType,
                                                  size_t                       gemmCount,
                                                  std::shared_ptr<void>        gemmData,
                                                  const rocblaslt_matmul_algo& algo,
                                                  void*                        deviceUserArgs,
                                                  void*                        workspace,
                                                  hipStream_t                  stream);

std::string getKernelNameFromData(rocblaslt_handle             handle,
                                  const rocblaslt::RocGemmType gemmType,
                                  std::shared_ptr<void>        gemmData);

std::string getSolutionNameFromData(rocblaslt_handle             handle,
                                    const rocblaslt::RocGemmType gemmType,
                                    std::shared_ptr<void>        gemmData);

std::string getKernelNameFromAlgoIndex(rocblaslt_handle handle, const rocblaslt_matmul_algo& algo);

std::string getSolutionNameFromAlgoIndex(rocblaslt_handle             handle,
                                         const rocblaslt_matmul_algo& algo);

/***********************************************************************************
 * Whether Tensile has been initialized for at least one device (used for
 *testing) *
 ***********************************************************************************/
std::atomic_bool& rocblaslt_internal_tensile_is_initialized();

/**********************************************
 * Whether to suppress Tensile error messages *
 **********************************************/
inline bool& rocblaslt_suppress_tensile_error_messages()
{
    thread_local bool t_suppress = false;
    return t_suppress;
}

rocblaslt_status getAllSolutions(RocblasltContractionProblem&                    prob,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes);

rocblaslt_status getAllSolutions(std::vector<RocblasltContractionProblem>&       probs,
                                 rocblaslt_handle                                handle,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes);

rocblaslt_status getAllSolutions(std::shared_ptr<void>                           gemmData,
                                 rocblaslt_handle                                handle,
                                 rocblaslt::RocGemmType                          gemmType,
                                 std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                                 size_t                                          maxWorkSpaceBytes);

rocblaslt_status
    getSolutionsFromIndex(rocblaslt_handle                                handle,
                          std::vector<int>&                               solutionIndex,
                          std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                          size_t                                          maxWorkSpaceBytes);

rocblaslt_status isSolutionSupported(rocblaslt_handle             handle,
                                     RocblasltContractionProblem& prob,
                                     std::shared_ptr<void>        gemmData,
                                     rocblaslt_matmul_algo*       algo,
                                     size_t*                      workspaceSizeInBytes);

template <typename Tuning>
rocblaslt_status isSolutionSupported(rocblaslt_handle              handle,
                                     const rocblaslt::RocGemmType& gemmType,
                                     std::shared_ptr<void>         gemmData,
                                     rocblaslt_matmul_algo&        algo,
                                     const Tuning*                 tuning,
                                     size_t&                       workspaceSizeInBytes);

std::vector<std::shared_ptr<TensileLite::ContractionSolution>>
    getBestRawSolutions(RocblasltContractionProblem const& prob,
                        rocblaslt_handle                   handle,
                        std::shared_ptr<void>              gemmData,
                        int                                requestedAlgoCount,
                        size_t                             maxWorkSpaceBytes);

/*******************************************************************************
 * getBestSolutions() calls finTopSolutions from Tensile and converts to       *
 * rocblaslt_matmul_heuristic_result                                           *
 *******************************************************************************/
rocblaslt_status getBestSolutions(RocblasltContractionProblem const& prob,
                                  rocblaslt_handle                   handle,
                                  std::shared_ptr<void>              gemmData,
                                  int                                requestedAlgoCount,
                                  rocblaslt_matmul_heuristic_result  heuristicResultsArray[],
                                  int*                               returnAlgoCount,
                                  size_t                             maxWorkSpaceBytes);

rocblaslt_status getBestSolutions(rocblaslt_handle       handle,
                                  rocblaslt::RocGemmType gemmType,
                                  std::shared_ptr<void>  gemmData,
                                  const int              workspaceBytes,
                                  const int              requestedAlgoCount,
                                  std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults);

/******************************************************
 * Map a hipblaslt data type to a corresponding Tensile type *
 ******************************************************/
inline rocisa::DataType hipDataType_to_tensile_type(hipDataType type)
{
    switch(type)
    {
    case HIP_R_16F:
        return rocisa::DataType::Half;
    case HIP_R_32F:
        return rocisa::DataType::Float;
    case HIP_R_64F:
        return rocisa::DataType::Double;
    case HIP_R_16BF:
        return rocisa::DataType::BFloat16;
    case HIP_R_8F_E4M3_FNUZ:
        return rocisa::DataType::Float8_fnuz;
    case HIP_R_8F_E5M2_FNUZ:
        return rocisa::DataType::BFloat8_fnuz;
    case HIP_R_8F_E4M3:
        return rocisa::DataType::Float8;
    case HIP_R_8F_E5M2:
        return rocisa::DataType::BFloat8;
    case HIP_R_8I:
        return rocisa::DataType::Int8;
    case HIP_R_32I:
        return rocisa::DataType::Int32;
    default:
        assert(!"hipDataType_to_tensile_type: non-supported type");
        return rocisa::DataType::None;
    }
}

inline rocisa::DataType rocComputeType_to_tensile_type(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f32_fast_xf32:
        return rocisa::DataType::XFloat32;
    case rocblaslt_compute_f32_fast_f16:
        return rocisa::DataType::Half;
    case rocblaslt_compute_f32_fast_bf16:
        return rocisa::DataType::BFloat16;
    case rocblaslt_compute_f16:
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
    case rocblaslt_compute_f32_fast_f8:
    case rocblaslt_compute_f32_fast_bf8:
    case rocblaslt_compute_f32_fast_f8bf8:
    case rocblaslt_compute_f32_fast_bf8f8:
        return rocisa::DataType::Float;
    case rocblaslt_compute_f64:
        return rocisa::DataType::Double;
    case rocblaslt_compute_i32:
        return rocisa::DataType::Int32;
    default:
        assert(!"rocDataType_to_tensile_type: non-supported type");
        return rocisa::DataType::None;
    }
}

namespace TensileLite
{
    class ProblemOverride;
}

TensileLite::ProblemOverride
    RocblasltContractionProblem2ProblemOverride(const RocblasltContractionProblem&);

TensileLite::ProblemOverride TensileDataGemm2ProblemOverride(std::shared_ptr<void>);

TensileLite::ContractionProblemGemm* ExtractProblemGemm(std::shared_ptr<void>);
