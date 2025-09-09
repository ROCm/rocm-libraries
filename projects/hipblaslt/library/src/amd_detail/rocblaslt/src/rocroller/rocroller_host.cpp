/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * The implementation of the rocblaslt<->rocRoller interface layer. *
 *********************************************************/

#include "gemm.hpp"
#include "kernel_type.hpp"
#include "rocroller_host.hpp"
#include "runtime_args_selection.hpp"
#include "parameter_selection.hpp"
#include "solution_selection.hpp"

#include "Debug.hpp"
#include "handle.h"
#include "utility.hpp"

using namespace rocRoller;

/**
 * @brief RocRollerHandle
 *
 * State that is needed for executing rocRoller kernels
 *
 */
struct RocRollerHandle
{
    // Map of kernels that have already been generated.
    // The first level of the map is indexed with a KernelType.
    // The second level of the map is indexed with a hash value of a
    // SolutionIndexParameters type.
    // The value is a GemmKernel.
    std::map<KernelType, std::map<int, std::shared_ptr<GemmKernel>>> generatedKernels;
};

/**
 * @brief Create a new rocRoller handle.
 *
 * This should be done whenever a hipBLASLt handle is created.
 *
 * @param handle
 */
void rocroller_create_handle(void** handle)
{
    *handle = new RocRollerHandle();
}

/**
 * @brief Destroy a rocRoller handle
 *
 * This should be done whenever a hipBLASLt handle is destroyed.
 *
 * @param handle
 */
void rocroller_destroy_handle(void* handle)
{
    delete static_cast<RocRollerHandle*>(handle);
}

inline std::string scaleModeOption(RocblasltContractionProblem::ScalingFormat scale)
{
    switch(scale)
    {
    case RocblasltContractionProblem::ScalingFormat::Scalar:
        return "1";
    case RocblasltContractionProblem::ScalingFormat::Vector:
        return "2";
    case RocblasltContractionProblem::ScalingFormat::Block:
        return "3";
    default:
        return "";
    }
}

inline void logBench(const RocblasltContractionProblem& prob,
                     const int&                         solutionIndex,
                     bool                               flush,
                     const int32_t&                     rotatingBufferSize,
                     const int32_t&                     coldIterations,
                     const int32_t&                     hotIterations)
{
    auto s = log_str(__func__,
                     "--api_method",
                     "c",
                     "-m",
                     prob.m,
                     "-n",
                     prob.n,
                     "-k",
                     prob.k,
                     "--lda",
                     prob.col_stride_a,
                     "--ldb",
                     prob.col_stride_b,
                     "--ldc",
                     prob.col_stride_c,
                     "--ldd",
                     prob.col_stride_d,
                     "--stride_a",
                     prob.batch_stride_a,
                     "--stride_b",
                     prob.batch_stride_b,
                     "--stride_c",
                     prob.batch_stride_c,
                     "--stride_d",
                     prob.batch_stride_d,
                     "--alpha",
                     *((float*)prob.alpha),
                     "--beta",
                     *((float*)prob.beta),
                     "--transA",
                     prob.trans_a ? "T" : "N",
                     "--transB",
                     prob.trans_b ? "T" : "N",
                     "--batch_count",
                     prob.batch_count,
                     "--scaleA",
                     scaleModeOption(prob.scaleAType),
                     "--scaleB",
                     scaleModeOption(prob.scaleBType),
                     "--a_type",
                     hipDataType_to_bench_string(prob.a_type),
                     "--b_type",
                     hipDataType_to_bench_string(prob.b_type),
                     "--c_type",
                     hipDataType_to_bench_string(prob.c_type),
                     "--d_type",
                     hipDataType_to_bench_string(prob.d_type),
                     "--compute_type",
                     "f32_r",
                     "--algo_method",
                     "index",
                     "--solution_index",
                     solutionIndex,
                     flush ? "--flush" : "",
                     "--rotating",
                     rotatingBufferSize,
                     "--cold_iters",
                     coldIterations,
                     "--iters",
                     hotIterations);

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
        log_bench_from_str(s);
    if(rocblaslt::Debug::Instance().printLogAsMarker())
    {
        rocblaslt::Debug::Instance().logMarkerStart(s.c_str());
        rocblaslt::Debug::Instance().logMarkerStop();
    }
}

inline void logProfile(const RocblasltContractionProblem& prob,
                       bool                               flush,
                       const int32_t&                     rotatingBufferSize,
                       const int32_t&                     coldIterations,
                       const int32_t&                     hotIterations)
{
    log_profile("matmul",
                "M",
                prob.m,
                "N",
                prob.n,
                "K",
                prob.k,
                "lda",
                prob.col_stride_a,
                "ldb",
                prob.col_stride_b,
                "ldc",
                prob.col_stride_c,
                "ldd",
                prob.col_stride_d,
                "stride_a",
                prob.batch_stride_a,
                "stride_b",
                prob.batch_stride_b,
                "stride_c",
                prob.batch_stride_c,
                "stride_d",
                prob.batch_stride_e,
                "alpha",
                *((float*)prob.alpha),
                "beta",
                *((float*)prob.beta),
                "transA",
                prob.trans_a == HIPBLAS_OP_T ? "T" : "N",
                "transB",
                prob.trans_b == HIPBLAS_OP_T ? "T" : "N",
                "batch_count",
                prob.batch_count,
                "scaleA",
                scaleModeOption(prob.scaleAType),
                "scaleB",
                scaleModeOption(prob.scaleBType),
                "a_type",
                hipDataType_to_bench_string(prob.a_type),
                "b_type",
                hipDataType_to_bench_string(prob.b_type),
                "c_type",
                hipDataType_to_bench_string(prob.c_type),
                "d_type",
                hipDataType_to_bench_string(prob.d_type),
                "compute_type",
                "f32_r",
                "flush",
                flush ? "true" : "false",
                "rotating",
                rotatingBufferSize,
                "cold_iters",
                coldIterations,
                "iters",
                hotIterations);
}

inline void logExtendedProfile(const RocblasltContractionProblem& prob,
                               const int&                         solutionIndex,
                               const std::string&                 kernelName,
                               bool                               flush,
                               const int32_t&                     rotatingBufferSize,
                               const int32_t&                     coldIterations,
                               const int32_t&                     hotIterations)
{
    log_profile("matmul",
                "M",
                prob.m,
                "N",
                prob.n,
                "K",
                prob.k,
                "lda",
                prob.col_stride_a,
                "ldb",
                prob.col_stride_b,
                "ldc",
                prob.col_stride_c,
                "ldd",
                prob.col_stride_d,
                "stride_a",
                prob.batch_stride_a,
                "stride_b",
                prob.batch_stride_b,
                "stride_c",
                prob.batch_stride_c,
                "stride_d",
                prob.batch_stride_e,
                "alpha",
                *((float*)prob.alpha),
                "beta",
                *((float*)prob.beta),
                "transA",
                prob.trans_a == HIPBLAS_OP_T ? "T" : "N",
                "transB",
                prob.trans_b == HIPBLAS_OP_T ? "T" : "N",
                "batch_count",
                prob.batch_count,
                "scaleA",
                scaleModeOption(prob.scaleAType),
                "scaleB",
                scaleModeOption(prob.scaleBType),
                "a_type",
                hipDataType_to_bench_string(prob.a_type),
                "b_type",
                hipDataType_to_bench_string(prob.b_type),
                "c_type",
                hipDataType_to_bench_string(prob.c_type),
                "d_type",
                hipDataType_to_bench_string(prob.d_type),
                "compute_type",
                "f32_r",
                "flush",
                flush ? "true" : "false",
                "rotating",
                rotatingBufferSize,
                "cold_iters",
                coldIterations,
                "iters",
                hotIterations,
                "solution_index",
                solutionIndex,
                "kernel_name",
                kernelName);
}

/**
 * @brief Convert hipDataType to a rocRoller::Datatype
 *
 * @param type
 * @return rocRoller::DataType
 */
rocRoller::DataType hipDataType_to_rocRoller_type(hipDataType type)
{
    // Older versions of ROCm do not have these types defined,
    // so they need to be handled specially.
    if(static_cast<int>(type) == HIP_R_6F_E2M3_EXT)
    {
        return rocRoller::DataType::FP6;
    }
    if(static_cast<int>(type) == HIP_R_6F_E3M2_EXT)
    {
        return rocRoller::DataType::BF6;
    }
    if(static_cast<int>(type) == HIP_R_4F_E2M1_EXT)
    {
        return rocRoller::DataType::FP4;
    }

    switch(type)
    {
    case HIP_R_16F:
        return rocRoller::DataType::Half;
    case HIP_R_32F:
        return rocRoller::DataType::Float;
    case HIP_R_16BF:
        return rocRoller::DataType::BFloat16;
    case HIP_R_8F_E4M3_FNUZ:
        return rocRoller::DataType::FP8;
    case HIP_R_8F_E5M2_FNUZ:
        return rocRoller::DataType::BF8;
    case HIP_R_8F_E4M3:
        return rocRoller::DataType::FP8;
    case HIP_R_8F_E5M2:
        return rocRoller::DataType::BF8;
    default:
        return rocRoller::DataType::None;
    }
}

/**
 * @brief Convert a rocblaslt_compute_type to a rocRoller::DataType
 *
 * @param type
 * @return rocRoller::DataType
 */
rocRoller::DataType rocblaslt_compute_type_to_rocRoller_type(rocblaslt_compute_type type)
{
    switch(type)
    {
    case rocblaslt_compute_f16:
        return rocRoller::DataType::Half;
    case rocblaslt_compute_f32:
    case rocblaslt_compute_f32_fast_xf32:
    case rocblaslt_compute_f32_fast_f16:
    case rocblaslt_compute_f32_fast_bf16:
    case rocblaslt_compute_f32_fast_f8_fnuz:
    case rocblaslt_compute_f32_fast_bf8_fnuz:
    case rocblaslt_compute_f32_fast_f8bf8_fnuz:
    case rocblaslt_compute_f32_fast_bf8f8_fnuz:
    case rocblaslt_compute_f32_fast_f8:
    case rocblaslt_compute_f32_fast_bf8:
    case rocblaslt_compute_f32_fast_f8bf8:
    case rocblaslt_compute_f32_fast_bf8f8:
        return rocRoller::DataType::Float;
    case rocblaslt_compute_f64:
        return rocRoller::DataType::Double;
    case rocblaslt_compute_i32:
        return rocRoller::DataType::Int32;
    default:
        return rocRoller::DataType::None;
    }
}

/**
 * @brief Generate a KernelType from a RocblasltContractionProblem
 *
 *
 * @param prob
 * @return kernelType
 */
KernelType genKernelType(const RocblasltContractionProblem& prob)
{
    KernelType kernelType;

    kernelType.typeA      = hipDataType_to_rocRoller_type(prob.a_type);
    kernelType.typeB      = hipDataType_to_rocRoller_type(prob.b_type);
    kernelType.typeC      = hipDataType_to_rocRoller_type(prob.c_type);
    kernelType.typeD      = hipDataType_to_rocRoller_type(prob.d_type);
    kernelType.typeAcc    = rocblaslt_compute_type_to_rocRoller_type(prob.compute_type);
    kernelType.transA     = prob.trans_a == HIPBLAS_OP_T;
    kernelType.transB     = prob.trans_b == HIPBLAS_OP_T;
    kernelType.scaleAMode = prob.scaleAType == RocblasltContractionProblem::ScalingFormat::Block
                                ? rocRoller::Operations::ScaleMode::Separate
                                : rocRoller::Operations::ScaleMode::None;
    kernelType.scaleBMode = prob.scaleBType == RocblasltContractionProblem::ScalingFormat::Block
                                ? rocRoller::Operations::ScaleMode::Separate
                                : rocRoller::Operations::ScaleMode::None;
    kernelType.scaleABlockRowSize = prob.scaleABlockRowSize;
    kernelType.scaleABlockColSize = prob.scaleABlockColSize;
    kernelType.scaleBBlockRowSize = prob.scaleBBlockRowSize;
    kernelType.scaleBBlockColSize = prob.scaleBBlockColSize;

    return kernelType;
}


/**
 * Generate a kernel from a given SolutionIndexParameters value.
 */
rocblaslt_status
    genKernelFromSolutionIndexParameters(RocRollerHandle*             rocroller_handle,
                                         KernelType                   kernelType,
                                         SolutionIndexParameters      solutionIndexParameter,
                                         int                          solutionIndex,
                                         std::shared_ptr<GemmKernel>& kernel)
{
    auto params = genSolutionParameters(kernelType, solutionIndexParameter);
    try
    {
        kernel                                                        = genGemmKernel(params);
        rocroller_handle->generatedKernels[kernelType][solutionIndex] = kernel;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return rocblaslt_status_not_implemented;
    }

    return rocblaslt_status_success;
}

/**
 * @brief Find the best rocRoller kernels for a given problem
 *
 * This mimics the functionality of getBestSolutions in tensile_host.cpp
 *
 * For a given kernel type and problem, determines the SolutionIndexParameters
 * that should be used.
 *
 * Checks to see if a kernel has already been generated for the chosen SolutionIndexParameters.
 *
 * If it hasn't, a new kernel will be generated and stored in generatedKernels.
 *
 * At the moment, only returns a single solution.
 *
 * @param handle
 * @param prob
 * @param requestedAlgoCount
 * @param heuristicResultsArray
 * @param maxWorkSpaceBytes
 * @param returnAlgoCount
 * @return rocblaslt_status
 */
rocblaslt_status
    getRocRollerBestSolutions(rocblaslt_handle                   handle,
                              const RocblasltContractionProblem& prob,
                              int                                requestedAlgoCount,
                              rocblaslt_matmul_heuristic_result  heuristicResultsArray[],
                              size_t                             maxWorkSpaceBytes,
                              int*                               returnAlgoCount)
{
    RocRollerHandle* rocroller_handle = static_cast<RocRollerHandle*>(handle->rocroller_handle);
    auto             kernelType       = genKernelType(prob);
    int              index;

    if(prob.bias != nullptr)
    {
        std::cerr << "rocRoller does not support bias" << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(prob.batch_count != 1)
    {
        std::cerr << "rocRoller only supports 1 batch_count not " << prob.batch_count << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(auto scale_type = hipDataType_to_rocRoller_type(prob.scale_type);
       scale_type != rocRoller::DataType::None && scale_type != rocRoller::DataType::Float)
    {
        std::cerr << "rocRoller only supports F32 as scale type not " << scale_type << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(kernelType.typeAcc != rocRoller::DataType::Float)
    {
        std::cerr << "rocRoller only supports F32 accumulation, not " << kernelType.typeAcc
                  << std::endl;
        return rocblaslt_status_invalid_value;
    }

    auto existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    if(existingKernelType == rocroller_handle->generatedKernels.end())
    {
        rocroller_handle->generatedKernels[kernelType] = {};
    }

    auto solutionIndexParameters
        = chooseSolutionIndexParameters(kernelType, prob, requestedAlgoCount);

    int i = 0;
    for(auto const& solutionIndexParameter : solutionIndexParameters)
    {
        if(requestedAlgoCount != -1 && i >= requestedAlgoCount)
            break;

        index = parametersToIndex(solutionIndexParameter);
        auto existingSolutionIndex = rocroller_handle->generatedKernels[kernelType].find(index);
        std::shared_ptr<GemmKernel> kernel;
        // If kernel doesn't already exist, generate it
        if(existingSolutionIndex == rocroller_handle->generatedKernels[kernelType].end())
        {
            auto                        status = genKernelFromSolutionIndexParameters(
                rocroller_handle, kernelType, solutionIndexParameter, index, kernel);
            if(status != rocblaslt_status_success)
                continue;
        }
        else
        {
            kernel = existingSolutionIndex->second;
        }

        // Fill out heuristicResultsArray
        // The most important thing to do is set the solutionIndex
        memset(heuristicResultsArray[i].algo.data, 0, sizeof(heuristicResultsArray[i].algo.data));
        int* solutionIndex = (int*)(heuristicResultsArray[i].algo.data);
        *solutionIndex     = index;
        heuristicResultsArray[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResultsArray[i].algo.fallback            = false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize            = workspaceRequired(kernel, prob);
        i++;
    }

    *returnAlgoCount = i;
    for(; i < requestedAlgoCount; i++)
    {
        heuristicResultsArray[i].state = rocblaslt_status_invalid_value;
    }

    return rocblaslt_status_success;
}

/**
 * Return all of the possible solutions for a KernelType
 */
rocblaslt_status
    getAllSolutionsRocRoller(RocblasltContractionProblem&                    prob,
                             rocblaslt_handle                                handle,
                             std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
                             size_t                                          maxWorkSpaceBytes)
{
    heuristicResults.resize(maxNumberSolutions());
    int  returnAlgoCount;
    auto result
        = getRocRollerBestSolutions(handle, prob, -1, heuristicResults.data(), maxWorkSpaceBytes, &returnAlgoCount);
    heuristicResults.resize(returnAlgoCount);
    return result;
}

/**
 * Return a list of heuristicResults for a given list of solution indices
 */
void getRocRollerSolutionsFromIndex(
    rocblaslt_handle                                handle,
    int                                             solutionIndex,
    std::vector<rocblaslt_matmul_heuristic_result>& heuristicResults,
    size_t                                          maxWorkSpaceBytes)
{
    rocblaslt_matmul_heuristic_result result;
    memset(&result, 0, sizeof(rocblaslt_matmul_heuristic_result));
    memset(result.algo.data, 0, sizeof(result.algo.data));
    int* index                      = (int*)(result.algo.data);
    *index                          = solutionIndex;
    result.algo.max_workspace_bytes = maxWorkSpaceBytes;
    result.algo.fallback            = false;
    result.state                    = rocblaslt_status_success;
    result.workspaceSize            = 0;
    heuristicResults.push_back(result);
}


/**
 * @brief Get a kernel based on the provided problem and algo.
 *
 * @param handle
 * @param prob
 * @param algo
 * @param kernel
 * @return rocblaslt_status
 */
rocblaslt_status getKernelFromAlgo(rocblaslt_handle                   handle,
                                   const RocblasltContractionProblem& prob,
                                   const rocblaslt_matmul_algo*       algo,
                                   std::shared_ptr<GemmKernel>&       kernel)
{
    int* solutionIndex = (int*)algo->data;

    if(solutionIndex == 0)
        return rocblaslt_status_not_implemented;

    RocRollerHandle* rocroller_handle = static_cast<RocRollerHandle*>(handle->rocroller_handle);
    auto             kernelType       = genKernelType(prob);

    auto existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    // If KernelType doesn't exist yet, add an empty container for it to map.
    if(existingKernelType == rocroller_handle->generatedKernels.end())
    {
        rocroller_handle->generatedKernels[kernelType] = {};
        existingKernelType = rocroller_handle->generatedKernels.find(kernelType);
    }

    auto existingKernel = existingKernelType->second.find(*solutionIndex);
    if(existingKernel != existingKernelType->second.end())
    {
        kernel = existingKernel->second;
        return rocblaslt_status_success;
    }
    else
    {
        auto solutionIndexParameter = indexToParameters(*solutionIndex);

        auto status = genKernelFromSolutionIndexParameters(
            rocroller_handle, kernelType, solutionIndexParameter, *solutionIndex, kernel);
        return status;
    }
}

rocblaslt_status isRocRollerSolutionSupported(rocblaslt_handle             handle,
                                              RocblasltContractionProblem& prob,
                                              rocblaslt_matmul_algo*       algo,
                                              size_t*                      workspaceSizeInBytes)
{
    std::shared_ptr<GemmKernel> kernel;
    auto                        status = getKernelFromAlgo(handle, prob, algo, kernel);
    if(status != rocblaslt_status_success)
        return status;

    auto workSpaceRequired = workspaceRequired(kernel, prob);

    if(workSpaceRequired > prob.workspaceSize)
        return rocblaslt_status_invalid_value;

    auto commandArgs = createCommandArguments(kernel, prob, DEFAULT_WGM);
    auto runtimeArgs = commandArgs.runtimeArguments();

    if(!kernel->commandKernel->matchesPredicates(runtimeArgs, LogLevel::Error))
    {
        return rocblaslt_status_invalid_value;
    }

    return rocblaslt_status_success;
}

/**
 * @brief Execute a contraction problem.
 *
 * This mimics the behavior of runContractionProblem in tensile_host.cpp
 *
 * If an algo has not been provided, call getRocRollerBestSolutions to find one.
 *
 * Find the kernel to run in generatedKernels and execute it.
 *
 * @param handle
 * @param algo
 * @param prob
 * @return rocblaslt_status
 */
rocblaslt_status runRocRollerContractionProblem(rocblaslt_handle                   handle,
                                                const rocblaslt_matmul_algo*       algo,
                                                const RocblasltContractionProblem& prob)
{
    rocblaslt_matmul_heuristic_result heuristicResult;
    if(algo == nullptr)
    {
        int  returnAlgoCount;
        auto status
            = getRocRollerBestSolutions(handle, prob, 1, &heuristicResult, prob.workspaceSize, &returnAlgoCount);
        if(status != rocblaslt_status_success)
            return status;
        if(returnAlgoCount == 0)
        {
            return rocblaslt_status_not_implemented;
        }
        algo = &heuristicResult.algo;
    }

    // Get the values of static member variables flush and rotating size from UserClientArguments
    UserClientArguments ClientArguments;
    bool                flush              = ClientArguments.GetFlushValue();
    int32_t             rotatingBufferSize = ClientArguments.GetRotatingBufferSizeValue();
    int32_t             hotIterations      = ClientArguments.GetHotIterationsValue();
    int32_t             coldIterations     = ClientArguments.GetColdIterationsValue();

    int* solutionIndex = (int*)algo->data;

    if((get_logger_layer_mode() & rocblaslt_layer_mode_log_bench)
       || rocblaslt::Debug::Instance().printLogAsMarker())
    {
        logBench(prob, *solutionIndex, flush, rotatingBufferSize, coldIterations, hotIterations);
    }

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_profile)
    {
        logProfile(prob, flush, rotatingBufferSize, coldIterations, hotIterations);
    }

    std::shared_ptr<GemmKernel> kernel;
    auto                        status = getKernelFromAlgo(handle, prob, algo, kernel);
    if(status != rocblaslt_status_success)
        return status;

    if(get_logger_layer_mode() & rocblaslt_layer_mode_log_extended_profile)
    {
        auto kernelName = genKernelName(kernel->params);
        logExtendedProfile(prob,
                           *solutionIndex,
                           kernelName,
                           flush,
                           rotatingBufferSize,
                           coldIterations,
                           hotIterations);
    }

    return runGemmKernel(kernel, prob);
}
