// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*********************************************************
 * The implementation of the rocblaslt<->rocRoller interface layer. *
 *********************************************************/

#include "rocroller_host.hpp"
#include "custom_kernels.hpp"
#include "gemm.hpp"
#include "kernel_type.hpp"
#include "parameter_selection.hpp"
#include "rocroller_interop.hpp"
#include "runtime_args_selection.hpp"
#include "solution_cache.hpp"
#include "solution_selection.hpp"

#include "Debug.hpp"
#include "handle.h"
#include "utility.hpp"

using namespace rocRoller;

std::string rocRollerShortKernelNameFromEncodedSolutionIndex(int encodedSolutionIndex)
{
    return shortRocRollerKernelNameFromSolutionIndex(indexToParameters(encodedSolutionIndex));
}

/**
 * @brief RocRollerHandle
 *
 * State that is needed for executing rocRoller kernels
 *
 */
struct RocRollerHandle
{
    SolutionCache cache;
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
    auto rr_handle = new RocRollerHandle();
    *handle        = rr_handle;

    if(getenv("HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL") == nullptr)
    {
        // If these kernels are loaded, they are used instead of RocRoller kernels
        preloadCustomKernels(rr_handle->cache);
    }
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
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0:
        return "3";
    case RocblasltContractionProblem::ScalingFormat::Block_32_UE8M0_32_8_EXT:
        return "1001";
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
        kernel = RocRollerGemmKernel::generate(params);
        rocroller_handle->cache.addKernel(kernelType, solutionIndexParameter, kernel);
    }
    catch(const std::exception& e)
    {
        std::stringstream msg;
        msg << "Error building the following kernel:" << std::endl;
        msg << params->toString() << std::endl;
        msg << e.what() << std::endl;
        log_info(__func__, msg.str());
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
    *returnAlgoCount = 0;

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

    auto scale_type = normalizeOperandType(prob.scale_type);
    if(scale_type != HIPBLASLT_DATATYPE_INVALID && scale_type != HIP_R_32F)
    {
        std::cerr << "rocRoller only supports F32 as scale type not "
                  << hipblaslt::hip2RocRollerType(scale_type) << std::endl;
        return rocblaslt_status_invalid_value;
    }

    if(kernelType.typeAcc != HIP_R_32F)
    {
        std::cerr << "rocRoller only supports F32 accumulation, not "
                  << hipblaslt::hip2RocRollerType(kernelType.typeAcc) << std::endl;
        return rocblaslt_status_invalid_value;
    }

    auto solutionIndexParameters
        = chooseSolutionIndexParameters(kernelType, prob, requestedAlgoCount);

    int i = 0;
    for(auto const& solutionIndexParameter : solutionIndexParameters)
    {
        if(requestedAlgoCount != -1 && i >= requestedAlgoCount)
            break;

        index = parametersToIndex(solutionIndexParameter);

        auto existingSolution = rocroller_handle->cache.getKernel(
            kernelType, solutionIndexParameter, ProblemDims{prob.m, prob.n, prob.k});
        std::shared_ptr<GemmKernel> kernel;
        // If kernel doesn't already exist, generate it
        if(!existingSolution)
        {
            // Validate problem dimensions match kernel tile requirements before generating kernel
            // to avoid rocRoller expression evaluation with invalid dimensions
            if(solutionIndexParameter.workgroupTile.m > 0 && solutionIndexParameter.workgroupTile.n > 0
                && solutionIndexParameter.workgroupTile.k > 0)
            {
                if(prob.m % solutionIndexParameter.workgroupTile.m != 0
                    || prob.n % solutionIndexParameter.workgroupTile.n != 0
                    || prob.k % solutionIndexParameter.workgroupTile.k != 0)
                {
                    continue;  // Skip this solution entirely
                }
            }

            auto status = genKernelFromSolutionIndexParameters(
                rocroller_handle, kernelType, solutionIndexParameter, index, kernel);
            if(status != rocblaslt_status_success)
                continue;
        }
        else
        {
            kernel = *existingSolution;
        }

        if (!kernel->isSupportedProblem(prob))
        {
            continue;
        }

        // Fill out heuristicResultsArray
        // The most important thing to do is set the solutionIndex
        memset(heuristicResultsArray[i].algo.data, 0, sizeof(heuristicResultsArray[i].algo.data));
        int* solutionIndex = (int*)(heuristicResultsArray[i].algo.data);
        *solutionIndex     = index;
        heuristicResultsArray[i].algo.max_workspace_bytes = maxWorkSpaceBytes;
        heuristicResultsArray[i].algo.fallback            = false;
        heuristicResultsArray[i].state                    = rocblaslt_status_success;
        heuristicResultsArray[i].workspaceSize            = kernel->workspaceRequired(prob);
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
    auto result = getRocRollerBestSolutions(
        handle, prob, -1, heuristicResults.data(), maxWorkSpaceBytes, &returnAlgoCount);
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

    auto solutionIndexParameters = indexToParameters(*solutionIndex);
    auto existingKernel          = rocroller_handle->cache.getKernel(
        kernelType, solutionIndexParameters, ProblemDims{prob.m, prob.n, prob.k});
    if(existingKernel)
    {
        kernel = *existingKernel;
        return rocblaslt_status_success;
    }
    else
    {
        auto status = genKernelFromSolutionIndexParameters(
            rocroller_handle, kernelType, solutionIndexParameters, *solutionIndex, kernel);
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

    if(!kernel->isSupportedProblem(prob))
    {
        return rocblaslt_status_invalid_value;
    }

    *workspaceSizeInBytes = kernel->workspaceRequired(prob);
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
        auto status = getRocRollerBestSolutions(
            handle, prob, 1, &heuristicResult, prob.workspaceSize, &returnAlgoCount);
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

    return kernel->run(prob);
}
