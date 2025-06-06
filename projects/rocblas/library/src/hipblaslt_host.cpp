/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************/

// The implementation of the rocBLAS<->HipBlasLT interface layer.

#include "rocblas.h"

extern "C" void rocblas_shutdown();

/*****************************************************************************
 * This is the only file in rocBLAS which should #include Hipblaslt headers    *
 * or reference Hipblaslt identifiers. hipblaslt_host.hpp defines the interface. *
 *****************************************************************************/

#include <functional>
#include <iostream>

#include "hipblaslt_host.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

namespace
{
    /********************************************************************
     * Variable template to map a rocBLAS type into a hipblasltDatatype_t *
     ********************************************************************/
    template <typename>
    constexpr auto hipblaslt_datatype = HIPBLASLT_DATATYPE_INVALID;

    template <>
    constexpr auto hipblaslt_datatype<int8_t> = HIP_R_8I;

    template <>
    constexpr auto hipblaslt_datatype<int32_t> = HIP_R_32I;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_half> = HIP_R_16F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_bfloat16> = HIP_R_16BF;

    template <>
    constexpr auto hipblaslt_datatype<float> = HIP_R_32F;

    template <>
    constexpr auto hipblaslt_datatype<double> = HIP_R_64F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_float_complex> = HIP_C_32F;

    template <>
    constexpr auto hipblaslt_datatype<rocblas_double_complex> = HIP_C_64F;

    /********************************************************************
     * Variable template to map a rocBLAS type into a hipblasLtComputeType_t *
     ********************************************************************/
    template <typename>
    constexpr auto hipblaslt_compute_type = HIPBLAS_COMPUTE_32F;

    template <>
    constexpr auto hipblaslt_compute_type<int32_t> = HIPBLAS_COMPUTE_32I;

    template <>
    constexpr auto hipblaslt_compute_type<rocblas_half> = HIPBLAS_COMPUTE_16F;

    template <>
    constexpr auto hipblaslt_compute_type<float> = HIPBLAS_COMPUTE_32F;

    template <>
    constexpr auto hipblaslt_compute_type<double> = HIPBLAS_COMPUTE_64F;

    /**************************************************************************
    * We normally print error messages only once, to avoid excessive logging *
    **************************************************************************/
    void print_if_verbose(const rocblas_internal_ostream& msg)
    {
        if(rocblas_suppress_tensile_error_messages())
            return;
        static constexpr char varname[] = "ROCBLAS_VERBOSE_HIPBLASLT_ERROR";
        static const char*    verbose   = getenv(varname);
        if(verbose)
        {
            rocblas_cerr << std::endl << msg << std::endl;
        }
    }

    /****************************************************************
     * Construct a HipBlasLT GEMM from a RocblasContractionProblem *
     ****************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructHipBlasLTGemm(const RocblasContractionProblem<Ti, To, Tc>& prob)
    {
        hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

        hipblaslt_ext::Gemm gemm(handle,
                                 (hipblasOperation_t)prob.trans_a,
                                 (hipblasOperation_t)prob.trans_b,
                                 hipblaslt_datatype<Ti>,
                                 hipblaslt_datatype<Ti>,
                                 hipblaslt_datatype<To>,
                                 hipblaslt_datatype<To>,
                                 hipblaslt_compute_type<Tc>);

        hipblaslt_ext::GemmProblemType problemType;
        problemType.setOpA((hipblasOperation_t)prob.trans_a);
        problemType.setOpB((hipblasOperation_t)prob.trans_b);
        problemType.setTypeA(hipblaslt_datatype<Ti>);
        problemType.setTypeB(hipblaslt_datatype<Ti>);
        problemType.setTypeC(hipblaslt_datatype<To>);
        problemType.setTypeD(hipblaslt_datatype<To>);
        problemType.setTypeCompute(hipblaslt_compute_type<Tc>);

        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   inputs;
        inputs.setA((void*)(prob.A + prob.buffer_offset_a));
        inputs.setB((void*)(prob.B + prob.buffer_offset_b));
        inputs.setC((void*)(prob.C + prob.buffer_offset_c));
        inputs.setD((void*)(prob.D + prob.buffer_offset_d));
        inputs.setAlpha((void*)prob.alpha);
        inputs.setBeta((void*)prob.beta);

        gemm.setProblem(prob.m,
                        prob.n,
                        prob.k,
                        prob.batch_count,
                        prob.col_stride_a,
                        prob.col_stride_b,
                        prob.col_stride_c,
                        prob.col_stride_d,
                        prob.batch_stride_a,
                        prob.batch_stride_b,
                        prob.batch_stride_c,
                        prob.batch_stride_d,
                        epilogue,
                        inputs,
                        problemType);
        return gemm;
    }

    /****************************************************************
     * Construct a HipBlasLT Groupped GEMM from a RocblasContractionProblem *
     ****************************************************************/
    template <typename Ti, typename To, typename Tc>
    auto ConstructHipBlasLTGroupedGemm(const RocblasContractionProblem<Ti, To, Tc>& prob)
    {
        hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

        hipblaslt_ext::GroupedGemm gemm(handle,
                                        (hipblasOperation_t)prob.trans_a,
                                        (hipblasOperation_t)prob.trans_b,
                                        hipblaslt_datatype<Ti>,
                                        hipblaslt_datatype<Ti>,
                                        hipblaslt_datatype<To>,
                                        hipblaslt_datatype<To>,
                                        hipblaslt_compute_type<Tc>);

        try
        {
            hipblaslt_ext::GemmProblemType problemType;
            problemType.setOpA((hipblasOperation_t)prob.trans_a);
            problemType.setOpB((hipblasOperation_t)prob.trans_b);
            problemType.setTypeA(hipblaslt_datatype<Ti>);
            problemType.setTypeB(hipblaslt_datatype<Ti>);
            problemType.setTypeC(hipblaslt_datatype<To>);
            problemType.setTypeD(hipblaslt_datatype<To>);
            problemType.setTypeCompute(hipblaslt_compute_type<Tc>);

            std::vector<int64_t>                     Ms(prob.batch_count);
            std::vector<int64_t>                     Ns(prob.batch_count);
            std::vector<int64_t>                     Ks(prob.batch_count);
            std::vector<int64_t>                     ldas(prob.batch_count);
            std::vector<int64_t>                     ldbs(prob.batch_count);
            std::vector<int64_t>                     ldcs(prob.batch_count);
            std::vector<int64_t>                     ldds(prob.batch_count);
            std::vector<int64_t>                     strideas(prob.batch_count);
            std::vector<int64_t>                     stridebs(prob.batch_count);
            std::vector<int64_t>                     stridecs(prob.batch_count);
            std::vector<int64_t>                     strideds(prob.batch_count);
            std::vector<int64_t>                     batch_counts(prob.batch_count);
            std::vector<hipblaslt_ext::GemmEpilogue> epilogues(prob.batch_count);
            std::vector<hipblaslt_ext::GemmInputs>   inputs(prob.batch_count);

            int batch_count = prob.batch_count;

            std::vector<Ti const*> A(batch_count, nullptr);
            std::vector<Ti const*> B(batch_count, nullptr);
            std::vector<To*>       C(batch_count, nullptr);
            std::vector<To*>       D(batch_count, nullptr);
            if(prob.batch_A)
            {
                THROW_IF_HIP_ERROR(hipMemcpy((void*)(&A[0]),
                                             prob.batch_A,
                                             sizeof(void*) * batch_count,
                                             hipMemcpyDeviceToHost));
            }
            if(prob.batch_B)
            {
                THROW_IF_HIP_ERROR(hipMemcpy((void*)(&B[0]),
                                             prob.batch_B,
                                             sizeof(void*) * batch_count,
                                             hipMemcpyDeviceToHost));
            }
            if(prob.batch_C)
            {
                THROW_IF_HIP_ERROR(hipMemcpy((void*)(&C[0]),
                                             prob.batch_C,
                                             sizeof(void*) * batch_count,
                                             hipMemcpyDeviceToHost));
            }
            if(prob.batch_D)
            {
                THROW_IF_HIP_ERROR(hipMemcpy((void*)(&D[0]),
                                             prob.batch_D,
                                             sizeof(void*) * batch_count,
                                             hipMemcpyDeviceToHost));
            }

            for(int batch = 0; batch < batch_count; batch++)
            {
                Ms[batch]           = prob.m;
                Ns[batch]           = prob.n;
                Ks[batch]           = prob.k;
                ldas[batch]         = prob.col_stride_a;
                ldbs[batch]         = prob.col_stride_b;
                ldcs[batch]         = prob.col_stride_c;
                ldds[batch]         = prob.col_stride_d;
                strideas[batch]     = prob.batch_stride_a;
                stridebs[batch]     = prob.batch_stride_b;
                stridecs[batch]     = prob.batch_stride_c;
                strideds[batch]     = prob.batch_stride_d;
                batch_counts[batch] = 1;
                inputs[batch].setA((void*)(A[batch] + prob.buffer_offset_a));
                inputs[batch].setB((void*)(B[batch] + prob.buffer_offset_b));
                inputs[batch].setC((void*)(C[batch] + prob.buffer_offset_c));
                inputs[batch].setD((void*)(D[batch] + prob.buffer_offset_d));
                inputs[batch].setAlpha((void*)prob.alpha);
                inputs[batch].setBeta((void*)prob.beta);
            }

            gemm.setProblem(Ms,
                            Ns,
                            Ks,
                            batch_counts,
                            ldas,
                            ldbs,
                            ldcs,
                            ldds,
                            strideas,
                            stridebs,
                            stridecs,
                            strideds,
                            epilogues,
                            inputs,
                            problemType);
        }
        catch(...)
        {
            rocblas_internal_ostream msg;
            print_if_verbose(
                msg << "rocBLAS warning: hipBlasLT grouped gemm construction exception");
        }
        return gemm;
    }

    /*
     * Combine common initialization functionality for GEMM and Ggrouped GEMM.
    */
    auto hipBlasLTInit(hipblaslt_ext::GemmInstance&      gemm,
                       rocblas_gemm_algo                 algo,
                       int32_t                           solution_index,
                       bool                              solution_query,
                       const rocblas_handle              probHandle,
                       size_t&                           workspace_size,
                       hipblasLtMatmulHeuristicResult_t& heuristicResult,
                       size_t                            extra_malloc = 0)
    {
        hipblaslt_ext::GemmPreference gemmPref;
        auto                          max_workspace_size = probHandle->get_available_workspace();
        gemmPref.setMaxWorkspaceBytes(max_workspace_size - extra_malloc);

        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;

        bool query_failure = false;

        if(algo == rocblas_gemm_algo_solution_index && solution_index > 0)
        {
            hipblasLtHandle_t& handle = *(probHandle->getHipblasLtHandle());
            // indx - 1 maps to zero based hipblasLt indices
            std::vector<int> solution_index_vec(1, solution_index - 1);
            if(hipblaslt_ext::getAlgosFromIndex(handle, solution_index_vec, heuristicResults)
               != HIPBLAS_STATUS_SUCCESS)
            {
                if(!solution_query)
                {
                    rocblas_internal_ostream msg;
                    print_if_verbose(
                        msg << "rocBLAS warning: hipBLASLt cannot find specified solution index!");
                    return rocblas_status_invalid_value;
                }
                else
                    query_failure = true;
            }

            if(heuristicResults.empty())
            {
                if(!solution_query)
                {
                    rocblas_internal_ostream msg;
                    print_if_verbose(msg << "rocBLAS warning: No hipBLASLt solution found");
                    return rocblas_status_invalid_value;
                }
                else
                    query_failure = true;
            }
        }
        else
        {
            const int request_solutions = 1;
            if(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResults)
               != HIPBLAS_STATUS_SUCCESS)
            {
                if(!solution_query)
                {
                    rocblas_internal_ostream msg;
                    print_if_verbose(msg
                                     << "rocBLAS error: hipBLASLt default heuristic fetch failed!");
                    return rocblas_status_internal_error;
                }
                else
                    query_failure = true;
            }

            if(heuristicResults.empty())
            {
                if(!solution_query)
                {
                    rocblas_internal_ostream msg;
                    print_if_verbose(msg << "rocBLAS warning: No hipBLASLt default solution found");
                    return rocblas_status_not_implemented;
                }
                else
                    query_failure = true;
            }
        }

        if(!query_failure)
        {
            heuristicResult = heuristicResults[0];

            if(algo == rocblas_gemm_algo_solution_index && solution_index > 0)
            {
                workspace_size = 0;
                if(gemm.isAlgoSupported(heuristicResult.algo, workspace_size)
                   != HIPBLAS_STATUS_SUCCESS)
                {
                    if(!solution_query)
                    {
                        rocblas_internal_ostream msg;
                        print_if_verbose(msg << "rocBLAS warning: hipBLASLt algo not supported.");
                        return rocblas_status_invalid_value;
                    }
                    else
                        query_failure = true;
                }
            }
            else
            {
                workspace_size = heuristicResult.workspaceSize;
            }

            workspace_size += extra_malloc;

            if(workspace_size > max_workspace_size)
            {
                if(!solution_query)
                {
                    rocblas_internal_ostream msg;
                    print_if_verbose(msg << "rocBLAS warning: hipBLASLt algo not supported: "
                                            "insufficient workspace.");
                    return rocblas_status_invalid_value;
                }
                else
                    query_failure = true;
            }
        }

        if(query_failure)
            return rocblas_status_invalid_value;
        else
            return rocblas_status_success;
    }

} // namespace

/******************************************************************************
 * runContractionProblemHipBlasLT calls Hipblaslt to run a contraction problem described *
 * by RocblasContractionProblem                                               *
 ******************************************************************************/
template <typename Ti, typename To, typename Tc>
rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<Ti, To, Tc>& prob,
                                              rocblas_gemm_algo                            algo,
                                              int32_t solution_index)
{
    bool solution_query = algo == rocblas_gemm_algo_solution_index
                          && prob.flags & rocblas_gemm_flags_check_solution_index;

    if(prob.batch_A == 0)
    {
        auto gemm = ConstructHipBlasLTGemm(prob);

        size_t                           workspace_size = 0;
        hipblasLtMatmulHeuristicResult_t heuristicResult;

        auto init = hipBlasLTInit(gemm,
                                  algo,
                                  solution_index,
                                  solution_query,
                                  prob.handle,
                                  workspace_size,
                                  heuristicResult);
        if(solution_query)
        {
            return init; // early return either success or invalid value
        }
        else if(rocblas_status_success != init)
        {
            return init;
        }

        auto gsu_malloc = prob.handle->gsu_malloc_by_size(workspace_size);

        if(!gsu_malloc)
        {
            return rocblas_status_memory_error;
        }

        void* d_workspace = prob.handle->gsu_workspace;

        if(gemm.initialize(heuristicResult.algo, d_workspace, false) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_if_verbose(msg << "rocBLAS error: hipBLASLt initialization failed!");
            return rocblas_status_internal_error;
        }
        if(gemm.run(prob.handle->get_stream()) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_if_verbose(msg << "rocBLAS warning: hipBLASLt execution failed!");
            return rocblas_status_internal_error;
        }
    }
    else
    {
        auto gemm         = ConstructHipBlasLTGroupedGemm(prob);
        auto userArgsSize = prob.batch_count * sizeof(hipblaslt_ext::UserArguments);

        size_t                           workspace_size = 0;
        hipblasLtMatmulHeuristicResult_t heuristicResult;

        auto init = hipBlasLTInit(gemm,
                                  algo,
                                  solution_index,
                                  solution_query,
                                  prob.handle,
                                  workspace_size,
                                  heuristicResult,
                                  userArgsSize);
        if(solution_query)
        {
            return init; // early return either success or invalid value
        }
        else if(rocblas_status_success != init)
        {
            return init;
        }

        auto gsu_malloc = prob.handle->gsu_malloc_by_size(workspace_size);

        if(!gsu_malloc)
        {
            return rocblas_status_memory_error;
        }
        void* d_workspace = prob.handle->gsu_workspace;

        if(gemm.initialize(heuristicResult.algo, d_workspace, false) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_if_verbose(msg << "rocBLAS error: hipBLASLt initialization failed!");
            return rocblas_status_internal_error;
        }

        hipblaslt_ext::UserArguments* userArgs;
        hipHostMalloc(&userArgs, userArgsSize);
        gemm.getDefaultValueForDeviceUserArguments(userArgs);

        // Copy them to device memory
        hipblaslt_ext::UserArguments* d_userArgs
            = (hipblaslt_ext::UserArguments*)((char*)(prob.handle->gsu_workspace)
                                              + (workspace_size - userArgsSize));
        hipMemcpy(d_userArgs, userArgs, userArgsSize, hipMemcpyHostToDevice);
        hipFree(userArgs);

        if(gemm.run(d_userArgs, prob.handle->get_stream()) != HIPBLAS_STATUS_SUCCESS)
        {
            rocblas_internal_ostream msg;
            print_if_verbose(msg << "rocBLAS warning: hipBLASLt execution failed!");
            return rocblas_status_internal_error;
        }
    }
    return rocblas_status_success;
}

template <typename Ti, typename To, typename Tc>
rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<Ti, To, Tc>& prob,
                                        rocblas_tensile_get_solution_option          option,
                                        rocblas_int*                                 list_array,
                                        rocblas_int*                                 list_size)
{

    constexpr bool is_complex = rocblas_is_complex<Ti> || rocblas_is_complex<Tc>;
    rocblas_status status     = rocblas_status_success;
    int            added_sols = 0;

    if(is_complex)
    {
        // TODO: revisit with any hipblaslt support changes, or with query of hipblaslt for support
        if(list_array == nullptr)
        {
            *list_size = 0;
        }
    }
    else
    {
        hipblasLtHandle_t& handle = *(prob.handle->getHipblasLtHandle());

        auto map_hipblaslt_to_rocblas_index = [](auto h) {
            // -1 getIndexFromAlgo if index is less than zero
            rocblas_int idx = hipblaslt_ext::getIndexFromAlgo(h.algo);
            if(idx < 0)
                return c_rocblas_bad_solution_index; // flag with bad index
            return idx + 1; // convert to one based index
        };

        if(option == MATCHES_TYPE)
        {
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
            std::vector<hipblasOperation_t> ops = {HIPBLAS_OP_N, HIPBLAS_OP_T, HIPBLAS_OP_C};
            hipblaslt_ext::GemmType         gemmType
                = prob.batch_A == 0 ? hipblaslt_ext::GemmType::HIPBLASLT_GEMM
                                    : hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM;
            for(auto op1 : ops)
            {
                for(auto op2 : ops)
                {
                    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults_temp;
                    auto fetch = hipblaslt_ext::getAllAlgos(handle,
                                                            gemmType,
                                                            op1,
                                                            op2,
                                                            hipblaslt_datatype<Ti>,
                                                            hipblaslt_datatype<Ti>,
                                                            hipblaslt_datatype<To>,
                                                            hipblaslt_datatype<To>,
                                                            hipblaslt_compute_type<Tc>,
                                                            heuristicResults_temp);

                    heuristicResults.insert(heuristicResults.end(),
                                            heuristicResults_temp.begin(),
                                            heuristicResults_temp.end());
                }
            }

            // Convert to indexes and remove duplicates.
            std::vector<rocblas_int> heuristicIndexes(heuristicResults.size());
            std::transform(heuristicResults.begin(),
                           heuristicResults.end(),
                           heuristicIndexes.begin(),
                           map_hipblaslt_to_rocblas_index);
            std::sort(heuristicIndexes.begin(), heuristicIndexes.end());
            auto itr = std::unique(heuristicIndexes.begin(), heuristicIndexes.end());
            heuristicIndexes.resize(std::distance(heuristicIndexes.begin(), itr));
            if(!heuristicIndexes.empty() && heuristicIndexes.back() == c_rocblas_bad_solution_index)
            {
                heuristicIndexes.pop_back();
            }

            if(list_array == nullptr)
            {
                *list_size = heuristicIndexes.size();
            }
            else
            {
                auto it = heuristicIndexes.begin();
                while(added_sols < *list_size && it != heuristicIndexes.end())
                {
                    list_array[added_sols++] = *it;
                    ++it;
                }
                int i = added_sols;
                while(i < *list_size)
                {
                    list_array[i++] = c_rocblas_default_solution;
                }
            }
        }
        else if(option == CAN_SOLVE)
        {
            hipblaslt_ext::GemmType gemmType
                = prob.batch_A == 0 ? hipblaslt_ext::GemmType::HIPBLASLT_GEMM
                                    : hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM;
            std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
            auto                                          fetch = hipblaslt_ext::getAllAlgos(handle,
                                                    gemmType,
                                                    (hipblasOperation_t)prob.trans_a,
                                                    (hipblasOperation_t)prob.trans_b,
                                                    hipblaslt_datatype<Ti>,
                                                    hipblaslt_datatype<Ti>,
                                                    hipblaslt_datatype<To>,
                                                    hipblaslt_datatype<To>,
                                                    hipblaslt_compute_type<Tc>,
                                                    heuristicResults);

            std::shared_ptr<hipblaslt_ext::GemmInstance> gemm;

            if(prob.batch_A == 0)
            {
                gemm = std::make_shared<hipblaslt_ext::GemmInstance>(ConstructHipBlasLTGemm(prob));
            }
            else
            {
                gemm = std::make_shared<hipblaslt_ext::GemmInstance>(
                    ConstructHipBlasLTGroupedGemm(prob));
            }

            size_t retSize   = heuristicResults.size();
            size_t iter_size = list_array == nullptr ? retSize : *list_size;

            auto   it = heuristicResults.begin();
            size_t tmpWorkspaceSize;
            while(added_sols < iter_size && it != heuristicResults.end())
            {
                if(gemm->isAlgoSupported(it->algo, tmpWorkspaceSize) == HIPBLAS_STATUS_SUCCESS)
                {
                    if(list_array != nullptr)
                    {
                        int solution_index = map_hipblaslt_to_rocblas_index(*it);
                        if(solution_index != c_rocblas_bad_solution_index)
                            list_array[added_sols++] = solution_index;
                        else
                            --retSize;
                    }
                }
                else
                {
                    --retSize;
                }
                ++it;
            }

            if(list_array == nullptr)
            {
                *list_size = retSize;
            }
            else
            {
                int i = added_sols;
                while(i < *list_size)
                {
                    list_array[i++] = c_rocblas_default_solution;
                }
            }
        }
        else
        {
            return rocblas_status_invalid_value;
        }
    }

    // inject rocblas source-code gemv if applicable
    rocblas_status rocblasSolStatus
        = getRocblasSolutions(prob, option, list_array, list_size, added_sols);

    if(rocblasSolStatus != rocblas_status_continue)
        return rocblasSolStatus;

    return status;
}

/******************************************************************************
 * Intantiate the cases of runContractionProblemHipBlasLT which are needed to satisfy  *
 * rocBLAS dependencies. This file's template functions are not defined in a  *
 * header file, in order to keep hipBLASLt and rocBLAS separate.                *
 ******************************************************************************/

// Non-HPA/GEMM types
template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_half>&, rocblas_gemm_algo algo, int32_t solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<float>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(const RocblasContractionProblem<double>&,
                                                       rocblas_gemm_algo algo,
                                                       int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_float_complex>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_double_complex>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

// HPA types
template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_half, rocblas_half, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_half, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status runContractionProblemHipBlasLT(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&,
    rocblas_gemm_algo algo,
    int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<rocblas_bfloat16, float, float>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

template rocblas_status
    runContractionProblemHipBlasLT(const RocblasContractionProblem<int8_t, int32_t, int32_t>&,
                                   rocblas_gemm_algo algo,
                                   int32_t           solution_index);

// ********** get all solutions explicits ********
// Non-HPA/GEMM types
template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<float>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(const RocblasContractionProblem<double>&,
                                                 rocblas_tensile_get_solution_option option,
                                                 rocblas_int*                        list_array,
                                                 rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_float_complex>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_double_complex>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

// HPA types
template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half, rocblas_half, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_half, float, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status getAllSolutionsHipBlasLT(
    const RocblasContractionProblem<rocblas_bfloat16, rocblas_bfloat16, float>&,
    rocblas_tensile_get_solution_option option,
    rocblas_int*                        list_array,
    rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<rocblas_bfloat16, float, float>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);

template rocblas_status
    getAllSolutionsHipBlasLT(const RocblasContractionProblem<int8_t, int32_t, int32_t>&,
                             rocblas_tensile_get_solution_option option,
                             rocblas_int*                        list_array,
                             rocblas_int*                        list_size);
