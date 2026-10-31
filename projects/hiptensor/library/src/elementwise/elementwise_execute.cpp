/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#include <algorithm>
#include <array>

#include <hiptensor/hiptensor.h>

#include "backends/ck/ck_backend.hpp"
#include "elementwise_execute.hpp"
#include "elementwise_solution.hpp"
#include "elementwise_solution_instances.hpp"
#include "elementwise_solution_registry.hpp"
#include "handle.hpp"
#include "hip_error.hpp"
#include "hiptensor_element_wise_operation.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "performance.hpp"
#include "reduction/reduction_epilogue.hpp"
#include "util.hpp"

namespace hiptensor
{
    namespace
    {
        // The operand tensors of an elementwise operation: inputs first, the output last.
        std::vector<hiptensorTensorDescriptor const*>
            elementwiseTensors(hiptensorOperationDescriptor const& desc)
        {
            switch(desc.mOperationType)
            {
            case HIPTENSOR_PERMUTATION:
                return {desc.mDescA, desc.mDescB};
            case HIPTENSOR_ELEMENTWISE_BINARY:
                return {desc.mDescA, desc.mDescC, desc.mDescD};
            case HIPTENSOR_ELEMENTWISE_TRINARY:
                return {desc.mDescA, desc.mDescB, desc.mDescC, desc.mDescD};
            default:
                return {};
            }
        }

        std::vector<std::vector<int32_t>>
            elementwiseModes(hiptensorOperationDescriptor const& desc)
        {
            switch(desc.mOperationType)
            {
            case HIPTENSOR_PERMUTATION:
                return {desc.mModeA, desc.mModeB};
            case HIPTENSOR_ELEMENTWISE_BINARY:
                return {desc.mModeA, desc.mModeC, desc.mModeD};
            case HIPTENSOR_ELEMENTWISE_TRINARY:
                return {desc.mModeA, desc.mModeB, desc.mModeC, desc.mModeD};
            default:
                return {};
            }
        }

        // The operators in the registry's query order: permutation {opA, opB}, binary
        // {opAC, opA, opC}, trinary {opABC, opAB, opA, opB, opC}.
        std::vector<hiptensorOperator_t>
            elementwiseQueryOperators(hiptensorOperationDescriptor const& desc)
        {
            switch(desc.mOperationType)
            {
            case HIPTENSOR_PERMUTATION:
                return {desc.mOpA, desc.mOpB};
            case HIPTENSOR_ELEMENTWISE_BINARY:
                return {desc.mOpAC, desc.mOpA, desc.mOpC};
            case HIPTENSOR_ELEMENTWISE_TRINARY:
                return {desc.mOpABC, desc.mOpAC, desc.mOpA, desc.mOpB, desc.mOpC};
            default:
                return {};
            }
        }
    } // namespace

    hiptensorStatus_t checkElementwiseDataTypes(hiptensorOperationDescriptor const& desc,
                                                char const*                         apiFuncName)
    {
        auto const typeScalar = desc.mScalarType;
        bool       supported  = false;

        switch(desc.mOperationType)
        {
        case HIPTENSOR_PERMUTATION:
        {
            // typeA, typeScalar
            constexpr std::array<std::array<hiptensorDataType_t, 2>, 4> validDataTypes
                = {{{HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                    {HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_64F, HIPTENSOR_R_64F}}};
            std::array<hiptensorDataType_t, 2> types = {desc.mDescA->mType, typeScalar};
            supported = std::find(validDataTypes.cbegin(), validDataTypes.cend(), types)
                        != validDataTypes.cend();
            break;
        }
        case HIPTENSOR_ELEMENTWISE_BINARY:
        {
            // typeA, typeC, typeScalar; D has C's type
            constexpr std::array<std::array<hiptensorDataType_t, 3>, 6> validDataTypes
                = {{{HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                    {HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_16BF},
                    {HIPTENSOR_R_16BF, HIPTENSOR_R_16BF, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}}};
            std::array<hiptensorDataType_t, 3> types
                = {desc.mDescA->mType, desc.mDescC->mType, typeScalar};
            supported = desc.mDescC->mType == desc.mDescD->mType
                        && std::find(validDataTypes.cbegin(), validDataTypes.cend(), types)
                               != validDataTypes.cend();
            break;
        }
        case HIPTENSOR_ELEMENTWISE_TRINARY:
        {
            // typeA, typeB, typeC, typeScalar
            constexpr std::array<std::array<hiptensorDataType_t, 4>, 3> validDataTypes
                = {{{HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F, HIPTENSOR_R_16F},
                    {HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F, HIPTENSOR_R_32F},
                    {HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F, HIPTENSOR_R_64F}}};
            std::array<hiptensorDataType_t, 4> types
                = {desc.mDescA->mType, desc.mDescB->mType, desc.mDescC->mType, typeScalar};
            supported = std::find(validDataTypes.cbegin(), validDataTypes.cend(), types)
                        != validDataTypes.cend();
            break;
        }
        default:
            break;
        }

        if(!supported)
        {
            char msg[256];
            snprintf(msg,
                     sizeof(msg),
                     "Unsupported Data Type Error : The combination of data types of the tensors "
                     "and the scalars is not supported (%s)",
                     hiptensorGetErrorString(HIPTENSOR_STATUS_NOT_SUPPORTED));
            Logger::instance()->logError(apiFuncName, msg);
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t elementwiseCheckPlan(hiptensorOperationDescriptor const& desc)
    {
        if(auto status = checkElementwiseDataTypes(desc, "hiptensorCreatePlan");
           status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        auto tensors = elementwiseTensors(desc);
        auto modes   = elementwiseModes(desc);

        std::vector<hiptensorDataType_t>  inTypes;
        std::vector<std::vector<int32_t>> inModes;
        for(std::size_t i = 0; i + 1 < tensors.size(); i++)
        {
            inTypes.push_back(tensors[i]->mType);
            inModes.push_back(modes[i]);
        }

        // Any scalar other than one selects the scaling kernels, which every execution can use.
        auto  scalars   = std::vector<double>(inTypes.size(), 2.0);
        auto& instances = ElementwiseSolutionInstances::instance();
        auto  solutions = instances->query(scalars,
                                          tensors[0]->mLengths,
                                          inTypes,
                                          {tensors.back()->mType},
                                          inModes,
                                          {modes.back()},
                                          elementwiseQueryOperators(desc),
                                          ElementwiseExecutionSpaceType_t::DEVICE);
        if(solutions.empty())
        {
            char msg[256];
            snprintf(msg,
                     sizeof(msg),
                     "No elementwise kernel for an output of rank %zu with these data types (%s)",
                     modes.back().size(),
                     hiptensorGetErrorString(HIPTENSOR_STATUS_NOT_SUPPORTED));
            Logger::instance()->logError("hiptensorCreatePlan", msg);
            return HIPTENSOR_STATUS_NOT_SUPPORTED;
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }

    hiptensorStatus_t applyZeroScales(std::vector<double> const&        scalars,
                                      std::vector<ElementwiseTensor>&   inputs,
                                      std::vector<std::size_t> const&   opIndex,
                                      std::vector<hiptensorOperator_t>& operators,
                                      ElementwiseTensor const&          output)
    {
        for(std::size_t i = 0; i < inputs.size(); i++)
        {
            if(scalars[i] != 0.0)
            {
                continue;
            }
            operators[opIndex[i]] = HIPTENSOR_OP_INTERNAL_ZERO;
            if(inputs[i].desc->mType == output.desc->mType)
            {
                inputs[i] = output;
            }
            else if(inputs[i].data == nullptr)
            {
                // The kernel still loads the input (only the operator discards the value), and
                // the output cannot stand in for an input of another type.
                Logger::instance()->logError(
                    "applyZeroScales",
                    "nullptr input with a zero scaling factor and a type other than the output's");
                return HIPTENSOR_STATUS_NOT_SUPPORTED;
            }
        }
        return HIPTENSOR_STATUS_SUCCESS;
    }

    bool elementwiseAllScalesZero(std::vector<double> const& scalars)
    {
        return !scalars.empty()
               && std::all_of(scalars.begin(), scalars.end(), [](double s) { return s == 0.0; });
    }

    hiptensorStatus_t
        elementwiseZeroOutput(hiptensorTensorDescriptor const& descD, void* D, hipStream_t stream)
    {
        auto const& lengths = descD.mLengths;
        auto const& strides = descD.mStrides;
        // The elements fill a contiguous range when, ordered by stride, every mode of extent
        // greater than one starts where the previous one ends.
        std::vector<std::size_t> order(lengths.size());
        for(std::size_t d = 0; d < order.size(); d++)
        {
            order[d] = d;
        }
        std::sort(
            order.begin(), order.end(), [&](auto a, auto b) { return strides[a] < strides[b]; });
        std::size_t expected   = 1;
        bool        contiguous = strides.size() == lengths.size();
        for(auto d : order)
        {
            if(!contiguous || lengths[d] == 1)
            {
                continue;
            }
            contiguous = strides[d] == expected;
            expected *= lengths[d];
        }
        if(contiguous)
        {
            auto const bytes = elementsFromLengths(lengths) * hiptensorDataTypeSize(descD.mType);
            if(bytes > 0)
            {
                HIPTENSOR_RETURN_IF_HIP_ERROR(hipMemsetAsync(D, 0, bytes, stream));
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
        // no input term: +0 at every element, through D's strides
        return reductionEpilogue(descD.mType,
                                 lengths,
                                 strides,
                                 D,
                                 {},
                                 nullptr,
                                 HIPTENSOR_OP_IDENTITY,
                                 0.0,
                                 {},
                                 nullptr,
                                 HIPTENSOR_OP_IDENTITY,
                                 0.0,
                                 stream);
    }

    hiptensorStatus_t runElementwise(hiptensorHandle_t                       handle,
                                     hiptensorPlan_t                         plan,
                                     AutotuneOps                             autotuneOp,
                                     char const*                             apiFuncName,
                                     std::vector<double> const&              scalars,
                                     std::vector<ElementwiseTensor> const&   inputs,
                                     ElementwiseTensor const&                output,
                                     std::vector<hiptensorOperator_t> const& operators,
                                     std::vector<hiptensorOperator_t> const& queryOperators,
                                     std::size_t                             flopsPerElement,
                                     backend::LaunchConfig const&            cfg)
    {
        auto& logger      = Logger::instance();
        auto& autotuneMgr = PlancacheAutotuneMgr::instance();
        char  msg[512];

        std::vector<std::vector<std::size_t>> inLengths, inStrides;
        std::vector<std::vector<int32_t>>     inModes;
        std::vector<hiptensorDataType_t>      inTypes;
        std::vector<void const*>              inData;
        std::size_t                           bytesPerElement
            = hiptensorDataTypeSize(output.desc->mType);
        for(auto const& in : inputs)
        {
            inLengths.push_back(in.desc->mLengths);
            inStrides.push_back(in.desc->mStrides);
            inModes.push_back(in.modes);
            inTypes.push_back(in.desc->mType);
            inData.push_back(in.data);
            bytesPerElement += hiptensorDataTypeSize(in.desc->mType);
        }

        // The plan keeps the kernel of an earlier execution, which was selected for that call's
        // scalars and operands. It is tried first; when it cannot run this call (a pass-through
        // instance and a scale other than one, or operands laid out differently after a zero
        // scale), the registry is queried. If the kernels the query picks for these scalars
        // cannot run the problem either, the scaling kernels, which apply any scalar, are tried.
        std::vector<std::vector<ElementwiseSolution*>> rounds;
        if(plan->mPref->mSolution != nullptr)
        {
            rounds.push_back({static_cast<ElementwiseSolution*>(plan->mPref->mSolution)});
        }
        auto& instances = ElementwiseSolutionInstances::instance();
        auto  query     = [&](std::vector<double> const& queryScalars) {
            return instances->query(queryScalars,
                                    inLengths[0],
                                    inTypes,
                                    {output.desc->mType},
                                    inModes,
                                    {output.modes},
                                    queryOperators,
                                    ElementwiseExecutionSpaceType_t::DEVICE);
        };
        rounds.push_back(query(scalars));
        rounds.push_back(query(std::vector<double>(scalars.size(), 2.0)));

        for(auto const& solutions : rounds)
        {
            for(auto* pSolution : solutions)
            {
                if(!pSolution->initArgs(scalars,
                                        inLengths,
                                        inStrides,
                                        inModes,
                                        {output.desc->mLengths},
                                        {output.desc->mStrides},
                                        {output.modes},
                                        operators,
                                        inData,
                                        {const_cast<void*>(output.data)}))
                {
                    continue;
                }

                float time = 0.0f;
                if(logger->getLogMask() & HIPTENSOR_LOG_LEVEL_PERF_TRACE)
                {
                    time = (*pSolution)(ckbackend::toStreamConfig(ckbackend::timedLaunch(cfg)));
                    if(time < 0)
                    {
                        return HIPTENSOR_STATUS_CK_ERROR;
                    }

                    auto flops = flopsPerElement * pSolution->problemSize();
                    auto bytes = bytesPerElement * pSolution->problemSize();

                    PerfMetrics metrics = {
                        pSolution->uid(), // id
                        pSolution->kernelName(), // name
                        time, // avg time
                        static_cast<float>(flops) / static_cast<float>(1.E9) / time, // tflops
                        static_cast<float>(bytes) / static_cast<float>(1.E6) / time // BW
                    };

                    snprintf(msg,
                             sizeof(msg),
                             "KernelId: %zu KernelName: %s, %0.3f ms, %0.3f TFlops/s, %0.3f GB/s",
                             metrics.mKernelUid,
                             metrics.mKernelName.c_str(),
                             metrics.mAvgTimeMs,
                             metrics.mTflops,
                             metrics.mBandwidth);
                    logger->logPerformanceTrace(apiFuncName, msg);
                }
                else
                {
                    time = (*pSolution)(ckbackend::toStreamConfig(cfg));
                    if(time < 0)
                    {
                        return HIPTENSOR_STATUS_CK_ERROR;
                    }
                }

                plan->mPref->mSolution = pSolution;
                autotuneMgr->saveAutotune<ElementwiseSolution>(autotuneOp, time, handle, plan);
                return HIPTENSOR_STATUS_SUCCESS;
            }
        }

        snprintf(msg,
                 sizeof(msg),
                 "Selected kernel is unable to solve the problem (%s)",
                 hiptensorGetErrorString(HIPTENSOR_STATUS_INTERNAL_ERROR));
        logger->logError(apiFuncName, msg);
        return HIPTENSOR_STATUS_INTERNAL_ERROR;
    }
} // namespace hiptensor
