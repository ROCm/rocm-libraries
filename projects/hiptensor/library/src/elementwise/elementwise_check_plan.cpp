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
#include <vector>

#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "elementwise_solution.hpp"
#include "elementwise_solution_instances.hpp"
#include "elementwise_solution_registry.hpp"
#include "elementwise_type_support.hpp"
#include "logger.hpp"

// Checks, when an elementwise plan (permutation, binary, trinary, or a reduction that keeps the
// rank) is created, that a kernel instance can run the operation, so that an unsupported problem
// is reported by hiptensorCreatePlan rather than when the plan is executed. Data types are checked
// with the same rule as the execute functions, and the registry is queried for the operation's
// output rank and data types.
hiptensorStatus_t elementwiseCheckPlan(const hiptensorOperationDescriptor_t desc)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    std::vector<hiptensorTensorDescriptor_t> inputs;
    std::vector<std::vector<int32_t>>        inModes;
    hiptensorTensorDescriptor_t              output = nullptr;
    std::vector<int32_t>                     outModes;
    // The operators in the order the execute functions pass them to the registry query.
    std::vector<hiptensorOperator_t> operators;
    bool                             typesSupported = false;

    switch(desc->mOperationType)
    {
    case HIPTENSOR_PERMUTATION:
        inputs         = {desc->mDescA};
        inModes        = {desc->mModeA};
        output         = desc->mDescB;
        outModes       = desc->mModeB;
        operators      = {desc->mOpA, desc->mOpB};
        typesSupported = hiptensor::isSupportedElementwiseTypeCombination(
            {desc->mDescA->mType, desc->mDescB->mType}, desc->mScalarType);
        break;
    case HIPTENSOR_ELEMENTWISE_BINARY:
        inputs         = {desc->mDescA, desc->mDescC};
        inModes        = {desc->mModeA, desc->mModeC};
        output         = desc->mDescD;
        outModes       = desc->mModeD;
        operators      = {desc->mOpAC, desc->mOpA, desc->mOpC};
        typesSupported = hiptensor::isSupportedElementwiseTypeCombination(
            {desc->mDescA->mType, desc->mDescC->mType, desc->mDescD->mType}, desc->mScalarType);
        break;
    case HIPTENSOR_ELEMENTWISE_TRINARY:
        inputs         = {desc->mDescA, desc->mDescB, desc->mDescC};
        inModes        = {desc->mModeA, desc->mModeB, desc->mModeC};
        output         = desc->mDescD;
        outModes       = desc->mModeD;
        operators      = {desc->mOpABC, desc->mOpAC, desc->mOpA, desc->mOpB, desc->mOpC};
        typesSupported = hiptensor::isSupportedElementwiseTypeCombination(
            {desc->mDescA->mType, desc->mDescB->mType, desc->mDescC->mType, desc->mDescD->mType},
            desc->mScalarType);
        break;
    case HIPTENSOR_REDUCTION:
        // Only a reduction that keeps the rank runs on the elementwise kernels: hiptensorReduce
        // executes it as the elementwise binary operation D = alpha * A + beta * C, with the
        // operation hiptensorCreatePlan sets to HIPTENSOR_OP_ADD.
        if(desc->mDescA->mLengths.size() != desc->mDescD->mLengths.size())
        {
            return HIPTENSOR_STATUS_SUCCESS;
        }
        inputs         = {desc->mDescA, desc->mDescC};
        inModes        = {desc->mModeA, desc->mModeC};
        output         = desc->mDescD;
        outModes       = desc->mModeD;
        operators      = {HIPTENSOR_OP_ADD, desc->mOpA, desc->mOpC};
        typesSupported = hiptensor::isSupportedElementwiseTypeCombination(
            {desc->mDescA->mType, desc->mDescC->mType, desc->mDescD->mType}, desc->mScalarType);
        break;
    default:
        return HIPTENSOR_STATUS_SUCCESS;
    }

    char msg[512];
    if(!typesSupported)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "Unsupported Data Type Error : The combination of data types of the tensors "
                 "and the scalars is not supported. "
                 "See the link for details "
                 "https://rocm.docs.amd.com/projects/hipTensor/en/latest/api-reference/"
                 "api-reference.html "
                 "(%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreatePlan", msg);
        return errorCode;
    }

    std::vector<hiptensorDataType_t> inTypes;
    for(auto const* input : inputs)
    {
        inTypes.push_back(input->mType);
    }

    // Scalars other than one select the scaling kernels, which can run every execution of the
    // plan whatever scalars it is given.
    // This relies on the pass-through (noop) instance set mirroring the scaling set.
    auto& instances = hiptensor::ElementwiseSolutionInstances::instance();
    auto  solutions = instances->query(std::vector<double>(inputs.size(), 2.0),
                                      inputs[0]->mLengths,
                                      inTypes,
                                       {output->mType},
                                      inModes,
                                       {outModes},
                                      operators,
                                      hiptensor::ElementwiseExecutionSpaceType_t::DEVICE);
    if(solutions.empty())
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_SUPPORTED;
        snprintf(msg,
                 sizeof(msg),
                 "No elementwise kernel supports an output of rank %zu with these data types (%s)",
                 outModes.size(),
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreatePlan", msg);
        return errorCode;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}
