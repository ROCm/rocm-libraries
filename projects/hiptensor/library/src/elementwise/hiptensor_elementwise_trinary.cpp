/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hiptensor/hiptensor.h>

#include "data_types.hpp"
#include "elementwise_execute.hpp"
#include "elementwise_solution.hpp"
#include "hip_error.hpp"
#include "logger.hpp"

#include "plancache_autotune.hpp"

hiptensorStatus_t hiptensorElementwiseTrinaryExecute(const hiptensorHandle_t handle,
                                                     const hiptensorPlan_t   plan,
                                                     const void*             alpha,
                                                     const void*             A,
                                                     const void*             beta,
                                                     const void*             B,
                                                     const void*             gamma,
                                                     const void*             C,
                                                     void*                   D,
                                                     hipStream_t             stream)
try
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_TrinaryOp);

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    hiptensorOperationDescriptor_t    opDes      = plan->mOpDesc;
    const hiptensorTensorDescriptor_t descA      = opDes->mDescA;
    const hiptensorTensorDescriptor_t descB      = opDes->mDescB;
    const hiptensorTensorDescriptor_t descC      = opDes->mDescC;
    const hiptensorTensorDescriptor_t descD      = opDes->mDescD;
    hiptensorOperator_t               opAB       = opDes->mOpAC;
    hiptensorOperator_t               opABC      = opDes->mOpABC;
    const hiptensorDataType_t         typeScalar = opDes->mScalarType;

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, "
             "beta=%p, B=%p, descB=%p, modeB=%p, "
             "gamma=%p, C=%p, descC=%p, modeC=%p, "
             "D=%p, descD=%p, modeD=%p, "
             "opAB=0x%02X, opABC=0x%02X, typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             opDes->mModeA.data(),
             beta,
             B,
             descB,
             opDes->mModeB.data(),
             gamma,
             C,
             descC,
             opDes->mModeC.data(),
             D,
             descD,
             opDes->mModeD.data(),
             opAB,
             opABC,
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorElementwiseTrinaryExecute", msg);

    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, beta);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, gamma);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, D);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    if(auto status
       = hiptensor::checkElementwiseDataTypes(*opDes, "hiptensorElementwiseTrinaryExecute");
       status != HIPTENSOR_STATUS_SUCCESS)
    {
        return status;
    }

    auto const computeType = hiptensor::convertToComputeType(typeScalar);
    auto const scalars     = std::vector<double>{hiptensor::readVal<double>(alpha, computeType),
                                                 hiptensor::readVal<double>(beta, computeType),
                                                 hiptensor::readVal<double>(gamma, computeType)};

    // An input is only read when its scaling factor is not zero (see applyZeroScales), so it
    // may be nullptr otherwise.
    char const* const names[] = {"A", "B", "C"};
    void const* const data[]  = {A, B, C};
    for(int i = 0; i < 3; i++)
    {
        if(scalars[i] != 0.0 && data[i] == nullptr)
        {
            snprintf(msg, sizeof(msg), "%s = nullptr with a non-zero scaling factor", names[i]);
            logger->logError("hiptensorElementwiseTrinaryExecute", msg);
            return HIPTENSOR_STATUS_NOT_INITIALIZED;
        }
    }

    auto output    = hiptensor::ElementwiseTensor{descD, opDes->mModeD, D};
    auto inputs    = std::vector<hiptensor::ElementwiseTensor>{
        {descA, opDes->mModeA, A}, {descB, opDes->mModeB, B}, {descC, opDes->mModeC, C}};
    auto operators = std::vector<hiptensorOperator_t>{
        opABC, opAB, opDes->mOpA, opDes->mOpB, opDes->mOpC};
    if(auto status = hiptensor::applyZeroScales(scalars, inputs, {2, 3, 4}, operators, output);
       status != HIPTENSOR_STATUS_SUCCESS)
    {
        return status;
    }

    autotuneMgr->setAutotune<hiptensor::ElementwiseSolution>(
        hiptensor::AutotuneOps::Autotune_TrinaryOp, handle, plan);

    return hiptensor::runElementwise(handle,
                                     plan,
                                     hiptensor::AutotuneOps::Autotune_TrinaryOp,
                                     "hiptensorElementwiseTrinaryExecute",
                                     scalars,
                                     inputs,
                                     output,
                                     operators,
                                     operators,
                                     8,
                                     stream);
}
catch(...)
{
    return hiptensor::exceptionToStatus("hiptensorElementwiseTrinaryExecute");
}
