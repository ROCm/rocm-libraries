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

#include "backend/registry.hpp"
#include "backends/ck/ck_backend.hpp"
#include "data_types.hpp"
#include "elementwise_execute.hpp"
#include "elementwise_solution.hpp"
#include "elementwise_solution_instances.hpp"
#include "elementwise_solution_registry.hpp"
#include "hip_error.hpp"
#include "logger.hpp"

#include "hiptensor_options.hpp"
#include "plancache_autotune.hpp"

hiptensorStatus_t hiptensorPermute(const hiptensorHandle_t handle,
                                   const hiptensorPlan_t   plan,
                                   const void*             alpha,
                                   const void*             A,
                                   void*                   B,
                                   const hipStream_t       stream)
try
{
    hiptensor::backend::LaunchConfig cfg;
    cfg.stream = stream;

    // Plans run on the backend selected when they were created.
    if(handle == nullptr || plan == nullptr || plan->mBackend == nullptr)
    {
        return hiptensor::ckbackend::permute(handle, plan, alpha, A, B, cfg);
    }

    hiptensor::backend::Scalars scalars;
    scalars.alpha = alpha;

    hiptensor::backend::Buffers buffers;
    buffers.A   = A;
    buffers.out = B;

    return hiptensor::backend::executePlan(handle, plan, scalars, buffers, nullptr, 0, cfg);
}
catch(...)
{
    return hiptensor::exceptionToStatus("hiptensorPermute");
}

hiptensorStatus_t hiptensor::ckbackend::permute(hiptensorHandle_t            handle,
                                                hiptensorPlan_t              plan,
                                                void const*                  alpha,
                                                void const*                  A,
                                                void*                        B,
                                                backend::LaunchConfig const& cfg)
{
    hipStream_t const stream = cfg.stream;

    using hiptensor::Logger;
    auto& logger = Logger::instance();

    using hiptensor::PlancacheAutotuneMgr;
    auto& autotuneMgr = PlancacheAutotuneMgr::instance();
    autotuneMgr->startAutotune(hiptensor::AutotuneOps::Autotune_Permutation);

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
    const hiptensorDataType_t         typeScalar = opDes->mScalarType;

    // Log API access
    char msg[2048];
    snprintf(msg,
             sizeof(msg),
             "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, B=%p, descB=%p, modeB=%p, "
             "typeScalar=0x%02X, stream=%p",
             handle,
             alpha,
             A,
             descA,
             opDes->mModeA.data(),
             B,
             descB,
             opDes->mModeB.data(),
             (unsigned int)typeScalar,
             stream);

    logger->logAPITrace("hiptensorPermute", msg);

    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, B);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descB);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    if(auto status = hiptensor::checkElementwiseDataTypes(*opDes, "hiptensorPermute");
       status != HIPTENSOR_STATUS_SUCCESS)
    {
        return status;
    }

    auto const scalars = std::vector<double>{
        hiptensor::readVal<double>(alpha, hiptensor::convertToComputeType(typeScalar))};

    // A is only read when alpha is not zero (see applyZeroScales), so it may be nullptr
    // otherwise.
    if(scalars[0] != 0.0 && A == nullptr)
    {
        logger->logError("hiptensorPermute", "A = nullptr with a non-zero scaling factor");
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    auto output    = hiptensor::ElementwiseTensor{descB, opDes->mModeB, B};
    auto inputs    = std::vector<hiptensor::ElementwiseTensor>{{descA, opDes->mModeA, A}};
    auto operators = std::vector<hiptensorOperator_t>{opDes->mOpA};
    if(auto status = hiptensor::applyZeroScales(scalars, inputs, {0}, operators, output);
       status != HIPTENSOR_STATUS_SUCCESS)
    {
        return status;
    }

    autotuneMgr->setAutotune<hiptensor::ElementwiseSolution>(
        hiptensor::AutotuneOps::Autotune_Permutation, handle, plan);

    // The plan keeps the kernel of an earlier execution, which was selected for that call's
    // alpha. runElementwise selects again when it cannot run this call, for example a
    // pass-through instance picked for alpha == 1 that cannot scale.
    return hiptensor::runElementwise(handle,
                                     plan,
                                     hiptensor::AutotuneOps::Autotune_Permutation,
                                     "hiptensorPermute",
                                     scalars,
                                     inputs,
                                     output,
                                     operators,
                                     {operators[0], opDes->mOpB},
                                     2,
                                     cfg);
}
