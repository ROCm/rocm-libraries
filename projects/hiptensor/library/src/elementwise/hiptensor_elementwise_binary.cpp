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
#include "hip_error.hpp"
#include "logger.hpp"

#include "plancache_autotune.hpp"

hiptensorStatus_t hiptensorElementwiseBinaryExecute(const hiptensorHandle_t handle,
                                                    const hiptensorPlan_t   plan,
                                                    const void*             alpha,
                                                    const void*             A,
                                                    const void*             gamma,
                                                    const void*             C,
                                                    void*                   D,
                                                    hipStream_t             stream)
try
{
    hiptensor::backend::LaunchConfig cfg;
    cfg.stream = stream;

    // Plans run on the backend selected when they were created.
    if(handle == nullptr || plan == nullptr || plan->mBackend == nullptr)
    {
        return hiptensor::ckbackend::elementwiseBinary(handle, plan, alpha, A, gamma, C, D, cfg);
    }

    hiptensor::backend::Scalars scalars;
    scalars.alpha = alpha;
    scalars.gamma = gamma;

    hiptensor::backend::Buffers buffers;
    buffers.A   = A;
    buffers.C   = C;
    buffers.out = D;

    return hiptensor::backend::executePlan(handle, plan, scalars, buffers, nullptr, 0, cfg);
}
catch(...)
{
    return hiptensor::exceptionToStatus("hiptensorElementwiseBinaryExecute");
}

hiptensorStatus_t hiptensor::ckbackend::elementwiseBinary(hiptensorHandle_t            handle,
                                                          hiptensorPlan_t              plan,
                                                          void const*                  alpha,
                                                          void const*                  A,
                                                          void const*                  gamma,
                                                          void const*                  C,
                                                          void*                        D,
                                                          backend::LaunchConfig const& cfg)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, handle);
    CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, plan);
    if(checkResult != HIPTENSOR_STATUS_SUCCESS)
    {
        return checkResult;
    }

    hiptensorOperationDescriptor_t opDes = plan->mOpDesc;
    return hiptensor::elementwiseBinaryExecute(handle,
                                               plan,
                                               opDes->mDescA,
                                               opDes->mModeA,
                                               opDes->mOpA,
                                               opDes->mDescC,
                                               opDes->mModeC,
                                               opDes->mOpC,
                                               opDes->mDescD,
                                               opDes->mModeD,
                                               opDes->mOpAC,
                                               alpha,
                                               A,
                                               gamma,
                                               C,
                                               D,
                                               cfg,
                                               "hiptensorElementwiseBinaryExecute");
}

namespace hiptensor
{
    hiptensorStatus_t elementwiseBinaryExecute(hiptensorHandle_t                handle,
                                               hiptensorPlan_t                  plan,
                                               hiptensorTensorDescriptor const* descA,
                                               std::vector<int32_t> const&      modeA,
                                               hiptensorOperator_t              opA,
                                               hiptensorTensorDescriptor const* descC,
                                               std::vector<int32_t> const&      modeC,
                                               hiptensorOperator_t              opC,
                                               hiptensorTensorDescriptor const* descD,
                                               std::vector<int32_t> const&      modeD,
                                               hiptensorOperator_t              opAC,
                                               void const*                      alpha,
                                               void const*                      A,
                                               void const*                      gamma,
                                               void const*                      C,
                                               void*                            D,
                                               backend::LaunchConfig const&     cfg,
                                               char const*                      apiFuncName)
    {
        hipStream_t const stream = cfg.stream;

        auto& logger = Logger::instance();

        auto& autotuneMgr = PlancacheAutotuneMgr::instance();
        autotuneMgr->startAutotune(AutotuneOps::Autotune_BinaryOp);

        const hiptensorDataType_t typeScalar = plan->mOpDesc->mScalarType;

        // Log API access
        char msg[2048];
        snprintf(msg,
                 sizeof(msg),
                 "handle=%p, alpha=%p, A=%p, descA=%p, modeA=%p, "
                 "gamma=%p, C=%p, descC=%p, modeC=%p, "
                 "D=%p, descD=%p, modeD=%p, "
                 "opAC=0x%02X, typeScalar=0x%02X, stream=%p",
                 handle,
                 alpha,
                 A,
                 descA,
                 modeA.data(),
                 gamma,
                 C,
                 descC,
                 modeC.data(),
                 D,
                 descD,
                 modeD.data(),
                 opAC,
                 (unsigned int)typeScalar,
                 stream);

        logger->logAPITrace(apiFuncName, msg);

        hiptensorStatus_t checkResult = HIPTENSOR_STATUS_SUCCESS;
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, alpha);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descA);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, gamma);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descC);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, D);
        CheckApiParams(checkResult, *logger, HIPTENSOR_STATUS_NOT_INITIALIZED, descD);
        if(checkResult != HIPTENSOR_STATUS_SUCCESS)
        {
            return checkResult;
        }

        if(auto status = checkElementwiseDataTypes(*plan->mOpDesc, apiFuncName);
           status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        auto const computeType = convertToComputeType(typeScalar);
        auto const scalars     = std::vector<double>{readVal<double>(alpha, computeType),
                                                     readVal<double>(gamma, computeType)};

        // An input is only read when its scaling factor is not zero (see applyZeroScales), so
        // it may be nullptr otherwise.
        bool const readsA = scalars[0] != 0.0;
        bool const readsC = scalars[1] != 0.0;
        if((readsA && A == nullptr) || (readsC && C == nullptr))
        {
            snprintf(msg,
                     sizeof(msg),
                     "%s = nullptr with a non-zero scaling factor",
                     (readsA && A == nullptr) ? "A" : "C");
            logger->logError(apiFuncName, msg);
            return HIPTENSOR_STATUS_NOT_INITIALIZED;
        }

        // alpha == gamma == 0: D = +0 without reading an input or running a kernel
        if(elementwiseAllScalesZero(scalars))
        {
            return elementwiseZeroOutput(*descD, D, stream);
        }

        auto output    = ElementwiseTensor{descD, modeD, D};
        auto inputs    = std::vector<ElementwiseTensor>{{descA, modeA, A}, {descC, modeC, C}};
        auto operators = std::vector<hiptensorOperator_t>{opAC, opA, opC};
        if(auto status = applyZeroScales(scalars, inputs, {1, 2}, operators, output);
           status != HIPTENSOR_STATUS_SUCCESS)
        {
            return status;
        }

        autotuneMgr->setAutotune<ElementwiseSolution>(AutotuneOps::Autotune_BinaryOp, handle, plan);

        return runElementwise(handle,
                              plan,
                              AutotuneOps::Autotune_BinaryOp,
                              apiFuncName,
                              scalars,
                              inputs,
                              output,
                              operators,
                              operators,
                              5,
                              cfg);
    }
} // namespace hiptensor
