/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>

#include <hiptensor/hiptensor.hpp>

#include "data_types.hpp"
#include "handle.hpp"
#include "hiptensor_options.hpp"
#include "logger.hpp"
#include "util.hpp"

hiptensorStatus_t hiptensorCreate(hiptensorHandle_t** handle)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(
        msg, sizeof(msg), "handle=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)handle);
    logger->logAPITrace("hiptensorCreate", msg);

    (*handle) = new hiptensorHandle_t;

    if(*handle == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_ALLOC_FAILED;
        snprintf(msg,
                 sizeof(msg),
                 "Initialization Error : handle = nullptr (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_ALLOC_FAILED;
    }

    auto hip_status = hipInit(0);

    if(hip_status == hipErrorInvalidDevice)
    {
        auto errorCode = HIPTENSOR_STATUS_HIP_ERROR;
        snprintf(msg,
                 sizeof(msg),
                 "Initialization error: invalid device (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_HIP_ERROR;
    }

    else if(hip_status == hipErrorInvalidValue)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(
            msg, sizeof(msg), "Initialization error: (%s)", hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorCreate", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    // Get the current device (handled by the Handle class)
    auto realHandle = hiptensor::Handle::createHandle((*handle)->fields);

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorDestroy(hiptensorHandle_t* handle)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(
        msg, sizeof(msg), "handle=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)handle);
    logger->logAPITrace("hiptensorDestroy", msg);

    hiptensor::Handle::destroyHandle(handle->fields);

    delete handle;
    handle = nullptr;

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorInitTensorDescriptor(const hiptensorHandle_t*     handle,
                                                hiptensorTensorDescriptor_t* desc,
                                                const uint32_t               numModes,
                                                const int64_t                lens[],
                                                const int64_t                strides[],
                                                hipDataType                  dataType,
                                                hiptensorOperator_t          unaryOp)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[256];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, desc=0x%llX, numModes=0x%02X, lens=0x%llX, strides=0x%llX,"
             "dataType=0x%02X, unaryOp=0x%02X",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)desc,
             (unsigned int)numModes,
             (unsigned long long)lens,
             (unsigned long long)strides,
             (unsigned int)dataType,
             (unsigned int)unaryOp);
    logger->logAPITrace("hiptensorInitTensorDescriptor", msg);

    if(handle == nullptr || desc == nullptr)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(handle == nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Initialization Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorInitTensorDescriptor", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    if((lens == nullptr && strides != nullptr)
       || ((dataType != HIP_R_16F) && (dataType != HIP_R_16BF) && (dataType != HIP_R_32F)
           && (dataType != HIP_R_64F) && (dataType != HIP_C_32F) && (dataType != HIP_C_64F))
       || ((unaryOp == HIPTENSOR_OP_ADD) || (unaryOp == HIPTENSOR_OP_MUL)
           || (unaryOp == HIPTENSOR_OP_MIN) || (unaryOp == HIPTENSOR_OP_MAX)
           || (unaryOp == HIPTENSOR_OP_UNKNOWN)))
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        if(lens == nullptr && strides != nullptr)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Tensor Initialization Error : lens = nullptr and strides != nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else if((unaryOp == HIPTENSOR_OP_ADD) || (unaryOp == HIPTENSOR_OP_MUL)
                || (unaryOp == HIPTENSOR_OP_MIN) || (unaryOp == HIPTENSOR_OP_MAX)
                || (unaryOp == HIPTENSOR_OP_UNKNOWN))
        {
            snprintf(msg,
                     sizeof(msg),
                     "Tensor Initialization Error : op is not a valid unary operator (%s) ",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Tensor Initialization Error : datatype should be float or double (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorInitTensorDescriptor", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    auto realHandle = hiptensor::Handle::toHandle((int64_t*)handle->fields);
    if(dataType == HIP_R_64F && !realHandle->getDevice().supportsF64())
    {
        return HIPTENSOR_STATUS_ARCH_MISMATCH;
    }

    if(strides)
    {
        // Construct with both given lengths and strides
        *desc = {dataType,
                 std::vector<std::size_t>(lens, lens + numModes),
                 std::vector<std::size_t>(strides, strides + numModes),
                 unaryOp};
    }
    else
    {
        // Re-construct strides from lengths, assuming packed.
        if(numModes > 0)
        {
            using hiptensor::HiptensorOptions;
            auto& options = HiptensorOptions::instance();

            auto                     lensVector = std::vector<std::size_t>(lens, lens + numModes);
            std::vector<std::size_t> stridesVector
                = hiptensor::stridesFromLengths(lensVector, options->isColMajorStrides());
            *desc = {dataType, lensVector, stridesVector, unaryOp};
        }
        else
        {
            *desc = {dataType, std::vector<std::size_t>(), std::vector<std::size_t>(), unaryOp};
        }
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

const char* hiptensorGetErrorString(const hiptensorStatus_t error)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg, sizeof(msg), "error=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)error);
    logger->logAPITrace("hiptensorGetErrorString", msg);

    if(error == HIPTENSOR_STATUS_SUCCESS)
        return "HIPTENSOR_STATUS_SUCCESS";
    else if(error == HIPTENSOR_STATUS_NOT_INITIALIZED)
        return "HIPTENSOR_STATUS_NOT_INITIALIZED";
    else if(error == HIPTENSOR_STATUS_ALLOC_FAILED)
        return "HIPTENSOR_STATUS_ALLOC_FAILED";
    else if(error == HIPTENSOR_STATUS_INVALID_VALUE)
        return "HIPTENSOR_STATUS_INVALID_VALUE";
    else if(error == HIPTENSOR_STATUS_ARCH_MISMATCH)
        return "HIPTENSOR_STATUS_ARCH_MISMATCH";
    else if(error == HIPTENSOR_STATUS_EXECUTION_FAILED)
        return "HIPTENSOR_STATUS_EXECUTION_FAILED";
    else if(error == HIPTENSOR_STATUS_INTERNAL_ERROR)
        return "HIPTENSOR_STATUS_INTERNAL_ERROR";
    else if(error == HIPTENSOR_STATUS_NOT_SUPPORTED)
        return "HIPTENSOR_STATUS_NOT_SUPPORTED";
    else if(error == HIPTENSOR_STATUS_CK_ERROR)
        return "HIPTENSOR_STATUS_CK_ERROR";
    else if(error == HIPTENSOR_STATUS_HIP_ERROR)
        return "HIPTENSOR_STATUS_HIP_ERROR";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE)
        return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    else if(error == HIPTENSOR_STATUS_INSUFFICIENT_DRIVER)
        return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
    else if(error == HIPTENSOR_STATUS_IO_ERROR)
        return "HIPTENSOR_STATUS_IO_ERROR";
    else
        return "HIPTENSOR_STATUS_UNKNOWN";
}

hiptensorStatus_t hiptensorGetAlignmentRequirement(const hiptensorHandle_t*           handle,
                                                   const void*                        ptr,
                                                   const hiptensorTensorDescriptor_t* desc,
                                                   uint32_t* alignmentRequirement)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[256];
    snprintf(msg,
             sizeof(msg),
             "handle=0x%0*llX, ptr=0x%llX, desc=0x%llX, alignmentRequirement=0x%02X",
             2 * (int)sizeof(void*),
             (unsigned long long)handle,
             (unsigned long long)ptr,
             (unsigned long long)desc,
             (unsigned int)*alignmentRequirement);

    logger->logAPITrace("hiptensorGetAlignmentRequirement", msg);

    if(!handle || !desc)
    {
        auto errorCode = HIPTENSOR_STATUS_NOT_INITIALIZED;
        if(!handle)
        {
            snprintf(msg,
                     sizeof(msg),
                     "Error : handle = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        else
        {
            snprintf(msg,
                     sizeof(msg),
                     "Error : contraction descriptor = nullptr (%s)",
                     hiptensorGetErrorString(errorCode));
        }
        logger->logError("hiptensorGetAlignmentRequirement", msg);
        return HIPTENSOR_STATUS_NOT_INITIALIZED;
    }

    *alignmentRequirement = 0u;
    for(auto i = hiptensor::hipDataTypeSize(desc->mType); i <= 16u; i *= 2)
    {
        if((std::size_t)ptr % (std::size_t)i == 0)
        {
            *alignmentRequirement = i;
        }
    }

    if(*alignmentRequirement == 0)
    {
        auto errorCode = HIPTENSOR_STATUS_INVALID_VALUE;
        snprintf(msg,
                 sizeof(msg),
                 "Error : alignment requirement is 0 (%s)",
                 hiptensorGetErrorString(errorCode));
        logger->logError("hiptensorGetAlignmentRequirement", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }
    else
    {
        return HIPTENSOR_STATUS_SUCCESS;
    }
}

hiptensorStatus_t hiptensorLoggerSetCallback(hiptensorLoggerCallback_t callback)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg,
             sizeof(msg),
             "callback=0x%0*llX",
             2 * (int)sizeof(void*),
             (unsigned long long)callback);
    logger->logAPITrace("hiptensorLoggerSetCallback", msg);

    // Check logger callback result
    auto loggerResult = logger->setCallback(callback);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Error : logger set callback not successful (%s)",
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetCallback", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetFile(FILE* file)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API access
    char msg[128];
    snprintf(msg, sizeof(msg), "file=0x%0*llX", 2 * (int)sizeof(void*), (unsigned long long)file);
    logger->logAPITrace("hiptensorLoggerSetFile", msg);

    // Check logger callback result
    auto loggerResult = logger->writeToStream(file);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "Error : logger set file not successful (%s)",
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetFile", msg);
        return HIPTENSOR_STATUS_IO_ERROR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerOpenFile(const char* logFile)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[2048];
    snprintf(msg, sizeof(msg), "logFile=%s", logFile);
    logger->logAPITrace("hiptensorLoggerOpenFile", msg);

    // Check logger open file result
    auto loggerResult = logger->openFileStream(logFile);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg, sizeof(msg), "fileName=%s (%s)", logFile, logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerOpenFile", msg);
        return HIPTENSOR_STATUS_IO_ERROR;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetLevel(hiptensorLogLevel_t level)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[128];
    snprintf(msg, sizeof(msg), "log level=0x%02X", (unsigned int)level);
    logger->logAPITrace("hiptensorLoggerSetLevel", msg);

    // Check logger level
    auto loggerResult = logger->setLogLevel(Logger::LogLevel_t(level));
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "level=0x%02X (%s)",
                 (unsigned int)level,
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetLevel", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerSetMask(int32_t mask)
{
    using hiptensor::Logger;
    auto& logger = Logger::instance();

    // Log API trace
    char msg[128];
    snprintf(msg, sizeof(msg), "mask=0x%02X", (unsigned int)mask);
    logger->logAPITrace("hiptensorLoggerSetMask", msg);

    // Check for logger error
    auto loggerResult = logger->setLogMask(mask);
    if(loggerResult != Logger::Status_t::SUCCESS)
    {
        snprintf(msg,
                 sizeof(msg),
                 "mask=0x%02X (%s)",
                 (unsigned int)mask,
                 logger->statusString(loggerResult));
        logger->logError("hiptensorLoggerSetMask", msg);
        return HIPTENSOR_STATUS_INVALID_VALUE;
    }

    return HIPTENSOR_STATUS_SUCCESS;
}

hiptensorStatus_t hiptensorLoggerForceDisable()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorLoggerForceDisable", "Logging Disabled");
    logger->disable();
    return HIPTENSOR_STATUS_SUCCESS;
}

int hiptensorGetHiprtVersion()
{
    // Log API trace
    auto& logger = hiptensor::Logger::instance();
    logger->logAPITrace("hiptensorGetHiprtVersion", "");

    int  version   = 0;
    auto hipResult = hipRuntimeGetVersion(&version);
    if(hipResult != hipError_t::hipSuccess)
    {
        char msg[256];
        snprintf(msg, sizeof(msg), "Hip error: (%s)", hipGetErrorString(hipResult));
        logger->logError("hiptensorGetHiprtVersion", msg);
        return -1;
    }

    return version;
}
