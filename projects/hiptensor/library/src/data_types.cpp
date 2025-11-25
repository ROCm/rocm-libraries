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

#include "data_types.hpp"

namespace hiptensor
{
    // Get data size in bytes from id
    uint32_t hiptensorDataTypeSize(hiptensorDataType_t id)
    {
        switch(id)
        {
        case HIPTENSOR_R_16BF:
            return sizeof(hip_bfloat16);
        case HIPTENSOR_R_16F:
            return sizeof(_Float16);
        case HIPTENSOR_R_32F:
            return sizeof(float);
        case HIPTENSOR_R_64F:
            return sizeof(double);
        case HIPTENSOR_R_8I:
            return sizeof(int8_t);
        case HIPTENSOR_R_8U:
            return sizeof(uint8_t);
        case HIPTENSOR_R_16I:
            return sizeof(int16_t);
        case HIPTENSOR_R_16U:
            return sizeof(uint16_t);
        case HIPTENSOR_R_32I:
            return sizeof(int32_t);
        case HIPTENSOR_R_32U:
            return sizeof(uint32_t);
        case HIPTENSOR_R_64I:
            return sizeof(int64_t);
        case HIPTENSOR_R_64U:
            return sizeof(uint64_t);
        case HIPTENSOR_C_32F:
            return sizeof(hipFloatComplex);
        case HIPTENSOR_C_64F:
            return sizeof(hipDoubleComplex);
        case NONE_TYPE:
            return 0;
        default:
        {
#if !NDEBUG
            std::cout << "Unhandled hip datatype: " << id << std::endl;
#endif // !NDEBUG
            return 0;
        }
        }
    }

    hiptensorComputeDescriptor_t convertToComputeType(hiptensorDataType_t hipType)
    {
        switch(hipType)
        {
        case HIPTENSOR_R_16BF:
            return HIPTENSOR_COMPUTE_DESC_16BF;
        case HIPTENSOR_R_16F:
            return HIPTENSOR_COMPUTE_DESC_16F;
        case HIPTENSOR_R_32F:
            return HIPTENSOR_COMPUTE_DESC_32F;
        case HIPTENSOR_R_64F:
            return HIPTENSOR_COMPUTE_DESC_64F;
        case HIPTENSOR_R_8I:
            return HIPTENSOR_COMPUTE_DESC_8I;
        case HIPTENSOR_R_8U:
            return HIPTENSOR_COMPUTE_DESC_8U;
        case HIPTENSOR_R_32I:
            return HIPTENSOR_COMPUTE_DESC_32I;
        case HIPTENSOR_R_32U:
            return HIPTENSOR_COMPUTE_DESC_32U;
        case HIPTENSOR_C_32F:
            return HIPTENSOR_COMPUTE_DESC_C32F;
        case HIPTENSOR_C_64F:
            return HIPTENSOR_COMPUTE_DESC_C64F;
        default:
            return HIPTENSOR_COMPUTE_DESC_NONE;
        }
    }

    std::optional<hiptensorDataType_t>
        convertToHipTensorDataType(hiptensorComputeDescriptor_t computeType)
    {
        switch(computeType)
        {
        case HIPTENSOR_COMPUTE_DESC_16BF:
            return HIPTENSOR_R_16BF;
        case HIPTENSOR_COMPUTE_DESC_16F:
            return HIPTENSOR_R_16F;
        case HIPTENSOR_COMPUTE_DESC_32F:
            return HIPTENSOR_R_32F;
        case HIPTENSOR_COMPUTE_DESC_64F:
            return HIPTENSOR_R_64F;
        case HIPTENSOR_COMPUTE_DESC_C32F:
            return HIPTENSOR_C_32F;
        case HIPTENSOR_COMPUTE_DESC_C64F:
            return HIPTENSOR_C_64F;
        default:
            return {}; // There is no invalid hiptensorDataType_t value
        }
    }

    // @cond
    template <>
    ScalarData readVal(void const* value, hiptensorComputeDescriptor_t id)
    {
        switch(id)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
        {
            return ScalarData(id, *(_Float16*)value);
        }
        case HIPTENSOR_COMPUTE_DESC_16BF:
        {
            return ScalarData(id, *(hip_bfloat16*)value);
        }
        case HIPTENSOR_COMPUTE_DESC_32F:
        {
            return ScalarData(id, *(float*)value);
        }
        case HIPTENSOR_COMPUTE_DESC_64F:
        {
            return ScalarData(id, *(double*)value);
        }
        case HIPTENSOR_COMPUTE_DESC_C32F:
        {
            auto complex = *(hipFloatComplex*)value;
            return {id, complex.x, complex.y};
        }
        case HIPTENSOR_COMPUTE_DESC_C64F:
        {
            auto complex = *(hipDoubleComplex*)value;
            return {id, complex.x, complex.y};
        }
        default:
        {
#if !NDEBUG
            std::cout << "Unhandled hiptensorComputeDescriptor_t: " << id << std::endl;
#endif // !NDEBUG
            return {HIPTENSOR_COMPUTE_DESC_NONE, 0, 0};
        }
        }
    }
    // @endcond

    void writeVal(void const* addr, hiptensorComputeDescriptor_t id, ScalarData value)
    {
        switch(id)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
        {
            *(_Float16*)addr = value.mReal;
            return;
        }
        case HIPTENSOR_COMPUTE_DESC_16BF:
        {
            *(hip_bfloat16*)addr = value.mReal;
            return;
        }
        case HIPTENSOR_COMPUTE_DESC_32F:
        {
            *(float*)addr = value.mReal;
            return;
        }
        case HIPTENSOR_COMPUTE_DESC_64F:
        {
            *(double*)addr = value.mReal;
            return;
        }
        case HIPTENSOR_COMPUTE_DESC_C32F:
        {
            *(hipFloatComplex*)addr = hipComplexDoubleToFloat(value.mComplex);
            return;
        }
        case HIPTENSOR_COMPUTE_DESC_C64F:
        {
            *(hipDoubleComplex*)addr = value.mComplex;
            return;
        }
        default:
        {
#if !NDEBUG
            std::cout << "Unhandled hiptensorComputeDescriptor_t: " << id << std::endl;
#endif // !NDEBUG
            return;
        }
        }
    }

    std::string computeTypeToString(hiptensorComputeDescriptor_t computeType)
    {
        switch(computeType)
        {
        case HIPTENSOR_COMPUTE_DESC_16BF:
            return "HIPTENSOR_COMPUTE_DESC_16BF";
        case HIPTENSOR_COMPUTE_DESC_16F:
            return "HIPTENSOR_COMPUTE_DESC_16F";
        case HIPTENSOR_COMPUTE_DESC_32F:
            return "HIPTENSOR_COMPUTE_DESC_32F";
        case HIPTENSOR_COMPUTE_DESC_64F:
            return "HIPTENSOR_COMPUTE_DESC_64F";
        case HIPTENSOR_COMPUTE_DESC_C32F:
            return "HIPTENSOR_COMPUTE_DESC_C32F";
        case HIPTENSOR_COMPUTE_DESC_C64F:
            return "HIPTENSOR_COMPUTE_DESC_C64F";
        default:
            return "HIPTENSOR_COMPUTE_DESC_NONE";
        }
    }

    std::string hipTypeToString(hiptensorDataType_t hipType)
    {
        switch(hipType)
        {
        case HIPTENSOR_R_16BF:
            return "HIPTENSOR_R_16BF";
        case HIPTENSOR_R_16F:
            return "HIPTENSOR_R_16F";
        case HIPTENSOR_R_32F:
            return "HIPTENSOR_R_32F";
        case HIPTENSOR_R_64F:
            return "HIPTENSOR_R_64F";
        case HIPTENSOR_R_8I:
            return "HIPTENSOR_R_8I";
        case HIPTENSOR_R_8U:
            return "HIPTENSOR_R_8U";
        case HIPTENSOR_R_32I:
            return "HIPTENSOR_R_32I";
        case HIPTENSOR_R_32U:
            return "HIPTENSOR_R_32U";
        case HIPTENSOR_C_32F:
            return "HIPTENSOR_C_32F";
        case HIPTENSOR_C_64F:
            return "HIPTENSOR_C_64F";
        default:
            return "HIP_TYPE_NONE";
        }
    }

    std::string opTypeToString(hiptensorOperator_t opType)
    {
        switch(opType)
        {
        case HIPTENSOR_OP_IDENTITY:
            return "HIPTENSOR_OP_IDENTITY";
        case HIPTENSOR_OP_SQRT:
            return "HIPTENSOR_OP_SQRT";
        case HIPTENSOR_OP_RELU:
            return "HIPTENSOR_OP_RELU";
        case HIPTENSOR_OP_CONJ:
            return "HIPTENSOR_OP_CONJ";
        case HIPTENSOR_OP_RCP:
            return "HIPTENSOR_OP_RCP";
        case HIPTENSOR_OP_SIGMOID:
            return "HIPTENSOR_OP_SIGMOID";
        case HIPTENSOR_OP_TANH:
            return "HIPTENSOR_OP_TANH";
        case HIPTENSOR_OP_EXP:
            return "HIPTENSOR_OP_EXP";
        case HIPTENSOR_OP_LOG:
            return "HIPTENSOR_OP_LOG";
        case HIPTENSOR_OP_ABS:
            return "HIPTENSOR_OP_ABS";
        case HIPTENSOR_OP_NEG:
            return "HIPTENSOR_OP_NEG";
        case HIPTENSOR_OP_SIN:
            return "HIPTENSOR_OP_SIN";
        case HIPTENSOR_OP_COS:
            return "HIPTENSOR_OP_COS";
        case HIPTENSOR_OP_TAN:
            return "HIPTENSOR_OP_TAN";
        case HIPTENSOR_OP_SINH:
            return "HIPTENSOR_OP_SINH";
        case HIPTENSOR_OP_COSH:
            return "HIPTENSOR_OP_COSH";
        case HIPTENSOR_OP_ASIN:
            return "HIPTENSOR_OP_ASIN";
        case HIPTENSOR_OP_ACOS:
            return "HIPTENSOR_OP_ACOS";
        case HIPTENSOR_OP_ATAN:
            return "HIPTENSOR_OP_ATAN";
        case HIPTENSOR_OP_ASINH:
            return "HIPTENSOR_OP_ASINH";
        case HIPTENSOR_OP_ACOSH:
            return "HIPTENSOR_OP_ACOSH";
        case HIPTENSOR_OP_ATANH:
            return "HIPTENSOR_OP_ATANH";
        case HIPTENSOR_OP_CEIL:
            return "HIPTENSOR_OP_CEIL";
        case HIPTENSOR_OP_FLOOR:
            return "HIPTENSOR_OP_FLOOR";
        case HIPTENSOR_OP_ADD:
            return "HIPTENSOR_OP_ADD";
        case HIPTENSOR_OP_MUL:
            return "HIPTENSOR_OP_MUL";
        case HIPTENSOR_OP_MAX:
            return "HIPTENSOR_OP_MAX";
        case HIPTENSOR_OP_MIN:
            return "HIPTENSOR_OP_MIN";
        default:
            return "HIPTENSOR_OP_UNKNOWN";
        }
    }

    std::string algoTypeToString(hiptensorAlgo_t algoType)
    {
        switch(algoType)
        {
        case HIPTENSOR_ALGO_ACTOR_CRITIC:
            return "HIPTENSOR_ALGO_ACTOR_CRITIC";
        case HIPTENSOR_ALGO_DEFAULT:
            return "HIPTENSOR_ALGO_DEFAULT";
        case HIPTENSOR_ALGO_DEFAULT_PATIENT:
            return "HIPTENSOR_ALGO_DEFAULT_PATIENT";
        default:
            return "HIPTENSOR_ALGO_UNKNOWN";
        }
    }

    std::string logLevelToString(hiptensorLogLevel_t logLevel)
    {
        switch(logLevel)
        {
        case HIPTENSOR_LOG_LEVEL_OFF:
            return "HIPTENSOR_LOG_LEVEL_OFF";
        case HIPTENSOR_LOG_LEVEL_ERROR:
            return "HIPTENSOR_LOG_LEVEL_ERROR";
        case HIPTENSOR_LOG_LEVEL_PERF_TRACE:
            return "HIPTENSOR_LOG_LEVEL_PERF_TRACE";
        case HIPTENSOR_LOG_LEVEL_PERF_HINT:
            return "HIPTENSOR_LOG_LEVEL_PERF_HINT";
        case HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE:
            return "HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE";
        case HIPTENSOR_LOG_LEVEL_API_TRACE:
            return "HIPTENSOR_LOG_LEVEL_API_TRACE";
        default:
            return "HIPTENSOR_LOG_LEVEL_UNKNOWN";
        }
    }

    std::string workSizePrefToString(hiptensorWorksizePreference_t workSize)
    {
        switch(workSize)
        {
        case HIPTENSOR_WORKSPACE_MIN:
            return "HIPTENSOR_WORKSPACE_MIN";
        case HIPTENSOR_WORKSPACE_DEFAULT:
            return "HIPTENSOR_WORKSPACE_DEFAULT";
        case HIPTENSOR_WORKSPACE_MAX:
            return "HIPTENSOR_WORKSPACE_MAX";
        default:
            return "HIPTENSOR_WORKSPACE_UNKNOWN";
        }
    }

} // namespace hiptensor

bool operator==(hiptensorDataType_t hipType, hiptensorComputeDescriptor_t computeType)
{
    if(hipType == HIPTENSOR_R_16BF)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_16BF);
    }
    else if(hipType == HIPTENSOR_R_16F)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_16F);
    }
    else if(hipType == HIPTENSOR_R_32F || hipType == HIPTENSOR_C_32F)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_32F);
    }
    else if(hipType == HIPTENSOR_R_64F || hipType == HIPTENSOR_C_64F)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_64F);
    }
    else if(hipType == HIPTENSOR_R_8I)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_8I);
    }
    else if(hipType == HIPTENSOR_R_8U)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_8U);
    }
    else if(hipType == HIPTENSOR_R_16I)
    {
        return false;
    }
    else if(hipType == HIPTENSOR_R_16U)
    {
        return false;
    }
    else if(hipType == HIPTENSOR_R_32I)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_32I);
    }
    else if(hipType == HIPTENSOR_R_32U)
    {
        return (computeType == HIPTENSOR_COMPUTE_DESC_32U);
    }
    else if(hipType == HIPTENSOR_R_64I)
    {
        return false;
    }
    else if(hipType == HIPTENSOR_R_64U)
    {
        return false;
    }
    else
    {
#if !NDEBUG
        std::cout << "Unhandled hip datatype: " << hipType << std::endl;
#endif // !NDEBUG
        return false;
    }
}

bool operator==(hiptensorComputeDescriptor_t computeType, hiptensorDataType_t hipType)
{
    return hipType == computeType;
}

bool operator!=(hiptensorDataType_t hipType, hiptensorComputeDescriptor_t computeType)
{
    return !(hipType == computeType);
}

bool operator!=(hiptensorComputeDescriptor_t computeType, hiptensorDataType_t hipType)
{
    return !(computeType == hipType);
}

namespace std
{
    std::string to_string(const hiptensor::ScalarData& value)
    {
        if(value.mType == HIPTENSOR_COMPUTE_DESC_C32F || value.mType == HIPTENSOR_COMPUTE_DESC_C64F)
        {
            return string() + "[" + to_string(value.mComplex.x) + ", " + to_string(value.mComplex.y)
                   + "]";
        }
        else
        {
            return to_string(value.mReal);
        }
    }
}
