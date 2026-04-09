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

#pragma once

// clang-format off
// Include order needs to be preserved
#include <memory>
#include <optional>
#include <vector>

#include <hip/library_types.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
// clang-format on

#include <hiptensor/hiptensor.h>
#include <hiptensor/hiptensor_types.h>

#include "hip_device.hpp"
#include "plan_cache.hpp"

typedef enum hiptensorOperationType_t
{
    HIPTENSOR_CONTRACTION         = 0,
    HIPTENSOR_PERMUTATION         = 1,
    HIPTENSOR_ELEMENTWISE_BINARY  = 2,
    HIPTENSOR_ELEMENTWISE_TRINARY = 3,
    HIPTENSOR_REDUCTION           = 4,
} hiptensorOperationType_t;

struct hiptensorOperationDescriptor
{
    uint32_t            mTag;
    hiptensorDataType_t mScalarType;
    float               mFlops;
    float               mMovedBytes;
    uint32_t            mPaddingLeft;
    uint32_t            mPaddingRighT;
    void*               mPaddingValue;

    hiptensorOperationType_t    mOperationType;
    int32_t                     mContractionOpId;
    hiptensorTensorDescriptor_t mDescA;
    std::vector<int32_t>        mModeA;
    hiptensorOperator_t         mOpA;

    hiptensorTensorDescriptor_t mDescB;
    std::vector<int32_t>        mModeB;
    hiptensorOperator_t         mOpB;

    hiptensorTensorDescriptor_t mDescC;
    std::vector<int32_t>        mModeC;
    hiptensorOperator_t         mOpC;

    hiptensorTensorDescriptor_t mDescD;
    std::vector<int32_t>        mModeD;

    hiptensorOperator_t mOpAC;
    hiptensorOperator_t mOpABC;

    hiptensorComputeDescriptor_t mDescCompute;
};

//! @brief hipTensor's library context
struct hiptensorHandle
{
    hiptensor::HipDevice getDevice()
    {
        return mDevice;
    }
    hiptensor::PlanCache* getPlanCache()
    {
        return planCache;
    }
    void setPlanCache(hiptensor::PlanCache* pt_PlanCache)
    {
        planCache = pt_PlanCache;
    }

private:
    hiptensor::HipDevice  mDevice;
    hiptensor::PlanCache* planCache = nullptr;
};

struct hiptensorPlan
{
    uint64_t                       mRequiredWorkspace;
    hiptensorOperationDescriptor_t mOpDesc;
    hiptensorPlanPreference_t      mPref;

    // Owned deep copies so the plan is self-contained and does not hold
    // dangling pointers when the caller destroys the original objects.
    std::unique_ptr<hiptensorTensorDescriptor>    mOwnedDescA;
    std::unique_ptr<hiptensorTensorDescriptor>    mOwnedDescB;
    std::unique_ptr<hiptensorTensorDescriptor>    mOwnedDescC;
    std::unique_ptr<hiptensorTensorDescriptor>    mOwnedDescD;
    std::unique_ptr<hiptensorOperationDescriptor> mOwnedOpDesc;
    std::unique_ptr<hiptensorPlanPreference>      mOwnedPref;
};

struct hiptensorPlanPreference
{
    hiptensorAutotuneMode_t mAutotuneMode;
    hiptensorCacheMode_t    mCacheMode;
    int32_t                 mIncrementalCount;
    int32_t                 mKernelRank;
    hiptensorJitMode_t      mJit;

    hiptensorAlgo_t mSelectionAlgorithm;
    //! A vector of the solver candidates
    std::vector<void*> mCandidates;
    void*              mSolution;
};

//! @brief Structure representing a tensor descriptor
//!
//! Represents a descriptor for the tensor with the given properties of
//! data type, lengths, strides and element-wise unary operation.
//! Constructed with hiptensorInitTensorDescriptor() function.
struct hiptensorTensorDescriptor
{
    //! Data type of the tensors enum selection
    hiptensorDataType_t mType;
    //! Lengths of the tensor
    std::vector<std::size_t> mLengths;
    //! Strides of the tensor
    std::vector<std::size_t> mStrides;
    uint32_t                 mAlignmentRequirement;
};

bool inline operator==(const hiptensorTensorDescriptor& lhs, const hiptensorTensorDescriptor& rhs)
{
    return lhs.mType == rhs.mType && lhs.mLengths == rhs.mLengths && lhs.mStrides == rhs.mStrides
           && lhs.mAlignmentRequirement == rhs.mAlignmentRequirement;
}

namespace hiptensor
{
    using index_t = int32_t;
    // Used to map to empty tensors
    struct NoneType;

    struct ScalarData
    {
        hiptensorComputeDescriptor_t mType;
        union
        {
            double           mReal;
            hipDoubleComplex mComplex;
        };

        ScalarData() = default;
        ScalarData(hiptensorComputeDescriptor_t type, double real, double imag = 0)
        {
            mType = type;
            if(type == HIPTENSOR_COMPUTE_DESC_C32F || type == HIPTENSOR_COMPUTE_DESC_C64F)
            {
                mComplex = make_hipDoubleComplex(real, imag);
            }
            else
            {
                mReal = real;
            }
        }
        operator float() const
        {
            return static_cast<float>(mReal);
        }
        operator hipDoubleComplex() const
        {
            return mComplex;
        }
    };

    static constexpr hiptensorDataType_t NONE_TYPE = (hiptensorDataType_t)31;

    // Map type to runtime HipTensorDataType
    template <typename T>
    struct HipTensorDataType;

    template <typename T>
    static constexpr auto HipTensorDataType_v = HipTensorDataType<T>::value;

    // Get data size in bytes from id
    HIPTENSOR_EXPORT uint32_t hiptensorDataTypeSize(hiptensorDataType_t id);

    // Convert hiptensorDataType_t to hiptensorComputeDescriptor_t
    HIPTENSOR_EXPORT hiptensorComputeDescriptor_t convertToComputeType(hiptensorDataType_t hipType);
    HIPTENSOR_EXPORT                              std::optional<hiptensorDataType_t>
        convertToHipTensorDataType(hiptensorComputeDescriptor_t computeType);

    // Read a single value from void pointer, casted to T
    template <typename T>
    T readVal(void const* value, hiptensorDataType_t id);

    template <typename T>
    T readVal(void const* value, hiptensorComputeDescriptor_t id);

    HIPTENSOR_EXPORT void
        writeVal(void const* addr, hiptensorComputeDescriptor_t id, ScalarData value);

    HIPTENSOR_EXPORT std::string computeTypeToString(hiptensorComputeDescriptor_t computeType);
    HIPTENSOR_EXPORT std::string hipTypeToString(hiptensorDataType_t hipType);
    HIPTENSOR_EXPORT std::string opTypeToString(hiptensorOperator_t opType);
    HIPTENSOR_EXPORT std::string algoTypeToString(hiptensorAlgo_t algoType);
    HIPTENSOR_EXPORT std::string logLevelToString(hiptensorLogLevel_t);
    HIPTENSOR_EXPORT std::string workSizePrefToString(hiptensorWorksizePreference_t workSize);
} // namespace hiptensor

bool operator==(hiptensorDataType_t hipType, hiptensorComputeDescriptor_t computeType);
bool operator==(hiptensorComputeDescriptor_t computeType, hiptensorDataType_t hipType);

bool operator!=(hiptensorDataType_t hipType, hiptensorComputeDescriptor_t computeType);
bool operator!=(hiptensorComputeDescriptor_t computeType, hiptensorDataType_t hipType);

namespace std
{
    std::string to_string(const hiptensor::ScalarData& value);
}

#include "data_types_impl.hpp"
